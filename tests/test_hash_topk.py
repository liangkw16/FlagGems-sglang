# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
import torch.nn.functional as F

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("hash_topk")


def reference(router_logits, input_ids, tid2eid, shared, scale, func):
    assert func == "sqrtsoftplus"
    num_tokens, num_routed = router_logits.shape
    expert_ids = tid2eid[input_ids.long()].long()
    logits = torch.gather(router_logits.float(), 1, expert_ids)
    w = torch.sqrt(F.softplus(logits))
    w = w / w.sum(dim=-1, keepdim=True)
    shared_w = torch.full(
        (num_tokens, shared), 1.0 / scale,
        dtype=torch.float32, device=router_logits.device,
    )
    shared_ids = (
        num_routed
        + torch.arange(shared, device=router_logits.device)
    ).expand(num_tokens, -1)
    weights = torch.cat([w, shared_w], dim=-1).float()
    ids = torch.cat(
        [expert_ids.to(torch.int32), shared_ids.to(torch.int32)], dim=-1
    )
    return weights.contiguous(), ids.contiguous()


def make_case(tokens=33, routed=64, topk=8, vocab=512, shared=2):
    logits = torch.randn(
        tokens, routed, dtype=torch.bfloat16, device="cuda"
    )
    ids = torch.randint(
        0, vocab, (tokens,), dtype=torch.int64, device="cuda"
    )
    table = torch.randint(
        0, routed, (vocab, topk), dtype=torch.int32, device="cuda"
    )
    return logits, ids, table, shared


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class HashTopkTest(unittest.TestCase):
    def check(self, args, scale=2.5):
        expected = reference(*args, scale, "sqrtsoftplus")
        for name, module in MODULES:
            with self.subTest(module=name):
                w, i = module.hash_topk(*args, scale, "sqrtsoftplus")
                torch.testing.assert_close(
                    w, expected[0], rtol=2e-2, atol=2e-2
                )
                torch.testing.assert_close(
                    i, expected[1], rtol=0, atol=0
                )

    def test_shapes(self):
        for tokens, routed, topk, shared in (
            (1, 64, 8, 2),
            (33, 64, 8, 2),
            (513, 256, 16, 4),
            (7, 32, 3, 1),
            (2049, 256, 4, 8),
        ):
            with self.subTest(shape=(tokens, routed, topk, shared)):
                self.check(make_case(tokens, routed, topk, shared, shared))

    def test_softplus_threshold(self):
        # logits above F.softplus's threshold-20 identity branch
        args = make_case(tokens=16, routed=64, topk=8, shared=2)
        logits = args[0].float()
        logits[:, :] = torch.linspace(21.0, 30.0, 64, device="cuda")
        args = (logits.to(torch.bfloat16),) + args[1:]
        self.check(args)

    def test_duplicate_and_unsorted_experts(self):
        # the in-kernel one-hot match-reduce (enflame e8) must behave
        # exactly like the tensor-index gather when a table row repeats
        # an expert or lists ids out of order: slot order is preserved
        # and both duplicate slots score the same logit.
        args = make_case(tokens=33, routed=64, topk=8, vocab=128, shared=2)
        logits, ids, table, shared = args
        table[:, 0] = 7
        table[:, 1] = 7
        table[:, 2] = 63
        table[:, 3:] = torch.flip(table[:, 3:], dims=[1])
        self.check(args)

    def test_wide_routed_tiling(self):
        # num_routed above the 1024 NRTILE cap: a partial tail tile
        # (1536) and an exact two-full-tile row (2048), with expert ids
        # pinned on the tile boundary lanes.
        for routed in (1536, 2048):
            with self.subTest(routed=routed):
                logits = torch.randn(
                    9, routed, dtype=torch.bfloat16, device="cuda"
                )
                ids = torch.randint(
                    0, 97, (9,), dtype=torch.int64, device="cuda"
                )
                table = torch.randint(
                    0, routed, (97, 6), dtype=torch.int32, device="cuda"
                )
                table[0, 0] = 1023
                table[0, 1] = 1024
                table[0, 2] = routed - 1
                self.check((logits, ids, table, 2))

    def test_row_gapped_strided_inputs(self):
        # row-gapped layouts (stride(1)==1, stride(0)>width) must run
        # through the real constexpr/runtime stride values without a
        # dense copy (T90 e6 round-2 contract class).
        base_logits = torch.randn(
            66, 64, dtype=torch.bfloat16, device="cuda"
        )
        base_table = torch.randint(
            0, 64, (256, 8), dtype=torch.int32, device="cuda"
        )
        ids = torch.randint(
            0, 128, (33,), dtype=torch.int64, device="cuda"
        )
        self.check(
            (base_logits[::2], ids, base_table[::2], 2)
        )

    def test_int32_input_ids_and_zero_shared(self):
        # int32 token ids (no i64 low-half path) and the zero shared
        # experts edge (NSHARED pow2 pad fully masked, width==topk).
        logits = torch.randn(
            5, 32, dtype=torch.bfloat16, device="cuda"
        )
        ids = torch.randint(
            0, 64, (5,), dtype=torch.int64, device="cuda"
        ).to(torch.int32)
        table = torch.randint(
            0, 32, (64, 4), dtype=torch.int32, device="cuda"
        )
        self.check((logits, ids, table, 0))


RELEASE_REQUIRED_TESTS = [
    "HashTopkTest.test_shapes",
    "HashTopkTest.test_softplus_threshold",
    "HashTopkTest.test_duplicate_and_unsorted_experts",
    "HashTopkTest.test_wide_routed_tiling",
    "HashTopkTest.test_row_gapped_strided_inputs",
    "HashTopkTest.test_int32_input_ids_and_zero_shared",
]


if __name__ == "__main__":
    unittest.main()
