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


RELEASE_REQUIRED_TESTS = [
    "HashTopkTest.test_shapes",
    "HashTopkTest.test_softplus_threshold",
]


if __name__ == "__main__":
    unittest.main()
