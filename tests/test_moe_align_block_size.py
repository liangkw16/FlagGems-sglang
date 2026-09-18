# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("moe_align_block_size")


def reference(topk_ids, num_experts, block_size, sorted_token_ids,
              expert_ids, num_tokens_post_pad, cumsum_buffer, pad_flag):
    num_routed = num_experts - 1
    flat = topk_ids.flatten()
    numel = flat.numel()
    sorted_ids = torch.full_like(sorted_token_ids, numel)
    eids = expert_ids.clone()
    offset = 0
    for e in range(num_routed):
        idx = (flat == e).nonzero().flatten()
        n = idx.numel()
        aligned = ((n + block_size - 1) // block_size) * block_size
        if n:
            sorted_ids[offset:offset + n] = idx.to(sorted_ids.dtype)
        nblocks = aligned // block_size
        if nblocks:
            beg = offset // block_size
            eids[beg:beg + nblocks] = e
        offset += aligned
    npost = torch.full_like(num_tokens_post_pad, offset)
    return sorted_ids, eids, npost


def make_case(tokens=128, topk=8, num_experts=33, block_size=16):
    ids = torch.randint(
        0, num_experts - 1, (tokens, topk), dtype=torch.int32,
        device="cuda",
    )
    buf = tokens * topk + block_size * num_experts
    return (
        ids,
        num_experts,
        block_size,
        torch.empty(buf, dtype=torch.int32, device="cuda"),
        torch.empty(
            buf // block_size, dtype=torch.int32, device="cuda"
        ),
        torch.empty(1, dtype=torch.int32, device="cuda"),
        torch.empty(num_experts + 1, dtype=torch.int32, device="cuda"),
        True,
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class MoeAlignBlockSizeTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        for name, module in MODULES:
            with self.subTest(module=name):
                got = module.moe_align_block_size(*args)
                self.assertEqual(
                    int(got[2].item()), int(expected[2].item())
                )
                npost = int(expected[2].item())
                # sorted prefix: per-expert multiset + sentinel tail
                torch.testing.assert_close(
                    got[0][:npost].sort().values.int(),
                    expected[0][:npost].sort().values.int(),
                    rtol=0, atol=0,
                )
                tail = got[0].numel() - npost
                if tail:
                    self.assertTrue(
                        (got[0][npost:] == args[0].numel()).all()
                    )
                torch.testing.assert_close(
                    got[1], expected[1], rtol=0, atol=0
                )

    def test_shapes_and_expert_grid(self):
        for tokens, topk, experts, bs in (
            (1, 1, 3, 4),
            (7, 8, 17, 16),
            (128, 8, 33, 16),
            (513, 4, 257, 32),
        ):
            with self.subTest(shape=(tokens, topk, experts, bs)):
                self.check(make_case(tokens, topk, experts, bs))

    def test_filtered_expert_bucket(self):
        # ids equal to the filtered slot never occupy blocks
        args = make_case(tokens=32, topk=4, num_experts=9, block_size=8)
        ids = args[0].clone()
        ids[:, 0] = 8
        ids[:, 1:] = 0
        self.check((ids,) + args[1:])


RELEASE_REQUIRED_TESTS = [
    "MoeAlignBlockSizeTest.test_shapes_and_expert_grid",
    "MoeAlignBlockSizeTest.test_filtered_expert_bucket",
]


if __name__ == "__main__":
    unittest.main()
