# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("moe_align_single_token")


def reference(topk_ids, block_size):
    topk = topk_ids.shape[1]
    numel = topk_ids.numel()
    device = topk_ids.device
    sorted_ids = torch.full(
        (topk * block_size,), numel, dtype=torch.int32, device=device
    )
    experts = torch.sort(topk_ids.flatten()).values.to(torch.int32)
    order = torch.argsort(topk_ids.flatten())
    for slot in range(topk):
        sorted_ids[slot * block_size] = int(order[slot])
    num_post = torch.full(
        (1,), topk * block_size, dtype=torch.int32, device=device
    )
    return sorted_ids, experts, num_post


def make_case(topk=8, block_size=128, num_experts=256):
    ids = torch.randperm(num_experts)[:topk].to(torch.int32).cuda()
    return ids.reshape(1, topk), block_size


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class MoeAlignSingleTokenTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.moe_align_single_token(*args)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(
                        got, want, rtol=0, atol=0
                    )

    def test_topk_and_block_grid(self):
        for topk in (1, 2, 3, 7, 8, 9, 16):
            for block_size in (16, 64, 128):
                with self.subTest(topk=topk, block_size=block_size):
                    self.check(make_case(topk, block_size))


RELEASE_REQUIRED_TESTS = [
    "MoeAlignSingleTokenTest.test_topk_and_block_grid",
]


if __name__ == "__main__":
    unittest.main()
