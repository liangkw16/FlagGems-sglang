# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("compute_src2dst")


def reference(reorder_ids, num_toks):
    out = torch.empty(num_toks, dtype=torch.int32, device=reorder_ids.device)
    out[reorder_ids.long()] = torch.arange(
        num_toks, dtype=torch.int32, device=reorder_ids.device
    )
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class ComputeSrc2DstTest(unittest.TestCase):
    def check(self, ids):
        before = ids.clone()
        expected = reference(ids, ids.numel())
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.compute_src2dst(ids, ids.numel())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(ids, before, rtol=0, atol=0)

    def test_permutations_and_boundaries(self):
        torch.manual_seed(61)
        for n in (0, 1, 255, 256, 257, 1023, 1024, 1025, 65537):
            with self.subTest(n=n):
                self.check(torch.randperm(n, device="cuda"))

    def test_route_sort_inverse(self):
        experts = torch.tensor([[3, 1], [0, 3], [1, 0]], device="cuda")
        self.check(experts.flatten().argsort(stable=True))
        for n in (1, 257, 4096):
            self.check(torch.arange(n, device="cuda"))
            self.check(torch.arange(n - 1, -1, -1, device="cuda"))

    def test_strided_and_index_dtypes(self):
        for dtype in (torch.int32, torch.int64):
            storage = torch.empty(515, dtype=dtype, device="cuda")
            ids = storage[1::2]
            ids.copy_(torch.randperm(ids.numel(), device="cuda"))
            self.check(ids)
            ids.copy_(torch.arange(ids.numel(), device="cuda"))
            self.check(ids)


RELEASE_REQUIRED_TESTS = [
    "ComputeSrc2DstTest.test_permutations_and_boundaries",
    "ComputeSrc2DstTest.test_route_sort_inverse",
    "ComputeSrc2DstTest.test_strided_and_index_dtypes",
]


if __name__ == "__main__":
    unittest.main()
