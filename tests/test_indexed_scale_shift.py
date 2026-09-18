# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("indexed_scale_shift")


def reference(x, shift, scale, indices):
    idx = indices.long()
    sh = shift[idx].float()
    sc = scale[idx].float()
    one_plus = (1.0 + sc).to(torch.bfloat16).float()
    scaled = (x.float() * one_plus).to(torch.bfloat16).float()
    return (scaled + sh).to(x.dtype)


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class IndexedScaleShiftTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.indexed_scale_shift(*args)
                torch.testing.assert_close(
                    bits(actual), bits(expected), rtol=0, atol=0
                )

    def test_shapes_and_variants(self):
        for rows, hdim, variants in (
            (1, 1, 1),
            (7, 1024, 5),
            (513, 5120, 256),
            (2049, 1023, 3),
        ):
            with self.subTest(shape=(rows, hdim, variants)):
                x = torch.randn(
                    rows, hdim, dtype=torch.bfloat16, device="cuda"
                )
                shift = torch.randn(
                    variants, hdim, dtype=torch.bfloat16, device="cuda"
                )
                scale = torch.randn(
                    variants, hdim, dtype=torch.bfloat16, device="cuda"
                ) * 0.1
                idx = torch.randint(
                    0, variants, (rows,), dtype=torch.int64, device="cuda"
                )
                self.check((x, shift, scale, idx))

    def test_double_round_boundary(self):
        # 1 + scale lands exactly between two bf16 neighbours: the
        # explicit intermediate round is the contract.
        x = torch.ones(16, dtype=torch.bfloat16, device="cuda")
        scale = torch.full(
            (1, 16), 2.0**-9, dtype=torch.bfloat16, device="cuda"
        )
        shift = torch.zeros(1, 16, dtype=torch.bfloat16, device="cuda")
        idx = torch.zeros(16, dtype=torch.int32, device="cuda")
        self.check((x, shift, scale, idx))

    def test_int32_indices(self):
        x = torch.randn(33, 2048, dtype=torch.bfloat16, device="cuda")
        shift = torch.randn(7, 2048, dtype=torch.bfloat16, device="cuda")
        scale = torch.randn(7, 2048, dtype=torch.bfloat16, device="cuda")
        idx = torch.randint(
            0, 7, (33,), dtype=torch.int32, device="cuda"
        )
        self.check((x, shift, scale, idx))


RELEASE_REQUIRED_TESTS = [
    "IndexedScaleShiftTest.test_shapes_and_variants",
    "IndexedScaleShiftTest.test_double_round_boundary",
    "IndexedScaleShiftTest.test_int32_indices",
]


if __name__ == "__main__":
    unittest.main()
