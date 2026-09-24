# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("concat_mla_absorb_q")


def reference(a, b):
    return torch.cat([a, b], dim=-1)


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class ConcatMlaAbsorbQTest(unittest.TestCase):
    def check(self, a, b):
        expected = reference(a, b)
        snapshots = [a.clone(), b.clone()]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.concat_mla_absorb_q(a, b)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(actual.is_contiguous())
                torch.testing.assert_close(
                    bits(actual), bits(expected), rtol=0, atol=0
                )
                for value, before in zip((a, b), snapshots):
                    torch.testing.assert_close(
                        bits(value), bits(before), rtol=0, atol=0
                    )

    def make_case(self, d0, d1, a_last, b_last, *, stride=False):
        if stride:
            a_big = torch.randn(d0, d1, a_last + 8, dtype=torch.bfloat16, device="cuda")
            a = a_big[..., :a_last]
            b_big = torch.randn(d0, d1 * 2, b_last, dtype=torch.bfloat16, device="cuda")
            b = b_big[:, ::2, :]
        else:
            a = torch.randn(d0, d1, a_last, dtype=torch.bfloat16, device="cuda")
            b = torch.randn(d0, d1, b_last, dtype=torch.bfloat16, device="cuda")
        return a, b

    def test_mla_shapes_and_tails(self):
        cases = [
            (1, 1, 512, 64),
            (4, 16, 192, 128),
            (2, 128, 576, 512),
            (7, 33, 129, 65),
            (3, 8, 1, 1),
            (2, 4, 511, 513),
        ]
        for d0, d1, a_last, b_last in cases:
            with self.subTest(d0=d0, d1=d1, a_last=a_last, b_last=b_last):
                self.check(*self.make_case(d0, d1, a_last, b_last))

    def test_strided_sources(self):
        # the contract calls out two differently strided source rows
        cases = [(2, 16, 512, 64), (4, 33, 192, 96)]
        for d0, d1, a_last, b_last in cases:
            with self.subTest(d0=d0, d1=d1):
                self.check(*self.make_case(d0, d1, a_last, b_last, stride=True))

    def test_fp16_dtype(self):
        a = torch.randn(2, 8, 128, dtype=torch.float16, device="cuda")
        b = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
        self.check(a, b)

    def test_innermost_stride_slice(self):
        # a sliced last dim carries stride(2)=2: the kernel must scale
        # column offsets by the runtime innermost strides
        a = torch.arange(12, device="cuda", dtype=torch.bfloat16).reshape(1, 1, 12)[
            ..., ::2
        ]
        b = torch.zeros(1, 1, 1, dtype=torch.bfloat16, device="cuda")
        self.assertEqual(a.stride(2), 2)
        self.check(a, b)
        a2 = torch.randn(2, 5, 96, dtype=torch.bfloat16, device="cuda")
        b2 = torch.randn(2, 5, 192 * 2, dtype=torch.bfloat16, device="cuda")[..., ::2]
        self.check(a2, b2)

    def test_wide_row_beyond_block_cap(self):
        # vendor whole-row vectors are capped at 65536 lanes; a wider row
        # must fall through the chunked column loop (review finding)
        a = torch.randn(1, 2, 70000, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(1, 2, 70000, dtype=torch.bfloat16, device="cuda")
        self.check(a, b)

    def test_huge_rows_ascend_grid_cap(self):
        # 300000 rows would launch 75000 programs at BLOCK_R=4, past the
        # Ascend coreDim cap of 65535; the wrapper must grow the row tile
        # (platform case 8 failure, EE1003 kernel-launch invalid coreDim)
        d0, d1 = 6000, 50
        self.assertEqual(d0 * d1, 300000)
        a = torch.randn(d0, d1, 8, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(d0, d1, 8, dtype=torch.bfloat16, device="cuda")
        self.check(a, b)

    def test_empty_rows(self):
        a = torch.randn(0, 8, 128, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(0, 8, 64, dtype=torch.bfloat16, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                out = module.concat_mla_absorb_q(a, b)
                self.assertEqual(out.shape, (0, 8, 192))


RELEASE_REQUIRED_TESTS = [
    "ConcatMlaAbsorbQTest.test_mla_shapes_and_tails",
    "ConcatMlaAbsorbQTest.test_strided_sources",
    "ConcatMlaAbsorbQTest.test_fp16_dtype",
    "ConcatMlaAbsorbQTest.test_innermost_stride_slice",
    "ConcatMlaAbsorbQTest.test_wide_row_beyond_block_cap",
    "ConcatMlaAbsorbQTest.test_huge_rows_ascend_grid_cap",
    "ConcatMlaAbsorbQTest.test_empty_rows",
]

if __name__ == "__main__":
    unittest.main()
