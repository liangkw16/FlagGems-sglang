# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("rmsnorm_hf")


def reference(input, weight, eps):
    x = input.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * y.to(input.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class RmsnormHfTest(unittest.TestCase):
    def check(self, x, weight, eps=1e-6):
        expected = reference(x, weight, eps)
        for name, module in MODULES:
            with self.subTest(module=name, dtype=x.dtype):
                actual = module.rmsnorm_hf(x, weight, eps)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)

    def test_shapes_and_dtypes(self):
        for dtype in (torch.bfloat16, torch.float16):
            for rows, hidden in ((1, 64), (3, 128), (128, 512), (17, 333), (4096, 1024)):
                with self.subTest(dtype=dtype, rows=rows, hidden=hidden):
                    x = torch.randn(rows, hidden, dtype=dtype, device="cuda") * 3
                    w = torch.randn(hidden, dtype=dtype, device="cuda")
                    self.check(x, w)

    def test_strided_input(self):
        big = torch.randn(64, 1024, dtype=torch.bfloat16, device="cuda")
        x = big[:, ::2]
        w = torch.randn(512, dtype=torch.bfloat16, device="cuda")
        self.check(x, w)

    def test_wider_weight_dtype_promotes(self):
        # weight * y follows torch promotion: an fp32 weight widens the
        # output beyond the bf16 input (review finding)
        x = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(256, dtype=torch.float32, device="cuda")
        expected_dtype = torch.promote_types(w.dtype, x.dtype)
        for name, module in MODULES:
            with self.subTest(module=name):
                out = module.rmsnorm_hf(x, w, 1e-6)
                self.assertEqual(out.dtype, expected_dtype)

    def test_eps_and_zero_row(self):
        x = torch.zeros(4, 128, dtype=torch.bfloat16, device="cuda")
        x[0, 0] = 1.0
        w = torch.ones(128, dtype=torch.bfloat16, device="cuda")
        for eps in (1e-6, 1e-2):
            self.check(x, w, eps)

    def test_exact_subblock_large_hidden(self):
        # E1 shape set (the subblock two-pass itself was rolled back;
        # after the rollback every shape runs the whole-row kernel, so
        # these pin large-hidden correctness on the current dispatch:
        # 2048/4096/8192 compile the EXACT naked kernel, 1280/1536/
        # 3072/5120 run the fp32-compare masked arm).
        for dtype in (torch.bfloat16, torch.float16):
            for rows, hidden in (
                (1, 1280),
                (4, 1536),
                (3, 2048),
                (4, 3072),
                (2, 4096),
                (5, 5120),
                (3, 8192),
                (33, 2048),
            ):
                with self.subTest(dtype=dtype, rows=rows, hidden=hidden):
                    x = (
                        torch.randn(rows, hidden, dtype=dtype, device="cuda")
                        * 3
                    )
                    w = torch.randn(hidden, dtype=dtype, device="cuda")
                    self.check(x, w)

    def test_subblock_gate_boundary(self):
        # ragged-hidden shapes around the old E1 gate (the 333 platform
        # probe among them): every one of them now runs the masked
        # whole-row arm with the fp32 compare and the other-less weight
        # load, and must still match the reference.
        for dtype in (torch.bfloat16, torch.float16):
            for rows, hidden in (
                (8, 1024),
                (7, 1025),
                (6, 1200),
                (5, 2176),
                (4, 333),
            ):
                with self.subTest(dtype=dtype, rows=rows, hidden=hidden):
                    x = (
                        torch.randn(rows, hidden, dtype=dtype, device="cuda")
                        * 3
                    )
                    w = torch.randn(hidden, dtype=dtype, device="cuda")
                    self.check(x, w)

    def test_strided_input_large_hidden_subblock(self):
        # column stride != 1 on a large hidden (2048, x_s1 = w_s0 = 2)
        # - now the EXACT naked arm walked with a stride of 2.
        big = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
        x = big[:, ::2]
        w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")[::2]
        self.check(x, w)

    def test_exact_pow2_unmasked_whole_row(self):
        # E2 exact-unmask-whole-row: every pow2 hidden compiles the
        # EXACT constexpr arm - zero bounds mask, zero `other` prefill,
        # zero per-lane compare. All tiers the platform exercises,
        # including the smallest EXACT shape (hidden 16 == BLOCK 16).
        for dtype in (torch.bfloat16, torch.float16):
            for rows, hidden in (
                (17, 16),
                (3, 64),
                (9, 128),
                (5, 256),
                (128, 512),
                (4096, 1024),
                (3, 2048),
                (2, 4096),
                (3, 8192),
            ):
                with self.subTest(dtype=dtype, rows=rows, hidden=hidden):
                    x = (
                        torch.randn(rows, hidden, dtype=dtype, device="cuda")
                        * 3
                    )
                    w = torch.randn(hidden, dtype=dtype, device="cuda")
                    self.check(x, w)

    def test_exact_block_minus_one_boundary(self):
        # hidden == BLOCK (EXACT naked arm) against hidden == BLOCK-1
        # (masked arm with a single masked lane per row), plus the
        # hidden < 16 shapes where block = max(16, next_pow2(hidden))
        # floors at 16 and the mask covers half the block or more.
        for dtype in (torch.bfloat16, torch.float16):
            for rows, hidden in (
                (8, 15),
                (8, 16),
                (5, 63),
                (5, 64),
                (6, 511),
                (6, 512),
                (4, 1023),
                (4, 1024),
                (2, 2047),
                (2, 2048),
                (2, 7),
                (2, 8),
            ):
                with self.subTest(dtype=dtype, rows=rows, hidden=hidden):
                    x = (
                        torch.randn(rows, hidden, dtype=dtype, device="cuda")
                        * 3
                    )
                    w = torch.randn(hidden, dtype=dtype, device="cuda")
                    self.check(x, w)

    def test_strided_exact_pow2(self):
        # strided forms that stay on the EXACT arm: column stride 2
        # (hidden 2048 pow2, x_s1 = w_s0 = 2) and a doubled row stride
        # (hidden 1024, x_s0 = 2 * hidden) - both must load/store every
        # lane unmasked without walking off the row.
        big = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
        x = big[:, ::2]
        w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")[::2]
        self.check(x, w)

        wide = torch.randn(16, 1024, dtype=torch.bfloat16, device="cuda")
        x_rows = wide[::2, :]
        w_rows = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
        self.check(x_rows, w_rows)


RELEASE_REQUIRED_TESTS = [
    "RmsnormHfTest.test_shapes_and_dtypes",
    "RmsnormHfTest.test_strided_input",
    "RmsnormHfTest.test_wider_weight_dtype_promotes",
    "RmsnormHfTest.test_eps_and_zero_row",
    "RmsnormHfTest.test_exact_subblock_large_hidden",
    "RmsnormHfTest.test_subblock_gate_boundary",
    "RmsnormHfTest.test_strided_input_large_hidden_subblock",
    "RmsnormHfTest.test_exact_pow2_unmasked_whole_row",
    "RmsnormHfTest.test_exact_block_minus_one_boundary",
    "RmsnormHfTest.test_strided_exact_pow2",
]

if __name__ == "__main__":
    unittest.main()
