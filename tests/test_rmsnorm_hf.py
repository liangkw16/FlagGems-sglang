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
        # E1 exact-subblock path: hidden > 1024 with 256 | hidden splits
        # the row into D_TILE = min(hidden & -hidden, 1024) exact tiles
        # (D_TILE 256/512/1024 and N_SUB 2..8 all covered), two passes,
        # zero masks on the hot path.
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
        # hidden <= 1024 or hidden % 256 != 0 keeps the whole-row path
        # (the platform 333 probe stays there); both sides of the gate
        # must match the reference.
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
        # column stride != 1 routed through the exact-subblock path
        # (hidden 2048, x_s1 = w_s0 = 2).
        big = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
        x = big[:, ::2]
        w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")[::2]
        self.check(x, w)


RELEASE_REQUIRED_TESTS = [
    "RmsnormHfTest.test_shapes_and_dtypes",
    "RmsnormHfTest.test_strided_input",
    "RmsnormHfTest.test_wider_weight_dtype_promotes",
    "RmsnormHfTest.test_eps_and_zero_row",
    "RmsnormHfTest.test_exact_subblock_large_hidden",
    "RmsnormHfTest.test_subblock_gate_boundary",
    "RmsnormHfTest.test_strided_input_large_hidden_subblock",
]

if __name__ == "__main__":
    unittest.main()
