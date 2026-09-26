# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("tiny_n_gemm")


def reference(x, w, out_dtype):
    return (x.float() @ w.float().t()).to(out_dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class TinyNGemmTest(unittest.TestCase):
    def check(self, m, k, n, out_dtype):
        gen = torch.Generator(device="cuda").manual_seed(m * 1000 + k)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.5
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.5
        expected = reference(x, w, out_dtype)
        for name, module in MODULES:
            with self.subTest(module=name, m=m, k=k, n=n):
                actual = module.tiny_n_gemm(x, w, out_dtype)
                self.assertEqual(actual.dtype, out_dtype)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_decode_shapes(self):
        for m, k, n in ((1, 64, 128), (8, 512, 256), (16, 1024, 512), (5, 333, 65), (16, 64, 32)):
            for out_dtype in (torch.bfloat16, torch.float32):
                with self.subTest(m=m, k=k, n=n, dt=out_dtype):
                    self.check(m, k, n, out_dtype)

    def test_strided_k_and_k0(self):
        gen = torch.Generator(device="cuda").manual_seed(21)
        x = torch.randn(64, 2, dtype=torch.bfloat16, device="cuda", generator=gen).t()  # [2,64] col-major
        w = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda", generator=gen)
        expected = reference(x, w, torch.float32)
        for name, module in MODULES:
            with self.subTest(module=name):
                torch.testing.assert_close(module.tiny_n_gemm(x, w, torch.float32), expected, rtol=2e-2, atol=2e-2)
        # K=0 -> zero output (review finding: uninitialized empty returned)
        x0 = torch.empty(2, 0, dtype=torch.bfloat16, device="cuda")
        w0 = torch.empty(32, 0, dtype=torch.bfloat16, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name, case="k0"):
                out = module.tiny_n_gemm(x0, w0, torch.float32)
                torch.testing.assert_close(out, torch.zeros(2, 32, device="cuda"), rtol=0, atol=0)

    def test_m1_and_m16_bounds(self):
        self.check(1, 128, 64, torch.bfloat16)
        self.check(16, 128, 64, torch.float32)
        self.check(16, 4096, 128, torch.bfloat16)


RELEASE_REQUIRED_TESTS = [
    "TinyNGemmTest.test_decode_shapes",
    "TinyNGemmTest.test_strided_k_and_k0",
    "TinyNGemmTest.test_m1_and_m16_bounds",
]

if __name__ == "__main__":
    unittest.main()
