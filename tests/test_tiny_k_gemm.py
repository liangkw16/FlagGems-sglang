# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("tiny_k_gemm")


def reference(x, w, out_dtype):
    return (x.float() @ w.float().t()).to(out_dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class TKGTest(unittest.TestCase):
    def check(self, x, w, od):
        want = reference(x, w, od)
        for name, module in MODULES:
            with self.subTest(module=name):
                got = module.tiny_k_gemm(x, w, od)
                torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)

    def test_matrix(self):
        for m in (1, 7, 16):
            for k in (128, 256):
                for n in (64, 1000, 4096):
                    with self.subTest(m=m, k=k, n=n):
                        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
                        self.check(x, w, torch.bfloat16)
                        self.check(x, w, torch.float32)

    def test_w_natural_layout_strides(self):
        # e9 enflame semantics regression: the w tile is loaded in w's
        # natural [BLOCK_N, K] row-major layout and transposed in-register
        # (tl.trans) before tl.dot. Lock that morphology with row-padded
        # (non-contiguous) x/w whose row strides (xs0/ws0) differ from k,
        # plus the ragged n tail that pins the mask orientation of the
        # natural-layout load (n % BLOCK_N != 0 masks rows, not columns).
        for m in (1, 7, 16):
            for k in (128, 256):
                for n in (64, 1000):
                    with self.subTest(m=m, k=k, n=n):
                        x = torch.randn(m, k + 16, dtype=torch.bfloat16, device="cuda")[:, :k]
                        w = torch.randn(n, k + 32, dtype=torch.bfloat16, device="cuda")[:, :k]
                        self.check(x, w, torch.bfloat16)
                        self.check(x, w, torch.float32)


RELEASE_REQUIRED_TESTS = [
    "TKGTest.test_matrix",
    "TKGTest.test_w_natural_layout_strides",
]


if __name__ == "__main__":
    unittest.main()
