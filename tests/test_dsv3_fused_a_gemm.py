# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("dsv3_fused_a_gemm")

TOL = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
    torch.float32: (1e-4, 1e-4),
}


def reference(mat_a, mat_b):
    return (mat_a.float() @ mat_b.float()).to(mat_a.dtype)


def make_case(
    m=7,
    hd_in=768,
    hd_out=80,
    dtype=torch.bfloat16,
    strided=False,
):
    a = torch.randn(m, hd_in, dtype=dtype, device="cuda")
    b_row = torch.randn(hd_out, hd_in, dtype=dtype, device="cuda")
    b = b_row.t()
    if strided:
        storage = torch.randn(m * 2 + 1, hd_in, dtype=dtype, device="cuda")
        a = storage[1::2]
    return a, b


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class DSV3FusedAGemmTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() for x in args]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.dsv3_fused_a_gemm(*args)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=TOL[args[0].dtype][0],
                    atol=TOL[args[0].dtype][1],
                )
                for value, before in zip(args, snapshots):
                    torch.testing.assert_close(
                        value, before, rtol=0, atol=0, equal_nan=True
                    )

    def test_dtypes(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                self.check(make_case(dtype=dtype))

    def test_shapes(self):
        for m in (1, 2, 15, 16):
            with self.subTest(m=m):
                self.check(make_case(m=m))
        for hd_in in (256, 512, 1024, 2048):
            with self.subTest(hd_in=hd_in):
                self.check(make_case(hd_in=hd_in))
        for hd_out in (16, 64, 256, 2112):
            with self.subTest(hd_out=hd_out):
                self.check(make_case(hd_out=hd_out))

    def test_strided_a(self):
        self.check(make_case(strided=True))

    def test_split_k_boundary(self):
        for dtype in (torch.bfloat16, torch.float16):
            for m in (1, 16):
                for hd_in in (3840, 4096, 4352):
                    for hd_out in (496, 512, 528):
                        with self.subTest(
                            dtype=dtype, m=m, hd_in=hd_in, hd_out=hd_out
                        ):
                            self.check(
                                make_case(
                                    m=m,
                                    hd_in=hd_in,
                                    hd_out=hd_out,
                                    dtype=dtype,
                                )
                            )

    def test_split_k_strided(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                a, _ = make_case(
                    m=15,
                    hd_in=4352,
                    hd_out=80,
                    dtype=dtype,
                    strided=True,
                )
                # Keep the contracted contiguous K axis, with gaps between
                # A rows and between B columns; K does not divide four ways.
                storage = torch.randn(161, 4352, dtype=dtype, device="cuda")
                b = storage[1::2].t()
                self.check((a, b))

    def test_split_k_cancellation(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                a = torch.ones(16, 4352, dtype=dtype, device="cuda")
                b_row = torch.zeros(80, 4352, dtype=dtype, device="cuda")
                large = 256 if dtype == torch.bfloat16 else 2048
                # Split partials are [large+1, -large, -large, large+1].
                # Casting each partial before summation loses the result 2.
                b_row[:, 0] = large
                b_row[:, 1] = 1
                b_row[:, 1152] = -large
                b_row[:, 2304] = -large
                b_row[:, 3456] = large
                b_row[:, 3457] = 1
                b = b_row.t()
                expected = torch.full((16, 80), 2, dtype=dtype, device="cuda")
                torch.testing.assert_close(
                    reference(a, b), expected, rtol=0, atol=0
                )
                self.check((a, b))
                for name, module in MODULES:
                    with self.subTest(module=name):
                        first = module.dsv3_fused_a_gemm(a, b)
                        for _ in range(3):
                            actual = module.dsv3_fused_a_gemm(a, b)
                            self.assertTrue(
                                torch.equal(
                                    actual.contiguous().view(torch.uint8),
                                    first.contiguous().view(torch.uint8),
                                ),
                                "split reduction must be deterministic",
                            )

    def test_special_values(self):
        a, b = make_case(m=4, dtype=torch.float32)
        a[0, :4] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0], device="cuda"
        )
        expected = reference(a, b)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.dsv3_fused_a_gemm(a, b)
                torch.testing.assert_close(
                    actual, expected, rtol=1e-4, atol=1e-4, equal_nan=True
                )


RELEASE_REQUIRED_TESTS = [
    "DSV3FusedAGemmTest.test_dtypes",
    "DSV3FusedAGemmTest.test_shapes",
    "DSV3FusedAGemmTest.test_strided_a",
    "DSV3FusedAGemmTest.test_split_k_boundary",
    "DSV3FusedAGemmTest.test_split_k_strided",
    "DSV3FusedAGemmTest.test_split_k_cancellation",
    "DSV3FusedAGemmTest.test_special_values",
]


if __name__ == "__main__":
    unittest.main()
