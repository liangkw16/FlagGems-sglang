# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("concat_and_cast_mha_k")

FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def reference(k, k_nope, k_rope):
    return torch.cat([k_nope, k_rope.expand(-1, k.shape[1], -1)], dim=-1).to(
        k.dtype
    )


def make_case(
    tokens=7,
    heads=128,
    nd=128,
    rd=64,
    strided=False,
    in_dtype=torch.bfloat16,
    out_dtype=torch.bfloat16,
):
    args = [
        torch.randn(shape, dtype=in_dtype, device="cuda")
        for shape in (
            (tokens, heads, nd + rd),
            (tokens, heads, nd),
            (tokens, 1, rd),
        )
    ]
    args[0] = args[0].to(out_dtype)
    if strided:
        args = [x.transpose(0, 1).contiguous().transpose(0, 1) for x in args]
        expanded = torch.empty(
            tokens, 1, rd * 2 + 1, dtype=in_dtype, device="cuda"
        )
        expanded[..., 1::2] = args[2]
        args[2] = expanded[..., 1::2]
    return tuple(args)


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class ConcatAndCastMHAKTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() for x in args]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.concat_and_cast_mha_k(*args)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                torch.testing.assert_close(
                    bits(actual), bits(expected), rtol=0, atol=0
                )
                for value, before in zip(args, snapshots):
                    torch.testing.assert_close(
                        bits(value), bits(before), rtol=0, atol=0
                    )

    def test_production_dimensions(self):
        for tokens in (1, 7, 127, 513):
            with self.subTest(tokens=tokens):
                self.check(make_case(tokens=tokens))

    def test_head_and_dimension_boundaries(self):
        for heads in (1, 3, 4, 5):
            for nd, rd in ((127, 63), (128, 64), (129, 65), (0, 64), (128, 0)):
                with self.subTest(heads=heads, nd=nd, rd=rd):
                    self.check(make_case(tokens=3, heads=heads, nd=nd, rd=rd))

    def test_dtype_cast_matrix(self):
        # Same-precision controls plus every cross-precision direction;
        # the bit-level comparison pins the cast to the reference's RTNE
        # (rounding midpoints, overflow-to-inf and subnormals included).
        for in_dtype in FLOAT_DTYPES:
            for out_dtype in FLOAT_DTYPES:
                with self.subTest(in_dtype=in_dtype, out_dtype=out_dtype):
                    args = make_case(
                        tokens=5,
                        heads=5,
                        in_dtype=in_dtype,
                        out_dtype=out_dtype,
                    )
                    args[1][0, 0, :3] = torch.tensor(
                        [1.0 + 2.0**-9, 1.0 + 2.0**-8, 1.0 + 3.0 * 2.0**-9],
                        dtype=in_dtype,
                        device="cuda",
                    )
                    if in_dtype == torch.float32:
                        args[1][0, 0, 3:6] = torch.tensor(
                            [3.0e38, 6.0e-8, 1.0e-45], dtype=in_dtype,
                            device="cuda",
                        )
                    self.check(args)

    def test_strides_and_special_bits(self):
        args = make_case(tokens=5, heads=5, strided=True)
        self.check(args)
        # Same-precision NaN/Inf/-0 passthrough (bit-identical move);
        # cross-precision NaN payloads are conversion-defined and stay
        # out of the exact matrix.
        args[1][0, 0, :4] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0], device="cuda"
        ).to(args[1].dtype)
        args[2][0, 0, :4] = args[1][0, 0, :4]
        self.check(args)
        for out_dtype in (torch.float16, torch.float32):
            with self.subTest(out_dtype=out_dtype):
                cross = make_case(
                    tokens=3,
                    heads=3,
                    in_dtype=torch.bfloat16,
                    out_dtype=out_dtype,
                )
                self.check(cross)


RELEASE_REQUIRED_TESTS = [
    "ConcatAndCastMHAKTest.test_production_dimensions",
    "ConcatAndCastMHAKTest.test_head_and_dimension_boundaries",
    "ConcatAndCastMHAKTest.test_dtype_cast_matrix",
    "ConcatAndCastMHAKTest.test_strides_and_special_bits",
]


if __name__ == "__main__":
    unittest.main()
