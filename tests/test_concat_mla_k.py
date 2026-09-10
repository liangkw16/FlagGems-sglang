# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("concat_mla_k")


def reference(k, k_nope, k_rope):
    return torch.cat([k_nope, k_rope.expand(-1, k.shape[1], -1)], dim=-1).to(
        k.dtype
    )


def make_case(tokens=7, heads=128, nd=128, rd=64, strided=False):
    args = [
        torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        for shape in (
            (tokens, heads, nd + rd),
            (tokens, heads, nd),
            (tokens, 1, rd),
        )
    ]
    if strided:
        args = [x.transpose(0, 1).contiguous().transpose(0, 1) for x in args]
        expanded = torch.empty(
            tokens, 1, rd * 2 + 1, dtype=torch.bfloat16, device="cuda"
        )
        expanded[..., 1::2] = args[2]
        args[2] = expanded[..., 1::2]
    return tuple(args)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class ConcatMLAKTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() for x in args]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.concat_mla_k(*args)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                torch.testing.assert_close(
                    actual.view(torch.int16),
                    expected.view(torch.int16),
                    rtol=0,
                    atol=0,
                )
                for value, before in zip(args, snapshots):
                    torch.testing.assert_close(
                        value.view(torch.int16),
                        before.view(torch.int16),
                        rtol=0,
                        atol=0,
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

    def test_strides_and_special_bits(self):
        args = make_case(tokens=5, heads=5, strided=True)
        self.check(args)
        args[1][0, 0, :4] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0], device="cuda"
        )
        args[2][0, 0, :4] = args[1][0, 0, :4]
        self.check(args)
        k, nope, rope = make_case(tokens=3, heads=5)
        storage = torch.empty(3, 5, 257, device="cuda", dtype=torch.bfloat16)
        storage[..., 1::2] = nope
        self.check((k, storage[..., 1::2], rope))

    def test_empty(self):
        for tokens, heads, nd, rd in (
            (0, 128, 128, 64),
            (3, 0, 128, 64),
            (2, 4, 0, 0),
        ):
            self.check(make_case(tokens, heads, nd, rd))


RELEASE_REQUIRED_TESTS = [
    "ConcatMLAKTest.test_production_dimensions",
    "ConcatMLAKTest.test_head_and_dimension_boundaries",
    "ConcatMLAKTest.test_strides_and_special_bits",
    "ConcatMLAKTest.test_empty",
]


if __name__ == "__main__":
    unittest.main()
