# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("ltx2_split_rotary")


def reference(x, cos, sin):
    batch, seq_len, inner = x.shape
    _, num_heads, _, half = cos.shape
    head_dim = half * 2
    xv = x.reshape(batch, seq_len, num_heads, head_dim)
    c = cos.permute(0, 2, 1, 3)
    s = sin.permute(0, 2, 1, 3)
    x1, x2 = xv[..., :half], xv[..., half:]
    o1 = (x1 * c).to(torch.bfloat16).float() - x2.float() * s.float()
    o2 = (x2 * c).to(torch.bfloat16).float() + x1.float() * s.float()
    out = torch.cat([o1, o2], dim=-1)
    return out.reshape(batch, seq_len, inner).to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class Ltx2SplitRotaryTest(unittest.TestCase):
    def check(self, x, cos, sin):
        expected = reference(x, cos, sin)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.ltx2_split_rotary(x, cos, sin)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)

    def make(self, B, T, H, half, dtype=torch.bfloat16, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(B, T, H * half * 2, dtype=dtype, device="cuda", generator=gen)
        cos = torch.randn(B, H, T, half, dtype=dtype, device="cuda", generator=gen)
        sin = torch.randn(B, H, T, half, dtype=dtype, device="cuda", generator=gen)
        return x, cos, sin

    def test_shapes(self):
        for B, T, H, half in ((1, 8, 4, 32), (2, 33, 16, 64), (1, 128, 8, 1), (3, 5, 2, 96)):
            with self.subTest(B=B, T=T, H=H, half=half):
                self.check(*self.make(B, T, H, half))

    def test_non_contiguous_tables(self):
        gen = torch.Generator(device="cuda").manual_seed(5)
        x = torch.randn(2, 16, 8 * 256, dtype=torch.bfloat16, device="cuda", generator=gen)
        big_c = torch.randn(2, 16, 8, 128, dtype=torch.bfloat16, device="cuda", generator=gen)
        cos = big_c.permute(0, 2, 1, 3)
        big_s = torch.randn(2, 16, 8, 128, dtype=torch.bfloat16, device="cuda", generator=gen)
        sin = big_s.permute(0, 2, 1, 3)
        self.assertFalse(cos.is_contiguous())
        self.check(x, cos, sin)

    def test_sliced_and_independent_layouts(self):
        # a sliced x (stride-2 inner) plus cos/sin built with different
        # independent layouts: the output must be contiguous and both
        # tables must be read through their OWN strides
        gen = torch.Generator(device="cuda").manual_seed(9)
        big = torch.randn(1, 8, 4, 128 * 2, dtype=torch.bfloat16, device="cuda", generator=gen)
        x3 = big[..., ::2].reshape(1, 8, 4, 128)  # sliced then reshaped
        x = x3.permute(0, 1, 3, 2).reshape(1, 8, 512).contiguous()
        cos = big[..., 0::2].permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
        sin_big = torch.randn(8, 4, 8, 128, dtype=torch.bfloat16, device="cuda", generator=gen)
        sin = sin_big.permute(2, 1, 0, 3)  # [8, 4, 8, 128] -> [T? no] keep shapes right
        # build legal tables: cos/sin [B=1, H=4, T=8, half=128]
        c = torch.randn(1, 4, 8, 128, dtype=torch.bfloat16, device="cuda", generator=gen)
        s2 = torch.randn(8, 4, 1, 128, dtype=torch.bfloat16, device="cuda", generator=gen)
        sin = s2.permute(2, 1, 0, 3)  # [1, 4, 8, 128] non-contiguous
        self.assertFalse(sin.is_contiguous())
        self.check(x, c, sin)

    def test_sliced_x(self):
        gen = torch.Generator(device="cuda").manual_seed(4)
        big = torch.randn(1, 8, 256, dtype=torch.bfloat16, device="cuda", generator=gen)
        x = big[..., ::2]  # inner stride 2
        self.assertFalse(x.is_contiguous())
        c = torch.randn(1, 4, 8, 64, dtype=torch.bfloat16, device="cuda", generator=gen)
        s2 = torch.randn(1, 4, 8, 64, dtype=torch.bfloat16, device="cuda", generator=gen)
        self.check(x, c, s2)

    def test_extreme_angles(self):
        x = torch.tensor(
            [[[100.0, -100.0, 1e-4, 5.0]]], dtype=torch.bfloat16, device="cuda"
        )
        cos = torch.tensor([[[[1.0, -1.0]]]], dtype=torch.bfloat16, device="cuda")
        sin = torch.tensor([[[[0.5, -0.5]]]], dtype=torch.bfloat16, device="cuda")
        self.check(x, cos, sin)


RELEASE_REQUIRED_TESTS = [
    "Ltx2SplitRotaryTest.test_shapes",
    "Ltx2SplitRotaryTest.test_non_contiguous_tables",
    "Ltx2SplitRotaryTest.test_sliced_and_independent_layouts",
    "Ltx2SplitRotaryTest.test_sliced_x",
    "Ltx2SplitRotaryTest.test_extreme_angles",
]

if __name__ == "__main__":
    unittest.main()
