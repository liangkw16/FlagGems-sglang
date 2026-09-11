# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("fused_eh_norm")

TOL = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


def _rmsnorm(x, weight, eps):
    xf = x.float()
    return (
        xf
        * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        * weight.float()
    )


def reference(inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps):
    e = _rmsnorm(inputs_embeds, enorm_weight, eps)
    h = _rmsnorm(previous_hidden, hnorm_weight, eps)
    return torch.cat([e, h], dim=-1).to(inputs_embeds.dtype)


def make_case(
    tokens=7,
    hidden=1024,
    dtype=torch.bfloat16,
    eps=1e-6,
    strided=False,
):
    e = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
    h = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
    ew = torch.randn(hidden, dtype=dtype, device="cuda")
    hw = torch.randn(hidden, dtype=dtype, device="cuda")
    if strided:
        storage = torch.randn(
            tokens * 2 + 1, hidden, dtype=dtype, device="cuda"
        )
        e = storage[1::2]
        w_storage = torch.randn(hidden * 2, dtype=dtype, device="cuda")
        ew = w_storage[1::2]
    return e, h, ew, hw, eps


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class FusedEHNormTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() if torch.is_tensor(x) else x for x in args[:4]]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.fused_eh_norm(*args)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=TOL[args[0].dtype][0],
                    atol=TOL[args[0].dtype][1],
                )
                for value, before in zip(args[:4], snapshots):
                    if torch.is_tensor(value):
                        torch.testing.assert_close(
                            value, before, rtol=0, atol=0, equal_nan=True
                        )

    def test_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            for eps in (1e-6, 1e-5):
                with self.subTest(dtype=dtype, eps=eps):
                    self.check(make_case(dtype=dtype, eps=eps))

    def test_hidden_range_and_tokens(self):
        for hidden in (512, 768, 2048, 7168, 8192):
            with self.subTest(hidden=hidden):
                self.check(make_case(hidden=hidden))
        for tokens in (1, 3, 128, 8193):
            with self.subTest(tokens=tokens):
                self.check(make_case(tokens=tokens, hidden=512))

    def test_strides(self):
        self.check(make_case(strided=True))

    def test_zero_row_and_empty(self):
        e, h, ew, hw, eps = make_case(tokens=4, hidden=512)
        e[:, :] = 0
        h[:, :] = 0
        self.check((e, h, ew, hw, eps))
        self.check(make_case(tokens=0, hidden=512))


RELEASE_REQUIRED_TESTS = [
    "FusedEHNormTest.test_dtypes",
    "FusedEHNormTest.test_hidden_range_and_tokens",
    "FusedEHNormTest.test_strides",
    "FusedEHNormTest.test_zero_row_and_empty",
]


if __name__ == "__main__":
    unittest.main()
