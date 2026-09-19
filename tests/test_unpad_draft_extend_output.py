# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("unpad_draft_extend_output")


def reference(raw_out, cu, lens, total):
    bs = lens.shape[0]
    out = torch.empty((total, raw_out.shape[2], raw_out.shape[3]), dtype=raw_out.dtype, device=raw_out.device)
    for b in range(bs):
        s = int(lens[b])
        beg = int(cu[b])
        out[beg : beg + s] = raw_out[b, :s]
    return out


def make_case(lens=(0, 3, 1, 5), heads=8, dim=128, tpb=None):
    bs = len(lens)
    if tpb is None:
        tpb = max(lens) + 2
    cum = [0]
    for n in lens:
        cum.append(cum[-1] + n)
    raw = torch.randn(bs, tpb, heads, dim, dtype=torch.bfloat16, device="cuda")
    return (
        raw,
        torch.tensor(cum, dtype=torch.int32, device="cuda"),
        torch.tensor(lens, dtype=torch.int32, device="cuda"),
        cum[-1],
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class UnpadTest(unittest.TestCase):
    def check(self, args):
        want = reference(*args)
        for name, module in MODULES:
            with self.subTest(module=name):
                got = module.unpad_draft_extend_output(*args)
                torch.testing.assert_close(got.view(torch.uint8), want.contiguous().view(torch.uint8), rtol=0, atol=0)

    def test_ragged_and_boundaries(self):
        self.check(make_case())
        self.check(make_case(lens=(0,), tpb=1))
        self.check(make_case(lens=(tpb_len := 64,), tpb=64, heads=2, dim=64))
        self.check(make_case(lens=(7,) * 40, heads=4, dim=96, tpb=7))


RELEASE_REQUIRED_TESTS = [
    "UnpadTest.test_ragged_and_boundaries",
]


if __name__ == "__main__":
    unittest.main()
