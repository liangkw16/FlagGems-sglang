# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("fixup_zero_kv")


def reference(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    out_c, lse_c = out.clone(), lse.clone()
    zero = (kv_lens == 0).nonzero().flatten().tolist()
    for i in zero:
        beg, end = int(cum_seq_lens[i]), int(cum_seq_lens[i + 1])
        out_c[beg:end] = 0
        lse_c[beg:end] = float("-inf")
    return out_c, lse_c


def make_case(kv_lens=(4, 0, 7, 0, 0), dtype=torch.bfloat16, heads=8, vdim=16):
    cum = [0]
    for n in kv_lens:
        cum.append(cum[-1] + n)
    total = cum[-1]
    out = torch.randn(total, heads, vdim, dtype=dtype, device="cuda")
    lse = torch.randn(total, heads, dtype=torch.float32, device="cuda")
    lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    cum_t = torch.tensor(cum, dtype=torch.int32, device="cuda")
    span = max(kv_lens, default=0)
    return out, lse, lens, cum_t, span


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class FixupZeroKVTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() for x in args[:2]]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.fixup_zero_kv(*args)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(
                        bits(got), bits(want), rtol=0, atol=0
                    )
                # Inputs stay untouched (the fix lands on the clones).
                for value, before in zip(args[:2], snapshots):
                    torch.testing.assert_close(
                        bits(value), bits(before), rtol=0, atol=0
                    )
                for got in actual:
                    self.assertNotEqual(got.data_ptr(), args[0].data_ptr())

    def test_mixed_zero_and_nonzero(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                self.check(make_case(dtype=dtype))

    def test_all_zero_all_nonzero_and_empty(self):
        self.check(make_case(kv_lens=(0, 0, 0)))
        self.check(make_case(kv_lens=(3, 1, 5)))
        self.check(make_case(kv_lens=()))
        self.check(make_case(kv_lens=(0,)))
        self.check(make_case(kv_lens=(0, 2, 0, 0, 9, 0)))

    def test_segment_length_boundaries(self):
        for lengths in ((7, 8, 9), (1,), (16, 15, 17), (8, 8, 8, 8)):
            with self.subTest(lengths=lengths):
                self.check(make_case(kv_lens=lengths))

    def test_lying_max_seq_len(self):
        # max_seq_len is advisory: an understated span must not lose
        # tokens (the in-kernel tile stride covers the remainder).
        args = make_case(kv_lens=(0, 33, 0))
        self.check(args)
        self.check((args[0], args[1], args[2], args[3], 1))

    def test_special_values_preserved(self):
        # Untouched rows keep NaN/Inf/-0 bytes; touched rows become
        # exact zeros / exact -inf.
        args = make_case(kv_lens=(0, 3))
        args[0][0, 0, :4] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0], device="cuda"
        ).to(args[0].dtype)
        args[1][3, :4] = torch.tensor(
            [float("nan"), float("inf"), -0.0, 1.0], device="cuda"
        )
        self.check(args)

    def test_repeated_calls_reread_inputs(self):
        args = make_case()
        self.check(args)
        args[0].fill_(2.0)
        args[1].fill_(-3.5)
        self.check(args)


RELEASE_REQUIRED_TESTS = [
    "FixupZeroKVTest.test_mixed_zero_and_nonzero",
    "FixupZeroKVTest.test_all_zero_all_nonzero_and_empty",
    "FixupZeroKVTest.test_segment_length_boundaries",
    "FixupZeroKVTest.test_lying_max_seq_len",
    "FixupZeroKVTest.test_special_values_preserved",
    "FixupZeroKVTest.test_repeated_calls_reread_inputs",
]


if __name__ == "__main__":
    unittest.main()
