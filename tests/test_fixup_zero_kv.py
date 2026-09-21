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


def make_case(
    kv_lens=(4, 0, 7, 0, 0),
    tok_lens=None,
    dtype=torch.bfloat16,
    heads=8,
    vdim=16,
):
    # cum_seq_lens is the TOKEN boundary vector and is independent of
    # kv_lens: a zero-KV request still owns output rows (its q tokens).
    if tok_lens is None:
        tok_lens = [n + 2 for n in kv_lens]
    cum = [0]
    for n in tok_lens:
        cum.append(cum[-1] + n)
    total = cum[-1]
    out = torch.randn(total, heads, vdim, dtype=dtype, device="cuda")
    lse = torch.randn(total, heads, dtype=torch.float32, device="cuda")
    lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    cum_t = torch.tensor(cum, dtype=torch.int32, device="cuda")
    span = max(tok_lens, default=0)
    return out, lse, lens, cum_t, span


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class FixupZeroKVTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        # E8 contract change: out/lse are the operator's output params
        # and the fix now lands in place (the task's own semantic - the
        # SGLang kernel zeroes the output rows in a single launch; the
        # platform accepted in-place out-params for moe_align_block_size
        # since T84 e12). The reference clones only to keep its own
        # result pristine, so value equality against the clone is the
        # full contract; buffer identity is no longer asserted.
        pristine_out, pristine_lse = args[0].clone(), args[1].clone()
        for name, module in MODULES:
            with self.subTest(module=name):
                # fresh inputs per module: the in-place generic would
                # otherwise pre-fix the buffers a broken vendor reads
                module_args = (
                    pristine_out.clone(),
                    pristine_lse.clone(),
                ) + args[2:]
                actual = module.fixup_zero_kv(*module_args)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(
                        bits(got), bits(want), rtol=0, atol=0
                    )

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

    def test_non_power_of_two_heads(self):
        # BLOCK_H rounds up to the next power of two; the unmasked
        # lanes must not overrun into the neighbouring token's row.
        for heads in (3, 5, 7):
            with self.subTest(heads=heads):
                self.check(
                    make_case(kv_lens=(0, 4), tok_lens=(3, 6), heads=heads)
                )
                self.check(
                    make_case(kv_lens=(2, 0), tok_lens=(5, 4), heads=heads)
                )

    def test_segment_length_boundaries(self):
        for tok_lens in ((7, 8, 9), (1,), (16, 15, 17), (8, 8, 8, 8)):
            with self.subTest(tok_lens=tok_lens):
                self.check(
                    make_case(
                        kv_lens=(0,) * len(tok_lens), tok_lens=tok_lens
                    )
                )
        # Zero-KV requests with zero q tokens and long mixed neighbours.
        self.check(
            make_case(kv_lens=(5, 0, 3, 0, 2), tok_lens=(9, 0, 17, 1, 5))
        )

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
        total = args[1].shape[0]
        args[1][total - 1, :4] = torch.tensor(
            [float("nan"), float("inf"), -0.0, 1.0], device="cuda"
        )
        self.check(args)

    def test_uncovered_cum_boundaries(self):
        # Tokens outside [cum[0], cum[batch]) keep the input bytes; the
        # outputs are allocated empty, so every token needs a writer.
        args = make_case(kv_lens=(0, 3), tok_lens=(3, 5))
        head, tail = 2, 11 - (2 + 3 + 5)
        self.check(args)
        out2 = torch.randn(
            11 + tail, 4, 8, dtype=torch.bfloat16, device="cuda"
        )
        lse2 = torch.randn(11 + tail, 4, dtype=torch.float32, device="cuda")
        cum2 = [2, 5, 9]
        cumt2 = torch.tensor(cum2 + [11], dtype=torch.int32, device="cuda")
        self.check(
            (
                out2,
                lse2,
                torch.tensor((0, 3, 0), dtype=torch.int32, device="cuda"),
                cumt2,
                5,
            )
        )

    def test_empty_batch_full_output(self):
        # batch == 0 with tokens present returns a full clone.
        out = torch.randn(7, 4, 8, dtype=torch.bfloat16, device="cuda")
        lse = torch.randn(7, 4, dtype=torch.float32, device="cuda")
        self.check(
            (
                out,
                lse,
                torch.empty(0, dtype=torch.int32, device="cuda"),
                torch.tensor([0], dtype=torch.int32, device="cuda"),
                7,
            )
        )

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
    "FixupZeroKVTest.test_non_power_of_two_heads",
    "FixupZeroKVTest.test_uncovered_cum_boundaries",
    "FixupZeroKVTest.test_empty_batch_full_output",
    "FixupZeroKVTest.test_repeated_calls_reread_inputs",
]


if __name__ == "__main__":
    unittest.main()
