# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
import triton

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

    def test_row_width_around_ascend_block(self):
        # B-1/B/B+1 around the ascend vendor's 8192 flat-store rung and
        # the empty-value-dim edge that must still write lse rows
        for heads, vdim in ((65, 127 - 16), (64, 128), (65, 128)):
            with self.subTest(hv=heads * vdim):
                self.check(
                    make_case(
                        kv_lens=(0, 3), tok_lens=(4, 5), heads=heads,
                        vdim=vdim,
                    )
                )

    def test_row_width_around_wide_block(self):
        # B-1/B/B+1 around the widest vendor flat-store rung (16384):
        # below it the tail mask folds away, above it the sweep loops
        for heads, vdim in ((129, 127), (128, 128), (129, 128)):
            with self.subTest(hv=heads * vdim):
                self.check(
                    make_case(
                        kv_lens=(0, 3), tok_lens=(4, 5), heads=heads,
                        vdim=vdim,
                    )
                )

    def test_segment_length_boundaries(self):
        for tok_lens in ((7, 8, 9), (1,), (16, 15, 17), (8, 8, 8, 8)):
            with self.subTest(tok_lens=tok_lens):
                self.check(
                    make_case(kv_lens=(0,) * len(tok_lens), tok_lens=tok_lens)
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
        tail = 11 - (2 + 3 + 5)
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

    def test_understated_span_zero_segment(self):
        # the advisory span sizes the launch; a zero-KV segment longer
        # than max_seq_len*BLOCK_T must still be fully fixed (the e12
        # enflame item mapping dropped the tile-stride guard)
        out = torch.randn(33, 96, 128, dtype=torch.float16, device="cuda")
        lse = torch.randn(33, 96, dtype=torch.float32, device="cuda")
        cum = torch.tensor([0, 33], dtype=torch.int32, device="cuda")
        lens = torch.tensor([0], dtype=torch.int32, device="cuda")
        self.check((out, lse, lens, cum, 1))

    def test_beyond_fp32_mask_domain(self):
        # Long spans must use exact integer addresses and masks rather
        # than rounding their tail boundaries through fp32.
        # out stream: 1400 rows x (96*128) = 17,203,200 elements; the
        # boundary offset 17,203,199 is odd and rounds up in fp32.
        args = make_case(kv_lens=(0,), tok_lens=(1400,), heads=96, vdim=128)
        self.assertGreater(args[0].numel(), 2**24)
        self.check(args)
        # lse stream: 131073 rows x 128 heads = 16,777,344 elements
        # (boundary offset 16,777,343 likewise rounds up); vdim=1 keeps
        # the out span at two bytes per row-head.
        args = make_case(kv_lens=(0,), tok_lens=(131073,), heads=128, vdim=1)
        self.assertGreater(args[1].numel(), 2**24)
        self.check(args)

    def test_long_segment_with_understated_span(self):
        # The persistent worker must cover a long zero segment even when
        # the advisory span underestimates its length.
        args = make_case(
            kv_lens=(0, 4, 0), tok_lens=(1400, 5, 33), heads=96, vdim=128
        )
        self.check(args)
        self.check((args[0], args[1], args[2], args[3], 500))

    def test_full_and_partial_token_tile(self):
        # Cover exact and ragged token-tile boundaries plus an empty
        # zero-KV segment next to a nonempty neighbour.
        self.check(make_case(kv_lens=(0,), tok_lens=(64,)))
        self.check(make_case(kv_lens=(0,), tok_lens=(65,)))
        self.check(make_case(kv_lens=(0, 3), tok_lens=(64, 2)))
        self.check(make_case(kv_lens=(0, 0), tok_lens=(64, 0)))

    def test_global_worker_grid_and_slot_coverage(self):
        # Four proxy workers exercise one long segment, multiple virtual
        # items per worker, and an advisory span shorter than the data.
        ascend = dict(MODULES).get("ascend")
        if ascend is None:
            return
        original_kernel = ascend._fixup_zero_kv
        original_workers = ascend._MAX_WORKERS
        grids = []

        class _GridRecorder:
            def __getitem__(self, grid):
                def _record(*args, **kwargs):
                    grids.append(grid)

                return _record

        cases = (
            ((0,), (33,), (4,)),
            ((0, 3, 0), (33, 1, 0), (4,)),
            ((3,) * 8 + (0,), (1,) * 8 + (33,), (4,)),
        )
        ascend._MAX_WORKERS = 4
        try:
            for kv_lens, tok_lens, want_grid in cases:
                with self.subTest(batch=len(kv_lens)):
                    args = make_case(kv_lens=kv_lens, tok_lens=tok_lens)
                    ascend._fixup_zero_kv = _GridRecorder()
                    try:
                        ascend.fixup_zero_kv(*args)
                        self.assertEqual(grids[-1], want_grid)
                    finally:
                        ascend._fixup_zero_kv = original_kernel
                    self.check((args[0], args[1], args[2], args[3], 1))
        finally:
            ascend._fixup_zero_kv = original_kernel
            ascend._MAX_WORKERS = original_workers

    def test_enflame_int32_domain_boundary(self):
        # e30: the enflame wrapper's addressing-domain predicate. At the
        # T80 geometry (HV=12288 >> NH=96) the out stream binds: the
        # last in-domain total_tokens is the largest t with
        # (t+BLOCK_T)*HV + HV + BLOCK_V < 2^31, and one token more must
        # flip the branch to the i64 kernel. Strides (not just shapes)
        # participate, and so does the flattened work-item count.
        enflame = dict(MODULES).get("enflame")
        if enflame is None:
            return
        hv, nh = 96 * 128, 96
        block_h = triton.next_power_of_2(nh)
        t_ok = (2**31 - 1 - hv - 4096) // hv - 8
        self.assertTrue(
            enflame._fits_int32(t_ok, hv, nh, hv, block_h, 1)
        )
        self.assertFalse(
            enflame._fits_int32(t_ok + 1, hv, nh, hv, block_h, 1)
        )
        # a padded row stride exits the domain even at the in-domain
        # token count, and an inflated lse stride binds on its own
        self.assertFalse(
            enflame._fits_int32(t_ok, hv * 2, nh, hv, block_h, 1)
        )
        self.assertFalse(
            enflame._fits_int32(1, hv, 2**30, hv, block_h, 1)
        )
        # the item count gate: at or beyond 2^31 the i64 kernel returns
        self.assertTrue(
            enflame._fits_int32(t_ok, hv, nh, hv, block_h, 2**31 - 1)
        )
        self.assertFalse(
            enflame._fits_int32(t_ok, hv, nh, hv, block_h, 2**31)
        )

    def test_enflame_domain_branch_selection(self):
        # e30: in-domain shapes must launch the all-int32 kernel;
        # out-of-domain shapes must keep the original i64 kernel. The
        # out-of-domain case inflates the out row stride on a one-row
        # tensor so the computed offsets cross 2^31 while only row 0 is
        # ever addressable (24KB of storage).
        enflame = dict(MODULES).get("enflame")
        if enflame is None:
            return
        counters = [0, 0]

        class _Recorder:
            def __init__(self, slot):
                self._slot = slot

            def __getitem__(self, grid):
                def _launch(*args, **kwargs):
                    counters[self._slot] += 1

                return _launch

        original32 = enflame._fixup_zero_kv32
        original64 = enflame._fixup_zero_kv
        enflame._fixup_zero_kv32 = _Recorder(0)
        enflame._fixup_zero_kv = _Recorder(1)
        try:
            enflame.fixup_zero_kv(*make_case())
            self.assertEqual(counters, [1, 0])
            heads, vdim = 96, 128
            hv = heads * vdim
            t_hi = 1 + 8  # total_tokens + BLOCK_T ragged-tail lanes
            stride0 = 2 * (-(-(2**31 - hv - 4096) // t_hi))
            row = torch.randn(1, 1, hv, dtype=torch.float16, device="cuda")
            out = row.as_strided((1, heads, vdim), (stride0, vdim, 1))
            lse = torch.randn(1, heads, dtype=torch.float32, device="cuda")
            args = (
                out,
                lse,
                torch.tensor((0,), dtype=torch.int32, device="cuda"),
                torch.tensor((0, 1), dtype=torch.int32, device="cuda"),
                1,
            )
            self.assertFalse(
                enflame._fits_int32(
                    1, stride0, lse.stride(0), hv,
                    triton.next_power_of_2(heads), 1,
                )
            )
            enflame.fixup_zero_kv(*args)
            self.assertEqual(counters, [1, 1])
        finally:
            enflame._fixup_zero_kv32 = original32
            enflame._fixup_zero_kv = original64

    def test_enflame_out_of_domain_i64_path(self):
        # e30 numeric regression for the kept i64 kernel through the new
        # branched wrapper: the inflated row stride routes this call to
        # _fixup_zero_kv (verified above), and the fix must still land
        # exactly. check() would clone the buffers and erase the stride,
        # so this compares the strided view directly against the
        # reference.
        enflame = dict(MODULES).get("enflame")
        if enflame is None:
            return
        heads, vdim = 96, 128
        hv = heads * vdim
        t_hi = 1 + 8
        stride0 = 2 * (-(-(2**31 - hv - 4096) // t_hi))
        row = torch.randn(1, 1, hv, dtype=torch.float16, device="cuda")
        out = row.as_strided((1, heads, vdim), (stride0, vdim, 1))
        lse = torch.randn(1, heads, dtype=torch.float32, device="cuda")
        lens = torch.tensor((0,), dtype=torch.int32, device="cuda")
        cum = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
        expected = reference(out, lse, lens, cum, 1)
        self.assertFalse(
            enflame._fits_int32(
                1, stride0, lse.stride(0), hv,
                triton.next_power_of_2(heads), 1,
            )
        )
        actual = enflame.fixup_zero_kv(out, lse, lens, cum, 1)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(bits(got), bits(want), rtol=0, atol=0)


RELEASE_REQUIRED_TESTS = [
    "FixupZeroKVTest.test_understated_span_zero_segment",
    "FixupZeroKVTest.test_mixed_zero_and_nonzero",
    "FixupZeroKVTest.test_all_zero_all_nonzero_and_empty",
    "FixupZeroKVTest.test_segment_length_boundaries",
    "FixupZeroKVTest.test_lying_max_seq_len",
    "FixupZeroKVTest.test_special_values_preserved",
    "FixupZeroKVTest.test_non_power_of_two_heads",
    "FixupZeroKVTest.test_uncovered_cum_boundaries",
    "FixupZeroKVTest.test_empty_batch_full_output",
    "FixupZeroKVTest.test_repeated_calls_reread_inputs",
    "FixupZeroKVTest.test_beyond_fp32_mask_domain",
    "FixupZeroKVTest.test_long_segment_with_understated_span",
    "FixupZeroKVTest.test_full_and_partial_token_tile",
    "FixupZeroKVTest.test_global_worker_grid_and_slot_coverage",
    "FixupZeroKVTest.test_enflame_int32_domain_boundary",
    "FixupZeroKVTest.test_enflame_domain_branch_selection",
    "FixupZeroKVTest.test_enflame_out_of_domain_i64_path",
]


if __name__ == "__main__":
    unittest.main()
