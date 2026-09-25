# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("get_mla_kv_buffer")


def reference(kv_buffer, loc, cache_k_nope, cache_k_rope):
    nope_dim = cache_k_nope.shape[-1]
    rows = kv_buffer[loc.long()]
    nope = rows[:, :nope_dim].to(cache_k_nope.dtype)
    rope = rows[:, nope_dim:].to(cache_k_rope.dtype)
    return nope, rope


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class GetMlaKvBufferTest(unittest.TestCase):
    def check(self, kv_buffer, loc, nope_dim, nope_dtype, rope_dtype):
        cache_k_nope = torch.empty(
            (loc.shape[0], nope_dim), dtype=nope_dtype, device=kv_buffer.device
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], kv_buffer.shape[-1] - nope_dim),
            dtype=rope_dtype,
            device=kv_buffer.device,
        )
        expected = reference(kv_buffer, loc, cache_k_nope, cache_k_rope)
        snapshots = [kv_buffer.clone(), loc.clone()]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.get_mla_kv_buffer(
                    kv_buffer, loc, cache_k_nope, cache_k_rope
                )
                self.assertEqual(len(actual), 2)
                for got, want in zip(actual, expected):
                    self.assertEqual(got.shape, want.shape)
                    self.assertEqual(got.dtype, want.dtype)
                    self.assertTrue(got.is_contiguous())
                    torch.testing.assert_close(
                        bits(got), bits(want), rtol=0, atol=0
                    )
                torch.testing.assert_close(
                    bits(kv_buffer), bits(snapshots[0]), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    bits(loc), bits(snapshots[1]), rtol=0, atol=0
                )

    def make_case(self, n, nope_dim, rope_dim, *, kv_dtype=torch.float16):
        total = n + 4  # unused tail rows in the page pool
        kv = torch.randn(total, nope_dim + rope_dim, dtype=kv_dtype, device="cuda")
        loc = torch.randperm(n, device="cuda").to(torch.int32)
        return kv, loc

    def test_mla_realistic_shapes(self):
        for n in (1, 3, 15, 16, 17, 255, 256, 257, 4096):
            with self.subTest(n=n):
                kv, loc = self.make_case(n, 512, 64)
                self.check(kv, loc, 512, torch.float16, torch.float16)

    def test_cross_dtype_store(self):
        for kv_dtype, nope_dtype, rope_dtype in (
            (torch.float16, torch.bfloat16, torch.float16),
            (torch.float32, torch.float16, torch.bfloat16),
            (torch.bfloat16, torch.float32, torch.bfloat16),
        ):
            with self.subTest(kv=kv_dtype, nope=nope_dtype, rope=rope_dtype):
                kv, loc = self.make_case(37, 128, 32, kv_dtype=kv_dtype)
                self.check(kv, loc, 128, nope_dtype, rope_dtype)

    def test_loc_int64_and_duplicates(self):
        kv, loc32 = self.make_case(64, 96, 24)
        loc64 = loc32.long()
        self.check(kv, loc64, 96, torch.float16, torch.float16)
        dup = torch.full((50,), 7, dtype=torch.int64, device="cuda")
        self.check(kv, dup, 96, torch.float16, torch.float16)

    def test_strided_kv_buffer(self):
        big = torch.randn(200, 576, dtype=torch.float16, device="cuda")
        kv = big[3:163]  # non-zero storage offset, same strides
        loc = torch.randperm(kv.shape[0], device="cuda").to(torch.int64)
        self.check(kv, loc, 512, torch.float16, torch.float16)

    def test_dim_edges(self):
        for nope_dim, rope_dim in ((1, 63), (575, 1), (32, 96), (64, 64)):
            with self.subTest(nope=nope_dim, rope=rope_dim):
                kv, loc = self.make_case(70, nope_dim, rope_dim)
                self.check(kv, loc, nope_dim, torch.float16, torch.float16)

    def test_column_strided_kv_buffer(self):
        # a column-sliced cache view carries stride(1)=2: both halves
        # must follow the runtime column stride (review finding)
        big = torch.arange(32, device="cuda", dtype=torch.float16).reshape(4, 8)
        kv = big[:, ::2]
        self.assertEqual(kv.stride(1), 2)
        loc = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
        self.check(kv, loc, 2, torch.float16, torch.float16)
        self.check(kv, loc, 1, torch.float16, torch.float16)

    def test_wide_half_beyond_block_cap(self):
        # nope_dim=70000 exceeds the 65536 vendor lane cap; the chunked
        # loop must cover it (review finding)
        kv = torch.randn(6, 70000 + 100, dtype=torch.float16, device="cuda")
        loc = torch.arange(5, dtype=torch.int32, device="cuda")
        self.check(kv, loc, 70000, torch.float16, torch.float16)

    def test_zero_width_half(self):
        # one half empty must not skip the kernel: the other half still
        # has to be written (review finding, b6641097)
        kv = torch.randn(8, 64, dtype=torch.float16, device="cuda")
        loc = torch.tensor([3, 1, 5], dtype=torch.int32, device="cuda")
        self.check(kv, loc, 0, torch.float16, torch.float16)
        self.check(kv, loc, 64, torch.float16, torch.float16)
        kv2 = torch.randn(8, 96, dtype=torch.bfloat16, device="cuda")
        loc2 = torch.arange(4, dtype=torch.int32, device="cuda")
        self.check(kv2, loc2, 1, torch.bfloat16, torch.bfloat16)

    def test_empty_rows(self):
        kv, _ = self.make_case(8, 64, 16)
        loc = torch.empty(0, dtype=torch.int64, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                nope, rope = module.get_mla_kv_buffer(
                    kv,
                    loc,
                    torch.empty(0, 64, dtype=torch.float16, device="cuda"),
                    torch.empty(0, 16, dtype=torch.float16, device="cuda"),
                )
                self.assertEqual(nope.shape, (0, 64))
                self.assertEqual(rope.shape, (0, 16))

    # ------------------------------------------------------------------
    # round-2 kunlun arm regressions: the kunlunxin vendor moved to the
    # upstream sglang single-row kernel (one row per program, exact
    # pow2 arange widths, zero masks/loops) behind a host shape branch;
    # these pin the new kernel semantics and the branch predicate.
    # ------------------------------------------------------------------

    def test_row_form_int64_loc_and_duplicates(self):
        # pow2 half widths take the single-row kernel: i64 loc values,
        # duplicates and unsorted gather order must survive the scalar
        # loc load (new-kernel semantics regression)
        kv = torch.randn(16, 576, dtype=torch.float16, device="cuda")
        loc = torch.tensor(
            [9, 9, 2, 15, 2, 0, 9], dtype=torch.int64, device="cuda"
        )
        self.check(kv, loc, 512, torch.float16, torch.float16)
        loc32 = torch.tensor([3, 3, 7], dtype=torch.int32, device="cuda")
        self.check(kv, loc32, 512, torch.float16, torch.float16)

    def test_row_form_dim_boundary(self):
        # exact pow2 widths take the single-row kernel; the +-1
        # neighbours must fall back to the masked rows kernel - both
        # arms match the reference (host branch boundary regression)
        for nope_dim, rope_dim in (
            (64, 64),
            (65, 63),
            (63, 65),
            (128, 32),
            (127, 33),
        ):
            with self.subTest(nope=nope_dim, rope=rope_dim):
                kv, loc = self.make_case(33, nope_dim, rope_dim)
                self.check(kv, loc, nope_dim, torch.float16, torch.float16)

    def test_row_form_stride_fallbacks(self):
        # pow2 widths with a column-strided kv view or a strided loc
        # view must leave the single-row kernel via the host branch and
        # stay exact through the masked rows kernel
        big = torch.randn(48, 256, dtype=torch.float16, device="cuda")
        kv = big[:, ::2]  # shape (48, 128), stride(1) == 2
        self.assertEqual(kv.stride(1), 2)
        loc = torch.randperm(kv.shape[0], device="cuda").to(torch.int32)
        self.check(kv, loc, 64, torch.float16, torch.float16)

        kv2 = torch.randn(64, 576, dtype=torch.float16, device="cuda")
        loc2 = (
            torch.randperm(64, device="cuda").to(torch.int32).repeat(2)[::2]
        )
        self.assertEqual(loc2.stride(0), 2)
        self.check(kv2, loc2, 512, torch.float16, torch.float16)

    def test_row_form_host_branch_predicate(self):
        # pin the host-branch predicate itself: exact pow2 widths within
        # the lane cap, unit column/loc strides, one program per row
        checked = 0
        for name, module in MODULES:
            predicate = getattr(module, "_upstream_row_form", None)
            if predicate is None:
                continue
            with self.subTest(module=name):
                checked += 1
                self.assertTrue(predicate(4096, 512, 64, 1, 1))
                self.assertTrue(predicate(1, 1, 1, 1, 1))
                self.assertFalse(predicate(4096, 575, 64, 1, 1))  # nope
                self.assertFalse(predicate(4096, 512, 96, 1, 1))  # rope
                self.assertFalse(predicate(4096, 512, 0, 1, 1))  # empty half
                self.assertFalse(predicate(4096, 512, 64, 1, 2))  # kv_s1
                self.assertFalse(predicate(4096, 512, 64, 2, 1))  # loc_s0
                self.assertFalse(predicate(0, 512, 64, 1, 1))  # n == 0
                self.assertFalse(predicate(65536, 512, 64, 1, 1))  # grid cap
                self.assertFalse(predicate(16, 131072, 64, 1, 1))  # lanes
        if not checked:
            self.skipTest("no applicable source exposes _upstream_row_form")

    def test_row_form_grid_cap_fallback(self):
        # n past the 65535-program cap drops to the masked rows kernel
        n = 65536 + 16
        kv = torch.randn(n + 4, 128, dtype=torch.float16, device="cuda")
        loc = torch.randperm(n, device="cuda").to(torch.int32)
        self.check(kv, loc, 64, torch.float16, torch.float16)


RELEASE_REQUIRED_TESTS = [
    "GetMlaKvBufferTest.test_mla_realistic_shapes",
    "GetMlaKvBufferTest.test_cross_dtype_store",
    "GetMlaKvBufferTest.test_loc_int64_and_duplicates",
    "GetMlaKvBufferTest.test_strided_kv_buffer",
    "GetMlaKvBufferTest.test_dim_edges",
    "GetMlaKvBufferTest.test_column_strided_kv_buffer",
    "GetMlaKvBufferTest.test_wide_half_beyond_block_cap",
    "GetMlaKvBufferTest.test_zero_width_half",
    "GetMlaKvBufferTest.test_empty_rows",
    "GetMlaKvBufferTest.test_row_form_int64_loc_and_duplicates",
    "GetMlaKvBufferTest.test_row_form_dim_boundary",
    "GetMlaKvBufferTest.test_row_form_stride_fallbacks",
    "GetMlaKvBufferTest.test_row_form_host_branch_predicate",
    "GetMlaKvBufferTest.test_row_form_grid_cap_fallback",
]

if __name__ == "__main__":
    unittest.main()
