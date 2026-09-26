# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("fused_pack_qkv")


def reference(q, k, v, indices):
    bs, seq, h, d = q.shape
    idx = indices.long()
    return (
        q.reshape(bs * seq, h, d)[idx],
        k.reshape(bs * seq, h, d)[idx],
        v.reshape(bs * seq, h, d)[idx],
    )


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class FusedPackQkvTest(unittest.TestCase):
    def check(self, q, k, v, indices):
        expected = reference(q, k, v, indices)
        snapshots = [x.clone() for x in (q, k, v, indices)]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.fused_pack_qkv(q, k, v, indices)
                self.assertEqual(len(actual), 3)
                for got, want in zip(actual, expected):
                    self.assertEqual(got.shape, want.shape)
                    self.assertEqual(got.dtype, want.dtype)
                    self.assertTrue(got.is_contiguous())
                    torch.testing.assert_close(
                        bits(got), bits(want), rtol=0, atol=0
                    )
                for value, before in zip((q, k, v, indices), snapshots):
                    torch.testing.assert_close(
                        bits(value), bits(before), rtol=0, atol=0
                    )

    def make_case(self, bs, seq, h, d, total_valid, *, dtype=torch.float16):
        q = torch.randn(bs, seq, h, d, dtype=dtype, device="cuda")
        k = torch.randn(bs, seq, h, d, dtype=dtype, device="cuda")
        v = torch.randn(bs, seq, h, d, dtype=dtype, device="cuda")
        perm = torch.randperm(bs * seq, device="cuda")[:total_valid]
        return q, k, v, perm.to(torch.int32)

    def test_diffusion_shapes_and_tails(self):
        cases = [
            (2, 128, 16, 64, 256),
            (1, 4096, 24, 64, 4096),
            (4, 256, 8, 128, 511),
            (2, 64, 16, 64, 1),
            (3, 333, 12, 40, 999),
        ]
        for bs, seq, h, d, total in cases:
            with self.subTest(bs=bs, seq=seq, h=h, d=d, total=total):
                args = self.make_case(bs, seq, h, d, total)
                self.check(*args)

    def test_indices_int64_and_duplicates(self):
        q, k, v, idx32 = self.make_case(2, 256, 8, 64, 100)
        self.check(q, k, v, idx32.long())
        dup = torch.full((64,), 5, dtype=torch.int64, device="cuda")
        self.check(q, k, v, dup)
        # a pure gather may also repeat indices so often that the
        # output exceeds the input (n > B*S); the iluvatar guard must
        # still bound the output side on its own (review r1 P1)
        inflate = torch.zeros(2048, dtype=torch.int32, device="cuda")
        self.assertGreater(inflate.shape[0], 2 * 256)
        self.check(q, k, v, inflate)

    def test_fp32_and_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                args = self.make_case(2, 128, 16, 64, 200, dtype=dtype)
                self.check(*args)

    def test_sliced_indices(self):
        # indices may arrive as a strided view: reads must follow
        # indices.stride(0) (review finding)
        q, k, v, _ = self.make_case(1, 8, 4, 16, 4)
        idx = torch.arange(8, dtype=torch.int32, device="cuda")[::2]
        self.assertEqual(idx.stride(0), 2)
        self.check(q, k, v, idx)

    def test_non_contiguous_qkv(self):
        # reference's reshape accepts non-contiguous inputs; the wrapper
        # must normalize instead of asserting (review finding)
        base = torch.arange(16, device="cuda", dtype=torch.float16).reshape(1, 2, 2, 4)
        qt = base.transpose(-1, -2)
        self.assertFalse(qt.is_contiguous())
        kt = torch.randn(1, 2, 2, 4, dtype=torch.float16, device="cuda").transpose(
            -1, -2
        )
        vt = torch.randn(1, 2, 2, 4, dtype=torch.float16, device="cuda").transpose(
            -1, -2
        )
        indices = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.check(qt, kt, vt, indices)

    def test_row_elems_beyond_block_cap(self):
        # H*D=70000 exceeds the 65536 vendor lane cap; the chunked
        # column loop must cover it (review finding)
        q = torch.randn(1, 2, 350, 200, dtype=torch.float16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        self.assertEqual(q.shape[-2] * q.shape[-1], 70000)
        idx = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        self.check(q, k, v, idx)

    def test_row_elems_wider_than_block(self):
        # E3 generic caps BLOCK_C at 2048 (muxi max_tile_size,
        # chip-rulesets.md:29): H*D=2048 sits exactly at the cap (single
        # segment) and H*D=4096 forces the chunked two-segment loop
        for h, d in ((16, 128), (32, 128)):
            with self.subTest(h=h, d=d):
                args = self.make_case(1, 512, h, d, 700)
                self.assertEqual(
                    args[0].shape[-2] * args[0].shape[-1], h * d
                )
                self.check(*args)

    def test_iluvatar_int32_domain_guard(self):
        # E2 iluvatar vendor host dispatch: int32 addressing is only
        # legal while every flat offset stays below 2**31. A wrong
        # boundary compare silently turns into out-of-bounds writes on
        # the target chip, so the boundary arithmetic itself is the
        # regression (the full matrix above already exercises the
        # int32 kernel path; the i64 twin only fires past 2**31
        # elements, which does not fit CI memory)
        iluvatar = next(
            (mod for name, mod in MODULES if name == "iluvatar"), None
        )
        # hard presence pin, not a skip: the vendor file ships in the
        # ZIP, so any release/screening of this tree must keep the
        # iluvatar source in FLAGOS_TEST_SOURCES (verify_release
        # rejects skipped tests; absence here is a misconfiguration)
        self.assertIsNotNone(
            iluvatar, "iluvatar vendor missing from FLAGOS_TEST_SOURCES"
        )
        # largest legal domain still dispatches int32
        self.assertTrue(iluvatar._use_int32((1 << 31) - 1, 4, 8, 1))
        # q.numel() == 2**31 must widen to int64 addressing
        self.assertFalse(iluvatar._use_int32(1 << 31, 4, 8, 1))
        # a strided indices view can push its own load offset past the
        # limit even when q is far below it
        self.assertFalse(iluvatar._use_int32(1024, 3, 8, 1 << 31))
        # inflated duplicate indices can push the OUTPUT past the
        # limit while q stays far below it (review r1 P1: q=(1,1,1,
        # 32768) with 65537 zero indices wraps dst=65536*32768=2**31
        # to a negative offset -> OOB write under the old guard)
        self.assertFalse(iluvatar._use_int32(32768, 65537, 32768, 1))
        # exact output-side boundary: n * row_elems == 2**31 widens,
        # one element below stays int32
        self.assertFalse(iluvatar._use_int32(1024, 2, 1 << 30, 1))
        self.assertTrue(iluvatar._use_int32(1024, 2, (1 << 30) - 1, 1))
        # empty gather never overflows
        self.assertTrue(iluvatar._use_int32(1024, 0, 8, 1))

    def test_generic_int32_domain_guard(self):
        # E3 generic host dispatch (same reviewed _use_int32 as the E2
        # iluvatar vendor, output side bounded on its own): int32
        # addressing is only legal while every flat offset stays below
        # 2**31. A wrong boundary compare silently turns into
        # out-of-bounds writes on the five generic-rider chips, so the
        # boundary arithmetic itself is the regression (the GPU matrix
        # above exercises the int32 kernel path; the i64 twin only
        # fires past 2**31 elements, which does not fit CI memory)
        generic = next(
            (mod for name, mod in MODULES if name == "generic"), None
        )
        # the generic module is always the first entry from
        # load_operator_modules; absence is a loader misconfiguration
        self.assertIsNotNone(generic, "generic module missing from MODULES")
        # largest legal domain still dispatches int32
        self.assertTrue(generic._use_int32((1 << 31) - 1, 4, 8, 1))
        # q.numel() == 2**31 must widen to int64 addressing
        self.assertFalse(generic._use_int32(1 << 31, 4, 8, 1))
        # a strided indices view can push its own load offset past the
        # limit even when q is far below it
        self.assertFalse(generic._use_int32(1024, 3, 8, 1 << 31))
        # inflated duplicate indices can push the OUTPUT past the
        # limit while q stays far below it (q=(1,1,1,32768) with 65537
        # zero indices wraps dst=65536*32768=2**31 to a negative
        # offset -> OOB write under a q.numel()-only guard)
        self.assertFalse(generic._use_int32(32768, 65537, 32768, 1))
        # exact output-side boundary: n * row_elems == 2**31 widens,
        # one element below stays int32
        self.assertFalse(generic._use_int32(1024, 2, 1 << 30, 1))
        self.assertTrue(generic._use_int32(1024, 2, (1 << 30) - 1, 1))
        # empty gather never overflows
        self.assertTrue(generic._use_int32(1024, 0, 8, 1))
        # muxi compliance pin: the per-row tile must never exceed the
        # max_tile_size=2048 elements/program rule (chip-rulesets.md:29);
        # the s0 2D tile (4x1024=4096) was 2x over the limit
        self.assertLessEqual(generic._MAX_LANES, 2048)

    def test_generic_rows_per_prog_multirow(self):
        # E3 generic grid fallback: the launch stays under the 65535
        # Ascend coreDim cap via rows_per_prog = cdiv(n, 65535). The
        # evaluation domain (n <= B*S) rides rows_per_prog == 1, so the
        # multi-row loop body is pinned here explicitly with row_elems=1
        # (memory-cheap) across the cap boundary: 65535 keeps the pure
        # per-row grid exactly at the cap, 65536/65537 take 2 rows per
        # program
        cap = 65535
        for total in (65535, 65536, 65537):
            with self.subTest(total=total):
                rows_per_prog = (total + cap - 1) // cap
                grid = (total + rows_per_prog - 1) // rows_per_prog
                self.assertLessEqual(grid, cap)
                if total > cap:
                    self.assertGreaterEqual(rows_per_prog, 2)
                args = self.make_case(1, 65540, 1, 1, total)
                self.check(*args)

    def test_all_tokens_and_empty(self):
        q, k, v, _ = self.make_case(2, 128, 8, 64, 128)
        full = torch.randperm(2 * 128, device="cuda").to(torch.int32)
        self.check(q, k, v, full)
        empty = torch.empty(0, dtype=torch.int64, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                out = module.fused_pack_qkv(q, k, v, empty)
                self.assertEqual(out[0].shape, (0, 8, 64))


RELEASE_REQUIRED_TESTS = [
    "FusedPackQkvTest.test_diffusion_shapes_and_tails",
    "FusedPackQkvTest.test_indices_int64_and_duplicates",
    "FusedPackQkvTest.test_fp32_and_bf16",
    "FusedPackQkvTest.test_row_elems_beyond_block_cap",
    "FusedPackQkvTest.test_row_elems_wider_than_block",
    "FusedPackQkvTest.test_sliced_indices",
    "FusedPackQkvTest.test_non_contiguous_qkv",
    "FusedPackQkvTest.test_all_tokens_and_empty",
    "FusedPackQkvTest.test_iluvatar_int32_domain_guard",
    "FusedPackQkvTest.test_generic_int32_domain_guard",
    "FusedPackQkvTest.test_generic_rows_per_prog_multirow",
]

if __name__ == "__main__":
    unittest.main()
