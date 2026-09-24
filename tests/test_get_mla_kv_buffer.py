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


RELEASE_REQUIRED_TESTS = [
    "GetMlaKvBufferTest.test_mla_realistic_shapes",
    "GetMlaKvBufferTest.test_cross_dtype_store",
    "GetMlaKvBufferTest.test_loc_int64_and_duplicates",
    "GetMlaKvBufferTest.test_strided_kv_buffer",
    "GetMlaKvBufferTest.test_dim_edges",
    "GetMlaKvBufferTest.test_empty_rows",
]

if __name__ == "__main__":
    unittest.main()
