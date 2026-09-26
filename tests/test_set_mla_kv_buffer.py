# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("set_mla_kv_buffer")


def reference(kv_buffer, loc, cache_k_nope, cache_k_rope):
    out = kv_buffer.clone()
    row = torch.cat([cache_k_nope, cache_k_rope], dim=-1).to(out.dtype)
    out[loc.long()] = row
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class SetMlaKvBufferTest(unittest.TestCase):
    def check(self, kv, loc, nd, rd, ndt=torch.float16, rdt=torch.float16):
        nope = torch.randn(loc.shape[0], nd, dtype=ndt, device="cuda")
        rope = torch.randn(loc.shape[0], rd, dtype=rdt, device="cuda")
        expected = reference(kv, loc, nope, rope)
        snap = kv.clone()
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.set_mla_kv_buffer(kv, loc, nope, rope)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(kv, snap, rtol=0, atol=0)

    def test_shapes_and_tails(self):
        for n, nd, rd in ((1, 512, 64), (37, 512, 64), (256, 576, 512), (1024, 512, 64), (70, 32, 96)):
            with self.subTest(n=n, nd=nd, rd=rd):
                kv = torch.randn(n + 8, nd + rd, dtype=torch.float16, device="cuda") * 9
                loc = torch.randperm(n, device="cuda").to(torch.int64)
                self.check(kv, loc, nd, rd)

    def test_cross_dtype_and_int32_loc(self):
        kv = torch.randn(16, 576, dtype=torch.bfloat16, device="cuda")
        loc = torch.randperm(8, device="cuda").to(torch.int32)
        self.check(kv, loc, 512, 64, torch.bfloat16, torch.float16)
        self.check(kv, loc.long(), 512, 64, torch.float32, torch.bfloat16)

    def test_transposed_sources(self):
        kv = torch.randn(8, 576, dtype=torch.float16, device="cuda")
        loc = torch.arange(4, dtype=torch.int64, device="cuda")
        nd = torch.randn(4, 512, dtype=torch.float16, device="cuda").t().contiguous().t()  # values keep, stride col-major
        nope = torch.randn(512, 4, dtype=torch.float16, device="cuda").t()
        rope = torch.randn(64, 4, dtype=torch.float16, device="cuda").t()
        self.assertFalse(nope.is_contiguous())
        expected = reference(kv, loc, nope, rope)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.set_mla_kv_buffer(kv, loc, nope, rope)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_duplicates_and_empty(self):
        # duplicate slots must carry IDENTICAL source rows: differing
        # values racing on one slot are unordered even in the reference
        kv = torch.randn(12, 576, dtype=torch.float16, device="cuda")
        dup = torch.full((5,), 7, dtype=torch.int64, device="cuda")
        nope = torch.randn(1, 512, dtype=torch.float16, device="cuda").expand(5, 512).contiguous()
        rope = torch.randn(1, 64, dtype=torch.float16, device="cuda").expand(5, 64).contiguous()
        expected = reference(kv, dup, nope, rope)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.set_mla_kv_buffer(kv, dup, nope, rope)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        empty = torch.empty(0, dtype=torch.int64, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                out = module.set_mla_kv_buffer(
                    kv, empty,
                    torch.empty(0, 512, dtype=torch.float16, device="cuda"),
                    torch.empty(0, 64, dtype=torch.float16, device="cuda"),
                )
                torch.testing.assert_close(out, kv, rtol=0, atol=0)


RELEASE_REQUIRED_TESTS = [
    "SetMlaKvBufferTest.test_shapes_and_tails",
    "SetMlaKvBufferTest.test_cross_dtype_and_int32_loc",
    "SetMlaKvBufferTest.test_transposed_sources",
    "SetMlaKvBufferTest.test_duplicates_and_empty",
]

if __name__ == "__main__":
    unittest.main()
