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
        q = torch.randn(1, 350, 200, 1, dtype=torch.float16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        idx = torch.tensor([3, 70000 - 1], dtype=torch.int64, device="cuda")
        self.check(q, k, v, idx)

    def test_row_elems_wider_than_block(self):
        # H*D=2048 == BLOCK_C forces at least two column iterations
        args = self.make_case(1, 512, 16, 128, 700)
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
]

if __name__ == "__main__":
    unittest.main()
