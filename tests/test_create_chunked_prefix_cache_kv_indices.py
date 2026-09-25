# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("create_chunked_prefix_cache_kv_indices")


def reference(
    req_to_token,
    req_pool_indices,
    chunk_start_idx,
    chunk_seq_lens,
    chunk_cu_seq_lens,
    chunk_kv_indices,
):
    out = chunk_kv_indices.clone()
    for i in range(req_pool_indices.shape[0]):
        beg = int(chunk_cu_seq_lens[i])
        n = int(chunk_seq_lens[i])
        start = int(chunk_start_idx[i])
        pool = int(req_pool_indices[i])
        out[beg : beg + n] = req_to_token[pool, start : start + n]
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class CreateChunkedPrefixCacheKvIndicesTest(unittest.TestCase):
    def check(self, req_to_token, pool_idx, starts, lens, cus, kv_indices):
        expected = reference(
            req_to_token, pool_idx, starts, lens, cus, kv_indices
        )
        snapshots = [x.clone() for x in (
            req_to_token, pool_idx, starts, lens, cus, kv_indices
        )]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.create_chunked_prefix_cache_kv_indices(
                    req_to_token, pool_idx, starts, lens, cus, kv_indices
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for value, before in zip(
                    (
                        req_to_token,
                        pool_idx,
                        starts,
                        lens,
                        cus,
                        kv_indices,
                    ),
                    snapshots,
                ):
                    torch.testing.assert_close(value, before, rtol=0, atol=0)

    def make_case(self, n_req, max_ctx, total_tokens, *, max_len=1024, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        req_to_token = torch.randint(
            0, 2**30, (128, max_ctx), dtype=torch.int32, device="cuda"
        )
        pool_idx = torch.randperm(128, generator=gen, device="cuda")[
            :n_req
        ].to(torch.int32)
        lens = torch.randint(
            1, max_len, (n_req,), dtype=torch.int32, device="cuda", generator=gen
        )
        cus = torch.zeros(n_req, dtype=torch.int32, device="cuda")
        cus[1:] = torch.cumsum(lens, 0)[:-1]
        starts = torch.randint(
            0,
            max(1, max_ctx - int(lens.max())),
            (n_req,),
            dtype=torch.int32,
            device="cuda",
            generator=gen,
        )
        sentinel = torch.tensor(
            [123456789] * total_tokens, dtype=torch.int32, device="cuda"
        )
        kv_indices = sentinel
        return req_to_token, pool_idx, starts, lens, cus, kv_indices

    def test_basic_chunks(self):
        self.check(*self.make_case(8, 4096, 8192, max_len=1024, seed=1))

    def test_single_request_and_lengths(self):
        for n_req in (1, 2):
            with self.subTest(n_req=n_req):
                self.check(*self.make_case(n_req, 2048, 4096, max_len=64, seed=2))

    def test_len_crossing_block(self):
        # lengths straddling BLOCK_C=1024
        n_req = 5
        req_to_token, pool_idx, _, _, _, _ = self.make_case(
            n_req, 4096, 8192, max_len=1024, seed=3
        )
        lens = torch.tensor(
            [1, 1023, 1024, 1025, 2048], dtype=torch.int32, device="cuda"
        )
        cus = torch.zeros(n_req, dtype=torch.int32, device="cuda")
        cus[1:] = torch.cumsum(lens, 0)[:-1]
        starts = torch.tensor(
            [0, 100, 200, 300, 400], dtype=torch.int32, device="cuda"
        )
        kv_indices = torch.full(
            (int(lens.sum()),), -1, dtype=torch.int32, device="cuda"
        )
        self.check(req_to_token, pool_idx, starts, lens, cus, kv_indices)

    def test_zero_length_row(self):
        n_req = 4
        req_to_token, pool_idx, _, _, _, _ = self.make_case(
            n_req, 2048, 4096, max_len=64, seed=4
        )
        lens = torch.tensor([32, 0, 17, 1], dtype=torch.int32, device="cuda")
        cus = torch.zeros(n_req, dtype=torch.int32, device="cuda")
        cus[1:] = torch.cumsum(lens, 0)[:-1]
        starts = torch.tensor([5, 0, 50, 10], dtype=torch.int32, device="cuda")
        kv_indices = torch.full(
            (int(lens.sum()) + 8,), -7, dtype=torch.int32, device="cuda"
        )
        self.check(req_to_token, pool_idx, starts, lens, cus, kv_indices)

    def test_tail_sentinel_preserved(self):
        # tokens beyond the written window keep the clone of the base
        n_req = 3
        args = self.make_case(n_req, 1024, 2048, max_len=128, seed=5)
        req_to_token, pool_idx, starts, lens, cus, kv_indices = args
        actual = MODULES[0][1].create_chunked_prefix_cache_kv_indices(
            req_to_token, pool_idx, starts, lens, cus, kv_indices
        )
        written = int(lens.sum())
        torch.testing.assert_close(
            actual[written:], kv_indices[written:], rtol=0, atol=0
        )

    def test_column_strided_req_to_token(self):
        # a column-sliced pool view carries stride(1)=2: token windows
        # must follow the runtime column stride (review finding)
        base = torch.arange(64, device="cuda", dtype=torch.int32).reshape(4, 16)
        req_to_token = base[:, ::2]
        self.assertEqual(req_to_token.stride(1), 2)
        pool_idx = torch.tensor([1], dtype=torch.int32, device="cuda")
        starts = torch.tensor([2], dtype=torch.int32, device="cuda")
        lens = torch.tensor([3], dtype=torch.int32, device="cuda")
        cus = torch.tensor([0], dtype=torch.int32, device="cuda")
        kv_indices = torch.full((5,), -1, dtype=torch.int32, device="cuda")
        self.check(req_to_token, pool_idx, starts, lens, cus, kv_indices)

    def test_unsorted_and_overlapping_windows(self):
        # cu is not guaranteed sorted and windows may not tile: the fill
        # must preserve base bytes everywhere outside the union, even
        # with out-of-order and zero-length requests (review findings)
        req_to_token = torch.arange(
            512, device="cuda", dtype=torch.int32
        ).reshape(8, 64)
        pool_idx = torch.tensor([2, 0, 5], dtype=torch.int32, device="cuda")
        starts = torch.tensor([1, 7, 3], dtype=torch.int32, device="cuda")
        lens = torch.tensor([3, 0, 2], dtype=torch.int32, device="cuda")
        cus = torch.tensor([6, 2, 2], dtype=torch.int32, device="cuda")
        kv_indices = torch.arange(12, device="cuda", dtype=torch.int32) * 7
        self.check(req_to_token, pool_idx, starts, lens, cus, kv_indices)

    def test_nested_windows_consistent_sources(self):
        # nested windows whose sources agree at the overlap: the fill
        # must not restore base bytes inside the union (protects the
        # intersection-scan fix; positions 5-7 would regress otherwise)
        base = torch.arange(16, device="cuda", dtype=torch.int32) * 100
        req_to_token = torch.arange(
            64, device="cuda", dtype=torch.int32
        ).reshape(1, 64)
        pool_idx = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
        starts = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
        lens = torch.tensor([7, 2], dtype=torch.int32, device="cuda")
        cus = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
        self.check(req_to_token, pool_idx, starts, lens, cus, base)

    def test_strided_base(self):
        # a non-contiguous base view: preserved bytes must come from the
        # base's own stride, not a flat memcpy (review finding)
        req_to_token = torch.arange(128, device="cuda", dtype=torch.int32).reshape(2, 64)
        pool_idx = torch.tensor([1], dtype=torch.int32, device="cuda")
        starts = torch.tensor([2], dtype=torch.int32, device="cuda")
        lens = torch.tensor([2], dtype=torch.int32, device="cuda")
        cus = torch.tensor([0], dtype=torch.int32, device="cuda")
        big = torch.arange(40, device="cuda", dtype=torch.int32) * 3
        kv_indices = big[::2]
        self.assertFalse(kv_indices.is_contiguous())
        self.check(req_to_token, pool_idx, starts, lens, cus, kv_indices)

    def test_gapped_cu_layout(self):
        # windows need not tile the buffer from zero: elements outside
        # every window keep the base bytes (fill kernel must cover gaps)
        req_to_token = torch.arange(256, device="cuda", dtype=torch.int32).reshape(4, 64)
        pool_idx = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
        starts = torch.tensor([2, 5], dtype=torch.int32, device="cuda")
        lens = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
        cus = torch.tensor([1, 5], dtype=torch.int32, device="cuda")
        kv_indices = torch.arange(10, device="cuda", dtype=torch.int32) * 10
        self.check(req_to_token, pool_idx, starts, lens, cus, kv_indices)

    def test_empty_requests(self):
        req_to_token = torch.randint(
            0, 100, (8, 64), dtype=torch.int32, device="cuda"
        )
        pool_idx = torch.empty(0, dtype=torch.int32, device="cuda")
        starts = torch.empty(0, dtype=torch.int32, device="cuda")
        lens = torch.empty(0, dtype=torch.int32, device="cuda")
        cus = torch.empty(0, dtype=torch.int32, device="cuda")
        kv_indices = torch.full((16,), -1, dtype=torch.int32, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.create_chunked_prefix_cache_kv_indices(
                    req_to_token, pool_idx, starts, lens, cus, kv_indices
                )
                torch.testing.assert_close(
                    actual, kv_indices.clone(), rtol=0, atol=0
                )


RELEASE_REQUIRED_TESTS = [
    "CreateChunkedPrefixCacheKvIndicesTest.test_basic_chunks",
    "CreateChunkedPrefixCacheKvIndicesTest.test_single_request_and_lengths",
    "CreateChunkedPrefixCacheKvIndicesTest.test_len_crossing_block",
    "CreateChunkedPrefixCacheKvIndicesTest.test_zero_length_row",
    "CreateChunkedPrefixCacheKvIndicesTest.test_tail_sentinel_preserved",
    "CreateChunkedPrefixCacheKvIndicesTest.test_column_strided_req_to_token",
    "CreateChunkedPrefixCacheKvIndicesTest.test_gapped_cu_layout",
    "CreateChunkedPrefixCacheKvIndicesTest.test_unsorted_and_overlapping_windows",
    "CreateChunkedPrefixCacheKvIndicesTest.test_strided_base",
    "CreateChunkedPrefixCacheKvIndicesTest.test_nested_windows_consistent_sources",
    "CreateChunkedPrefixCacheKvIndicesTest.test_empty_requests",
]

if __name__ == "__main__":
    unittest.main()
