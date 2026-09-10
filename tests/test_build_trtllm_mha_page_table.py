# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("build_trtllm_mha_page_table")


def reference(
    req_to_token, req_pool_indices, cache_seqlens, page_table, page_size
):
    out = page_table.clone()
    bs = req_pool_indices.shape[0]
    pages = torch.arange(out.shape[1], device=out.device).unsqueeze(0)
    mask = pages < ((cache_seqlens + page_size - 1) // page_size).unsqueeze(1)
    slots = req_to_token[
        req_pool_indices.long().unsqueeze(1),
        (pages * page_size).expand(bs, -1),
    ]
    out[mask] = (slots // page_size)[mask].to(out.dtype)
    return out


def make_case(
    bs=7, columns=257, page_size=64, dtype=torch.int32, strided=False
):
    context = max(1, columns * page_size)
    pool = torch.randint(
        -4097, 2**20, (8, context), dtype=torch.int32, device="cuda"
    )
    requests = torch.arange(bs, device="cuda", dtype=dtype) % 8
    choices = [
        0,
        1,
        page_size - 1,
        page_size,
        page_size + 1,
        context - 1,
        context,
    ]
    lengths = torch.tensor(
        [choices[i % len(choices)] for i in range(bs)],
        dtype=dtype,
        device="cuda",
    )
    old = torch.randint(
        -10000, 10000, (bs, columns), dtype=torch.int32, device="cuda"
    )
    if strided:
        pool = pool.t().contiguous().t()
        old = old.t().contiguous().t()

        def spaced(x):
            storage = torch.empty(
                x.numel() * 2 + 1, dtype=x.dtype, device=x.device
            )
            storage[1::2].copy_(x)
            return storage[1::2]

        requests, lengths = spaced(requests), spaced(lengths)
    return pool, requests, lengths, old, page_size


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class PageTableTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() for x in args[:-1]]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.build_trtllm_mha_page_table(*args)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for value, before in zip(args[:-1], snapshots):
                    torch.testing.assert_close(value, before, rtol=0, atol=0)
                if actual.numel():
                    self.assertNotEqual(actual.data_ptr(), args[3].data_ptr())

    def test_page_sizes_and_sentinels(self):
        for shift in range(13):
            with self.subTest(page_size=1 << shift):
                self.check(make_case(columns=17, page_size=1 << shift))

    def test_tiles_and_multiple_pages(self):
        for columns in (255, 256, 257, 511, 512, 513, 1025):
            with self.subTest(columns=columns):
                args = make_case(bs=1, columns=columns, page_size=1)
                args[2].fill_(columns)
                self.check(args)
        self.check(make_case(bs=9, columns=1031, page_size=32))

    def test_strides_and_index_dtypes(self):
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                self.check(make_case(dtype=dtype, strided=True))
        args = make_case(bs=5, columns=17)
        expanded_pool = torch.empty(
            8, args[0].shape[1] * 2, dtype=torch.int32, device="cuda"
        )
        expanded_pool[:, ::2] = args[0]
        self.check((expanded_pool[:, ::2], *args[1:]))

    def test_empty_and_zero_lengths(self):
        for bs, columns in ((0, 0), (0, 17), (3, 0)):
            self.check(make_case(bs=bs, columns=columns))
        args = make_case()
        args[2].zero_()
        self.check(args)


RELEASE_REQUIRED_TESTS = [
    "PageTableTest.test_page_sizes_and_sentinels",
    "PageTableTest.test_tiles_and_multiple_pages",
    "PageTableTest.test_strides_and_index_dtypes",
    "PageTableTest.test_empty_and_zero_lengths",
]


if __name__ == "__main__":
    unittest.main()
