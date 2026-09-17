# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("create_flashmla_kv_indices")


def reference(
    req_to_token,
    req_pool_indices,
    page_kernel_lens,
    kv_start_idx,
    kv_indices,
    page_size,
):
    out = kv_indices.clone()
    for i in range(req_pool_indices.shape[0]):
        n = int(page_kernel_lens[i])
        start = int(kv_start_idx[i]) if kv_start_idx is not None else 0
        pool = int(req_pool_indices[i])
        num_pages = (n + page_size - 1) // page_size
        slots = req_to_token[
            pool, start : start + num_pages * page_size : page_size
        ]
        out[i, :num_pages] = slots // page_size
    return out


def make_case(
    lengths=(0, 1, 63, 64, 65, 511, 512, 513, 1025),
    page_size=64,
    has_start=True,
    strided=False,
    dtype=torch.int32,
    context=None,
    width=None,
):
    bs = len(lengths)
    starts = [i % 17 for i in range(bs)]
    if context is None:
        context = (
            max(lengths, default=0)
            + max(starts, default=0)
            + 3 * page_size
        )
    if width is None:
        width = (
            max(
                [(n + page_size - 1) // page_size for n in lengths],
                default=0,
            )
            + 5
        )
    pool = torch.randint(
        0, 2**30, (5, context), dtype=torch.int32, device="cuda"
    )
    vectors = [
        torch.tensor(values, dtype=dtype, device="cuda")
        for values in ([i % 5 for i in range(bs)], lengths, starts)
    ]
    old = torch.randint(
        -10000, 10000, (bs, width), dtype=torch.int32, device="cuda"
    )
    if strided:
        pool = pool.t().contiguous().t()

        def spaced(x):
            storage = torch.empty(
                x.numel() * 2 + 1, dtype=x.dtype, device=x.device
            )
            storage[1::2].copy_(x)
            return storage[1::2]

        vectors = [spaced(x) for x in vectors]
    requests, lens, start = vectors
    return pool, requests, lens, start if has_start else None, old, page_size


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class FlashMLAKVIndicesTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() if x is not None else None for x in args[:-1]]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.create_flashmla_kv_indices(*args)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for value, before in zip(args[:-1], snapshots):
                    if value is not None:
                        torch.testing.assert_close(
                            value, before, rtol=0, atol=0
                        )
                if actual.numel():
                    self.assertNotEqual(actual.data_ptr(), args[-2].data_ptr())

    def test_page_sizes_and_boundaries(self):
        for page_size in (1, 3, 16, 64, 128):
            for has_start in (False, True):
                with self.subTest(page_size=page_size, has_start=has_start):
                    self.check(
                        make_case(page_size=page_size, has_start=has_start)
                    )

    def test_nondivisible_and_long_rows(self):
        self.check(make_case(lengths=(8193,), page_size=64))
        self.check(make_case(lengths=(32769,), page_size=128))
        self.check(make_case(lengths=(65, 1, 2) * 43, page_size=3))

    def test_strides_and_index_dtypes(self):
        for dtype in (torch.int32, torch.int64):
            for has_start in (False, True):
                with self.subTest(dtype=dtype, has_start=has_start):
                    self.check(
                        make_case(
                            strided=True,
                            dtype=dtype,
                            has_start=has_start,
                        )
                    )
        args = make_case()
        pool = torch.empty(
            5, args[0].shape[1] * 2, dtype=torch.int32, device="cuda"
        )
        pool[:, ::2] = args[0]
        self.check((pool[:, ::2], *args[1:]))

    def test_empty_batch_and_zero_rows(self):
        for lengths in ((), (0,), (0, 0, 0)):
            self.check(make_case(lengths=lengths))
        args = make_case(lengths=())
        self.check(
            (
                *args[:-2],
                torch.empty(0, 7, dtype=torch.int32, device="cuda"),
                args[-1],
            )
        )

    def test_tail_preservation_and_changed_base(self):
        # Row tails past each row's valid pages must keep the base
        # tensor's bytes, including after the base is mutated between
        # calls (the kernel re-reads it, no stale clone).
        args = make_case(lengths=(64, 129, 0), page_size=64, width=16)
        self.check(args)
        args[4].fill_(123456)
        args[0].fill_(7777)
        self.check(args)

    def test_width_equals_pages(self):
        # width exactly covering every row's page count: zero-length
        # tails everywhere, including the width-0 edge.
        self.check(
            make_case(lengths=(64, 128), page_size=64, width=2)
        )
        args = make_case(lengths=(0,), page_size=64, width=0)
        self.check(args)

    def test_split_geometry_boundaries(self):
        for width in (1, 255, 256, 257, 511, 512, 513):
            with self.subTest(width=width):
                self.check(
                    make_case(
                        lengths=(width * 64,),
                        page_size=64,
                        width=width,
                        has_start=False,
                    )
                )
        for batch in (31, 32, 33, 127, 128, 129, 257):
            with self.subTest(batch=batch):
                self.check(
                    make_case(
                        lengths=(513,) * batch, page_size=64, has_start=True
                    )
                )


RELEASE_REQUIRED_TESTS = [
    "FlashMLAKVIndicesTest.test_page_sizes_and_boundaries",
    "FlashMLAKVIndicesTest.test_nondivisible_and_long_rows",
    "FlashMLAKVIndicesTest.test_strides_and_index_dtypes",
    "FlashMLAKVIndicesTest.test_empty_batch_and_zero_rows",
    "FlashMLAKVIndicesTest.test_tail_preservation_and_changed_base",
    "FlashMLAKVIndicesTest.test_width_equals_pages",
    "FlashMLAKVIndicesTest.test_split_geometry_boundaries",
]


if __name__ == "__main__":
    unittest.main()
