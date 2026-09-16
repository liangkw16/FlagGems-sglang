# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("create_flashinfer_kv_indices")


def reference(
    req_to_token,
    req_pool_indices,
    page_kernel_lens,
    kv_indptr,
    kv_start_idx,
    kv_indices,
):
    out = kv_indices.clone()
    for i in range(req_pool_indices.numel()):
        beg = int(kv_indptr[i])
        n = int(page_kernel_lens[i])
        start = int(kv_start_idx[i]) if kv_start_idx is not None else 0
        pool = int(req_pool_indices[i])
        out[beg : beg + n] = req_to_token[pool, start : start + n]
    return out


def make_case(
    lengths=(0, 1, 511, 512, 513, 1025),
    has_start=True,
    strided=False,
    dtype=torch.int32,
    context=None,
    head=1,
    gap=3,
    tail=5,
):
    bs = len(lengths)
    starts = [i % 17 for i in range(bs)]
    if context is None:
        context = max(lengths, default=0) + 17
    pool = torch.randint(
        -(2**30), 2**30, (5, context), dtype=torch.int32, device="cuda"
    )
    pointers = [head]
    for n in lengths:
        pointers.append(pointers[-1] + n + gap)
    vectors = [
        torch.tensor(values, dtype=dtype, device="cuda")
        for values in ([i % 5 for i in range(bs)], lengths, pointers, starts)
    ]
    old = torch.randint(
        -10000, 10000, (pointers[-1] + tail,), dtype=torch.int32, device="cuda"
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
        old = spaced(old)
    requests, lens, indptr, start = vectors
    return pool, requests, lens, indptr, start if has_start else None, old


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class KVIndicesTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() if x is not None else None for x in args]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.create_flashinfer_kv_indices(*args)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for value, before in zip(args, snapshots):
                    if value is not None:
                        torch.testing.assert_close(
                            value, before, rtol=0, atol=0
                        )
                if actual.numel():
                    self.assertNotEqual(actual.data_ptr(), args[-1].data_ptr())

    def test_segments_offsets_and_tails(self):
        for has_start in (False, True):
            with self.subTest(has_start=has_start):
                self.check(make_case(has_start=has_start))
                self.check(make_case(lengths=(8193,), has_start=has_start))
                self.check(make_case(lengths=(32769,), has_start=has_start))
        self.check(make_case(lengths=(0, 1, 513) * 43))

    def test_strides_and_index_dtypes(self):
        for dtype in (torch.int32, torch.int64):
            for has_start in (False, True):
                with self.subTest(dtype=dtype, has_start=has_start):
                    self.check(
                        make_case(
                            strided=True, dtype=dtype, has_start=has_start
                        )
                    )
        args = make_case(lengths=(1, 513))
        pool = torch.empty(
            5, args[0].shape[1] * 2, dtype=torch.int32, device="cuda"
        )
        pool[:, ::2] = args[0]
        self.check((pool[:, ::2], *args[1:]))

    def test_empty_and_all_zero_segments(self):
        for lengths in ((), (0,), (0, 0, 0)):
            self.check(make_case(lengths=lengths))
        args = make_case(lengths=())
        self.check(
            (*args[:-1], torch.empty(0, dtype=torch.int32, device="cuda"))
        )

    def test_repeated_calls_preserve_changed_base(self):
        args = make_case()
        self.check(args)
        args[-1].fill_(123456)
        args[0].fill_(-17)
        self.check(args)

    def test_split_geometry_boundaries(self):
        for width in (1, 511, 512, 513, 4095, 4096, 4097, 8191, 8192, 8193):
            with self.subTest(width=width):
                self.check(
                    make_case(lengths=(width,), has_start=False, context=width)
                )
        for batch in (31, 32, 33, 127, 128, 129, 257):
            with self.subTest(batch=batch):
                self.check(
                    make_case(
                        lengths=(513,) * batch, context=530, has_start=True
                    )
                )

    def test_long_preserved_regions(self):
        # These regions can exceed the pool width. Every split must stride
        # through all its old-buffer tiles after the split count shrinks.
        for strided in (False, True):
            with self.subTest(strided=strided):
                self.check(
                    make_case(
                        lengths=(17, 0, 29),
                        context=4096,
                        head=8193,
                        gap=16385,
                        tail=32769,
                        strided=strided,
                        dtype=torch.int64,
                    )
                )


RELEASE_REQUIRED_TESTS = [
    "KVIndicesTest.test_segments_offsets_and_tails",
    "KVIndicesTest.test_strides_and_index_dtypes",
    "KVIndicesTest.test_empty_and_all_zero_segments",
    "KVIndicesTest.test_repeated_calls_preserve_changed_base",
    "KVIndicesTest.test_split_geometry_boundaries",
    "KVIndicesTest.test_long_preserved_regions",
]


if __name__ == "__main__":
    unittest.main()
