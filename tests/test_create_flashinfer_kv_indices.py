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
):
    bs = len(lengths)
    starts = [i % 17 for i in range(bs)]
    context = max(lengths, default=0) + 17
    pool = torch.randint(
        -(2**30), 2**30, (5, context), dtype=torch.int32, device="cuda"
    )
    pointers = [1]
    for n in lengths:
        pointers.append(pointers[-1] + n + 3)
    vectors = [
        torch.tensor(values, dtype=dtype, device="cuda")
        for values in ([i % 5 for i in range(bs)], lengths, pointers, starts)
    ]
    old = torch.randint(
        -10000, 10000, (pointers[-1] + 5,), dtype=torch.int32, device="cuda"
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


RELEASE_REQUIRED_TESTS = [
    "KVIndicesTest.test_segments_offsets_and_tails",
    "KVIndicesTest.test_strides_and_index_dtypes",
    "KVIndicesTest.test_empty_and_all_zero_segments",
    "KVIndicesTest.test_repeated_calls_preserve_changed_base",
]


if __name__ == "__main__":
    unittest.main()
