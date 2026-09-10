# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("deepep_permute")


def reference(input, gateup_input, src2dst, topk_ids, topk, hidden_size):
    out = gateup_input.clone()
    dst = src2dst.reshape(-1)
    valid = dst >= 0
    rows = input.repeat_interleave(topk, dim=0).to(out.dtype)
    out[dst[valid].long()] = rows[valid]
    return out


def make_case(
    tokens=7,
    topk=4,
    hidden=1025,
    dtype=torch.bfloat16,
    out_dtype=None,
    strided=False,
):
    x = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
    out = torch.randn(
        tokens * topk, hidden, dtype=out_dtype or dtype, device="cuda"
    )
    routes = (
        torch.randperm(tokens * topk, device="cuda")
        .to(torch.int32)
        .reshape(tokens, topk)
    )
    routes.reshape(-1)[::5] = -1
    ids = torch.full((tokens, topk), -1, dtype=torch.int64, device="cuda")
    if strided:
        x = x.t().contiguous().t()
        out = out.t().contiguous().t()
        storage = torch.empty(
            tokens * 2 + 1, topk * 2 + 1, device="cuda", dtype=torch.int32
        )
        storage[1::2, 1::2] = routes
        routes = storage[1::2, 1::2]
    return x, out, routes, ids, topk, hidden


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class DeepEPPermuteTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() if x is not None else None for x in args[:4]]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.deepep_permute(*args)
                torch.testing.assert_close(
                    actual, expected, rtol=0, atol=0, equal_nan=True
                )
                for value, before in zip(args[:4], snapshots):
                    if value is not None:
                        torch.testing.assert_close(
                            value, before, rtol=0, atol=0, equal_nan=True
                        )
                if actual.numel():
                    self.assertNotEqual(actual.data_ptr(), args[1].data_ptr())

    def test_dtypes_and_cast(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self.check(make_case(dtype=dtype))
                self.check(make_case(dtype=torch.float32, out_dtype=dtype))

    def test_topk_tiles_and_large_hidden(self):
        for topk in (1, 2, 8):
            for hidden in (1, 511, 512, 513, 4096, 7168):
                with self.subTest(topk=topk, hidden=hidden):
                    self.check(make_case(tokens=3, topk=topk, hidden=hidden))

    def test_strides_and_unused_topk_ids(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            args = make_case(dtype=dtype, strided=True)
            self.check(args)
            self.check((*args[:3], None, *args[4:]))
        x, out, routes, ids, topk, hidden = make_case(hidden=513)
        storage = torch.empty(
            x.shape[0], hidden * 2 + 1, device="cuda", dtype=x.dtype
        )
        storage[:, 1::2] = x
        self.check(
            (storage[:, 1::2], out, routes.to(torch.int64), ids, topk, hidden)
        )

    def test_empty_and_all_invalid(self):
        for tokens, topk, hidden in ((0, 4, 513), (3, 0, 513), (3, 4, 0)):
            self.check(make_case(tokens=tokens, topk=topk, hidden=hidden))
        args = make_case()
        args[2].fill_(-1)
        self.check(args)
        args[1].fill_(37)
        self.check(args)

    def test_special_values(self):
        args = make_case(dtype=torch.float32)
        args[0][:, :4] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0], device="cuda"
        )
        self.check(args)


RELEASE_REQUIRED_TESTS = [
    "DeepEPPermuteTest.test_dtypes_and_cast",
    "DeepEPPermuteTest.test_topk_tiles_and_large_hidden",
    "DeepEPPermuteTest.test_strides_and_unused_topk_ids",
    "DeepEPPermuteTest.test_empty_and_all_invalid",
    "DeepEPPermuteTest.test_special_values",
]


if __name__ == "__main__":
    unittest.main()
