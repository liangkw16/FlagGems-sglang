# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("deepep_post_reorder")


def reference(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    hidden_size,
    routed_scaling_factor,
):
    acc = torch.zeros(output.shape, dtype=torch.float32, device=output.device)
    for i in range(topk):
        dst = src2dst[:, i]
        valid = (dst >= 0).float()
        rows = down_output[dst.clamp(min=0).long()].float()
        w = (
            topk_weights[:, i].to(down_output.dtype).float()
            * routed_scaling_factor
        )
        acc += rows * (w * valid)[:, None]
    return acc.to(output.dtype)


def make_case(
    tokens=7,
    topk=4,
    hidden=1025,
    dtype=torch.bfloat16,
    out_dtype=None,
    weights_dtype=None,
    scaling=2.5,
    strided=False,
):
    down = torch.randn(tokens * topk, hidden, dtype=dtype, device="cuda") * (
        torch.rand(tokens * topk, hidden, device="cuda") < 0.5
    )
    out = torch.randn(tokens, hidden, dtype=out_dtype or dtype, device="cuda")
    routes = (
        torch.randperm(tokens * topk, device="cuda")
        .to(torch.int32)
        .reshape(tokens, topk)
    )
    routes.reshape(-1)[::5] = -1
    weights = torch.rand(
        tokens, topk, dtype=weights_dtype or dtype, device="cuda"
    )
    ids = torch.full((tokens, topk), -1, dtype=torch.int64, device="cuda")
    if strided:
        down = down.t().contiguous().t()
        storage = torch.empty(
            tokens * 2 + 1, topk * 2 + 1, device="cuda", dtype=torch.int32
        )
        storage[1::2, 1::2] = routes
        routes = storage[1::2, 1::2]
    return down, out, routes, ids, weights, topk, hidden, scaling


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class DeepEPPostReorderTest(unittest.TestCase):
    def check(self, args):
        expected = reference(*args)
        snapshots = [x.clone() if torch.is_tensor(x) else x for x in args[:5]]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.deepep_post_reorder(*args)
                torch.testing.assert_close(
                    actual, expected, rtol=3e-2, atol=3e-2
                )
                for value, before in zip(args[:5], snapshots):
                    if torch.is_tensor(value):
                        torch.testing.assert_close(
                            value, before, rtol=0, atol=0, equal_nan=True
                        )
                if actual.numel():
                    self.assertNotEqual(actual.data_ptr(), args[1].data_ptr())

    def test_dtypes_and_cast(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self.check(make_case(dtype=dtype))
                self.check(make_case(dtype=dtype, out_dtype=torch.bfloat16))
                self.check(make_case(dtype=dtype, weights_dtype=torch.float32))

    def test_topk_tiles_and_large_hidden(self):
        for topk in (1, 2, 8):
            for hidden in (1, 511, 512, 513, 4096, 7168):
                with self.subTest(topk=topk, hidden=hidden):
                    self.check(make_case(tokens=3, topk=topk, hidden=hidden))
        self.check(make_case(tokens=8193, topk=1, hidden=4097))
        for scaling in (1.0, 0.0625):
            with self.subTest(scaling=scaling):
                self.check(make_case(scaling=scaling))

    def test_strides(self):
        args = make_case(strided=True)
        self.check(args)
        x, out, routes, ids, w, topk, hidden, s = args
        self.check((x, out, routes.to(torch.int64), ids, w, topk, hidden, s))

    def test_empty_and_all_invalid(self):
        for tokens, topk, hidden in ((0, 4, 513), (3, 0, 513), (3, 4, 0)):
            self.check(make_case(tokens=tokens, topk=topk, hidden=hidden))
        args = make_case()
        args[2].fill_(-1)
        self.check(args)


RELEASE_REQUIRED_TESTS = [
    "DeepEPPostReorderTest.test_dtypes_and_cast",
    "DeepEPPostReorderTest.test_topk_tiles_and_large_hidden",
    "DeepEPPostReorderTest.test_strides",
    "DeepEPPostReorderTest.test_empty_and_all_invalid",
]


if __name__ == "__main__":
    unittest.main()
