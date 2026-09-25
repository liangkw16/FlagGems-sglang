# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("pad_draft_extend_query")


def reference(q, padded_q, seq_lens_q, cu_seqlens_q):
    out = padded_q.clone()
    bs = cu_seqlens_q.shape[0] - 1
    for b in range(bs):
        s = int(seq_lens_q[b])
        beg = int(cu_seqlens_q[b])
        out[b, :s] = q[beg : beg + s]
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class PadDraftExtendQueryTest(unittest.TestCase):
    def check(self, q, padded_q, lens, cu):
        expected = reference(q, padded_q, lens, cu)
        snap = padded_q.clone()
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.pad_draft_extend_query(q, padded_q, lens, cu)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(padded_q, snap, rtol=0, atol=0)

    def make(self, bs, max_seq, h, d, lens_list, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        total = sum(lens_list)
        q = torch.randn(total, h, d, dtype=torch.float16, device="cuda", generator=gen)
        pad = torch.randn(bs, max_seq, h, d, dtype=torch.float16, device="cuda", generator=gen) * 100
        lens = torch.tensor(lens_list, dtype=torch.int32, device="cuda")
        cu = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
        cu[1:] = torch.cumsum(lens, 0)
        return q, pad, lens, cu

    def test_typical_and_edges(self):
        cases = [
            (4, 64, 8, 64, [64, 32, 1, 64]),
            (2, 128, 16, 128, [128, 0]),
            (3, 33, 4, 40, [33, 17, 1]),
            (1, 256, 2, 32, [256]),
        ]
        for args in cases:
            with self.subTest(bs=args[0], lens=args[4]):
                self.check(*self.make(*args))

    def test_all_full_and_all_empty(self):
        self.check(*self.make(2, 16, 4, 8, [16, 16]))
        q = torch.empty(0, 4, 8, dtype=torch.float16, device="cuda")
        pad = torch.randn(3, 8, 4, 8, dtype=torch.float16, device="cuda")
        lens = torch.zeros(3, dtype=torch.int32, device="cuda")
        cu = torch.zeros(4, dtype=torch.int32, device="cuda")
        self.check(q, pad, lens, cu)

    def test_bf16(self):
        gen = torch.Generator(device="cuda").manual_seed(3)
        q = torch.randn(10, 4, 16, dtype=torch.bfloat16, device="cuda", generator=gen)
        pad = torch.randn(2, 8, 4, 16, dtype=torch.bfloat16, device="cuda", generator=gen) * 7
        lens = torch.tensor([8, 2], dtype=torch.int32, device="cuda")
        cu = torch.tensor([0, 8, 10], dtype=torch.int32, device="cuda")
        self.check(q, pad, lens, cu)


RELEASE_REQUIRED_TESTS = [
    "PadDraftExtendQueryTest.test_typical_and_edges",
    "PadDraftExtendQueryTest.test_all_full_and_all_empty",
    "PadDraftExtendQueryTest.test_bf16",
]

if __name__ == "__main__":
    unittest.main()
