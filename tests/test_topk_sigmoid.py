# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("topk_sigmoid")


def reference(topk_weights, topk_ids, gating_output, renormalize, routed_scaling_factor):
    k = topk_weights.shape[1]
    scores = torch.sigmoid(gating_output.float())
    w, ids = torch.topk(scores, k, dim=-1)
    if renormalize:
        w = w * routed_scaling_factor / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w.contiguous(), ids.to(torch.int32).contiguous()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class TopkSigmoidTest(unittest.TestCase):
    def check(self, T, E, k, renorm, dtype, scale=2.5, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        gating = torch.randn(T, E, dtype=dtype, device="cuda", generator=gen)
        ew = torch.zeros(T, k, dtype=torch.float32, device="cuda")
        ei = torch.zeros(T, k, dtype=torch.int32, device="cuda")
        ew.fill_(0)  # ensure in-place kernel fully overwrites
        exp_w, exp_i = reference(ew.clone(), ei.clone(), gating, renorm, scale)
        for name, module in MODULES:
            with self.subTest(module=name, T=T, E=E, k=k, renorm=renorm):
                w = torch.full((T, k), 7.0, dtype=torch.float32, device="cuda")
                i = torch.full((T, k), 99, dtype=torch.int32, device="cuda")
                module.topk_sigmoid(w, i, gating, renorm, scale)
                torch.testing.assert_close(w, exp_w, rtol=2e-2, atol=2e-3)
                torch.testing.assert_close(i, exp_i.to(torch.int32), rtol=0, atol=0)

    def test_matrix(self):
        for T, E, k in ((8, 16, 4), (64, 8, 2), (128, 32, 8), (5, 257, 4), (33, 64, 1)):
            for renorm in (True, False):
                with self.subTest(T=T, E=E, k=k, renorm=renorm):
                    self.check(T, E, k, renorm, torch.float32)

    def test_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            self.check(32, 16, 4, True, dtype, seed=3)
            self.check(32, 16, 4, False, dtype, seed=4)

    def test_k17_renorm_and_strided(self):
        # k>16 renorm must cover every weight (review finding: arange(0,16)
        # left the 17th raw) and a transposed logits matrix must read
        # through its column stride
        self.check(8, 32, 17, True, torch.float32, seed=11)
        self.check(16, 64, 32, True, torch.float32, seed=12)
        gen = torch.Generator(device="cuda").manual_seed(13)
        base = torch.randn(16, 8, dtype=torch.float32, device="cuda", generator=gen)
        gating = base.t()  # [8, 16] non-contiguous
        ew = torch.zeros(8, 4, dtype=torch.float32, device="cuda")
        ei = torch.zeros(8, 4, dtype=torch.int32, device="cuda")
        exp_w, exp_i = reference(ew.clone(), ei.clone(), gating, True, 2.5)
        for name, module in MODULES:
            with self.subTest(module=name):
                w = torch.zeros(8, 4, dtype=torch.float32, device="cuda")
                i = torch.zeros(8, 4, dtype=torch.int32, device="cuda")
                module.topk_sigmoid(w, i, gating, True, 2.5)
                torch.testing.assert_close(w, exp_w, rtol=2e-2, atol=2e-3)
                torch.testing.assert_close(i, exp_i, rtol=0, atol=0)

    def test_extreme_logits(self):
        T, E, k = 4, 8, 3
        gating = torch.tensor(
            [[8.0, -8.0, 0.0, 1.0, 2.0, -1.0, 6.0, -6.0]] * T,
            dtype=torch.float32, device="cuda",
        )
        for renorm in (True, False):
            with self.subTest(renorm=renorm):
                self.check(T, E, k, renorm, torch.float32, seed=9)
        # directly verify saturated gates
        for name, module in MODULES:
            w = torch.zeros(T, k, dtype=torch.float32, device="cuda")
            i = torch.zeros(T, k, dtype=torch.int32, device="cuda")
            module.topk_sigmoid(w, i, gating, False, 1.0)
            self.assertEqual(i[0, 0].item(), 0)  # sigmoid(8) > sigmoid(6) in fp32


RELEASE_REQUIRED_TESTS = [
    "TopkSigmoidTest.test_matrix",
    "TopkSigmoidTest.test_dtypes",
    "TopkSigmoidTest.test_k17_renorm_and_strided",
    "TopkSigmoidTest.test_extreme_logits",
]

if __name__ == "__main__":
    unittest.main()
