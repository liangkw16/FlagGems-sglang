# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("recompute_w_u")


import math


def reference(k, v, beta, g_cumsum, A, cu_seqlens):
    assert cu_seqlens is None
    B, T, Hg, K = k.shape
    _, _, H, V = v.shape
    BT = A.shape[-1]
    heads_per_kv = H // Hg
    w = torch.empty_like(k.repeat_interleave(heads_per_kv, dim=2)[..., :K]) if H != Hg else torch.empty_like(k)
    w = torch.empty(B, T, H, K, dtype=k.dtype, device=k.device)
    u = torch.empty_like(v)
    for c in range(T // BT):
        rows = slice(c * BT, (c + 1) * BT)
        Ac = A[:, rows].permute(0, 2, 1, 3).float()
        b_beta = beta[:, rows].permute(0, 2, 1).float()
        b_g = torch.exp(g_cumsum[:, rows].permute(0, 2, 1).float())
        vb = v[:, rows].permute(0, 2, 1, 3).float() * b_beta[..., None]
        u[:, rows] = (Ac @ vb.to(v.dtype).float()).permute(0, 2, 1, 3).to(v.dtype)
        kh = k[:, rows].permute(0, 2, 1, 3).float()
        kh = kh.repeat_interleave(heads_per_kv, dim=1)
        kb = kh * b_beta[..., None] * b_g[..., None]
        w[:, rows] = (Ac @ kb.to(k.dtype).float()).permute(0, 2, 1, 3).to(k.dtype)
    return w, u


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class RecomputeWUTest(unittest.TestCase):
    def check(self, k, v, beta, g, A):
        ew, eu = reference(k, v, beta, g, A, None)
        for name, module in MODULES:
            with self.subTest(module=name):
                w, u = module.recompute_w_u(k, v, beta, g, A, None)
                torch.testing.assert_close(w, ew, rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(u, eu, rtol=2e-2, atol=2e-2)

    def make(self, B, T, Hg, H, K, V, BT, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        k = torch.randn(B, T, Hg, K, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.5
        v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.5
        beta = torch.rand(B, T, H, dtype=torch.float32, device="cuda", generator=gen)
        g = torch.randn(B, T, H, dtype=torch.float32, device="cuda", generator=gen) * 0.1
        # A: well-conditioned UT-ish matrices via cholesky of random SPD
        raw = torch.randn(B, T, H, BT, BT, dtype=torch.float32, device="cuda", generator=gen) * 0.1
        eye = torch.eye(BT, device="cuda", dtype=torch.float32)
        A = (raw.triu() + eye).to(torch.bfloat16)
        return k, v, beta, g, A

    def test_mqa_and_gqa(self):
        # H == Hg (MQA-free) and 2:1 GQA
        for B, T, Hg, H, K, V, BT in (
            (1, 64, 4, 4, 64, 64, 64),
            (2, 128, 4, 8, 32, 64, 64),
        ):
            with self.subTest(B=B, T=T, Hg=Hg, H=H):
                self.check(*self.make(B, T, Hg, H, K, V, BT))

    def test_odd_dims_and_tail(self):
        # non-pow2 K/V, BT=32
        self.check(*self.make(1, 96, 2, 4, 48, 40, 32, seed=3))

    def test_single_chunk(self):
        self.check(*self.make(1, 64, 2, 2, 64, 64, 64, seed=7))


RELEASE_REQUIRED_TESTS = [
    "RecomputeWUTest.test_mqa_and_gqa",
    "RecomputeWUTest.test_odd_dims_and_tail",
    "RecomputeWUTest.test_single_chunk",
]

if __name__ == "__main__":
    unittest.main()
