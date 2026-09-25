# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("post_reorder_deepgemm")


def reference(down_output, output, src2dst, topk_ids, topk_weights, topk, num_tokens, hidden_size, routed_scaling_factor):
    acc = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=output.device)
    for i in range(topk):
        valid = (topk_ids[:, i] >= 0).float()
        rows = down_output[src2dst[:, i].clamp(min=0).long()].float()
        acc += rows * (topk_weights[:, i].float() * valid)[:, None]
    return (acc * routed_scaling_factor).to(output.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class PostReorderDeepgemmTest(unittest.TestCase):
    def check(self, down, out_tpl, s2d, ids, w, topk, N, H, scaling):
        expected = reference(down, out_tpl, s2d, ids, w, topk, N, H, scaling)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.post_reorder_deepgemm(down, out_tpl, s2d, ids, w, topk, N, H, scaling)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)

    def make(self, N, E, topk, H, seed=0, padding=True):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        down = torch.randn(N * topk, H, dtype=torch.bfloat16, device="cuda", generator=gen)
        out_tpl = torch.empty(N, H, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(-1 if padding else 0, E + 2, (N, topk), dtype=torch.int32, device="cuda", generator=gen)
        if padding:
            ids[:, 0] = -1  # some padded slots
            ids[:, 1] = E   # fused shared expert stays VALID (>= 0)
        w = torch.randn(N, topk, dtype=torch.float32, device="cuda", generator=gen)
        s2d = torch.randint(0, N * topk, (N, topk), dtype=torch.int32, device="cuda", generator=gen)
        s2d[ids < 0] = -1  # invalid slots may carry -1 dst
        return down, out_tpl, s2d, ids, w, topk, N, H, 2.5

    def test_matrix(self):
        for N, E, topk, H in ((8, 16, 4, 128), (64, 8, 2, 333), (128, 32, 8, 64)):
            with self.subTest(N=N, topk=topk, H=H):
                self.check(*self.make(N, E, topk, H))

    def test_all_valid(self):
        self.check(*self.make(32, 8, 4, 64, seed=7, padding=False))

    def test_fp16(self):
        down, out_tpl, s2d, ids, w, topk, N, H, sc = self.make(48, 8, 4, 64, seed=9)
        down = down.to(torch.float16)
        out_tpl = out_tpl.to(torch.float16)
        self.check(down, out_tpl, s2d, ids, w, topk, N, H, sc)


RELEASE_REQUIRED_TESTS = [
    "PostReorderDeepgemmTest.test_matrix",
    "PostReorderDeepgemmTest.test_all_valid",
    "PostReorderDeepgemmTest.test_fp16",
]

if __name__ == "__main__":
    unittest.main()
