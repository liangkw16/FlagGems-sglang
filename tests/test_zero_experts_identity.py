# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("zero_experts_identity")


def reference(expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states):
    zero = (expert_indices >= num_experts).to(hidden_states.dtype)
    s = (expert_scales.to(hidden_states.dtype) * zero).sum(-1, keepdim=True)
    return hidden_states * s


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class ZeroExpertsIdentityTest(unittest.TestCase):
    def check(self, T, topk, E, H, dtype, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        idx = torch.randint(0, E + 4, (T, topk), dtype=torch.int32, device="cuda", generator=gen)
        sc = torch.rand(T, topk, dtype=torch.float32, device="cuda", generator=gen)
        h = torch.randn(T, H, dtype=dtype, device="cuda", generator=gen) * 2
        expected = reference(idx, sc, E, 1, h)
        for name, module in MODULES:
            with self.subTest(module=name, T=T, H=H):
                out = module.zero_experts_identity(idx, sc, E, 1, h)
                self.assertEqual(out.dtype, h.dtype)
                torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

    def test_shapes(self):
        for T, topk, E, H in ((8, 4, 16, 512), (64, 8, 8, 1024), (33, 2, 32, 256), (128, 6, 64, 2048)):
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(T=T, topk=topk, E=E, H=H, dt=dtype):
                    self.check(T, topk, E, H, dtype)

    def test_all_and_none_zero(self):
        T, topk, E, H = 16, 4, 8, 512
        # all slots identity
        idx = torch.full((T, topk), E + 1, dtype=torch.int32, device="cuda")
        sc = torch.rand(T, topk, dtype=torch.float32, device="cuda")
        h = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        expected = reference(idx, sc, E, 1, h)
        for name, module in MODULES:
            with self.subTest(module=name, case="all"):
                torch.testing.assert_close(module.zero_experts_identity(idx, sc, E, 1, h), expected, rtol=2e-2, atol=2e-2)
        # no identity slots
        idx2 = torch.randint(0, E, (T, topk), dtype=torch.int32, device="cuda")
        expected2 = reference(idx2, sc, E, 1, h)
        for name, module in MODULES:
            with self.subTest(module=name, case="none"):
                torch.testing.assert_close(module.zero_experts_identity(idx2, sc, E, 1, h), expected2, rtol=0, atol=0)


RELEASE_REQUIRED_TESTS = [
    "ZeroExpertsIdentityTest.test_shapes",
    "ZeroExpertsIdentityTest.test_all_and_none_zero",
]

if __name__ == "__main__":
    unittest.main()
