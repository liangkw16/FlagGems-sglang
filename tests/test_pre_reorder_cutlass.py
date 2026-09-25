# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("pre_reorder_cutlass")


def reference(input, gateup_input, src2dst, topk_ids, a1_scales, num_local_experts, topk, num_tokens, hidden_size):
    out = gateup_input.clone()
    inv = 1.0 / float(a1_scales) if a1_scales is not None else 1.0
    scaled = (input.float() * inv).to(out.dtype)
    flat_ids = topk_ids.reshape(-1)
    flat_dst = src2dst.reshape(-1)
    valid = flat_ids != num_local_experts
    out[flat_dst[valid].long()] = scaled.repeat_interleave(topk, dim=0)[valid]
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class PreReorderCutlassTest(unittest.TestCase):
    def check(self, x, gate, s2d, ids, scale, E, topk, N, H):
        expected = reference(x, gate, s2d, ids, scale, E, topk, N, H)
        snap = gate.clone()
        for name, module in MODULES:
            with self.subTest(module=name, scale=None if scale is None else "tensor"):
                actual = module.pre_reorder_cutlass(x, gate, s2d, ids, scale, E, topk, N, H)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
                torch.testing.assert_close(gate, snap, rtol=0, atol=0)

    def make(self, N, E, topk, H, seed=0, with_scale=True, offrank=True):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(N, H, dtype=torch.bfloat16, device="cuda", generator=gen)
        gate = torch.randn(N * topk, H, dtype=torch.bfloat16, device="cuda", generator=gen) * 9
        ids = torch.randint(0, E, (N, topk), dtype=torch.int32, device="cuda", generator=gen)
        if offrank:
            ids[:, -1] = E
        # destinations: a permutation slice of [0, N*topk)
        perm = torch.randperm(N * topk, generator=gen, device="cuda")[: N * (topk - 1)].to(torch.int32)
        s2d = torch.zeros(N, topk, dtype=torch.int32, device="cuda")
        mask = ids != E
        s2d[mask] = perm
        s2d[~mask] = 0
        scale = torch.tensor(0.5, dtype=torch.float32, device="cuda") if with_scale else None
        return x, gate, s2d, ids, scale, E, topk, N, H

    def test_matrix(self):
        for with_scale in (True, False):
            for N, E, topk, H in ((8, 16, 4, 128), (64, 8, 2, 333), (128, 32, 8, 64)):
                with self.subTest(scale=with_scale, N=N, topk=topk, H=H):
                    self.check(*self.make(N, E, topk, H, with_scale=with_scale))

    def test_no_offrank_slots(self):
        args = self.make(32, 8, 4, 64, seed=5)
        x, gate, s2d, ids, scale, E, topk, N, H = args
        ids[:, :] = torch.randint(0, E, ids.shape, dtype=torch.int32, device="cuda")
        perm = torch.randperm(N * topk, device="cuda")[: N * topk].to(torch.int32)
        s2d[:, :] = perm.view(N, topk)
        self.check(x, gate, s2d, ids, scale, E, topk, N, H)

    def test_transposed_destination(self):
        # a transposed gateup buffer: unwritten rows must keep their
        # base VALUES (the kernel writes through a contiguous clone)
        x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16, device="cuda")
        gate_t = torch.full((3, 2), 99.0, dtype=torch.bfloat16, device="cuda").t()
        self.assertFalse(gate_t.is_contiguous())
        ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        s2d = torch.tensor([[2, 1]], dtype=torch.int32, device="cuda")
        self.check(x, gate_t, s2d, ids, None, 5, 2, 1, 3)

    def test_empty_tokens(self):
        x = torch.empty(0, 64, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        ids = torch.empty(0, 2, dtype=torch.int32, device="cuda")
        s2d = torch.empty(0, 2, dtype=torch.int32, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                out = module.pre_reorder_cutlass(x, gate, s2d, ids, None, 8, 2, 0, 64)
                torch.testing.assert_close(out, gate, rtol=0, atol=0)


RELEASE_REQUIRED_TESTS = [
    "PreReorderCutlassTest.test_matrix",
    "PreReorderCutlassTest.test_no_offrank_slots",
    "PreReorderCutlassTest.test_transposed_destination",
    "PreReorderCutlassTest.test_empty_tokens",
]

if __name__ == "__main__":
    unittest.main()
