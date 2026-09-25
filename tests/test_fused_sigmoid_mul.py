# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("fused_sigmoid_mul")


def reference(attn_output, gate):
    g = gate.reshape(attn_output.shape).float()
    out = attn_output.float() * torch.sigmoid(g)
    return out.to(attn_output.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class FusedSigmoidMulTest(unittest.TestCase):
    def check(self, attn, gate, rtol, atol):
        expected = reference(attn, gate)
        snapshots = [attn.clone(), gate.clone()]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.fused_sigmoid_mul(attn, gate)
                self.assertEqual(actual.dtype, attn.dtype)
                self.assertEqual(actual.shape, attn.shape)
                torch.testing.assert_close(
                    actual, expected, rtol=rtol, atol=atol
                )
                for value, before in zip((attn, gate), snapshots):
                    torch.testing.assert_close(
                        value, before, rtol=0, atol=0
                    )

    def test_flat_same_shape(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape in ((1, 1), (17, 129), (256, 512), (4096, 4096)):
                with self.subTest(dtype=dtype, shape=shape):
                    attn = torch.randn(shape, dtype=dtype, device="cuda")
                    gate = torch.randn(shape, dtype=dtype, device="cuda")
                    if dtype == torch.float32:
                        self.check(attn, gate, 1e-6, 1e-7)
                    else:
                        self.check(attn, gate, 2e-2, 1e-3)

    def test_strided_3d_gate(self):
        # gate is 3D with a non-contiguous inner layout: the kernel must
        # reach elements through the runtime strides
        for n, heads, dim in ((128, 16, 64), (33, 7, 40), (256, 8, 128)):
            with self.subTest(n=n, heads=heads, dim=dim):
                attn = torch.randn(n, heads * dim, dtype=torch.bfloat16, device="cuda")
                gate = torch.randn(
                    n, heads, dim * 2, dtype=torch.bfloat16, device="cuda"
                )[..., ::2]
                self.assertFalse(gate.is_contiguous())
                self.check(attn, gate, 2e-2, 1e-3)

    def test_contiguous_3d_gate(self):
        attn = torch.randn(256, 1024, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(256, 16, 64, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(gate.is_contiguous())
        self.check(attn, gate, 2e-2, 1e-3)

    def test_strided_attn(self):
        big = torch.randn(256, 1024 * 2, dtype=torch.float16, device="cuda")
        attn = big[:, ::2]
        self.assertFalse(attn.is_contiguous())
        gate = torch.randn(256, 1024, dtype=torch.float16, device="cuda")
        self.check(attn, gate, 2e-2, 1e-3)

    def test_grid_boundaries(self):
        # BLOCK=2048: every case must fully cover the flat range
        for numel in (2047, 2048, 2049, 4096, 4097, 8191):
            with self.subTest(numel=numel):
                attn = torch.randn(1, numel, dtype=torch.bfloat16, device="cuda")
                gate = torch.randn(1, numel, dtype=torch.bfloat16, device="cuda")
                self.check(attn, gate, 2e-2, 1e-3)

    def test_two_segment_block_boundaries(self):
        # regression for the flat two-segment no-mask hot path
        # (_ascend vendor, BLOCK=16384): numel at B-1/B/B+1 and
        # 2B-1/2B/2B+1 straddles the scalar branch, and exact
        # multiples (32768, 65536) run a grid with no tail program
        # at all - pure unmasked tiles. A single sub-BLOCK numel
        # (8191 also covered by test_grid_boundaries) runs the
        # tail arm alone (n_full=0).
        for numel in (16383, 16384, 16385, 32767, 32768, 32769, 65536):
            with self.subTest(numel=numel):
                attn = torch.randn(1, numel, dtype=torch.bfloat16, device="cuda")
                gate = torch.randn(1, numel, dtype=torch.bfloat16, device="cuda")
                self.check(attn, gate, 2e-2, 1e-3)

    def test_transposed_attn(self):
        # empty_like would keep the dense-transpose strides; the output
        # must still be value-correct for a non-contiguous attn input
        base = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
        attn = base.t().to(torch.bfloat16)
        self.assertFalse(attn.is_contiguous())
        gate = torch.randn(4, 3, dtype=torch.bfloat16, device="cuda")
        self.check(attn, gate, 2e-2, 1e-3)

    def test_extreme_values(self):
        # sigmoid saturates; inf/nan propagate per fp32 math
        attn = torch.tensor(
            [[100.0, -100.0, 0.0, 1e30]], dtype=torch.bfloat16, device="cuda"
        )
        gate = torch.tensor(
            [[100.0, -100.0, 0.0, -1e30]], dtype=torch.bfloat16, device="cuda"
        )
        self.check(attn, gate, 2e-2, 1e-3)

    def test_empty(self):
        attn = torch.empty(0, 512, dtype=torch.bfloat16, device="cuda")
        gate = torch.empty(0, 512, dtype=torch.bfloat16, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                out = module.fused_sigmoid_mul(attn, gate)
                self.assertEqual(out.shape, (0, 512))


RELEASE_REQUIRED_TESTS = [
    "FusedSigmoidMulTest.test_flat_same_shape",
    "FusedSigmoidMulTest.test_strided_3d_gate",
    "FusedSigmoidMulTest.test_contiguous_3d_gate",
    "FusedSigmoidMulTest.test_strided_attn",
    "FusedSigmoidMulTest.test_grid_boundaries",
    "FusedSigmoidMulTest.test_two_segment_block_boundaries",
    "FusedSigmoidMulTest.test_transposed_attn",
    "FusedSigmoidMulTest.test_extreme_values",
    "FusedSigmoidMulTest.test_empty",
]

if __name__ == "__main__":
    unittest.main()
