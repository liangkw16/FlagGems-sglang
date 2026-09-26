# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

# The release runner (verify_release.py run_suite) loads ONLY this
# module and consumes only THIS module's RELEASE_REQUIRED_TESTS
# (verify_release.py:225-251); the e4 candidate matrix in
# tests/test_e4_amd_two_segment_flat.py would never enter the receipt
# on its own -- --dependency only stages and hashes the file
# (verify_release.py:113-118). Import its classes here so
# loadTestsFromModule collects them into this suite, and rebind
# __module__: test.id() is built from class.__module__, so without the
# rebind the imported tests carry tests.test_e4_amd_two_segment_flat.*
# ids while the runner's required check only accepts
# f"{module.__name__}.{name}" == tests.test_fused_sigmoid_mul.* ->
# "required contract test missing from suite". The e4 file itself must
# still be staged via the runner's --dependency flag or the post-run
# unbound-imported-dependency check (verify_release.py:264-277)
# fails. Rebinding mutates the class objects' reported module for the
# whole process; the e4 module's own RELEASE_REQUIRED_TESTS remains
# the standalone manifest of that file and is not consumed by any
# runner path.
from tests.test_e4_amd_two_segment_flat import (
    E4AmdTwoSegmentFlatTest,
    E4AmdVendorSeamTest,
)

E4AmdVendorSeamTest.__module__ = __name__
E4AmdTwoSegmentFlatTest.__module__ = __name__

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

    def test_two_segment_dispatch_boundary(self):
        # numel=2^31 boundary dispatch regression (review r2): BLOCK
        # divides 2^31, so numel <= 2^31 is exactly the no-overflow
        # domain of the int32 hot path (at 2^31 the grid has no tail
        # program and max offs == INT32_MAX); above it the host must
        # route to the i64 cold twin. Analytical wraparound check:
        # at 2^31+1 the tail base wraps to -2^31 and every lane
        # passes `offs < numel` (OOB) - the bug this dispatch fixes.
        # A real 2^31-element execution needs 3x4GiB (bf16) and does
        # not fit the release device; the cold kernel's numerics are
        # covered by test_two_segment_i64_cold_variant_numerics.
        for name, module in MODULES:
            if not hasattr(module, "_INT32_NUMEL_MAX"):
                # generic/kunlunxin are single all-i64 kernels (no
                # dispatch seam to guard)
                continue
            with self.subTest(module=name):
                bound = module._INT32_NUMEL_MAX
                self.assertEqual(bound % module._BLOCK, 0)
                self.assertFalse(module._flat_cold(bound - 1))
                self.assertFalse(module._flat_cold(bound))
                self.assertTrue(module._flat_cold(bound + 1))
                self.assertTrue(module._flat_cold(bound + module._BLOCK))

    def test_two_segment_i64_cold_variant_numerics(self):
        # numel=2^31 boundary regression (review r2), execution arm:
        # force the i64 cold twin through the public seam by lowering
        # the module's dispatch bound (restored in finally), then run
        # the full numeric check on both a tail grid (2*BLOCK+7) and
        # a no-tail grid (3*BLOCK).
        for name, module in MODULES:
            if not hasattr(module, "_INT32_NUMEL_MAX"):
                continue
            block = module._BLOCK
            saved = module._INT32_NUMEL_MAX
            module._INT32_NUMEL_MAX = block
            try:
                for numel in (2 * block + 7, 3 * block):
                    with self.subTest(module=name, numel=numel):
                        self.assertTrue(module._flat_cold(numel))
                        attn = torch.randn(
                            1, numel, dtype=torch.bfloat16, device="cuda"
                        )
                        gate = torch.randn(
                            1, numel, dtype=torch.bfloat16, device="cuda"
                        )
                        self.check(attn, gate, 2e-2, 1e-3)
            finally:
                module._INT32_NUMEL_MAX = saved

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
    "FusedSigmoidMulTest.test_two_segment_dispatch_boundary",
    "FusedSigmoidMulTest.test_two_segment_i64_cold_variant_numerics",
    "FusedSigmoidMulTest.test_transposed_attn",
    "FusedSigmoidMulTest.test_extreme_values",
    "FusedSigmoidMulTest.test_empty",
    # e4 amd-arm classes imported above (review r2 fix: these entries
    # are only enforceable because the classes are imported AND their
    # __module__ is rebound to this module's name)
    "E4AmdVendorSeamTest.test_amd_vendor_registered_with_dispatch_seam",
    "E4AmdTwoSegmentFlatTest.test_flat_same_shape",
    "E4AmdTwoSegmentFlatTest.test_strided_3d_gate",
    "E4AmdTwoSegmentFlatTest.test_contiguous_3d_gate",
    "E4AmdTwoSegmentFlatTest.test_strided_attn",
    "E4AmdTwoSegmentFlatTest.test_transposed_attn",
    "E4AmdTwoSegmentFlatTest.test_extreme_values",
    "E4AmdTwoSegmentFlatTest.test_empty",
    "E4AmdTwoSegmentFlatTest.test_amd_flat_block_boundaries",
    "E4AmdTwoSegmentFlatTest.test_amd_strided_block_boundaries",
    "E4AmdTwoSegmentFlatTest.test_amd_i64_cold_variant_numerics",
]

if __name__ == "__main__":
    unittest.main()
