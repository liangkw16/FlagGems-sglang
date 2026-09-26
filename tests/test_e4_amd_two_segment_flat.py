# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""e4 candidate matrix for Task 97 fused_sigmoid_mul (_amd vendor).

Carries over the tests/test_fused_sigmoid_mul.py matrix (dtypes, flat
shapes, strided 3D gate, strided/transposed attn, extremes, empty),
focused on the amd variant whose flat contiguous hot path is the e4
two-segment no-mask rewrite, plus new regressions the main file's
fixed-16384 grid cannot provide once the preregistered
BLOCK/warps pre-screen (8192/16384 x w4/w8) retiers the module:

- the amd vendor is registered in the operator matrix and exposes the
  dispatch seam names (``_BLOCK``/``_INT32_NUMEL_MAX``/``_flat_cold``)
  the shared two-segment guards key on;
- flat BLOCK-relative boundaries derived from ``AMD._BLOCK`` at
  runtime (B-1/B/B+1, 2B-1/2B/2B+1, exact-multiple 3B with no tail
  program, and a sub-BLOCK single-tail numel);
- the strided arm (this vendor carries its own copy of the generic
  strided kernel bytes) at its own BLOCK=2048 boundaries;
- the i64 cold twin forced through the public seam by lowering the
  module's dispatch bound (restored in ``finally``).

Release wiring (review r2 fix): the gate runner loads ONLY
tests/test_fused_sigmoid_mul.py and consumes only that entry module's
RELEASE_REQUIRED_TESTS (verify_release.py:225-251) - a bare
``--dependency`` of this file only stages and hashes it
(verify_release.py:113-118) and would never execute these tests. The
entry module therefore IMPORTS the two classes below (rebinding their
``__module__`` to the entry module's name: ``test.id()`` is built from
``class.__module__``, and the runner's required check only accepts
entry-module ids), and this file is staged via the runner's
``--dependency`` flag so the post-run unbound-imported-dependency
check (verify_release.py:264-277) passes. The RELEASE_REQUIRED_TESTS
below remains this module's standalone manifest.
"""

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = dict(load_operator_modules("fused_sigmoid_mul"))
AMD = MODULES.get("amd")


def reference(attn_output, gate):
    g = gate.reshape(attn_output.shape).float()
    out = attn_output.float() * torch.sigmoid(g)
    return out.to(attn_output.dtype)


class E4AmdVendorSeamTest(unittest.TestCase):
    def test_amd_vendor_registered_with_dispatch_seam(self):
        # regression: the amd variant exists, is discovered by the
        # operator matrix, and exposes the dispatch seam the shared
        # two-segment guards key on. BLOCK divisibility of the bound
        # is the domain proof itself (BLOCK | 2^31 iff numel <= 2^31
        # is exactly the int32 no-overflow domain), so a pre-screen
        # retier to a non-dividing BLOCK must fail here, not on the
        # platform.
        self.assertIsNotNone(AMD, "amd vendor missing from the matrix")
        for name in ("_BLOCK", "_INT32_NUMEL_MAX", "_flat_cold"):
            self.assertTrue(hasattr(AMD, name), f"amd seam lost: {name}")
        bound = AMD._INT32_NUMEL_MAX
        self.assertEqual(
            bound % AMD._BLOCK,
            0,
            "int32 domain broken: BLOCK no longer divides the bound",
        )
        self.assertFalse(AMD._flat_cold(bound - 1))
        self.assertFalse(AMD._flat_cold(bound))
        self.assertTrue(AMD._flat_cold(bound + 1))
        self.assertTrue(AMD._flat_cold(bound + AMD._BLOCK))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class E4AmdTwoSegmentFlatTest(unittest.TestCase):
    def check(self, attn, gate, rtol, atol):
        expected = reference(attn, gate)
        snapshots = [attn.clone(), gate.clone()]
        with self.subTest(module="amd"):
            actual = AMD.fused_sigmoid_mul(attn, gate)
            self.assertEqual(actual.dtype, attn.dtype)
            self.assertEqual(actual.shape, attn.shape)
            torch.testing.assert_close(
                actual, expected, rtol=rtol, atol=atol
            )
            for value, before in zip((attn, gate), snapshots):
                torch.testing.assert_close(
                    value, before, rtol=0, atol=0
                )

    # --- carried matrix (tests/test_fused_sigmoid_mul.py), amd arm ---

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
                attn = torch.randn(
                    n, heads * dim, dtype=torch.bfloat16, device="cuda"
                )
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

    def test_transposed_attn(self):
        # the strided arm must stay value-correct for a non-contiguous
        # attn input whose empty_like output is plain flat
        base = torch.arange(12, device="cuda", dtype=torch.float32).reshape(
            3, 4
        )
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
        out = AMD.fused_sigmoid_mul(attn, gate)
        self.assertEqual(out.shape, (0, 512))

    # --- new e4 regressions ---

    def test_amd_flat_block_boundaries(self):
        # hot-path boundaries derived from the module's CURRENT block
        # so the regression keeps guarding after the preregistered
        # 8192/16384 pre-screen retier: B-1/B/B+1 and 2B-1/2B/2B+1
        # straddle the scalar branch, 3B runs a no-tail grid of pure
        # unmasked tiles, and 7 runs the tail arm alone (n_full=0)
        block = AMD._BLOCK
        for numel in (
            7,
            block - 1,
            block,
            block + 1,
            2 * block - 1,
            2 * block,
            2 * block + 1,
            3 * block,
        ):
            with self.subTest(numel=numel):
                attn = torch.randn(
                    1, numel, dtype=torch.bfloat16, device="cuda"
                )
                gate = torch.randn(
                    1, numel, dtype=torch.bfloat16, device="cuda"
                )
                self.check(attn, gate, 2e-2, 1e-3)

    def test_amd_strided_block_boundaries(self):
        # the amd file carries its own copy of the generic strided
        # kernel (BLOCK=2048/w8) - pin its tail handling through the
        # public seam with non-contiguous attn at 2048-relative
        # boundaries (sub-block, exact multiple, one-past-multiple)
        for numel in (2047, 2048, 2049):
            with self.subTest(numel=numel):
                big = torch.randn(
                    1, numel * 2, dtype=torch.bfloat16, device="cuda"
                )
                attn = big[:, ::2]
                self.assertFalse(attn.is_contiguous())
                gate = torch.randn(
                    1, numel, dtype=torch.bfloat16, device="cuda"
                )
                self.check(attn, gate, 2e-2, 1e-3)

    def test_amd_i64_cold_variant_numerics(self):
        # force the i64 cold twin through the public seam by lowering
        # the module's dispatch bound (restored in finally), then run
        # the full numeric check on a tail grid (2*BLOCK+7) and a
        # no-tail grid (3*BLOCK)
        block = AMD._BLOCK
        saved = AMD._INT32_NUMEL_MAX
        AMD._INT32_NUMEL_MAX = block
        try:
            for numel in (2 * block + 7, 3 * block):
                with self.subTest(numel=numel):
                    self.assertTrue(AMD._flat_cold(numel))
                    attn = torch.randn(
                        1, numel, dtype=torch.bfloat16, device="cuda"
                    )
                    gate = torch.randn(
                        1, numel, dtype=torch.bfloat16, device="cuda"
                    )
                    self.check(attn, gate, 2e-2, 1e-3)
        finally:
            AMD._INT32_NUMEL_MAX = saved


RELEASE_REQUIRED_TESTS = [
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
