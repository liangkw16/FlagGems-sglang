# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("sigmoid_gate_mul_broadcast")


def reference(x, gate):
    g = torch.sigmoid(gate.reshape(-1, 1).float())
    return (x.float() * g).to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class SGMBTest(unittest.TestCase):
    def check(self, x, gate):
        want = reference(x, gate)
        for name, module in MODULES:
            with self.subTest(module=name):
                got = module.sigmoid_gate_mul_broadcast(x, gate)
                torch.testing.assert_close(
                    got.float(), want.float(), rtol=2e-2, atol=2e-2
                )

    def test_shapes_and_saturation(self):
        for n, d in ((1, 1), (7, 1024), (513, 5120), (2049, 1023)):
            with self.subTest(shape=(n, d)):
                x = torch.randn(n, d, dtype=torch.bfloat16, device="cuda")
                gate = torch.randn(n, 1, dtype=torch.bfloat16, device="cuda") * 3
                self.check(x, gate)
        x = torch.randn(32, 2048, dtype=torch.float16, device="cuda")
        gate = torch.tensor(
            [0.0, -100.0, 100.0] * 10 + [0.0, 0.0], dtype=torch.float16, device="cuda"
        ).reshape(32, 1)
        self.check(x, gate)

    def test_flat_block_boundary_gate_gather(self):
        # Boundary axes for the _enflame streaming vendor, kept from the
        # e6 flat form (BLOCK=65536 streams, gate via offs // hdim) and
        # re-read under the e7 [RB, W] row-block tile: an exact block
        # with no tail, a one-row tail block, an odd-hdim degenerate
        # tile, more row blocks than the 12-CTA grid-stride cap, and
        # hdim=1 where every element addresses its own gate row. Per-row
        # distinct gates (sigmoid spread 0.047..0.953) make any
        # gate-indexing off-by-one blow the 2e-2 tolerance.
        for n, d in (
            (64, 1024),  # e7: rows == RB(64): one exact row block, no tail
            (65, 1024),  # e7: second row block carrying a single row
            (33, 2047),  # e7: W=1 degenerate tile, 2047 column iterations
            (128, 8192),  # e7: 16 row blocks over 12 CTAs: 2nd pass
            (70000, 1),  # hdim=1: every element its own gate row
        ):
            with self.subTest(shape=(n, d)):
                x = torch.randn(n, d, dtype=torch.bfloat16, device="cuda")
                gate = (
                    (
                        torch.arange(n, device="cuda", dtype=torch.float32)
                        % 7
                        - 3
                    )
                    .to(torch.bfloat16)
                    .reshape(n, 1)
                )
                self.check(x, gate)

    def test_rowblock_tile_boundary(self):
        # The _enflame e7 vendor tiles [RB, W] with W = the largest
        # power-of-two factor of hdim (capped at 65536) and
        # RB * W = 65536, so the new misalignment axes are row blocks
        # ending exactly at/past the rows boundary, multi-column-block
        # widths (W < hdim: the compile-time-counted column loop must
        # cover the full row), the sub-512 W band, and the W=65536 cap
        # where RB collapses to 1. Per-row distinct gates make any
        # [RB]-vector-load or broadcast off-by-one blow the 2e-2
        # tolerance; a missed column block fails zero-filled regions.
        for n, d in (
            (63, 1024),  # rows = RB-1: single row block, 1 masked lane
            (64, 1024),  # rows = RB exactly: no row mask ever live
            (65, 1024),  # rows = RB+1: tail row block with one row
            (769, 1024),  # 13 row blocks > 12 CTAs: grid-stride 2nd pass
            (513, 5120),  # W=1024: 5 column blocks + 1-row tail block
            (3, 7168),  # W=1024: 7 column blocks
            (5, 96),  # W=32 (sub-512 band): 3 column blocks
            (3, 65536),  # W capped at 65536, RB=1: one row per block
        ):
            with self.subTest(shape=(n, d)):
                x = torch.randn(n, d, dtype=torch.bfloat16, device="cuda")
                gate = (
                    (
                        torch.arange(n, device="cuda", dtype=torch.float32)
                        % 7
                        - 3
                    )
                    .to(torch.bfloat16)
                    .reshape(n, 1)
                )
                self.check(x, gate)

    def test_row_gated_strided_x(self):
        # Row-gapped x (stride(1) == 1, stride(0) > hdim) is inside the
        # generic contract (xs0 addressing); the _enflame tile vendor
        # must not reject it - it takes a layout copy and still computes
        # the gating multiply in the Triton kernel.
        for n, d in ((7, 1023), (130, 1024)):
            with self.subTest(shape=(n, d)):
                base = torch.randn(
                    2 * n, d, dtype=torch.bfloat16, device="cuda"
                )
                x = base[::2]
                self.assertFalse(x.is_contiguous())
                gate = (
                    (
                        torch.arange(n, device="cuda", dtype=torch.float32)
                        % 7
                        - 3
                    )
                    .to(torch.bfloat16)
                    .reshape(n, 1)
                )
                self.check(x, gate)


RELEASE_REQUIRED_TESTS = [
    "SGMBTest.test_shapes_and_saturation",
    "SGMBTest.test_flat_block_boundary_gate_gather",
    "SGMBTest.test_rowblock_tile_boundary",
    "SGMBTest.test_row_gated_strided_x",
]


if __name__ == "__main__":
    unittest.main()
