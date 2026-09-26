# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

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

    def test_transposed_routing_tables(self):
        # transposed [topk, N] routing tables address through column
        # strides; the reference still reads them as [N, topk]
        gen = torch.Generator(device="cuda").manual_seed(11)
        N, topk, H = 4, 2, 8
        down = torch.arange(1, N * topk * H + 1, dtype=torch.float32, device="cuda").reshape(N * topk, H)
        out_tpl = torch.empty(N, H, dtype=torch.float32, device="cuda")
        ids = torch.zeros(N, topk, dtype=torch.int32, device="cuda")
        w = torch.ones(N, topk, dtype=torch.float32, device="cuda")
        s2d = torch.tensor([[0, 2], [1, 3], [2, 0], [3, 1]], dtype=torch.int32, device="cuda")
        # transpose-of-contiguous view: logical [N, topk] with col stride N
        ids_t = ids.t().contiguous().t()
        s2d_t = s2d.t().contiguous().t()
        w_t = w.t().contiguous().t()
        self.assertFalse(s2d_t.is_contiguous())
        self.check(down, out_tpl, s2d_t, ids_t, w_t, topk, N, H, 1.0)

    def test_fp16(self):
        down, out_tpl, s2d, ids, w, topk, N, H, sc = self.make(48, 8, 4, 64, seed=9)
        down = down.to(torch.float16)
        out_tpl = out_tpl.to(torch.float16)
        self.check(down, out_tpl, s2d, ids, w, topk, N, H, sc)

    def test_hidden_program_boundaries(self):
        # E1: hidden blocks live on grid.y with BLOCK=2048 - cover the
        # tile edges, multi-tile strides and the gy=255 cap boundary.
        for h in (
            2047,
            2048,
            2049,
            4095,
            4096,
            4097,
            7168,
            255 * 2048 - 1,
            255 * 2048,
            255 * 2048 + 1,
        ):
            with self.subTest(hidden=h):
                self.check(*self.make(1, 8, 2, h))

    def test_two_segment_hidden_semantics(self):
        # _ascend vendor: each hidden tl.range iteration branches on the
        # uniform scalar `h0 + BLOCK <= hdim` - full blocks take the
        # no-mask/no-other load/store, only the tail block keeps the
        # masked load (without other) and masked store. Pin both arms
        # with multi-row, multi-expert gathers including padded slots
        # (-1 dst), and force the tail block to land in the SAME
        # grid-stride program as full blocks (hidden > 255*2048 gives
        # some pid a second tl.range pass). A dropped tail-store mask
        # would spill uninit lanes into the next row and fail here.
        for n, e, topk, h in (
            (3, 8, 3, 2048 + 1),
            (5, 16, 4, 2 * 2048 - 1),
            (2, 8, 2, 255 * 2048 + 2049),
            (2, 8, 2, 255 * 2048 + 4097),
        ):
            with self.subTest(N=n, topk=topk, H=h):
                self.check(*self.make(n, e, topk, h))

    def test_slot_skip_reference_gate(self):
        # e3 slot-skip: an invalid slot (topk_ids < 0) contributes
        # nothing - the whole gather is skipped. The gate stays on
        # topk_ids (the reference validity predicate), NOT on the
        # src2dst value (upstream SGLang main gates on dst_idx >= 0):
        # row 3 below is a VALID slot with dst == -1, which must still
        # contribute down[clamp(-1)] = down[0] - a dst-gated port would
        # silently drop it. Row 0 has every slot invalid and must read
        # exactly zero. All data is finite, so the branchless s0
        # semantics and the skip semantics agree; this pins the gate
        # and the skip branch against every loaded module.
        gen = torch.Generator(device="cuda").manual_seed(21)
        N, topk, H = 8, 4, 96
        down = torch.randn(N * topk, H, dtype=torch.bfloat16, device="cuda", generator=gen)
        down[0].fill_(0.25)  # clamp target: finite and nonzero
        out_tpl = torch.empty(N, H, dtype=torch.bfloat16, device="cuda")
        ids = torch.full((N, topk), 3, dtype=torch.int32, device="cuda")
        w = torch.randn(N, topk, dtype=torch.float32, device="cuda", generator=gen)
        # valid dsts avoid row 0 so the clamped contribution of row 3 is
        # the only place down[0] may enter
        s2d = torch.randint(1, N * topk, (N, topk), dtype=torch.int32, device="cuda", generator=gen)
        ids[0] = -1  # row 0: every slot invalid -> output row == 0
        s2d[0] = -1
        ids[1, 0] = -1  # row 1: mixed padding
        s2d[1, 0] = -1
        ids[2, 1:] = -1  # row 2: only slot 0 stays valid
        s2d[2, 1:] = -1
        s2d[3, 2] = -1  # row 3: VALID eid with dst=-1 -> clamped row 0
        self.check(down, out_tpl, s2d, ids, w, topk, N, H, 2.5)
        # pipelined tl.range row loop x live skip branches: rows beyond
        # the grid.x cap revisit the row loop for a second grid-stride
        # iteration with half the slots padded
        self.check(*self.make(65537, 8, 2, 64, seed=22))

    def test_slot_skip_nonfinite_semantics(self):
        # e3 disclosed divergence: an INVALID slot whose clamped gather
        # row (dst=-1 -> row 0) or weight is non-finite. The literal
        # torch reference computes down[clamp(dst)] * (w * 0), and
        # 0 * NaN/Inf propagates NaN; the slot-skip kernel skips the
        # slot entirely and returns the bare valid-slot sum. Platform
        # harness data is randn (finite), so both semantics agree on
        # every platform case - this regression pins the skip form.
        # Modules WITHOUT SLOT_SKIP_SEMANTICS (the frozen kunlunxin s0
        # branchless vendor) keep the literal NaN-propagating semantics
        # and are checked against the literal reference with equal_nan.
        gen = torch.Generator(device="cuda").manual_seed(23)
        N, topk, H = 5, 4, 96
        down = torch.randn(N * topk, H, dtype=torch.bfloat16, device="cuda", generator=gen)
        down[0].fill_(float("nan"))  # clamp target poisoned
        out_tpl = torch.empty(N, H, dtype=torch.bfloat16, device="cuda")
        ids = torch.full((N, topk), 3, dtype=torch.int32, device="cuda")
        w = torch.randn(N, topk, dtype=torch.float32, device="cuda", generator=gen)
        # valid dsts avoid the poisoned row 0
        s2d = torch.randint(1, N * topk, (N, topk), dtype=torch.int32, device="cuda", generator=gen)
        ids[:, 0] = -1  # invalid slots clamp onto the poisoned row 0
        s2d[:, 0] = -1
        ids[:, 2] = -1
        s2d[:, 2] = -1
        w[:, 2] = float("inf")  # non-finite weight on an invalid slot
        scaling = 2.5
        # skip semantics: only the valid slots 1 and 3 gather
        skip_expected = torch.zeros(N, H, dtype=torch.float32, device="cuda")
        for i in (1, 3):
            skip_expected += down[s2d[:, i].long()].float() * w[:, i][:, None]
        skip_expected = (skip_expected * scaling).to(out_tpl.dtype)
        literal = reference(down, out_tpl, s2d, ids, w, topk, N, H, scaling)
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.post_reorder_deepgemm(
                    down, out_tpl, s2d, ids, w, topk, N, H, scaling
                )
                if getattr(module, "SLOT_SKIP_SEMANTICS", False):
                    torch.testing.assert_close(
                        actual, skip_expected, rtol=2e-2, atol=2e-3
                    )
                else:
                    torch.testing.assert_close(
                        actual, literal, rtol=2e-2, atol=2e-3, equal_nan=True
                    )

    def test_token_program_boundaries(self):
        # E1: grid.x is capped at 65535 // gy, so token counts above the
        # cap must be covered by the grid-stride row loop.
        for n in (65535, 65536, 65537):
            with self.subTest(num_tokens=n):
                self.check(*self.make(n, 8, 1, 8, padding=False))

    def test_grid_product_boundary(self):
        # rows=32768, hidden=2049 -> grid (32767, 2): the final row is
        # only reachable through the grid-stride iteration.
        args = list(self.make(32768, 8, 1, 2049, padding=False))
        args[0][-1, :].fill_(0.25)
        args[2][-1, 0] = 32767
        args[4][-1, 0] = 1.0
        args[8] = 1.0
        self.check(*args)


class PostReorderDeepgemmGridTest(unittest.TestCase):
    def test_grid_product_cap(self):
        # Metadata stubs capture launch dimensions on CPU without
        # allocating the large token x hidden combinations or claiming
        # numerical coverage.
        class TensorMetadata:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
                self.ndim = len(shape)
                # the wrappers evaluate output.device as a torch.empty
                # argument before the mocked call can intercept it
                self.device = torch.device("cpu")

            def stride(self, dimension):
                return self.shape[1] if dimension == 0 else 1

        class CaptureKernel:
            def __init__(self):
                self.grids = []

            def __getitem__(self, grid):
                self.grids.append(grid)
                return lambda *args, **kwargs: None

        for rows, hidden, expected_grid in (
            (32768, 2049, (32767, 2)),
            (258, 255 * 2048, (257, 255)),
            (65537, 255 * 2048 + 1, (257, 255)),
        ):
            down = TensorMetadata((rows, hidden), torch.bfloat16)
            out_tpl = TensorMetadata((rows, hidden), torch.bfloat16)
            s2d = TensorMetadata((rows, 1), torch.int32)
            ids = TensorMetadata((rows, 1), torch.int32)
            w = TensorMetadata((rows, 1), torch.float32)
            for name, module in MODULES:
                with self.subTest(module=name, rows=rows, hidden=hidden):
                    capture = CaptureKernel()
                    with mock.patch.object(
                        module, "_post_reorder_deepgemm", capture
                    ), mock.patch.object(
                        module.torch, "empty", return_value=out_tpl
                    ):
                        module.post_reorder_deepgemm(
                            down, out_tpl, s2d, ids, w, 1, rows, hidden, 1.0
                        )
                    self.assertEqual(len(capture.grids), 1)
                    grid = capture.grids[0]
                    product = 1
                    for size in grid:
                        self.assertGreater(size, 0)
                        product *= size
                    self.assertLessEqual(product, 65535)
                    if name == "generic":
                        self.assertEqual(grid, expected_grid)


RELEASE_REQUIRED_TESTS = [
    "PostReorderDeepgemmTest.test_matrix",
    "PostReorderDeepgemmTest.test_all_valid",
    "PostReorderDeepgemmTest.test_transposed_routing_tables",
    "PostReorderDeepgemmTest.test_fp16",
    "PostReorderDeepgemmTest.test_hidden_program_boundaries",
    "PostReorderDeepgemmTest.test_two_segment_hidden_semantics",
    "PostReorderDeepgemmTest.test_slot_skip_reference_gate",
    "PostReorderDeepgemmTest.test_slot_skip_nonfinite_semantics",
    "PostReorderDeepgemmTest.test_token_program_boundaries",
    "PostReorderDeepgemmTest.test_grid_product_boundary",
    "PostReorderDeepgemmGridTest.test_grid_product_cap",
]

if __name__ == "__main__":
    unittest.main()
