# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("fused_gate_sigmoid_mul_add")


def reference(hidden_states, gate_weight, shared_output, final_hidden_states):
    gate = (hidden_states.float() * gate_weight.float()).sum(dim=-1)
    out = final_hidden_states.float() + torch.sigmoid(gate)[:, None] * (
        shared_output.float()
    )
    return out.to(final_hidden_states.dtype)


def make_case(
    rows=7, hidden=1024, dtype=torch.bfloat16, strided=False, seed=0
):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    args = [
        torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
        for shape in (
            (rows, hidden),
            (hidden,),
            (rows, hidden),
            (rows, hidden),
        )
    ]
    if strided:
        # Row-gap strides keep the inner dim contiguous (the kernel's
        # contract); the transpose trick would break stride(1) == 1.
        strided_args = []
        for i, x in enumerate(args):
            if x.ndim == 2:
                buffer = torch.zeros(
                    x.shape[0] * 2, x.shape[1], dtype=x.dtype, device=x.device
                )
                buffer[1::2] = x
                strided_args.append(buffer[1::2])
            else:
                strided_args.append(x)
        args = strided_args
    return tuple(args)


# Reduction-order noise: any Triton reordering of the row dot moves
# sigmoid outputs by a few low-order bits, and cancellation points
# (final ~ -gate*shared) turn tiny absolute diffs into large relative
# ones. These tolerances cover that noise floor while staying far
# tighter than the platform's per-dtype standard.
TOLERANCES = {
    torch.bfloat16: (2e-2, 2e-2),
    torch.float16: (1e-2, 1e-2),
    torch.float32: (1e-3, 1e-3),
}


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class FusedGateSigmoidMulAddTest(unittest.TestCase):
    def check(self, args, equal_nan=False):
        expected = reference(*args)
        snapshots = [x.clone() for x in args]
        for name, module in MODULES:
            with self.subTest(module=name):
                actual = module.fused_gate_sigmoid_mul_add(*args)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                rtol, atol = TOLERANCES[args[0].dtype]
                torch.testing.assert_close(
                    actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan
                )
                for value, before in zip(args, snapshots):
                    torch.testing.assert_close(
                        value, before, rtol=0, atol=0, equal_nan=True
                    )

    def test_dtype_matrix_and_shapes(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for rows, hidden in (
                (1, 1),
                (7, 1024),
                (127, 1023),
                (128, 1025),
                (513, 5120),
                (2049, 7168),
                (128, 8191),
                (128, 8192),
                (128, 8193),
            ):
                with self.subTest(dtype=dtype, rows=rows, hidden=hidden):
                    self.check(
                        make_case(rows=rows, hidden=hidden, dtype=dtype)
                    )

    def test_strided_rows(self):
        self.check(make_case(rows=33, hidden=2048, strided=True))

    def test_gate_saturation_and_zero_rows(self):
        args = make_case(rows=4, hidden=512)
        args[0][0] *= 200.0  # sigmoid saturates to 1
        args[0][1] *= -200.0  # sigmoid saturates to 0
        args[0][2] = 0.0  # gate = 0 -> sigmoid = 0.5 exactly
        self.check(args)

    def test_cancellation_and_tails(self):
        # Row 0: severe cancellation in the fp32 dot; rows 1-2 cross the
        # BLOCK_H tail with a hidden width no tile divides.
        args = list(
            make_case(rows=3, hidden=1535, seed=7)
        )
        args[0] = args[0].clone()
        args[0][0] = torch.linspace(
            -1.0, 1.0, 1535, dtype=args[0].dtype, device="cuda"
        )
        args[1] = torch.linspace(
            1.0, -1.0, 1535, dtype=args[1].dtype, device="cuda"
        )
        self.check(tuple(args))

    def test_nan_propagation(self):
        args = make_case(rows=2, hidden=256)
        args[0][0, 0] = float("nan")
        self.check(args, equal_nan=True)

    def test_single_wave_grid_routing(self):
        # E12 host routing: the upstream single-wave launch (one program
        # per row, pid * HDIM int32 addressing) is legal only inside the
        # 65535 grid.x limit and while the element span rows * hdim
        # stays below 2**31; anything larger must route multi-wave. The
        # 2**31 arm cannot be allocated on the 16GB proxy, so the
        # boundary is asserted on the routing arithmetic itself.
        for name, module in MODULES:
            if not hasattr(module, "_single_wave_grid"):
                # Out-of-scope variants: e13 rolls generic and kunlunxin
                # back to the e10 multi-wave-only bytes (the e12 single
                # wave hung tianshu and cost kunlun -75%); ascend keeps
                # the e6 form, enflame keeps its own e11 single-wave
                # policy.
                continue
            with self.subTest(module=name):
                route = module._single_wave_grid
                self.assertEqual(route(1, 1), 1)
                self.assertEqual(route(65535, 1), 65535)
                # 65535 * 32768 == 2**31 - 2**15: largest legal span.
                self.assertEqual(route(65535, 32768), 65535)
                # One row past the grid.x limit routes multi-wave.
                self.assertIsNone(route(65536, 64))
                # 65536 * 32768 == 2**31 exactly: int32 span overflow.
                self.assertIsNone(route(65536, 32768))
                # Span overflow with tiny row counts routes multi-wave.
                self.assertIsNone(route(2, 2**30))
                self.assertIsNone(route(2**31, 1))

    def test_multiwave_fallback_above_grid_limit(self):
        # rows one past the 65535 grid.x limit: every module must route
        # to its multi-wave grid-stride kernel and stay correct.
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self.check(
                    make_case(
                        rows=65536, hidden=64, dtype=dtype, seed=11
                    )
                )

    def test_single_wave_multi_wave_parity(self):
        # Identical bytes through both branches: contiguous rows take
        # the single-wave kernel where the module has one (e13: metax
        # and hygon; enflame runs its own single-wave policy either
        # way), row-gapped strides take the multi-wave kernel - branch
        # choice must not move a bit. Modules already on multi-wave-only
        # bytes (e13 generic/kunlunxin, ascend) hold the same parity
        # through the identical kernel.
        dense = make_case(rows=97, hidden=2048, seed=5)
        gapped = make_case(rows=97, hidden=2048, strided=True, seed=5)
        for name, module in MODULES:
            with self.subTest(module=name):
                a = module.fused_gate_sigmoid_mul_add(*dense)
                b = module.fused_gate_sigmoid_mul_add(*gapped)
                torch.testing.assert_close(a, b, rtol=0, atol=0)


RELEASE_REQUIRED_TESTS = [
    "FusedGateSigmoidMulAddTest.test_dtype_matrix_and_shapes",
    "FusedGateSigmoidMulAddTest.test_strided_rows",
    "FusedGateSigmoidMulAddTest.test_gate_saturation_and_zero_rows",
    "FusedGateSigmoidMulAddTest.test_cancellation_and_tails",
    "FusedGateSigmoidMulAddTest.test_nan_propagation",
    "FusedGateSigmoidMulAddTest.test_single_wave_grid_routing",
    "FusedGateSigmoidMulAddTest.test_multiwave_fallback_above_grid_limit",
    "FusedGateSigmoidMulAddTest.test_single_wave_multi_wave_parity",
]


if __name__ == "__main__":
    unittest.main()
