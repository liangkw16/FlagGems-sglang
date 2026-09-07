# Copyright 2026 FlagOS Contributors
# (same license)
import unittest

import torch

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("log_scaling_tau")
MODULE = dict(MODULES)["generic"]


def reference(x, tau):
    rows = x.shape[0]
    tau_r = tau.reshape(rows).float()
    shape = [rows] + [1] * (x.dim() - 1)
    return (x.float() * tau_r.view(shape)).to(x.dtype)


def make_case(T, tail, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((T,) + tail, generator=g).cuda().to(dtype)
    tau = torch.randn(T, generator=g).cuda().float()
    return x, tau


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class LogScalingTauTest(unittest.TestCase):
    def _check(self, x, tau):
        expected = reference(x, tau)
        for name, mod in MODULES:
            with self.subTest(module=name):
                actual = mod.log_scaling_tau(x, tau)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                torch.testing.assert_close(
                    actual.float(), expected.float(), atol=1e-2, rtol=1e-2
                )

    def test_platform_shapes(self):
        for T, tail, dtype in (
            (4, (64,), torch.float16),
            (8, (128,), torch.float16),
            (16, (64, 128), torch.float16),
            (32, (256,), torch.float16),
            (4, (64,), torch.bfloat16),
        ):
            with self.subTest(T=T, tail=tail, dtype=dtype):
                self._check(*make_case(T, tail, dtype))

    def test_strided_input_and_tau(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.randn(3, 2, device="cuda", dtype=dtype).t()
            tau = torch.tensor([2.0, 99.0, -3.0, 99.0], device="cuda")[::2]
            with self.subTest(dtype=dtype):
                self._check(x, tau)

    def test_tail_boundaries(self):
        for n in (63, 64, 65, 1023, 1024, 1025, 256 * 1024):
            with self.subTest(n=n):
                self._check(*make_case(3, (n,), torch.float32))
        self._check(*make_case(2, (3, 5), torch.bfloat16))
        self._check(*make_case(3, (), torch.float32))
        self._check(*make_case(0, (64,), torch.float16))

    def test_fp32(self):
        self._check(*make_case(7, (320,), torch.float32, seed=2))

    def test_repeated_calls_direct_launch(self):
        # Second and later calls for a shape key take the prebound
        # CompiledKernel launcher; repeated calls and fresh tensors
        # must stay exact.
        for T, tail, dtype in (
            (4, (64,), torch.float16),
            (16, (64, 128), torch.float16),
            (4, (64,), torch.bfloat16),
        ):
            x, tau = make_case(T, tail, dtype)
            expected = reference(x, tau)
            for name, mod in MODULES:
                with self.subTest(module=name, T=T, dtype=dtype):
                    first = mod.log_scaling_tau(x, tau)
                    second = mod.log_scaling_tau(x, tau)
                    third = mod.log_scaling_tau(x.clone(), tau.clone())
                    for actual in (first, second, third):
                        torch.testing.assert_close(
                            actual.float(),
                            expected.float(),
                            atol=1e-2,
                            rtol=1e-2,
                        )
        # Misaligned contiguous views (odd storage offset) must
        # re-enter the standard dispatch path after the aligned key is
        # already cached and stay correct.
        storage = torch.randn(4 * 64 + 1, device="cuda", dtype=torch.float16)
        x = storage[1:].view(4, 64)
        tau = torch.randn(4, device="cuda").float()
        expected = reference(x, tau)
        for name, mod in MODULES:
            with self.subTest(module=name, case="misaligned"):
                actual = mod.log_scaling_tau(x, tau)
                torch.testing.assert_close(
                    actual.float(), expected.float(), atol=1e-2, rtol=1e-2
                )


RELEASE_REQUIRED_TESTS = [
    "LogScalingTauTest.test_platform_shapes",
    "LogScalingTauTest.test_fp32",
    "LogScalingTauTest.test_strided_input_and_tau",
    "LogScalingTauTest.test_tail_boundaries",
    "LogScalingTauTest.test_repeated_calls_direct_launch",
]


if __name__ == "__main__":
    unittest.main()
