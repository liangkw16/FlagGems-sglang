# Copyright 2026 FlagOS Contributors
# (same license)
import importlib.util
import unittest
from pathlib import Path
import torch

MODULE_PATH = (
    Path(__file__).parents[1] / "src" / "flaggems_sglang" / "ops" / "log_scaling_tau.py"
)
SPEC = importlib.util.spec_from_file_location("log_scaling_tau_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


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
        actual = MODULE.log_scaling_tau(x, tau)
        expected = reference(x, tau)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)

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

    def test_fp32(self):
        self._check(*make_case(7, (320,), torch.float32, seed=2))


if __name__ == "__main__":
    unittest.main()
