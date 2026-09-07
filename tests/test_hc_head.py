# Copyright 2026 FlagOS Contributors
# (same license)
import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from tests._op_variants import load_operator_modules

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "hc_head.py"
)
SPEC = importlib.util.spec_from_file_location("hc_head_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


def make_case(T, hc_mult, hidden, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, hc_mult, hidden, generator=g).cuda().to(torch.bfloat16)
    hc_fn = (
        torch.randn(hc_mult, hc_mult * hidden, generator=g).cuda().float()
        * 0.02
    )
    hc_scale = torch.randn(1, generator=g).cuda().float()
    hc_base = torch.randn(hc_mult, generator=g).cuda().float()
    return x, hc_fn, hc_scale, hc_base, 1e-6, 1e-3


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class HcHeadTest(unittest.TestCase):
    def _check(self, *args):
        actual = MODULE.hc_head(*args)
        expected = reference(*args)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=1.5e-2, rtol=1.5e-2
        )

    def test_shapes(self):
        for T, hc_mult, hidden in (
            (4, 2, 128),
            (8, 4, 512),
            (16, 4, 1024),
            (3, 8, 2048),
            (1, 4, 320),
            (33, 4, 256),
        ):
            with self.subTest(T=T, hc_mult=hc_mult, hidden=hidden):
                self._check(*make_case(T, hc_mult, hidden, seed=T + hc_mult))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class HcHeadVariantsTest(unittest.TestCase):
    """Core matrix across every backend variant (generic + vendors)."""

    MODULES = load_operator_modules("hc_head")

    def test_variants_match_reference(self):
        for T, hc_mult, hidden in (
            (4, 2, 128),
            (3, 8, 2048),
            (1, 4, 320),
            (33, 4, 256),
        ):
            with self.subTest(T=T, hc_mult=hc_mult, hidden=hidden):
                args = make_case(T, hc_mult, hidden, seed=T * hc_mult)
                expected = reference(*args)
                for name, module in self.MODULES:
                    with self.subTest(module=name):
                        actual = module.hc_head(*args)
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(actual.dtype, expected.dtype)
                        torch.testing.assert_close(
                            actual.float(),
                            expected.float(),
                            atol=1.5e-2,
                            rtol=1.5e-2,
                        )


RELEASE_REQUIRED_TESTS = [
    "HcHeadTest.test_shapes",
    "HcHeadVariantsTest.test_variants_match_reference",
]


if __name__ == "__main__":
    unittest.main()
