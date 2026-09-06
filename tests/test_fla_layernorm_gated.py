# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import unittest
from pathlib import Path

import torch

from tests._op_variants import load_operator_modules

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "fla_layernorm_gated.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fla_layernorm_gated_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


def reference(
    x, g, weight, bias, activation="swish", eps=1e-5, is_rms_norm=True
):
    out_dtype = x.dtype
    xf = x.float()
    if is_rms_norm:
        var = (xf**2).mean(dim=-1, keepdim=True)
        x_hat = xf * (var + eps).rsqrt()
    else:
        mean = xf.mean(dim=-1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (xf - mean) * (var + eps).rsqrt()
    y = x_hat
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    gf = g.float()
    if activation in ("swish", "silu"):
        y = y * gf * gf.sigmoid()
    elif activation == "sigmoid":
        y = y * gf.sigmoid()
    return y.to(out_dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FlaLayernormGatedTest(unittest.TestCase):
    def _check(
        self,
        x,
        g,
        weight=None,
        bias=None,
        activation="swish",
        eps=1e-5,
        is_rms_norm=True,
    ):
        actual = MODULE.fla_layernorm_gated(
            x, g, weight, bias, activation, eps, is_rms_norm
        )
        expected = reference(x, g, weight, bias, activation, eps, is_rms_norm)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        atol, rtol = TOL[x.dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_dtypes_and_activations(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for act in ("swish", "silu", "sigmoid"):
                with self.subTest(dtype=dtype, act=act):
                    x = torch.randn(32, 512, device="cuda", dtype=dtype)
                    g = torch.randn(32, 512, device="cuda", dtype=dtype)
                    self._check(x, g, activation=act)

    def test_norm_modes(self):
        for rms in (True, False):
            with self.subTest(rms=rms):
                x = torch.randn(16, 1024, device="cuda")
                g = torch.randn(16, 1024, device="cuda")
                self._check(x, g, is_rms_norm=rms)

    def test_weight_bias_combos(self):
        x = torch.randn(8, 256, device="cuda")
        g = torch.randn(8, 256, device="cuda")
        w = torch.randn(256, device="cuda")
        b = torch.randn(256, device="cuda")
        for has_w, has_b in (
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ):
            with self.subTest(w=has_w, b=has_b):
                self._check(x, g, w if has_w else None, b if has_b else None)

    def test_shapes(self):
        for T, D in ((1, 16), (100, 1023), (5, 1024), (3, 4096)):
            with self.subTest(T=T, D=D):
                x = torch.randn(T, D, device="cuda")
                g = torch.randn(T, D, device="cuda")
                self._check(x, g)

    def test_empty(self):
        x = torch.randn(0, 128, device="cuda")
        g = torch.randn(0, 128, device="cuda")
        out = MODULE.fla_layernorm_gated(x, g, None, None)
        self.assertEqual(out.shape, (0, 128))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FlaLayernormGatedVariantsTest(unittest.TestCase):
    """Core matrix across every backend variant (generic + vendors)."""

    MODULES = load_operator_modules("fla_layernorm_gated")

    def test_variants_match_reference(self):
        for T, D in ((8, 256), (32, 512)):
            for act in ("swish", "sigmoid"):
                x = torch.randn(T, D, device="cuda")
                g = torch.randn(T, D, device="cuda")
                w = torch.randn(D, device="cuda")
                ref = reference(x, g, w, None, activation=act)
                for name, module in self.MODULES:
                    with self.subTest(module=name, act=act, T=T):
                        out = module.fla_layernorm_gated(
                            x, g, w, None, activation=act
                        )
                        torch.testing.assert_close(
                            out, ref, atol=1e-4, rtol=1e-4
                        )


if __name__ == "__main__":
    unittest.main()
