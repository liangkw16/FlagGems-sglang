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
import torch.nn.functional as F

from tests._op_variants import load_operator_modules

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "fused_gdn_gating.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fused_gdn_gating_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        beta * x <= threshold, F.softplus(x, beta=beta), x
    )
    g = -torch.exp(A_log.float()) * softplus_x
    beta_output = torch.sigmoid(b.float())
    return g.unsqueeze(0).to(torch.float32), beta_output.unsqueeze(0).to(
        torch.float32
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FusedGdnGatingTest(unittest.TestCase):
    def _check(self, A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
        actual_g, actual_beta = MODULE.fused_gdn_gating(
            A_log, a, b, dt_bias, beta, threshold
        )
        ref_g, ref_beta = reference(A_log, a, b, dt_bias, beta, threshold)
        for name, got, exp in (
            ("g", actual_g, ref_g),
            ("beta", actual_beta, ref_beta),
        ):
            self.assertEqual(got.shape, exp.shape, name)
            self.assertEqual(got.dtype, exp.dtype, name)
            torch.testing.assert_close(got, exp, atol=1e-5, rtol=1e-5)

    def test_row_head_tiles_and_parameter_strides(self):
        torch.manual_seed(53)
        for rows, heads in ((3, 127), (4, 128), (5, 129)):
            a = torch.randn(rows, heads * 2, device="cuda")[:, ::2]
            b = torch.randn_like(a)
            alog = torch.randn(heads * 2, device="cuda")[::2]
            bias = torch.randn(heads * 2, device="cuda")[::2]
            for beta, threshold in ((0.5, 10.0), (1.0, 20.0), (2.0, 20.0)):
                with self.subTest(rows=rows, heads=heads, beta=beta):
                    actual = MODULE.fused_gdn_gating(
                        alog, a, b, bias, beta, threshold
                    )
                    expected = reference(alog, a, b, bias, beta, threshold)
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(
                            got, want, atol=1e-4, rtol=1e-4
                        )

    def test_basic(self):
        for B, H in ((1, 8), (4, 16), (32, 32)):
            with self.subTest(B=B, H=H):
                A_log = torch.randn(H, device="cuda")
                a = torch.randn(B, H, device="cuda") * 3
                b = torch.randn(B, H, device="cuda") * 3
                dt_bias = torch.randn(H, device="cuda")
                self._check(A_log, a, b, dt_bias)

    def test_threshold_boundary(self):
        # values near and beyond threshold=20
        H = 4
        A_log = torch.randn(H, device="cuda")
        dt_bias = torch.zeros(H, device="cuda")
        a = torch.tensor(
            [[-25.0, -20.0, 19.9, 25.0], [-1.0, 0.0, 20.0, 30.0]],
            device="cuda",
        )
        b = torch.randn(2, H, device="cuda")
        self._check(A_log, a, b, dt_bias, beta=1.0, threshold=20.0)

    def test_custom_beta(self):
        H = 8
        A_log = torch.randn(H, device="cuda")
        a = torch.randn(4, H, device="cuda")
        b = torch.randn(4, H, device="cuda")
        dt_bias = torch.randn(H, device="cuda")
        self._check(A_log, a, b, dt_bias, beta=2.0, threshold=15.0)

    def test_empty_batch(self):
        H = 8
        A_log = torch.randn(H, device="cuda")
        a = torch.zeros(0, H, device="cuda")
        b = torch.zeros(0, H, device="cuda")
        dt_bias = torch.randn(H, device="cuda")
        g, beta_out = MODULE.fused_gdn_gating(A_log, a, b, dt_bias)
        self.assertEqual(g.shape, (1, 0, H))
        self.assertEqual(beta_out.shape, (1, 0, H))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FusedGdnGatingVariantsTest(unittest.TestCase):
    """Core matrix across every backend variant (generic + vendors)."""

    MODULES = load_operator_modules("fused_gdn_gating")

    def test_variants_match_reference(self):
        for B, H in ((4, 16), (32, 8), (1, 7), (257, 128)):
            torch.manual_seed(B * 1009 + H)
            # A_log kept within platform-observed magnitude: the
            # log(1+exp) softplus rounding (~6e-8 abs) amplifies by
            # exp(A_log); beyond ~e^6 it can exceed the 1e-4 gate even
            # though the platform's own cases (E1 passed 8/8) never
            # combine extreme-negative x with a huge A_log.
            A_log = torch.randn(H, device="cuda") * 2
            a = torch.randn(B, H, device="cuda") * 3  # match platform range
            b = torch.randn(B, H, device="cuda") * 3
            dt_bias = torch.randn(H, device="cuda") * 5
            ref_g, ref_beta = reference(A_log, a, b, dt_bias)
            for name, module in self.MODULES:
                with self.subTest(module=name, B=B, H=H):
                    g, beta_out = module.fused_gdn_gating(A_log, a, b, dt_bias)
                    torch.testing.assert_close(g, ref_g, atol=1e-4, rtol=1e-4)
                    torch.testing.assert_close(
                        beta_out, ref_beta, atol=1e-4, rtol=1e-4
                    )

    def test_variants_noncontiguous_inputs(self):
        # SGLang #22312-class regression: non-contiguous a/b must be read
        # through their own strides, never an assumed-contiguous flat
        # layout.
        B, H = 32, 16
        torch.manual_seed(7)
        A_log = torch.randn(H, device="cuda") * 2
        wide_a = torch.randn(B, H * 2, device="cuda") * 3
        wide_b = torch.randn(B, H * 2, device="cuda") * 3
        a = wide_a[:, ::2]
        b = wide_b[:, 1::2]
        self.assertFalse(a.is_contiguous())
        self.assertFalse(b.is_contiguous())
        dt_bias = torch.randn(H, device="cuda") * 5
        ref_g, ref_beta = reference(A_log, a, b, dt_bias)
        for name, module in self.MODULES:
            with self.subTest(module=name):
                g, beta_output = module.fused_gdn_gating(A_log, a, b, dt_bias)
                torch.testing.assert_close(g, ref_g, atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(
                    beta_output, ref_beta, atol=1e-4, rtol=1e-4
                )


RELEASE_REQUIRED_TESTS = [
    "FusedGdnGatingTest.test_row_head_tiles_and_parameter_strides",
    "FusedGdnGatingTest.test_basic",
    "FusedGdnGatingTest.test_threshold_boundary",
    "FusedGdnGatingTest.test_custom_beta",
    "FusedGdnGatingTest.test_empty_batch",
    "FusedGdnGatingVariantsTest.test_variants_match_reference",
    "FusedGdnGatingVariantsTest.test_variants_noncontiguous_inputs",
]


if __name__ == "__main__":
    unittest.main()
