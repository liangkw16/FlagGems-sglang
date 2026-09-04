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
    / "causal_conv1d_update.py"
)
SPEC = importlib.util.spec_from_file_location(
    "causal_conv1d_update_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


def reference(x, conv_state, weight, bias=None, activation="silu"):
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    conv_state = conv_state.clone()

    x_cat = torch.cat([conv_state.float(), x.float()], dim=-1)
    out = torch.zeros_like(x, dtype=torch.float32)

    for t in range(seqlen):
        window = x_cat[:, :, t : t + state_len + 1][:, :, -width:]
        val = (window * weight.float().unsqueeze(0)).sum(-1)
        if bias is not None:
            val = val + bias.float()
        if activation in ("silu", "swish"):
            val = val * torch.sigmoid(val)
        out[:, :, t] = val

    new_conv_state = x_cat[:, :, -state_len:]
    conv_state.copy_(new_conv_state.to(conv_state.dtype))

    out = out.to(x.dtype)
    if unsqueeze:
        out = out.squeeze(-1)
    return out, conv_state


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class CausalConv1dUpdateTest(unittest.TestCase):
    def _check(self, x, conv_state, weight, bias=None, activation="silu"):
        x_snapshot = x.clone()
        state_snapshot = conv_state.clone()
        w_snapshot = weight.clone()
        actual_out, actual_state = MODULE.causal_conv1d_update(
            x, conv_state, weight, bias=bias, activation=activation
        )
        expected_out, expected_state = reference(
            x, conv_state, weight, bias=bias, activation=activation
        )
        self.assertEqual(actual_out.shape, expected_out.shape)
        self.assertEqual(actual_out.dtype, expected_out.dtype)
        self.assertEqual(actual_state.shape, expected_state.shape)
        self.assertEqual(actual_state.dtype, expected_state.dtype)
        atol, rtol = TOLERANCES[x.dtype]
        torch.testing.assert_close(actual_out, expected_out, atol=atol, rtol=rtol)
        atol_s, rtol_s = TOLERANCES[conv_state.dtype]
        torch.testing.assert_close(
            actual_state, expected_state, atol=atol_s, rtol=rtol_s
        )
        # Inputs must be untouched (reference clones before updating).
        torch.testing.assert_close(x, x_snapshot)
        torch.testing.assert_close(conv_state, state_snapshot)
        torch.testing.assert_close(weight, w_snapshot)
        return actual_out, actual_state

    def test_dtypes_with_bias_and_activation(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for activation in ("silu", "swish", "identity"):
                with self.subTest(dtype=dtype, activation=activation):
                    g = torch.Generator(device="cuda").manual_seed(0)
                    x = torch.randn(8, 512, device="cuda", dtype=dtype)
                    state = torch.randn(
                        8, 512, 3, device="cuda", dtype=dtype, generator=g
                    )
                    w = torch.randn(512, 4, device="cuda", dtype=dtype, generator=g)
                    b = torch.randn(512, device="cuda", dtype=dtype, generator=g)
                    self._check(x, state, w, bias=b, activation=activation)

    def test_widths_state_lengths_and_seqlens(self):
        for width, state_len, seqlen in (
            (2, 1, 1),
            (2, 4, 1),
            (3, 2, 1),
            (3, 5, 2),
            (4, 3, 1),
            (4, 3, 3),
            (4, 7, 5),
            (5, 4, 2),
            (8, 7, 1),
        ):
            with self.subTest(width=width, state_len=state_len, seqlen=seqlen):
                x = torch.randn(3, 200, seqlen, device="cuda") * 2.0
                state = torch.randn(3, 200, state_len, device="cuda")
                w = torch.randn(200, width, device="cuda")
                self._check(x, state, w)

    def test_2d_input_is_seqlen_one(self):
        x2d = torch.randn(7, 300, device="cuda")
        state = torch.randn(7, 300, 3, device="cuda")
        w = torch.randn(300, 4, device="cuda")
        out, new_state = self._check(x2d, state, w)
        self.assertEqual(out.dim(), 2)
        self.assertEqual(out.shape, (7, 300))
        self.assertTrue(new_state.shape == (7, 300, 3))

    def test_batch_and_dim_boundaries(self):
        for batch, dim in ((1, 1), (1, 64), (3, 100), (17, 255), (2, 256), (5, 2049)):
            with self.subTest(batch=batch, dim=dim):
                x = torch.randn(batch, dim, 1, device="cuda")
                state = torch.randn(batch, dim, 3, device="cuda")
                w = torch.randn(dim, 4, device="cuda")
                self._check(x, state, w)

    def test_non_contiguous_inputs(self):
        x_base = torch.randn(4, 128, 8, device="cuda")
        x = x_base[:, :, ::2]
        state_base = torch.randn(4, 128, 12, device="cuda")
        state = state_base[:, :, 1:]
        w = torch.randn(128, 4, device="cuda")
        self.assertFalse(x.is_contiguous())
        self.assertFalse(state.is_contiguous())
        self._check(x, state, w)

    def test_seqlen_larger_than_state_len(self):
        # Full state replacement path: new state comes entirely from x.
        x = torch.randn(2, 96, 6, device="cuda")
        state = torch.randn(2, 96, 3, device="cuda")
        w = torch.randn(96, 4, device="cuda")
        out, new_state = self._check(x, state, w)
        # Last state_len columns of the fp32 x_cat equal x[:, :, -3:].
        torch.testing.assert_close(
            new_state, x[:, :, -3:].to(state.dtype), atol=1e-6, rtol=1e-6
        )

    def test_special_values(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.full((2, 64, 1), -92.0, device="cuda", dtype=dtype)
                x[0, 0, 0] = 90.0
                x[1, 3, 0] = float("inf")
                state = torch.full((2, 64, 3), -90.0, device="cuda", dtype=dtype)
                state[0, 1, 2] = 1e4
                w = torch.ones(64, 4, device="cuda", dtype=dtype)
                # silu(-inf) and silu(+large) can produce inf/NaN in both
                # implementations; compare with matching NaNs allowed.
                actual_out, _ = MODULE.causal_conv1d_update(x, state, w)
                expected_out, _ = reference(x, state, w)
                atol, rtol = TOLERANCES[dtype]
                torch.testing.assert_close(
                    actual_out,
                    expected_out,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=True,
                )

    def test_empty_batch(self):
        x = torch.randn(0, 64, 1, device="cuda")
        state = torch.randn(0, 64, 3, device="cuda")
        w = torch.randn(64, 4, device="cuda")
        out, new_state = MODULE.causal_conv1d_update(x, state, w)
        self.assertEqual(out.shape, (0, 64, 1))
        self.assertEqual(new_state.shape, (0, 64, 3))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class CausalConv1dUpdateVariantsTest(unittest.TestCase):
    """Core matrix across every backend variant (generic + vendors)."""

    MODULES = load_operator_modules("causal_conv1d_update")

    def test_variants_match_reference(self):
        for width, state_len, seqlen in ((4, 3, 1), (3, 5, 2), (4, 3, 3)):
            x = torch.randn(5, 200, seqlen, device="cuda") * 2.0
            state = torch.randn(5, 200, state_len, device="cuda")
            w = torch.randn(200, width, device="cuda")
            b = torch.randn(200, device="cuda")
            e_out, e_state = reference(x, state, w, bias=b)
            for name, module in self.MODULES:
                with self.subTest(module=name, w=width, sl=state_len, sq=seqlen):
                    out, new_state = module.causal_conv1d_update(x, state, w, bias=b)
                    torch.testing.assert_close(out, e_out, atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(new_state, e_state, atol=1e-5, rtol=1e-5)
