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

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "selective_state_update.py"
)
SPEC = importlib.util.spec_from_file_location(
    "selective_state_update_module", MODULE_PATH
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def reference(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    state = state.clone()
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    ratio = nheads // ngroups

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = F.softplus(dt_f)

    dA = torch.exp(dt_f.unsqueeze(-1) * A.float().unsqueeze(0).unsqueeze(2))
    B_exp = B.float().repeat_interleave(ratio, dim=1)
    dB = dt_f.unsqueeze(-1) * B_exp.unsqueeze(2)

    new_state = state.float() * dA + dB * x.float().unsqueeze(-1)
    state = new_state.to(state.dtype)

    C_exp = C.float().repeat_interleave(ratio, dim=1)
    y = torch.einsum("bhpn,bhn->bhp", new_state, C_exp)

    if D is not None:
        y = y + D.float() * x.float()
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(x.dtype), state


def make_inputs(B, H, P, N, G, dtype, seed, softplus=False):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(
            dtype=dtype, device="cuda"
        )

    state = r(B, H, P, N)
    x = r(B, H, P)
    dt = (r(B, H, P, scale=0.3) + 0.5).to(dtype)
    A = (-torch.rand(H, N, generator=g) - 0.1).to(dtype=dtype, device="cuda")
    Bm = r(B, G, N)
    C = r(B, G, N)
    return state, x, dt, A, Bm, C


class TestSelectiveStateUpdate(unittest.TestCase):
    TOL = {
        torch.float32: dict(rtol=1e-4, atol=1e-4),
        torch.float16: dict(rtol=1e-2, atol=1e-2),
        torch.bfloat16: dict(rtol=1.5e-2, atol=1.5e-2),
    }

    def _check(self, dtype, B, H, P, N, G, use_D, use_z, use_bias, sp):
        state, x, dt, A, Bm, C = make_inputs(B, H, P, N, G, dtype, B * 7 + H)
        D = torch.randn(H, P, device="cuda").to(dtype) if use_D else None
        z = torch.randn(B, H, P, device="cuda").to(dtype) if use_z else None
        dt_bias = (
            torch.randn(H, P, device="cuda").to(dtype) * 0.1
            if use_bias
            else None
        )
        state_in = state.clone()
        y, ns = MOD.selective_state_update(
            state, x, dt, A, Bm, C, D, z, dt_bias, sp
        )
        ry, rns = reference(state, x, dt, A, Bm, C, D, z, dt_bias, sp)
        tol = self.TOL[dtype]
        torch.testing.assert_close(y.float(), ry.float(), **tol)
        torch.testing.assert_close(ns.float(), rns.float(), **tol)
        self.assertTrue(torch.equal(state, state_in))

    def test_matrix(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for B, H, P, N, G in [
                (2, 8, 64, 128, 1),
                (4, 16, 64, 64, 4),
                (1, 2, 1, 1, 1),
                (128, 32, 64, 128, 8),
                (3, 5, 33, 96, 1),
                (8, 64, 64, 128, 8),
            ]:
                for flags in [
                    (False, False, False, False),
                    (True, True, True, True),
                    (True, False, True, False),
                    (False, True, False, True),
                ]:
                    with self.subTest(
                        dtype=dtype, B=B, H=H, P=P, N=N, G=G, flags=flags
                    ):
                        self._check(dtype, B, H, P, N, G, *flags)

    def test_large_batch(self):
        state, x, dt, A, Bm, C = make_inputs(
            2048, 8, 64, 128, 1, torch.bfloat16, 99
        )
        y, ns = MOD.selective_state_update(state, x, dt, A, Bm, C)
        ry, rns = reference(state, x, dt, A, Bm, C)
        torch.testing.assert_close(
            y.float(), ry.float(), rtol=1.5e-2, atol=1.5e-2
        )
        torch.testing.assert_close(
            ns.float(), rns.float(), rtol=1.5e-2, atol=1.5e-2
        )
        del state, x, dt, A, Bm, C, y, ns, ry, rns
        torch.cuda.empty_cache()

    def test_noncontiguous(self):
        state, x, dt, A, Bm, C = make_inputs(
            2, 8, 64, 128, 2, torch.float32, 5
        )
        state_s = state[:, :, ::2]
        x_s = x[:, :, ::2]
        dt_s = dt[:, :, ::2]
        y, ns = MOD.selective_state_update(state_s, x_s, dt_s, A, Bm, C)
        ry, rns = reference(
            state_s.contiguous(),
            x_s.contiguous(),
            dt_s.contiguous(),
            A,
            Bm,
            C,
        )
        torch.testing.assert_close(y, ry, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(ns, rns, rtol=1e-4, atol=1e-4)

    def test_softplus_extremes(self):
        dt = torch.full((1, 2, 4), 30.0, dtype=torch.float32, device="cuda")
        dt[0, 0, :2] = -30.0
        state, _, _, A, Bm, C = make_inputs(1, 2, 4, 8, 1, torch.float32, 3)
        x = torch.ones(1, 2, 4, dtype=torch.float32, device="cuda")
        y, ns = MOD.selective_state_update(
            state, x, dt, A, Bm, C, dt_softplus=True
        )
        ry, rns = reference(state, x, dt, A, Bm, C, dt_softplus=True)
        torch.testing.assert_close(y, ry, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(ns, rns, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
