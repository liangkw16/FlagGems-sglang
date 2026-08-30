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

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "sigmoid_gate_topk_renorm.py"
)
SPEC = importlib.util.spec_from_file_location(
    "sigmoid_gate_topk_renorm_module", MODULE_PATH
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def reference(logits, k, n_shared_experts, route_scale, global_scale, bias):
    T, E = logits.shape
    N = E - n_shared_experts
    routed_logits = logits[:, :N]
    shared_logits = logits[:, N:]
    sel = torch.sigmoid(routed_logits.float()) + bias.float()[None, :]
    indices = torch.argsort(sel, dim=-1, descending=True)[:, :k]
    routed_vals = torch.gather(routed_logits.float(), 1, indices)
    active = torch.cat([routed_vals, shared_logits.float()], dim=-1)
    probs = torch.sigmoid(active)
    weights = probs / probs.sum(dim=-1, keepdim=True)
    weights = weights * route_scale * float(global_scale)
    return (
        weights[:, :k].to(logits.dtype),
        indices.to(torch.int32),
        weights[:, k:].to(logits.dtype),
    )


class TestSigmoidGateTopkRenorm(unittest.TestCase):
    TOL = {
        torch.float32: dict(rtol=1e-4, atol=1e-4),
        torch.float16: dict(rtol=1e-2, atol=1e-2),
        torch.bfloat16: dict(rtol=1.5e-2, atol=1.5e-2),
    }

    def _run(self, T, N, S, k, dtype, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed + T + N)
        logits = torch.randn(T, N + S, generator=g).to(
            dtype=dtype, device="cuda"
        )
        bias = torch.randn(N, generator=g).to(
            dtype=torch.float32, device="cuda"
        )
        gs = torch.tensor([1.7], dtype=torch.float32, device="cuda")
        rw, idx, sw = MOD.sigmoid_gate_topk_renorm(logits, k, S, 2.5, gs, bias)
        rrw, ridx, rsw = reference(logits, k, S, 2.5, 1.7, bias)
        self.assertEqual(idx.dtype, torch.int32)
        self.assertTrue(
            torch.equal(idx, ridx)
            or torch.equal(idx.sort(dim=-1).values, ridx.sort(dim=-1).values),
            f"idx mismatch {idx[0]} vs {ridx[0]}",
        )
        tol = self.TOL[dtype]
        torch.testing.assert_close(rw.float(), rrw.float(), **tol)
        torch.testing.assert_close(sw.float(), rsw.float(), **tol)

    def test_matrix(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for T, N, S, k in [
                (64, 256, 2, 8),
                (1024, 512, 1, 6),
                (16, 128, 4, 4),
                (4096, 256, 8, 16),
                (7, 33, 3, 5),
                (128, 64, 0, 8),
                (1, 8, 1, 1),
            ]:
                with self.subTest(dtype=dtype, T=T, N=N, S=S, k=k):
                    self._run(T, N, S, k, dtype)

    def test_scale_tensor_and_invariance(self):
        logits = torch.randn(32, 66, device="cuda").to(torch.float32)
        bias = torch.randn(64, device="cuda")
        lc = logits.clone()
        MOD.sigmoid_gate_topk_renorm(logits, 4, 2, 1.0, 2.0, bias)
        self.assertTrue(torch.equal(logits, lc))

    def test_empty(self):
        logits = torch.empty(0, 16, device="cuda").to(torch.float16)
        bias = torch.zeros(14, device="cuda")
        rw, idx, sw = MOD.sigmoid_gate_topk_renorm(
            logits, 4, 2, 1.0, 1.0, bias
        )
        self.assertEqual(rw.shape, (0, 4))
        self.assertEqual(sw.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
