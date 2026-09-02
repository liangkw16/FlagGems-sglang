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

import unittest

import torch
import torch.nn.functional as F

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("moe_fused_gate")


def reference(
    scores,
    bias,
    topk,
    scoring_func="sigmoid",
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
    moe_softcapping=0.0,
    num_expert_group=1,
    topk_group=1,
):
    scores = scores.float()
    bias = bias.float()
    M, N = scores.shape
    K = topk
    K_routed = topk - num_fused_shared_experts
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    if scoring_func == "sigmoid":
        activated = torch.sigmoid(scores)
        biased = activated + bias[None, :]
    elif scoring_func == "sqrtsoftplus":
        activated = torch.sqrt(F.softplus(scores))
        biased = activated + bias[None, :]
    else:
        logit = scores
        if moe_softcapping != 0.0:
            logit = moe_softcapping * torch.tanh(logit / moe_softcapping)
        biased = logit + bias[None, :]
        activated = torch.softmax(biased, dim=-1)

    if num_expert_group > 1:
        experts_per_group = N // num_expert_group
        biased_g = biased.view(M, num_expert_group, experts_per_group)
        top2 = torch.topk(biased_g, 2, dim=-1).values
        group_score = top2.sum(dim=-1)
        keep_idx = torch.topk(group_score, topk_group, dim=-1).indices
        keep_mask_g = torch.zeros(
            M, num_expert_group, dtype=torch.bool, device=scores.device
        )
        keep_mask_g.scatter_(1, keep_idx, True)
        keep_mask = (
            keep_mask_g.unsqueeze(-1)
            .expand(M, num_expert_group, experts_per_group)
            .reshape(M, N)
        )
        biased = torch.where(
            keep_mask, biased, torch.full_like(biased, -float("inf"))
        )

    _, top_idx = torch.topk(biased, K_routed, dim=-1)
    selected_vals = torch.gather(activated, 1, top_idx)
    routed_sum = selected_vals.sum(dim=-1, keepdim=True)

    weights = torch.zeros(M, K, dtype=torch.float32, device=scores.device)
    indices = torch.zeros(M, K, dtype=torch.int32, device=scores.device)
    weights[:, :K_routed] = selected_vals
    indices[:, :K_routed] = top_idx.to(torch.int32)

    num_shared = K - K_routed
    if num_shared > 0:
        shared_weight = routed_sum / routed_scaling_factor
        shared_idx = N + torch.arange(
            num_shared, device=scores.device, dtype=torch.int32
        )
        weights[:, K_routed:] = shared_weight.expand(M, num_shared)
        indices[:, K_routed:] = shared_idx[None, :].expand(M, num_shared)

    if renormalize:
        norm = torch.where(
            routed_sum > 0, routed_sum, torch.ones_like(routed_sum)
        )
        weights = weights / norm
    if apply_routed_scaling_factor_on_output:
        weights = weights * routed_scaling_factor

    return weights, indices


class TestMoeFusedGate(unittest.TestCase):
    def _run(self, M, N, dtype, **kw):
        torch.manual_seed(M * 131 + N)
        scores = (torch.randn(M, N, device="cuda") * 2).to(dtype)
        bias = torch.randn(N, device="cuda").to(torch.float32)
        rw, ri = reference(scores, bias, **kw)
        tolerance = {
            torch.float32: 1e-4,
            torch.float16: 1e-2,
            torch.bfloat16: 1.5e-2,
        }[dtype]
        for variant, mod in MODULES:
            with self.subTest(variant=variant):
                w, i = mod.moe_fused_gate(scores, bias, **kw)
                self.assertEqual(w.dtype, torch.float32)
                self.assertEqual(i.dtype, torch.int32)
                self.assertTrue(
                    torch.equal(i, ri),
                    f"{variant}: index mismatch {i} vs {ri}",
                )
                torch.testing.assert_close(
                    w, rw, rtol=tolerance, atol=tolerance
                )

    def test_matrix(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for sf in ("sigmoid", "sqrtsoftplus", "softmax"):
                for kw in [
                    dict(topk=8),
                    dict(topk=8, renormalize=False),
                    dict(
                        topk=6,
                        num_fused_shared_experts=2,
                        routed_scaling_factor=2.5,
                    ),
                    dict(
                        topk=8,
                        num_fused_shared_experts=2,
                        routed_scaling_factor=2.5,
                        renormalize=False,
                        apply_routed_scaling_factor_on_output=True,
                    ),
                    dict(topk=8, scoring_func=sf, moe_softcapping=30.0),
                ]:
                    kw = dict(kw)
                    kw["scoring_func"] = kw.get("scoring_func", sf)
                    with self.subTest(dtype=dtype, sf=sf, kw=kw):
                        self._run(64, 128, dtype, **kw)

    def test_grouped(self):
        for kw in [
            dict(topk=8, num_expert_group=8, topk_group=4),
            dict(
                topk=4,
                num_expert_group=4,
                topk_group=2,
                num_fused_shared_experts=1,
                routed_scaling_factor=2.0,
            ),
            dict(
                topk=8,
                num_expert_group=8,
                topk_group=7,
                scoring_func="softmax",
            ),
        ]:
            with self.subTest(kw=kw):
                self._run(32, 256, torch.float32, **kw)

    def test_shapes_and_order(self):
        # ties: duplicated scores must pick lower index first
        scores = torch.zeros(2, 8, device="cuda").to(torch.float32)
        scores[0] = torch.tensor([1.0, 3.0, 3.0, 2.0, 0, 0, 0, 0]).cuda()
        bias = torch.zeros(8, device="cuda")
        rw, ri = reference(scores, bias, topk=3)
        for variant, mod in MODULES:
            with self.subTest(variant=variant):
                w, i = mod.moe_fused_gate(scores, bias, topk=3)
                # torch.topk tie order is unspecified on CUDA; compare sets and
                # require our tie order to be ascending (lower index first)
                self.assertEqual(sorted(i[0].tolist()), sorted(ri[0].tolist()))
                self.assertEqual(i[0].tolist(), sorted(i[0].tolist()))
                torch.testing.assert_close(
                    w[:, 1:], rw[:, 1:], rtol=1e-4, atol=1e-4
                )

    def test_kunlun_grouped_edges(self):
        mod = dict(MODULES)["kunlunxin"]

        scores = torch.zeros(1, 8, device="cuda")
        selector = torch.tensor(
            [3.0, 3.0, 0.0, 0.0, 4.0, 1.5, 0.0, 0.0],
            device="cuda",
        )
        _, indices = mod.moe_fused_gate(
            scores,
            selector - 0.5,
            topk=1,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
        )
        self.assertLess(indices[0, 0].item(), 4)

        scores = torch.tensor([[10.0, 0.0, 6.0, 6.0]], device="cuda")
        weights, indices = mod.moe_fused_gate(
            scores,
            torch.zeros(4, device="cuda"),
            topk=1,
            scoring_func="softmax",
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
        )
        ref_weights, _ = reference(
            scores,
            torch.zeros(4, device="cuda"),
            topk=1,
            scoring_func="softmax",
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
        )
        self.assertIn(indices[0, 0].item(), (2, 3))
        torch.testing.assert_close(weights, ref_weights)

    def test_nonpow2_and_large(self):
        with self.subTest("N=96"):
            self._run(7, 96, torch.float32, topk=4)
        with self.subTest("M=4096"):
            self._run(4096, 128, torch.bfloat16, topk=8)
        with self.subTest("M=70000 grid fold"):
            self._run(70000, 17, torch.float32, topk=4)
        with self.subTest("N=65 grouped"):
            self._run(
                4, 65, torch.float32, topk=4, num_expert_group=5, topk_group=2
            )

    def test_empty(self):
        scores = torch.empty(0, 16, device="cuda")
        bias = torch.zeros(16, device="cuda")
        for variant, mod in MODULES:
            with self.subTest(variant=variant):
                w, i = mod.moe_fused_gate(scores, bias, topk=4)
                self.assertEqual(w.shape, (0, 4))
                self.assertEqual(i.shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
