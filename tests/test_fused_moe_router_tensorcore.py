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

from tests._op_variants import load_operator_modules

MODULES = load_operator_modules("fused_moe_router_tensorcore")


def reference(x, router_weight, topk, moe_softcapping, correction_bias=None):
    logits = x.float() @ router_weight.float().t()
    if moe_softcapping != 0:
        logits = torch.tanh(logits / moe_softcapping) * moe_softcapping
    if correction_bias is not None:
        logits = logits + correction_bias.float()

    probs = torch.softmax(logits, dim=-1)
    topk_logits, topk_ids = torch.topk(logits, topk, dim=-1)
    topk_weights = torch.gather(probs, -1, topk_ids)

    return topk_weights, topk_ids.to(torch.int32)


TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FusedMoeRouterTensorcoreTest(unittest.TestCase):
    def _check(self, module, x, w, topk, cap, bias):
        actual_w, actual_ids = module.fused_moe_router_tensorcore(
            x, w, topk, cap, bias
        )
        exp_w, exp_ids = reference(x, w, topk, cap, bias)
        self.assertEqual(actual_w.shape, exp_w.shape)
        self.assertEqual(actual_w.dtype, exp_w.dtype)
        self.assertEqual(actual_ids.dtype, exp_ids.dtype)
        torch.testing.assert_close(actual_ids, exp_ids, atol=0, rtol=0)
        atol, rtol = TOLERANCES[x.dtype]
        torch.testing.assert_close(actual_w, exp_w, atol=atol, rtol=rtol)
        return actual_ids

    def test_dtype_shape_matrix_match_reference(self):
        for name, module in MODULES:
            torch.manual_seed(0)
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for rows, experts, hidden in (
                    (1, 8, 128),
                    (7, 64, 64),
                    (64, 8, 512),
                    (33, 100, 128),
                    (5, 256, 192),
                    (3, 65, 576),
                ):
                    for topk in (1, 2):
                        with self.subTest(
                            module=name,
                            dtype=dtype,
                            rows=rows,
                            experts=experts,
                            hidden=hidden,
                            topk=topk,
                        ):
                            x = torch.randn(rows, hidden, device="cuda").to(
                                dtype
                            )
                            w = (
                                torch.randn(experts, hidden, device="cuda")
                                * 0.05
                            ).to(dtype)
                            bias = torch.randn(
                                experts, device="cuda", dtype=torch.float32
                            )
                            self._check(module, x, w, topk, 30.0, bias)
                            self._check(module, x, w, topk, 0.0, bias)
                            self._check(module, x, w, topk, 30.0, None)
                            self._check(module, x, w, topk, 0.0, None)

    def test_large_hidden_and_rows(self):
        x = torch.randn(1000, 4096, device="cuda", dtype=torch.float16)
        w = (torch.randn(128, 4096, device="cuda") * 0.02).to(torch.float16)
        bias = torch.randn(128, device="cuda", dtype=torch.float32)
        for name, module in MODULES:
            with self.subTest(module=name):
                self._check(module, x, w, 2, 30.0, bias)
                self._check(module, x, w, 1, 0.0, None)

    def test_platform_case7_regression(self):
        torch.manual_seed(1234)
        x = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
        w = torch.randn(256, 4096, device="cuda", dtype=torch.float16) * 0.05
        bias = torch.randn(256, device="cuda", dtype=torch.float32)
        for name, module in MODULES:
            with self.subTest(module=name):
                self._check(module, x, w, 2, 0.0, bias)

    def test_near_tie_order(self):
        hidden = 128
        w = torch.zeros(8, hidden, device="cuda")
        x = torch.zeros(4, hidden, device="cuda")
        x[:, 0] = 1.0
        w[3, 0] = 5.0
        w[6, 0] = 5.0 - 1e-3
        w[1, 0] = 4.0
        w[5, 0] = 4.0 - 1e-3
        for name, module in MODULES:
            with self.subTest(module=name):
                ids = self._check(module, x, w, 2, 0.0, None)
                for row in range(4):
                    self.assertEqual(ids[row, 0].item(), 3)
                    self.assertEqual(ids[row, 1].item(), 6)

    def test_exact_tie_informational(self):
        # torch.topk tie order on CUDA is implementation-defined (observed
        # returning the higher index first); exact-tie inputs are documented
        # as a known divergence risk in the ledger, not asserted here.
        hidden = 128
        w = torch.zeros(8, hidden, device="cuda")
        x = torch.zeros(4, hidden, device="cuda")
        x[:, 0] = 1.0
        w[3, 0] = 5.0
        w[6, 0] = 5.0
        for name, module in MODULES:
            with self.subTest(module=name):
                ids = module.fused_moe_router_tensorcore(x, w, 2, 0.0, None)[1]
                for row in range(4):
                    self.assertEqual(set(ids[row].tolist()), {3, 6})

    def test_single_expert(self):
        x = torch.randn(2, 64, device="cuda")
        w = torch.randn(1, 64, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                self._check(module, x, w, 1, 0.0, None)

    def test_softcap_bias_order_and_global_softmax(self):
        x = torch.zeros(1, 64, device="cuda")
        w = torch.zeros(4, 64, device="cuda")
        x[0, 0] = 1.0
        w[:, 0] = torch.tensor([10.0, 0.0, -1.0, -2.0], device="cuda")
        bias = torch.tensor([-2.0, 0.0, 0.0, 0.0], device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                weights, ids = module.fused_moe_router_tensorcore(
                    x, w, 2, 1.0, bias
                )
                self.assertEqual(ids.tolist(), [[1, 2]])
                self.assertLess(weights.sum().item(), 1.0)

    def test_kunlun_softcap_keeps_padded_experts_masked(self):
        x = torch.zeros(1, 64, device="cuda")
        w = torch.zeros(3, 64, device="cuda")
        bias = torch.tensor([-10.0, -11.0, -12.0], device="cuda")
        for name, module in MODULES:
            if name == "kunlunxin":
                self._check(module, x, w, 2, 1.0, bias)

    def test_non_contiguous_inputs(self):
        x_base = torch.randn(8, 512, device="cuda")
        x = x_base[:, ::2]
        w_base = torch.randn(16, 256, device="cuda")
        w = w_base[::2]
        self.assertFalse(x.is_contiguous())
        self.assertFalse(w.is_contiguous())
        for name, module in MODULES:
            with self.subTest(module=name):
                self._check(module, x, w, 2, 30.0, None)

    def test_inputs_not_modified(self):
        x = torch.randn(16, 256, device="cuda", dtype=torch.float16)
        w = torch.randn(32, 256, device="cuda", dtype=torch.float16)
        bias = torch.randn(32, device="cuda", dtype=torch.float32)
        xs, ws, bs = x.clone(), w.clone(), bias.clone()
        for name, module in MODULES:
            with self.subTest(module=name):
                self._check(module, x, w, 2, 7.0, bias)
                self.assertTrue(torch.equal(x, xs))
                self.assertTrue(torch.equal(w, ws))
                self.assertTrue(torch.equal(bias, bs))

    def test_empty_rows(self):
        x = torch.randn(0, 128, device="cuda")
        w = torch.randn(8, 128, device="cuda")
        for name, module in MODULES:
            with self.subTest(module=name):
                out_w, out_ids = module.fused_moe_router_tensorcore(
                    x, w, 2, 0.0, None
                )
                self.assertEqual(out_w.shape, (0, 2))
                self.assertEqual(out_ids.shape, (0, 2))

    def test_row_grid_fold_path(self):
        rows = 70000
        x = torch.randn(rows, 64, device="cuda", dtype=torch.float16)
        w = (torch.randn(8, 64, device="cuda") * 0.1).to(torch.float16)
        self.assertGreater(rows, 65535)
        for name, module in MODULES:
            with self.subTest(module=name):
                self._check(module, x, w, 2, 0.0, None)


if __name__ == "__main__":
    unittest.main()
