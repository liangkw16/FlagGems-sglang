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
    / "sgemm_lora_a.py"
)
SPEC = importlib.util.spec_from_file_location(
    "sgemm_lora_a_module", MODULE_PATH
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class BatchInfo:
    def __init__(self, seg_indptr, weight_indices, permutation=None):
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices
        self.permutation = permutation
        self.bs = len(weight_indices)


def reference(x, weights, batch_info, stack_num=1):
    S, K = x.shape
    R = weights.shape[1]
    out = torch.zeros(S, R, dtype=x.dtype, device=x.device)
    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation
    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        if permutation is not None:
            rows = permutation[start:end].long()
        else:
            rows = torch.arange(start, end, device=x.device)
        x_seg = x[rows].float()
        w = weights[w_idx].float()
        val = x_seg @ w.t()
        out[rows] = val.to(x.dtype)
    return out


def make_case(S, K, R, num_lora, num_segs, dtype, seed, with_perm):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(S, K, generator=g).to(dtype=dtype, device="cuda")
    weights = (torch.randn(num_lora, R, K, generator=g) * 0.05).to(
        dtype=dtype, device="cuda"
    )
    # random non-decreasing segment boundaries covering all S rows
    if num_segs > 1 and S > 1:
        cuts = sorted(
            torch.randint(1, S, (num_segs - 1,), generator=g).tolist()
        )
    else:
        cuts = []
    indptr = torch.tensor([0] + cuts + [S], dtype=torch.int64, device="cuda")
    widx = torch.randint(
        0, num_lora, (num_segs,), generator=g, dtype=torch.int64
    ).to("cuda")
    perm = None
    if with_perm:
        perm = torch.randperm(S, generator=g).to(device="cuda")
    return x, weights, BatchInfo(indptr, widx, perm)


class TestSgemmLoraA(unittest.TestCase):
    TOL = {
        torch.float32: dict(rtol=1e-3, atol=1e-3),
        torch.float16: dict(rtol=1e-2, atol=1e-2),
        torch.bfloat16: dict(rtol=1.5e-2, atol=1.5e-2),
    }

    def _check(self, x, weights, info, stack_num=1):
        out = MOD.sgemm_lora_a(x, weights, info, stack_num)
        ref = reference(x, weights, info, stack_num)
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, x.dtype)
        torch.testing.assert_close(
            out.float(), ref.float(), **self.TOL[x.dtype]
        )

    def test_matrix(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for S, K, R, num_lora, segs, perm in [
                (128, 512, 64, 4, 3, False),
                (1024, 2048, 128, 8, 16, False),
                (256, 1024, 192, 3, 7, True),
                (4096, 4096, 256, 16, 32, True),
                (17, 33, 16, 2, 2, False),
                (1, 64, 32, 1, 1, False),
                (512, 128, 8, 4, 5, True),
            ]:
                with self.subTest(dtype=dtype, S=S, K=K, R=R, perm=perm):
                    x, w, info = make_case(
                        S, K, R, num_lora, segs, dtype, S + K, perm
                    )
                    self._check(x, w, info)

    def test_stack_num(self):
        x, w, info = make_case(256, 512, 128, 4, 5, torch.float16, 9, True)
        # stack_num is reflected in weights' first dim already; call parity
        self._check(x, w, info, stack_num=2)

    def test_empty_and_invariance(self):
        x, w, info = make_case(64, 128, 32, 2, 2, torch.float32, 3, False)
        xc, wc = x.clone(), w.clone()
        self._check(x, w, info)
        self.assertTrue(torch.equal(x, xc))
        self.assertTrue(torch.equal(w, wc))
        # zero rows segments still produce zeros elsewhere
        empty = BatchInfo(
            torch.tensor([0, 0, 5, 5], device="cuda"),
            torch.tensor([0, 1], device="cuda"),
        )
        self._check(x[:5], w, empty)


if __name__ == "__main__":
    unittest.main()
