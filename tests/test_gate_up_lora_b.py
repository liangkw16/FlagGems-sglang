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
from types import SimpleNamespace

import torch

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "gate_up_lora_b.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gate_up_lora_b_module", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(x, gate_up_lora_b, batch_info, output_dim, base_output):
    out = base_output.clone().float()
    r = gate_up_lora_b.shape[-1]

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks
    scalings = batch_info.scalings
    permutation = batch_info.permutation

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        if int(lora_ranks[w_idx].item()) == 0:
            continue
        scaling = float(scalings[w_idx].item())
        if permutation is not None:
            rows = permutation[start:end].long()
        else:
            rows = torch.arange(start, end, device=x.device)

        x_seg = x[rows].float()
        for i in range(2):
            o_start = i * output_dim
            o_end = o_start + output_dim
            x_slice = x_seg[:, i * r : (i + 1) * r]
            w_slice = gate_up_lora_b[w_idx, o_start:o_end, :].float()
            out[rows, o_start:o_end] += scaling * (x_slice @ w_slice.t())

    return out.to(base_output.dtype)


def make_batch_info(device, lengths, adapters, ranks, scalings, perm=None):
    seg_indptr = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0)), device=device
    )
    weight_indices = torch.tensor(adapters, device=device)
    lora_ranks = torch.tensor(ranks, device=device)
    scalings_t = torch.tensor(scalings, device=device)
    permutation = (
        torch.tensor(perm, device=device) if perm is not None else None
    )
    return SimpleNamespace(
        seg_indptr=seg_indptr.to(torch.int64),
        weight_indices=weight_indices.to(torch.int64),
        lora_ranks=lora_ranks.to(torch.int64),
        scalings=scalings_t.to(torch.float32),
        permutation=permutation,
        bs=len(lengths),
        max_len=max(lengths) if lengths else 0,
    )


TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class GateUpLoraBTest(unittest.TestCase):
    def _check(self, x, w, info, od, base):
        actual = MODULE.gate_up_lora_b(x, w, info, od, base)
        expected = reference(x, w, info, od, base)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        atol, rtol = TOLERANCES[base.dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return actual

    def test_dtype_and_shape_matrix(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for rank, od, seg_lens in (
                (16, 64, [70]),
                (16, 65, [64, 1, 63]),
                (32, 127, [130]),
                (32, 128, [3, 125, 65]),
                (64, 129, [1, 127, 129]),
                (16, 256, [300, 12]),
            ):
                with self.subTest(dtype=dtype, rank=rank, od=od):
                    s = sum(seg_lens)
                    num_lora = 4
                    x = torch.randn(s, 2 * rank, device="cuda").to(dtype)
                    w = (
                        torch.randn(num_lora, 2 * od, rank, device="cuda")
                        * 0.05
                    ).to(dtype)
                    base = torch.randn(s, 2 * od, device="cuda").to(dtype)
                    info = make_batch_info(
                        "cuda",
                        seg_lens,
                        [0, 1, 2][: len(seg_lens)],
                        [rank] * num_lora,
                        [1.0, 0.5, 2.0, 0.25][:num_lora],
                        perm=list(range(s)),
                    )
                    self._check(x, w, info, od, base)

    def test_rank_zero_and_empty_segments(self):
        rank, od = 16, 64
        s = 200
        x = torch.randn(s, 2 * rank, device="cuda")
        w = torch.randn(3, 2 * od, rank, device="cuda") * 0.05
        base = torch.randn(s, 2 * od, device="cuda")
        info = make_batch_info(
            "cuda",
            [50, 0, 80, 70],
            [0, 1, 2, 0],
            [0, rank, 0, rank],
            [1.0, 1.0, 1.0, 1.0],
        )
        self._check(x, w, info, od, base)

    def test_no_permutation(self):
        rank, od = 16, 128
        s = 150
        x = torch.randn(s, 2 * rank, device="cuda", dtype=torch.float16)
        w = (torch.randn(2, 2 * od, rank, device="cuda") * 0.1).to(
            torch.float16
        )
        base = torch.randn(s, 2 * od, device="cuda", dtype=torch.float16)
        info = make_batch_info(
            "cuda",
            [100, 50],
            [0, 1],
            [rank, rank],
            [0.7, 1.3],
            perm=None,
        )
        self._check(x, w, info, od, base)

    def test_shuffled_permutation(self):
        rank, od = 16, 96
        s = 120
        perm = torch.randperm(s).tolist()
        x = torch.randn(s, 2 * rank, device="cuda")
        w = torch.randn(2, 2 * od, rank, device="cuda") * 0.1
        base = torch.randn(s, 2 * od, device="cuda")
        info = make_batch_info(
            "cuda",
            [60, 60],
            [0, 1],
            [rank, rank],
            [1.0, 2.0],
            perm=perm,
        )
        self._check(x, w, info, od, base)

    def test_non_contiguous_inputs(self):
        rank, od = 16, 64
        s = 100
        x = torch.randn(s, 4 * rank, device="cuda")[:, ::2]
        base = torch.randn(s, 4 * od, device="cuda")[:, ::2]
        w = torch.randn(2, 2 * od, rank, device="cuda") * 0.1
        self.assertFalse(x.is_contiguous())
        self.assertFalse(base.is_contiguous())
        info = make_batch_info(
            "cuda", [s], [0], [rank], [1.0], perm=list(range(s))
        )
        self._check(x, w, info, od, base)

    def test_inputs_not_modified(self):
        rank, od = 16, 64
        s = 64
        x = torch.randn(s, 2 * rank, device="cuda", dtype=torch.float16)
        w = torch.randn(2, 2 * od, rank, device="cuda", dtype=torch.float16)
        base = torch.randn(s, 2 * od, device="cuda", dtype=torch.float16)
        info = make_batch_info(
            "cuda", [s], [0], [rank], [1.0], perm=list(range(s))
        )
        xs, ws, bs = x.clone(), w.clone(), base.clone()
        self._check(x, w, info, od, base)
        torch.testing.assert_close(x, xs)
        torch.testing.assert_close(w, ws)
        torch.testing.assert_close(base, bs)

    def test_empty_tokens(self):
        rank, od = 16, 64
        x = torch.randn(0, 2 * rank, device="cuda")
        w = torch.randn(2, 2 * od, rank, device="cuda")
        base = torch.randn(0, 2 * od, device="cuda")
        info = make_batch_info("cuda", [0], [0], [rank, rank], [1.0, 1.0])
        out = MODULE.gate_up_lora_b(x, w, info, od, base)
        self.assertEqual(out.shape, (0, 2 * od))

    def test_large_case(self):
        rank, od = 64, 1024
        s = 8192
        x = torch.randn(s, 2 * rank, device="cuda", dtype=torch.float16)
        w = (torch.randn(16, 2 * od, rank, device="cuda") * 0.02).to(
            torch.float16
        )
        base = torch.randn(s, 2 * od, device="cuda", dtype=torch.float16)
        lengths = [512] * 16
        info = make_batch_info(
            "cuda",
            lengths,
            [i % 16 for i in range(16)],
            [rank] * 16,
            [0.5 + 0.1 * i for i in range(16)],
            perm=list(range(s)),
        )
        self._check(x, w, info, od, base)


if __name__ == "__main__":
    unittest.main()
