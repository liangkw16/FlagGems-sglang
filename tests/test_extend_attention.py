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

TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
}


def reference(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_len_extend,
):
    B = qo_indptr.size(0) - 1
    _, H_Q, D = q_extend.shape
    _, H_KV, _ = k_extend.shape
    group_size = H_Q // H_KV
    scale = 1.0 / D**0.5
    o = torch.empty_like(q_extend, dtype=torch.float32)
    for i in range(B):
        q_start = int(qo_indptr[i].item())
        q_end = int(qo_indptr[i + 1].item())
        kv_start = int(kv_indptr[i].item())
        kv_end = int(kv_indptr[i + 1].item())
        prefix_indices = kv_indices[kv_start:kv_end]
        k_prefix = k_buffer[prefix_indices]
        v_prefix = v_buffer[prefix_indices]
        k_ext = k_extend[q_start:q_end]
        v_ext = v_extend[q_start:q_end]
        q_ext = q_extend[q_start:q_end]
        k_full = torch.cat([k_prefix, k_ext], dim=0).float()
        v_full = torch.cat([v_prefix, v_ext], dim=0).float()
        if group_size != 1:
            k_full = k_full.repeat_interleave(group_size, dim=1)
            v_full = v_full.repeat_interleave(group_size, dim=1)
        prefix_len = k_prefix.size(0)
        extend_len = k_ext.size(0)
        total_len = prefix_len + extend_len
        pos_keys = torch.arange(total_len, device=q_extend.device)
        t = prefix_len + torch.arange(extend_len, device=q_extend.device)
        causal_mask = pos_keys.unsqueeze(0) <= t.unsqueeze(1)
        attn_scores = (
            torch.einsum("qhd,khd->qhk", q_ext.float(), k_full) * scale
        )
        attn_scores = attn_scores.masked_fill(
            ~causal_mask.unsqueeze(1), float("-inf")
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        o[q_start:q_end] = torch.einsum("qhk,khd->qhd", attn_weights, v_full)
    return o


def make_case(
    extend_lens,
    prefix_lens,
    H_Q=8,
    H_KV=2,
    D=128,
    total_buffer=4096,
    dtype=torch.float32,
    seed=0,
):
    g = torch.Generator().manual_seed(seed)
    E = sum(extend_lens)
    q = torch.randn(E, H_Q, D, dtype=dtype, generator=g).cuda().to(dtype)
    k_ext = torch.randn(E, H_KV, D, dtype=dtype, generator=g).cuda().to(dtype)
    v_ext = torch.randn(E, H_KV, D, dtype=dtype, generator=g).cuda().to(dtype)
    kb = (
        torch.randn(total_buffer, H_KV, D, dtype=dtype, generator=g)
        .cuda()
        .to(dtype)
    )
    vb = (
        torch.randn(total_buffer, H_KV, D, dtype=dtype, generator=g)
        .cuda()
        .to(dtype)
    )
    qo_indptr = torch.tensor(
        [0] + list(torch.tensor(extend_lens).cumsum(0).tolist()),
        dtype=torch.int64,
    ).cuda()
    kv_offsets = [0]
    all_indices = []
    for pl in prefix_lens:
        idx = torch.randperm(total_buffer, generator=g)[:pl].sort().values
        all_indices.append(idx)
        kv_offsets.append(kv_offsets[-1] + pl)
    kv_indptr = torch.tensor(kv_offsets, dtype=torch.int64).cuda()
    kv_indices = torch.cat(all_indices).cuda()
    return q, k_ext, v_ext, kb, vb, qo_indptr, kv_indptr, kv_indices


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ExtendAttentionTest(unittest.TestCase):
    MODULES = load_operator_modules("extend_attention")

    def _check(self, q, ke, ve, kb, vb, qoi, kvi, kvidx):
        mle = int((qoi[1:] - qoi[:-1]).max().item()) if qoi.numel() > 1 else 0
        expected = reference(q, ke, ve, kb, vb, qoi, kvi, kvidx, mle)
        atol, rtol = TOL[q.dtype]
        for name, module in self.MODULES:
            with self.subTest(module=name):
                actual = module.extend_attention(
                    q, ke, ve, kb, vb, qoi, kvi, kvidx, mle
                )
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                torch.testing.assert_close(
                    actual, expected, atol=atol, rtol=rtol
                )

    def test_basic(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                args = make_case([8, 16, 4], [32, 64, 16], dtype=dtype)
                self._check(*args)

    def test_gqa(self):
        for hq, hkv in ((8, 2), (16, 4), (4, 4), (8, 8)):
            with self.subTest(hq=hq, hkv=hkv):
                args = make_case([8, 8], [16, 32], H_Q=hq, H_KV=hkv, seed=1)
                self._check(*args)

    def test_edge_cases(self):
        for ext, pre in (
            ([1], [0]),
            ([1], [100]),
            ([32], [0]),
            ([0, 8, 0], [16, 0, 32]),
        ):
            with self.subTest(ext=ext, pre=pre):
                args = make_case(ext, pre, seed=2)
                self._check(*args)

    def test_shapes(self):
        for ext, pre, D in (
            ([4, 8], [16, 32], 64),
            ([16], [128], 128),
            ([32, 16, 8], [64, 32, 16], 256),
        ):
            with self.subTest(ext=ext, pre=pre, D=D):
                args = make_case(ext, pre, D=D, seed=3)
                self._check(*args)


if __name__ == "__main__":
    unittest.main()
