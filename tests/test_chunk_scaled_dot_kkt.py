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
    / "chunk_scaled_dot_kkt.py"
)
SPEC = importlib.util.spec_from_file_location(
    "chunk_scaled_dot_kkt_module", MODULE_PATH
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


def reference(k, beta, g_cumsum=None, chunk_size=64):
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    ratio = H // Hg
    BT = chunk_size
    NT = T // BT

    k_c = k.float().view(B, NT, BT, Hg, K)
    k_c = k_c.repeat_interleave(ratio, dim=3)  # (B, NT, BT, H, K)
    k_c = k_c.permute(0, 1, 3, 2, 4)  # (B, NT, H, BT, K)

    A = torch.einsum("bnhik,bnhjk->bnhij", k_c, k_c)

    if g_cumsum is not None:
        g_c = g_cumsum.float().view(B, NT, BT, H).permute(0, 1, 3, 2)
        g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
        A = A * torch.where(g_diff <= 0, torch.exp(g_diff), torch.zeros_like(g_diff))

    beta_c = beta.float().view(B, NT, BT, H).permute(0, 1, 3, 2)
    A = A * beta_c.unsqueeze(-1)

    causal = torch.tril(
        torch.ones(BT, BT, dtype=torch.bool, device=k.device), diagonal=-1
    )
    A = torch.where(causal, A, torch.zeros_like(A))

    out = A.permute(0, 1, 3, 2, 4).reshape(B, T, H, BT)
    return out


def make_case(
    batch=2,
    nchunks=3,
    chunk_size=64,
    num_k_heads=2,
    ratio=2,
    k_dim=64,
    dtype=torch.float32,
    seed=0,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    seqlen = nchunks * chunk_size
    num_heads = num_k_heads * ratio
    k = (
        torch.randn(batch, seqlen, num_k_heads, k_dim, dtype=dtype, generator=g)
        .to("cuda")
        .to(dtype)
    )
    beta = (
        torch.randn(batch, seqlen, num_heads, dtype=dtype, generator=g)
        .to("cuda")
        .to(dtype)
    )
    return k, beta


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class ChunkScaledDotKktTest(unittest.TestCase):
    def _check(self, k, beta, g_cumsum=None, chunk_size=64):
        k_snapshot = k.clone()
        beta_snapshot = beta.clone()
        g_snapshot = g_cumsum.clone() if g_cumsum is not None else None
        actual = MODULE.chunk_scaled_dot_kkt(
            k, beta, g_cumsum=g_cumsum, chunk_size=chunk_size
        )
        expected = reference(k, beta, g_cumsum=g_cumsum, chunk_size=chunk_size)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, torch.float32)
        atol, rtol = TOLERANCES[k.dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(k, k_snapshot)
        torch.testing.assert_close(beta, beta_snapshot)
        if g_snapshot is not None:
            torch.testing.assert_close(g_cumsum, g_snapshot)
        return actual

    def test_dtypes_without_g(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                k, beta = make_case(dtype=dtype)
                self._check(k, beta)

    def test_gqa_ratios(self):
        for num_k_heads, ratio in ((2, 1), (2, 2), (1, 4), (4, 2)):
            with self.subTest(num_k_heads=num_k_heads, ratio=ratio):
                k, beta = make_case(num_k_heads=num_k_heads, ratio=ratio, seed=1)
                self._check(k, beta)

    def test_with_g_cumsum(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                k, beta = make_case(dtype=dtype, seed=2)
                B, T, H = beta.shape
                # Mostly-decreasing g keeps exponents <= 0; random rows
                # also exercise the exponent > 0 -> exact zero branch.
                g = torch.randn(B, T, H, device="cuda") * -0.5
                g[:, ::2] = torch.randn(B, (T + 1) // 2, H, device="cuda")
                self._check(k, beta, g_cumsum=g)

    def test_chunk_sizes_and_k_dims(self):
        for chunk_size, k_dim, nchunks in (
            (32, 32, 2),
            (32, 100, 1),
            (64, 16, 2),
            (64, 128, 1),
            (128, 64, 1),
        ):
            with self.subTest(chunk_size=chunk_size, k_dim=k_dim, nchunks=nchunks):
                k, beta = make_case(
                    nchunks=nchunks,
                    chunk_size=chunk_size,
                    k_dim=k_dim,
                    seed=3,
                )
                self._check(k, beta, chunk_size=chunk_size)

    def test_strict_lower_triangular_zeros(self):
        k, beta = make_case(seed=4)
        out = self._check(k, beta)
        B, T, H, BT = out.shape
        view = out.view(B, T // BT, BT, H, BT).permute(0, 1, 3, 2, 4)
        upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device="cuda"))
        self.assertTrue(torch.all(view.masked_select(upper) == 0))

    def test_batch_one_and_non_contiguous(self):
        k, beta = make_case(batch=1, seed=5)
        self._check(k, beta)
        big = torch.randn(
            k.shape[0], k.shape[1], k.shape[2], k.shape[3] * 2, device="cuda"
        )
        k_nc = big[..., ::2]
        self.assertFalse(k_nc.is_contiguous())
        self._check(k_nc, beta)

    def test_invalid_shapes_raise(self):
        k, beta = make_case(nchunks=3, chunk_size=64, seed=6)
        with self.assertRaises(ValueError):
            MODULE.chunk_scaled_dot_kkt(k, beta, chunk_size=50)
        k_bad, beta_bad = make_case(num_k_heads=2, ratio=2, seed=6)
        beta_bad = beta_bad[..., :3]  # H=3 not divisible by Hg=2
        with self.assertRaises(ValueError):
            MODULE.chunk_scaled_dot_kkt(k_bad, beta_bad)


if __name__ == "__main__":
    unittest.main()
