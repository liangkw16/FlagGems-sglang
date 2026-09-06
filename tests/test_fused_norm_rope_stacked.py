# Copyright 2026 FlagOS Contributors
# (same license)
import importlib.util
import unittest
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "flaggems_sglang"
    / "ops"
    / "fused_norm_rope_stacked.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fused_norm_rope_stacked_module", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference(
    kv,
    k_norm_weight,
    eps,
    cos_sin_cache,
    positions,
    num_kv_heads,
    head_dim,
    rotary_dim,
):
    T, L, _ = kv.shape
    H, D = num_kv_heads, head_dim
    kv_size = H * D
    half_rd = rotary_dim // 2

    k_all = kv[..., :kv_size].float().view(T, L, H, D)
    v_all = kv[..., kv_size:].view(T, L, H, D)

    w = k_norm_weight.float().view(1, L, 1, D)
    eps_l = eps.float().view(1, L, 1, 1)
    inv_rms = (k_all.pow(2).mean(dim=-1, keepdim=True) + eps_l).rsqrt()
    k_normed = k_all * inv_rms * w

    pos = positions.long()
    cos = cos_sin_cache[pos, :half_rd].float().view(T, 1, 1, half_rd)
    sin = cos_sin_cache[pos, half_rd:rotary_dim].float().view(T, 1, 1, half_rd)

    k1 = k_normed[..., :half_rd]
    k2 = k_normed[..., half_rd:rotary_dim]
    rot1 = k1 * cos - k2 * sin
    rot2 = k2 * cos + k1 * sin

    k_out = k_normed.clone()
    k_out[..., :half_rd] = rot1
    k_out[..., half_rd:rotary_dim] = rot2

    k_out = k_out.permute(1, 0, 2, 3).contiguous().to(kv.dtype)
    v_out = v_all.permute(1, 0, 2, 3).contiguous()
    return k_out, v_out


def make_case(T, L, H, D, rotary_dim, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    max_pos = 4096
    kv = torch.randn(T, L, H * D * 2, generator=g).cuda().to(dtype)
    w = (torch.randn(L, D, generator=g) * 0.1 + 1.0).cuda().to(dtype)
    eps = torch.full((L,), 1e-6).cuda()
    cache = (
        torch.randn(max_pos, rotary_dim, generator=g).cuda().to(torch.float32)
    )
    cache = torch.cat(
        [torch.cos(cache), torch.sin(cache)], dim=-1
    ).contiguous()
    positions = (
        torch.randint(0, max_pos, (T,), generator=g).cuda().to(torch.int64)
    )
    return kv, w, eps, cache, positions, H, D, rotary_dim


TOLS = {
    torch.float32: (1e-4, 1e-4),
    torch.bfloat16: (1.5e-2, 1.5e-2),
    torch.float16: (1e-2, 1e-2),
}


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FusedNormRopeStackedTest(unittest.TestCase):
    def _check(self, *args):
        kv = args[0]
        k_m, v_m = MODULE.fused_norm_rope_stacked(*args)
        k_r, v_r = reference(*args)
        atol, rtol = TOLS[kv.dtype]
        for actual, expected, name in ((k_m, k_r, "k"), (v_m, v_r, "v")):
            self.assertEqual(actual.shape, expected.shape, name)
            self.assertEqual(actual.dtype, expected.dtype, name)
            torch.testing.assert_close(
                actual.float(), expected.float(), atol=atol, rtol=rtol
            )

    def test_basic(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                self._check(*make_case(64, 4, 8, 128, 64, dtype))

    def test_rotary_equals_dim(self):
        self._check(*make_case(32, 3, 8, 128, 128, torch.bfloat16))

    def test_non_pow2_dim(self):
        self._check(*make_case(16, 2, 4, 96, 48, torch.bfloat16))

    def test_single_token_and_large_heads(self):
        self._check(*make_case(1, 5, 16, 64, 32, torch.bfloat16))

    def test_non_multiple_of_four_heads(self):
        # H not a multiple of HEADS_TILE=4: unmasked head rows would read the
        # V region and corrupt the next token's output (platform case 0 was H=2).
        for H, T, L in ((2, 1, 2), (3, 5, 2), (5, 3, 3), (7, 2, 2)):
            with self.subTest(H=H, T=T, L=L):
                self._check(
                    *make_case(T, L, H, 64, 32, torch.bfloat16, seed=H)
                )

    def test_odd_tokens(self):
        self._check(*make_case(257, 2, 8, 128, 64, torch.float16, seed=3))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FusedNormRopeStackedVariantsTest(unittest.TestCase):
    """Core matrix across every backend variant (generic + vendors)."""

    from tests._op_variants import load_operator_modules

    MODULES = load_operator_modules("fused_norm_rope_stacked")

    def test_variants_match_reference(self):
        for T, L, H, D, rd, dtype in (
            (1, 2, 2, 64, 32, torch.bfloat16),
            (64, 4, 8, 128, 64, torch.bfloat16),
            (32, 3, 8, 128, 128, torch.bfloat16),
            (16, 2, 4, 96, 48, torch.bfloat16),
            (33, 5, 5, 64, 32, torch.float16),
        ):
            args = make_case(T, L, H, D, rd, dtype, seed=T)
            k_r, v_r = reference(*args)
            atol, rtol = TOLS[dtype]
            for name, module in self.MODULES:
                with self.subTest(module=name, T=T, H=H):
                    k_m, v_m = module.fused_norm_rope_stacked(*args)
                    torch.testing.assert_close(
                        k_m.float(), k_r.float(), atol=atol, rtol=rtol
                    )
                    torch.testing.assert_close(
                        v_m.float(), v_r.float(), atol=atol, rtol=rtol
                    )


if __name__ == "__main__":
    unittest.main()
