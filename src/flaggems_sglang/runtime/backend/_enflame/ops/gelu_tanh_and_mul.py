# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Enflame vendor, e9: the whole field clusters at 3.7-4.1x here while the
# width ladder plateaued (4096->8192 only +7.7%, warps flat). FlagGems
# production gelu on gcu resolves a native libdevice tanh through
# triton_lang_helper's module chain - the one structural form never tried
# on this chip. This vendor adopts that resolution chain verbatim
# (backend extras first, __triton_builtin__ marker required to skip the
# typing-stub tanh, exp identity as the in-kernel fallback), keeping the
# e8 launch geometry (BLOCK 8192, cap 24, num_warps=4) byte-identical.

import importlib

import torch
import triton
import triton.language as tl

_BLOCK_COL = 8192
_MAX_PROGS = 24
_MAX_GRID = 65535


def _resolve_native_tanh():
    # Native tanh only from the ACTIVE triton backend's extras (a static
    # scan cross-picks foreign symbols - extra.amd's tanh is a real
    # triton builtin that lowers to __ocml_tanh_f32 and dies in NVIDIA
    # ptxas), builtin-marker required to skip typing stubs, and accepted
    # only after a one-element compile probe matches the exp identity.
    try:
        from triton.runtime.driver import driver as _driver

        backend = str(_driver.active.get_current_target().backend)
    except Exception:
        return None
    aliases = {
        "cuda": ("cuda",),
        "hip": ("hip", "amd"),
        "amd": ("amd", "hip"),
        "gcu": ("gcu", "enflame"),
        "enflame": ("gcu", "enflame"),
        "npu": ("npu", "ascend"),
        "ascend": ("npu", "ascend"),
        "xpu": ("xpu", "kunlunxin"),
        "kunlunxin": ("xpu", "kunlunxin"),
        "metax": ("metax", "maca"),
        "maca": ("metax", "maca"),
        "tianshu": ("tianshu", "iluvatar"),
        "iluvatar": ("tianshu", "iluvatar"),
        "hygon": ("hygon", "dcu"),
        "haiguang": ("hygon", "dcu"),
    }
    names = [
        f"triton.language.extra.{name}.libdevice"
        for name in aliases.get(backend, (backend,))
    ]
    names.append("triton.language.extra.libdevice")
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        fn = getattr(module, "tanh", None)
        if fn is None or not getattr(fn, "__triton_builtin__", False):
            continue
        if _probe_native_tanh(fn):
            return fn
    return None


@triton.jit
def _tanh_exp_identity(x):
    # Saturates safely: x -> +inf gives tanh -> 1, -inf -> -1.
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


_tanh_probe_target = None


@triton.jit
def _tanh_probe_kernel(x_ptr, y_ptr):
    v = tl.load(x_ptr)
    tl.store(y_ptr, _tanh_probe_target(v))


def _probe_native_tanh(fn):
    global _tanh_probe_target
    try:
        if not torch.cuda.is_available():
            return False
        _tanh_probe_target = fn
        x = torch.tensor([0.7], device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        _tanh_probe_kernel[(1,)](x, y)
        torch.cuda.synchronize()
        ref = 2.0 / (1.0 + torch.exp(-2.0 * x)) - 1.0
        return bool(torch.allclose(y, ref, atol=1e-5, rtol=1e-5))
    except Exception:
        return False


_tanh_impl = _resolve_native_tanh() or _tanh_exp_identity


@triton.jit
def _gelu_tanh_and_mul_kernel(
    x_ptr,
    output_ptr,
    rows,
    half_width,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_col_blocks = tl.cdiv(half_width, BLOCK_COL)
    total_blocks = rows * num_col_blocks
    grid_size = tl.num_programs(0)
    for block_id in range(pid, total_blocks, grid_size):
        row_id = block_id // num_col_blocks
        col_block = block_id - row_id * num_col_blocks
        col_offsets = col_block * BLOCK_COL + tl.arange(0, BLOCK_COL)
        col_mask = col_offsets < half_width
        row_base = row_id.to(tl.int64) * (2 * half_width)
        gate = tl.load(
            x_ptr + row_base + col_offsets,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            x_ptr + row_base + half_width + col_offsets,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        # E9: native tanh when the local Triton wheel exposes a real
        # extern (FlagGems production shim), exp identity otherwise.
        inner = 0.7978845608028654 * gate * (1.0 + 0.044715 * gate * gate)
        tanh_inner = _tanh_impl(inner)
        gelu = 0.5 * gate * (1.0 + tanh_inner)
        tl.store(
            output_ptr + row_id.to(tl.int64) * half_width + col_offsets,
            (gelu * up).to(output_ptr.dtype.element_ty),
            mask=col_mask,
        )


# E8 probe: num_warps=4 launch pin (single variable; enflame 1.74 vs
# second-tier 3.67 on this task - the PR corpus's cheapest per-chip knob).
def gelu_tanh_and_mul(input):
    x = input.contiguous()
    last_dim = x.shape[-1]
    assert last_dim % 2 == 0
    half_width = last_dim // 2
    output = torch.empty(
        x.shape[:-1] + (half_width,), dtype=x.dtype, device=x.device
    )
    rows = output.numel() // half_width if half_width else 0
    if rows and half_width:
        _gelu_tanh_and_mul_kernel[
            (min(rows * triton.cdiv(half_width, _BLOCK_COL), _MAX_PROGS),)
        ](
            x,
            output,
            rows,
            half_width,
            BLOCK_COL=_BLOCK_COL,
            num_warps=4,
        )
    return output


__all__ = ["gelu_tanh_and_mul"]
