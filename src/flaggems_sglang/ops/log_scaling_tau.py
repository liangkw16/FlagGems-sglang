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

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _triton_driver


@triton.jit
def _log_scaling_tau_kernel(
    x_ptr,
    tau_ptr,
    out_ptr,
    n_cols,
    x_stride_row,
    o_stride_row,
    tau_stride,
    COL_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    row = tl.program_id(0) // COL_BLOCKS
    col_block = tl.program_id(0) % COL_BLOCKS
    x_stride_row = tl.cast(x_stride_row, tl.int64)
    o_stride_row = tl.cast(o_stride_row, tl.int64)

    tau = tl.load(tau_ptr + row * tau_stride).to(tl.float32)
    offs = col_block * BLOCK + tl.arange(0, BLOCK)
    # Full blocks skip the per-lane int compare entirely - the
    # Ascend vector-CMP scalar-degradation fix (T40 E16 pattern);
    # only ragged tails keep the mask.
    if EVEN:
        x = tl.load(x_ptr + row * x_stride_row + offs).to(tl.float32)
        tl.store(
            out_ptr + row * o_stride_row + offs,
            (x * tau).to(out_ptr.dtype.element_ty),
        )
    else:
        mask = offs < n_cols
        x = tl.load(
            x_ptr + row * x_stride_row + offs, mask=mask, other=0.0
        ).to(tl.float32)
        tl.store(
            out_ptr + row * o_stride_row + offs,
            (x * tau).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _log_scaling_tau_fast_kernel(
    x_ptr,
    tau_ptr,
    out_ptr,
    N_COLS: tl.constexpr,
    COL_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Contiguous rows with unit-stride tau: every address term folds
    # into constexpr int32 arithmetic, so the launcher binds only the
    # three pointers. int32 offsets are safe because the wrapper only
    # selects this kernel when numel < 2**31.
    row = tl.program_id(0) // COL_BLOCKS
    col_block = tl.program_id(0) % COL_BLOCKS
    tau = tl.load(tau_ptr + row).to(tl.float32)
    offs = col_block * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + row * N_COLS + offs).to(tl.float32)
        tl.store(
            out_ptr + row * N_COLS + offs,
            (x * tau).to(out_ptr.dtype.element_ty),
        )
    else:
        mask = offs < N_COLS
        x = tl.load(x_ptr + row * N_COLS + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        tl.store(
            out_ptr + row * N_COLS + offs,
            (x * tau).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


_LAUNCH_PLANS = {}


def _launch_plan(n_cols):
    plan = _LAUNCH_PLANS.get(n_cols)
    if plan is None:
        block = min(triton.next_power_of_2(max(n_cols, 1)), 1024)
        col_blocks = triton.cdiv(n_cols, block)
        plan = (block, col_blocks, n_cols % block == 0)
        _LAUNCH_PLANS[n_cols] = plan
    return plan


_TRITON_KNOBS = getattr(triton, "knobs", None)
_FAST_LAUNCHERS = {}


def _hook_chain_empty(hook):
    # triton >= 3.6 represents launch hooks as HookChain objects; the
    # chain is inert while its call list is empty. Anything else that
    # is not None is treated as an active hook.
    calls = getattr(hook, "calls", None)
    return isinstance(calls, list) and not calls


def _direct_launcher(compiled, grid0):
    # Prebound launcher for the already-compiled fast kernel. The first
    # call for a shape goes through the standard Triton JIT dispatch,
    # which compiles the kernel and returns its CompiledKernel handle;
    # this closure then launches that SAME Triton kernel on later calls
    # with only a stream query and the C launcher call, skipping the
    # per-call Python binder (signature hashing and argument binding).
    # This is the same launch shape torch.compile's generated wrappers
    # use. Any Triton build without the documented CompiledKernel API
    # (or with launch hooks registered) keeps the standard dispatch -
    # there is no torch fallback in any path.
    launcher = getattr(compiled, "run", None)
    function = getattr(compiled, "function", None)
    packed = getattr(compiled, "packed_metadata", None)
    active = getattr(_triton_driver, "active", None)
    if (
        not callable(launcher)
        or function is None
        or packed is None
        or not hasattr(active, "get_current_stream")
        or not hasattr(active, "get_current_device")
    ):
        return False
    enter = exit_hook = None
    if _TRITON_KNOBS is not None:
        enter = getattr(_TRITON_KNOBS.runtime, "launch_enter_hook", None)
        exit_hook = getattr(_TRITON_KNOBS.runtime, "launch_exit_hook", None)
        for hook in (enter, exit_hook):
            if hook is not None and not _hook_chain_empty(hook):
                # An active profiling hook keeps the fully instrumented
                # standard dispatch; triton >= 3.6 exposes no-op
                # HookChain objects whose empty call list is safe to
                # pass straight to the C launcher.
                return False

    def _launch(x, tau, out):
        device = active.get_current_device()
        stream = active.get_current_stream(device)
        launcher(
            grid0,
            1,
            1,
            stream,
            function,
            packed,
            None,
            enter,
            exit_hook,
            x,
            tau,
            out,
        )

    return _launch


def log_scaling_tau(x, tau):
    x = x.contiguous()
    out = torch.empty_like(x)
    numel = out.numel()
    if numel == 0:
        return out

    rows = x.shape[0]
    n_cols = numel // rows
    block, col_blocks, even = _launch_plan(n_cols)
    grid = (rows * col_blocks,)
    if tau.stride(0) == 1 and numel < 2**31:
        # Contiguous tau and 32-bit-safe sizes take the constexpr-only
        # fast kernel; any other layout keeps the fully general strided
        # kernel below. Both paths are Triton kernels executing the
        # same fp32 row-scale math - there is no torch fallback.
        key = (x.dtype, rows, n_cols, x.get_device())
        entry = _FAST_LAUNCHERS.get(key)
        aligned = (
            x.data_ptr() % 16 == 0
            and tau.data_ptr() % 16 == 0
            and out.data_ptr() % 16 == 0
        )
        if entry is not None and entry and aligned:
            entry(x, tau, out)
            return out
        ck = _log_scaling_tau_fast_kernel[grid](
            x,
            tau,
            out,
            N_COLS=n_cols,
            COL_BLOCKS=col_blocks,
            BLOCK=block,
            EVEN=even,
            num_warps=4,
            num_stages=1,
        )
        if entry is None:
            # Cache a prebound launcher only for binaries compiled from
            # 16-byte-aligned pointers (the Triton pointer
            # specialization the direct path must preserve); later
            # misaligned calls re-enter the standard dispatch above.
            _FAST_LAUNCHERS[key] = (
                _direct_launcher(ck, grid[0]) if aligned else False
            )
        return out
    _log_scaling_tau_kernel[grid](
        x,
        tau,
        out,
        n_cols,
        x.stride(0),
        out.stride(0),
        tau.stride(0),
        COL_BLOCKS=col_blocks,
        BLOCK=block,
        EVEN=even,
        num_warps=4,
        num_stages=2,
    )
    return out


__all__ = ["log_scaling_tau"]
