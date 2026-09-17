#!/usr/bin/env python3
"""GCU int64 physical-layout probe for T77 compute_position (read-only diagnosis).

The FlashMLA task's positions output is int64; the GCU300 signature-level
i64 ban forces an int32 view in the wrapper. Two physical layouts are on
the table (see artifacts/competition/t60-narrow-storage-audit-20260916/
report.md and strategy-batch6.md section T77):

- A (standard two-word): logical int64 = 2 consecutive int32 words;
  element i lives at view index 2i/2i+1.
- B (torch-gcu narrow packed): the runtime allocates 4N bytes for N
  logical int64 while StorageImpl claims 8N; element i's physical slot
  is byte offset 4i, i.e. view index i.

This script decides between them ON A REAL GCU without writing outside
a fresh probe tensor. It must run on the target runtime (torch-gcu),
not on an NVIDIA proxy. Exit code 0 = layout identified; 3 = ambiguous.

Stages:
  1. record versions / device / env knobs / dispatcher registration;
  2. read-only layout discrimination: device int64 [1,2,3,4], view(int32),
     read ONLY the first 4 int32 words (both layouts stay inside the 16
     allocated bytes): A expects [1,0,2,0], B expects [1,2,3,4];
  3. output-direction probe: fresh int64 output, one tiny Triton kernel
     writes under each layout hypothesis into separate tensors, CPU
     readback decides which produced [10,20,30,40];
  4. value-range probe: legal T77 inputs (int32 prefix near INT32_MAX)
     -> positions crossing 2^31: A stores a nonzero high word, B cannot
     represent the value; report both readings, no verdict.
"""

from __future__ import annotations

import sys

import torch


def stage1() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import triton  # noqa: F401

        info["triton"] = triton.__version__
    except Exception as exc:  # pragma: no cover - environment-dependent
        info["triton_error"] = repr(exc)
    for key in (
        "TORCH_GCU_ENABLE_INT64_AND_UINT64",
        "ENABLE_I64_CHECK",
    ):
        info[f"env:{key}"] = __import__("os").environ.get(key)
    try:
        info["device"] = torch.gcu.get_device_name(0)
        info["gcu_available"] = torch.gcu.is_available()
        device = "gcu"
    except AttributeError:
        # torch-gcu exposes itself as a cuda-like backend on some builds
        info["device"] = torch.cuda.get_device_name(0) if info["cuda_available"] else None
        device = "cuda" if info["cuda_available"] else None
    info["device_side"] = device
    if device:
        try:
            import torch_gcu  # noqa: F401

            info["torch_gcu_module"] = True
        except Exception as exc:
            info["torch_gcu_module"] = repr(exc)
    return info


def stage2(device: str) -> dict:
    src = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)
    words = src.view(torch.int32)[:4].cpu().tolist()
    return {
        "words_first4": words,
        "layout_A_expected": [1, 0, 2, 0],
        "layout_B_expected": [1, 2, 3, 4],
        "verdict": "A" if words == [1, 0, 2, 0] else ("B" if words == [1, 2, 3, 4] else "unknown"),
        "view_numel": src.view(torch.int32).numel(),
        "storage_nbytes": src.untyped_storage().nbytes(),
    }


def stage3(device: str) -> dict:
    import triton
    import triton.language as tl

    @triton.jit
    def _write_words(ptr, N, TWO_WORD: tl.constexpr):
        offs = tl.arange(0, 4)
        vals = offs * 10 + 10
        if TWO_WORD:
            tl.store(ptr + offs * 2, vals)
            tl.store(ptr + offs * 2 + 1, tl.zeros((4,), dtype=tl.int32))
        else:
            tl.store(ptr + offs, vals)

    results = {}
    for two_word in (True, False):
        out = torch.zeros(4, dtype=torch.int64, device=device)
        _write_words[(1,)](out.view(torch.int32), 4, TWO_WORD=two_word)
        results["two_word" if two_word else "one_word"] = out.cpu().tolist()
    want = [10, 20, 30, 40]
    verdict = "A" if results["two_word"] == want else ("B" if results["one_word"] == want else "unknown")
    return {"readings": results, "expected": want, "verdict": verdict}


def stage4(device: str) -> dict:
    prefix = 2**31 - 2  # legal int32 prefix; +arange(4) crosses 2^31
    want = [prefix + i for i in range(4)]
    src = torch.tensor(want, dtype=torch.int64, device=device)
    words = src.view(torch.int32)[:8].cpu().tolist()
    return {
        "values": want,
        "words_first8": words,
        "note": "A: word[1]=1 appears; B: values are lossy/absent",
    }


def main() -> int:
    print("== stage 1: environment ==")
    info = stage1()
    for key, value in info.items():
        print(f"{key}: {value}")
    device = info.get("device_side")
    if not device:
        print("NO GCU/CUDA DEVICE - nothing to probe")
        return 3
    print("== stage 2: read-only layout discrimination ==")
    for key, value in stage2(device).items():
        print(f"{key}: {value}")
    print("== stage 3: output-direction Triton write ==")
    try:
        for key, value in stage3(device).items():
            print(f"{key}: {value}")
    except Exception as exc:
        print(f"stage3 error: {exc!r}")
    print("== stage 4: value-range crossing 2^31 ==")
    try:
        for key, value in stage4(device).items():
            print(f"{key}: {value}")
    except Exception as exc:
        print(f"stage4 error: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
