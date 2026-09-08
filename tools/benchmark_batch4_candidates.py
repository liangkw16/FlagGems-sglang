"""Paired, wrapper-inclusive NVIDIA screening; never a target-vendor score.

Usage: python tools/benchmark_batch4_candidates.py BASELINE_TREE CANDIDATE_TREE OUT
Trees must contain src/ and tests/. The output binds every source and this script.
"""

import hashlib
import importlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import triton
from triton.runtime.jit import JITFunction


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixtures(op):
    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    if op == "causal_conv1d_update":
        for seq in (1, 4, 16):
            for width in (2, 4):
                yield f"seq{seq}-w{width}", (
                    rand(32, 2048, seq),
                    rand(32, 2048, 4),
                    rand(2048, width),
                    rand(2048),
                    "silu",
                )
    elif op == "fla_layernorm_gated":
        for rows, dim in (
            (32, 128),
            (256, 1024),
            (128, 2048),
            (64, 1536),
            (32, 8192),
        ):
            for rms in (False, True):
                yield f"{rows}x{dim}-rms{rms}", (
                    rand(rows, dim),
                    rand(rows, dim),
                    rand(dim),
                    rand(dim),
                    "swish",
                    1e-5,
                    rms,
                )
    elif op == "l2norm":
        for rows, dim in (
            (32, 64),
            (4096, 64),
            (4096, 128),
            (4096, 1024),
            (65536, 64),
            (65536, 128),
        ):
            yield f"{rows}x{dim}", (rand(rows, dim),)
    elif op == "w8a8_block_int8_matmul":

        def ints(*shape, seed):
            g = torch.Generator().manual_seed(seed)
            return torch.randint(
                -128, 128, shape, generator=g, dtype=torch.int8
            ).cuda()

        for m, n, k, bn, bk in (
            (64, 4096, 4096, 128, 128),
            (128, 2048, 7168, 128, 128),
            (16, 1024, 4096, 128, 128),
            (64, 4096, 4096, 128, 64),
        ):
            kt = (k + bk - 1) // bk
            nt = (n + bn - 1) // bn
            yield f"{m}x{n}x{k}-g{bn}x{bk}", (
                ints(m, k, seed=m),
                ints(n, k, seed=n),
                torch.randn(m, kt).cuda().abs() * 0.01,
                torch.randn(nt, kt).cuda().abs() * 0.01,
                [bn, bk],
                torch.bfloat16,
            )
    elif op == "act_and_mul":
        for rows in (1, 1024):
            for stride in (1, 2):
                yield f"rows{rows}-stride{stride}", (
                    rand(rows, 4096 * stride)[:, ::stride],
                    "silu",
                    None,
                )
    elif op == "fused_gdn_gating":
        for rows, heads in (
            (1, 32),
            (32, 128),
            (33, 128),
            (128, 128),
            (256, 128),
            (4096, 128),
        ):
            yield f"{rows}x{heads}", (
                rand(heads),
                rand(rows, heads),
                rand(rows, heads),
                rand(heads),
            )
    elif op == "ernie45_rope_fused":
        for tokens, heads in ((32, 8), (256, 32), (3, 17)):
            yield f"tokens{tokens}-heads{heads}", (
                rand(tokens, heads * 128),
                rand(tokens, 4 * 128),
                rand(1024, 64),
                torch.randint(0, 1024, (3, tokens), device="cuda"),
                [8, 8, 16],
                128,
                64,
            )
    elif op == "chunked_embedding_lora_a":
        test = importlib.import_module("tests.test_" + op)
        for segments in ([8] * 32, [256] * 32):
            yield f"segments32-length{segments[0]}", test.make_case(
                segments,
                [64, 128],
                max_rank=128,
                dtype=torch.bfloat16,
                seed=46,
            )[:3] + (1024,)
    elif op == "chunk_scaled_dot_kkt":
        test = importlib.import_module("tests.test_" + op)
        for chunks in (1, 16):
            k, beta = test.make_case(
                nchunks=chunks, dtype=torch.bfloat16, seed=45
            )
            yield f"chunks{chunks}", (k, beta, None, 64)


def main():
    baseline, candidate, out = map(Path, sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(candidate.resolve()))
    torch.manual_seed(20260908)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "script_sha256": digest(Path(__file__)),
        "scope": "NVIDIA wrapper screening; vendor performance is unverified",
        "timing": (
            "6 alternating AB/BA pairs, 30 complete wrapper calls per measurement; "
            "wall clock with CUDA synchronization"
        ),
        "cases": [],
        "compiled": [],
    }
    vendors = {
        "causal_conv1d_update": None,
        "fla_layernorm_gated": "ascend",
        "l2norm": None,
        "act_and_mul": None,
        "fused_gdn_gating": None,
        "ernie45_rope_fused": "kunlunxin",
        "chunked_embedding_lora_a": "ascend",
        "chunk_scaled_dot_kkt": "kunlunxin",
        "log_scaling_tau": None,
        "w8a8_block_int8_matmul": None,
    }
    run = JITFunction.run
    label = ""
    seen = set()

    def capture(self, *args, **kwargs):
        compiled = run(self, *args, **kwargs)
        key = (label, compiled.hash)
        if key not in seen:
            seen.add(key)
            stem = f"{label}-{len(seen)}"
            files = {}
            for kind in ("ptx", "ttgir", "llir"):
                if kind in compiled.asm:
                    path = out / f"{stem}.{kind}"
                    path.write_text(compiled.asm[kind])
                    files[kind] = {"path": path.name, "sha256": digest(path)}
            report["compiled"].append(
                {
                    "label": label,
                    "kernel": self.__name__,
                    "hash": compiled.hash,
                    "metadata": compiled.metadata._asdict(),
                    "files": files,
                }
            )
        return compiled

    if len(sys.argv) > 4:
        picked = {}
        for item in sys.argv[4].split(","):
            op, _, vendor = item.partition("@")
            if vendor:
                vendors[op] = vendor
            picked[op] = vendors[op]
        vendors = picked
    for op, vendor in vendors.items():
        rel = Path("src/flaggems_sglang")
        if vendor:
            rel /= f"runtime/backend/_{vendor}"
        rel /= f"ops/{op}.py"
        modules = [
            load(root / rel, f"bench_{op}_{i}")
            for i, root in enumerate((baseline, candidate))
        ]
        funcs = [getattr(m, op) for m in modules]
        if op == "log_scaling_tau":
            # Existing generated code determines whether the 16-byte axis exists.
            cases = [
                (
                    str(dt),
                    (
                        torch.randn(32, 1024, device="cuda", dtype=dt),
                        torch.randn(32, device="cuda"),
                    ),
                )
                for dt in (torch.float32, torch.bfloat16)
            ]
        else:
            cases = fixtures(op)
        for case, args in cases:
            outputs = []
            for i, fn in enumerate(funcs):
                label = f"{op}-{i}-{case}"
                JITFunction.run = capture
                try:
                    outputs.append(fn(*args))
                    torch.cuda.synchronize()
                finally:
                    JITFunction.run = run

            def tensors(value):
                return value if isinstance(value, tuple) else (value,)

            for a, b in zip(tensors(outputs[0]), tensors(outputs[1])):
                torch.testing.assert_close(a, b, atol=0.02, rtol=0.02)
            if op == "log_scaling_tau":
                continue
            for fn in funcs:
                for _ in range(10):
                    fn(*args)
            samples = [[], []]
            for pair in range(6):
                for i in ((0, 1) if pair % 2 == 0 else (1, 0)):
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    for _ in range(30):
                        result = funcs[i](*args)
                        del result
                    torch.cuda.synchronize()
                    samples[i].append(
                        (time.perf_counter_ns() - start) / 30 / 1000
                    )
            row = {
                "operator": op,
                "case": case,
                "vendor_source": vendor or "generic",
                "source_sha256": [
                    digest(root / rel) for root in (baseline, candidate)
                ],
                "baseline_us": samples[0],
                "candidate_us": samples[1],
                "median_paired_speedup": statistics.median(
                    a / b for a, b in zip(*samples)
                ),
            }
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
            (out / "results.json").write_text(
                json.dumps(report, indent=2, default=str) + "\n"
            )
    (out / "results.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
