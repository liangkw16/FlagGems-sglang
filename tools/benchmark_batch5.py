# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Five alternating reference/candidate measurements; NVIDIA proxy only.

Run from a release directory containing this script and the selected test:
python tools/benchmark_batch5.py --operator clamp_position --output benchmark.json
Shapes below are public-production examples or proxy assumptions, not platform cases.
"""

import argparse
import hashlib
import importlib
import json
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.testing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cases(op, module):
    if op == "clamp_position":
        for n in (32, 32768):
            yield str(n), (torch.arange(n, device="cuda", dtype=torch.int64),)
    elif op == "compute_src2dst":
        for n in (256, 131072):
            yield str(n), (torch.randperm(n, device="cuda"), n)
    elif op == "build_trtllm_mha_page_table":
        for bs, columns in ((8, 128), (64, 2048)):
            yield f"{bs}x{columns}", module.make_case(bs=bs, columns=columns)
    elif op == "create_flashinfer_kv_indices":
        for lengths in ((8193,), (1024,) * 32):
            yield f"{len(lengths)}x{lengths[0]}", module.make_case(
                lengths=lengths
            )
    elif op == "concat_mla_k":
        for tokens in (32, 1024):
            yield f"{tokens}x128x192", module.make_case(tokens=tokens)
    elif op == "deepep_permute":
        for tokens in (8, 512):
            yield f"{tokens}x4x4096", module.make_case(
                tokens=tokens, hidden=4096
            )
    else:
        raise ValueError(op)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    module = importlib.import_module(f"tests.test_{args.operator}")
    candidate = getattr(dict(module.MODULES)["generic"], args.operator)
    torch.manual_seed(5)
    results = []
    for label, inputs in cases(args.operator, module):
        expected = module.reference(*inputs)
        actual = candidate(*inputs)
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0, equal_nan=True
        )
        functions = {
            "reference": lambda: module.reference(*inputs),
            "candidate": lambda: candidate(*inputs),
        }
        samples = []
        for round_no in range(5):
            order = (
                ("reference", "candidate")
                if round_no % 2 == 0
                else ("candidate", "reference")
            )
            sample = {"order": list(order)}
            for key in order:
                sample[key + "_ms"] = triton.testing.do_bench(
                    functions[key], warmup=20, rep=50
                )
            sample["speedup"] = sample["reference_ms"] / sample["candidate_ms"]
            assert sample["reference_ms"] > 0 and sample["candidate_ms"] > 0
            samples.append(sample)
        result = {
            "shape_label": label,
            "median_speedup": statistics.median(s["speedup"] for s in samples),
            "candidate_median_ms": statistics.median(
                s["candidate_ms"] for s in samples
            ),
            "samples": samples,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    source = ROOT / "src/flaggems_sglang/ops" / f"{args.operator}.py"
    report = {
        "operator": args.operator,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "method": "wrapper-inclusive triton.testing.do_bench, five alternating AB/BA rounds; proxy shapes",
        "cases": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
