"""Replay one immutable input in separate reference/candidate/trace processes.

Example (from a checkout, GPU work belongs in a bounded remote background job):
  python tools/diagnose_batch4.py fused_dual_residual_rmsnorm make input.pt
  python tools/diagnose_batch4.py fused_dual_residual_rmsnorm reference input.pt --out ref
  python tools/diagnose_batch4.py fused_dual_residual_rmsnorm candidate input.pt --out plain --source PATH
  python tools/diagnose_batch4.py fused_dual_residual_rmsnorm trace input.pt --out trace --source PATH

Trace adds prints to an artifact copy, never to the submitted source. Compare its
final output with the plain run: instrumentation can change compiler lowering.
Device prints contain FP32 bits; stages are diagnostics, not performance evidence.
"""

import argparse
import ast
import faulthandler
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
from triton.runtime.jit import JITFunction

ROOT = Path(__file__).resolve().parents[1]
OPERATORS = (
    "fused_dual_residual_rmsnorm",
    "fused_norm_rope_stacked",
    "hc_head",
    "extend_attention",
    "chain_speculative_sampling",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(x) for x in value)
    return value


def load(path):
    spec = importlib.util.spec_from_file_location("diagnostic_source", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def instrument(source, op, row, col):
    # Explicit variable names bind this probe to the reviewed kernel structure.
    if op == "fused_dual_residual_rmsnorm":
        condition = f"row == {row}"
        axes = dict.fromkeys(("var1", "rms1", "var2", "rms2"))
        axes.update(dict.fromkeys(("y1", "y1_cast", "mid_val", "out"), "offs"))
    elif op == "fused_norm_rope_stacked":
        condition = f"row == {row}"
        axes = dict.fromkeys(("var", "inv_rms"))
        axes.update(
            {"k_normed": "offs_d", "rot1": "r", "rot2": "r", "v": "offs_d"}
        )
    elif op == "extend_attention":
        condition = f"(global_q_idx == {row}) & (q_head == 0)"
        axes = {
            "score": "offs_n",
            "m_val": None,
            "p": "offs_n",
            "l_val": None,
            "acc": "offs_d",
            "result": "offs_d",
        }
    elif op == "chain_speculative_sampling":
        condition = f"b == {row}"
        axes = {
            "norm_sum": None,
            "target_u": None,
            "prefixes": "offs",
            "final_token": None,
        }
    else:
        return source  # hc_head already exposes all three kernel boundaries.
    tree = ast.parse(source)
    count = 0
    kernels = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "triton.jit" for d in node.decorator_list)
    ]
    for node in (child for kernel in kernels for child in ast.walk(kernel)):
        for field, items in ast.iter_fields(node):
            if not isinstance(items, list):
                continue
            rewritten = []
            for stmt in items:
                rewritten.append(stmt)
                target = None
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                elif isinstance(stmt, ast.AugAssign):
                    target = stmt.target
                if not isinstance(target, ast.Name) or target.id not in axes:
                    continue
                name = target.id
                axis = axes[name]
                value = f"tl.cast({name}, tl.float32)"
                guard = f"({condition})"
                if axis:
                    value = f"tl.sum(tl.where({axis} == {col}, {value}, 0.0))"
                    guard += f" & tl.sum(({axis} == {col}).to(tl.int32)).to(tl.int1)"
                code = (
                    f'if {guard}:\n    tl.device_print("TRACE:{name}:fp32bits", '
                    f"({value}).to(tl.int32, bitcast=True))"
                )
                rewritten.extend(ast.parse(code).body)
                count += 1
            setattr(node, field, rewritten)
    if count == 0:
        raise ValueError(
            "source has no recognized trace sites; review its kernel"
        )
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator", choices=OPERATORS)
    parser.add_argument(
        "mode", choices=("make", "reference", "candidate", "trace")
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--import-runtime")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--col", type=int, default=0)
    parser.add_argument("--large", action="store_true")
    opts = parser.parse_args()
    if opts.row < 0 or opts.col < 0:
        parser.error("row and col must be nonnegative")
    sys.path.insert(0, str(ROOT))
    if opts.import_runtime:
        importlib.import_module(opts.import_runtime)
    test_path = ROOT / f"tests/test_{opts.operator}.py"
    test = importlib.import_module("tests.test_" + opts.operator)
    if opts.mode == "make":
        if opts.device != "cuda":
            parser.error(
                "existing fixtures generate on CUDA; replay the saved CPU input on the target"
            )
        torch.manual_seed(42)
        if opts.operator == "fused_dual_residual_rmsnorm":
            rows, dim = (4096, 8192) if opts.large else (33, 1024)
            x = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            w1 = torch.randn(dim, device="cuda", dtype=x.dtype)
            args = (x, residual, w1, torch.randn_like(w1), 1e-5)
        elif opts.operator == "fused_norm_rope_stacked":
            # Large is a 67M-element stress input, not an unavailable platform fixture.
            shape = (
                (2048, 8, 16, 128, 64) if opts.large else (33, 2, 3, 96, 48)
            )
            args = test.make_case(*shape, torch.bfloat16, seed=54)
        elif opts.operator == "hc_head":
            args = test.make_case(3, 8, 2048, seed=55)
        elif opts.operator == "extend_attention":
            args = test.make_case([8], [65], H_Q=8, H_KV=2, D=96, seed=50) + (
                8,
            )
        else:
            args = test.make_case(dtype=torch.bfloat16, seed=1)
        with opts.input.open("xb") as file:
            torch.save(cpu(args), file)
        print(json.dumps({"input_sha256": sha(opts.input)}), flush=True)
        return
    if opts.out is None:
        parser.error("--out is required for replay")
    opts.out.mkdir(parents=True, exist_ok=False)
    sync = torch.get_device_module(opts.device).synchronize
    args = tuple(
        x.to(opts.device) if isinstance(x, torch.Tensor) else x
        for x in torch.load(opts.input, weights_only=True, map_location="cpu")
    )
    meta = {
        "operator": opts.operator,
        "mode": opts.mode,
        "device": opts.device,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "input_sha256": sha(opts.input),
        "script_sha256": sha(Path(__file__)),
        "reference_sha256": sha(test_path),
        "row": opts.row,
        "col": opts.col,
    }
    if opts.source:
        meta["source_sha256"] = sha(opts.source)
    (opts.out / "identity.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"phase": "inputs_ready", **meta}), flush=True)
    if opts.mode == "reference":
        calls = 0

        def capture_reference(frame, event, result):
            nonlocal calls
            if (
                event == "return"
                and frame.f_code.co_filename == str(test_path)
                and frame.f_code.co_name in ("reference", "_rmsnorm32")
            ):
                stages = {
                    k: cpu(v)
                    for k, v in frame.f_locals.items()
                    if isinstance(v, torch.Tensor)
                    and k
                    not in frame.f_code.co_varnames[: frame.f_code.co_argcount]
                }
                stages["return"] = cpu(result)
                torch.save(stages, opts.out / f"reference-stages-{calls}.pt")
                calls += 1

        sys.setprofile(capture_reference)
        print(json.dumps({"phase": "reference_start"}), flush=True)
        try:
            result = test.reference(*args)
            sync()
        finally:
            sys.setprofile(None)
    else:
        if opts.source is None:
            parser.error("--source is required for candidate/trace")
        source = opts.source
        if opts.mode == "trace":
            source = opts.out / "instrumented.py"
            source.write_text(
                instrument(
                    opts.source.read_text(), opts.operator, opts.row, opts.col
                )
            )
            meta["instrumented_sha256"] = sha(source)
            (opts.out / "identity.json").write_text(
                json.dumps(meta, indent=2) + "\n"
            )
        module = load(source)
        run = JITFunction.run
        calls = 0

        def execute(self, *kernel_args, **kwargs):
            nonlocal calls
            index = calls
            calls += 1
            print(
                json.dumps(
                    {
                        "phase": "compile_or_launch_start",
                        "kernel": self.__name__,
                        "index": index,
                    }
                ),
                flush=True,
            )
            compiled = run(self, *kernel_args, **kwargs)
            sync()
            print(
                json.dumps(
                    {
                        "phase": "kernel_completed",
                        "kernel": self.__name__,
                        "index": index,
                    }
                ),
                flush=True,
            )
            # Existing HC scratch buffers allow inspection without new kernels.
            outputs = {
                "_hc_sumsq_kernel": 1,
                "_hc_mix_kernel": 2,
                "_hc_fold_kernel": 5,
            }
            if self.__name__ in outputs:
                torch.save(
                    cpu(kernel_args[outputs[self.__name__]]),
                    opts.out / f"stage-{index}.pt",
                )
            return compiled

        JITFunction.run = execute
        try:
            result = getattr(module, opts.operator)(*args)
            sync()
        finally:
            JITFunction.run = run
        if not calls:
            raise RuntimeError("nonempty diagnostic executed no JIT kernel")
    torch.save(cpu(result), opts.out / "output.pt")
    print(
        json.dumps(
            {
                "phase": "completed",
                "output_sha256": sha(opts.out / "output.pt"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
