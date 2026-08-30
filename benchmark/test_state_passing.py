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
import statistics
from pathlib import Path

import torch
import triton

MODULE_PATH = (
    Path(__file__).parents[1] / "src" / "flaggems_sglang" / "ops" / "state_passing.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "state_passing_benchmark",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def reference(states, dA_cumsum, initial_states):
    batch, nchunks, nheads, dim = states.shape
    current = initial_states.float().clone()
    out = torch.empty_like(states)
    states_f = states.float()
    dA_last = dA_cumsum[..., -1].float().permute(0, 2, 1)
    for chunk in range(nchunks):
        out[:, chunk] = current.to(states.dtype)
        current = (
            current * torch.exp(dA_last[:, chunk]).unsqueeze(-1) + states_f[:, chunk]
        )
    return out, current


def make_inputs(shape, dtype):
    batch, nchunks, nheads, dim, length = shape
    states = torch.randn(
        (batch, nchunks, nheads, dim),
        device="cuda",
        dtype=dtype,
    )
    dA_cumsum = (
        -torch.rand(
            (batch, nheads, nchunks, length),
            device="cuda",
            dtype=dtype,
        )
        * 0.2
    )
    initial_states = torch.randn(
        (batch, nheads, dim),
        device="cuda",
        dtype=torch.float32,
    )
    return states, dA_cumsum, initial_states


def bench(call):
    return triton.testing.do_bench(call, warmup=3, rep=10)


def run_case(shape, dtype):
    inputs = make_inputs(shape, dtype)
    reference_call = lambda: reference(*inputs)
    kernel_call = lambda: MODULE.state_passing(*inputs)
    reference_call()
    kernel_call()

    samples = []
    for round_index in range(5):
        if round_index % 2 == 0:
            reference_ms = bench(reference_call)
            kernel_ms = bench(kernel_call)
        else:
            kernel_ms = bench(kernel_call)
            reference_ms = bench(reference_call)
        samples.append((reference_ms, kernel_ms, reference_ms / kernel_ms))

    speedups = [sample[2] for sample in samples]
    print(
        f"shape={shape} dtype={dtype} "
        f"reference_ms={statistics.median(x[0] for x in samples):.6f} "
        f"kernel_ms={statistics.median(x[1] for x in samples):.6f} "
        f"speedup={statistics.median(speedups):.6f}x "
        f"samples={','.join(f'{x:.6f}' for x in speedups)}"
    )
    return statistics.median(speedups)


def main():
    cases = (
        ((1, 1, 3, 257, 17), torch.float16),
        ((4, 16, 16, 256, 16), torch.float32),
        ((2, 8, 64, 8192, 256), torch.bfloat16),
        ((2, 64, 8, 4096, 64), torch.float16),
    )
    speedups = [run_case(shape, dtype) for shape, dtype in cases]
    print(
        f"average_speedup={statistics.mean(speedups):.6f}x "
        f"best_speedup={max(speedups):.6f}x "
        f"worst_speedup={min(speedups):.6f}x"
    )


if __name__ == "__main__":
    main()
