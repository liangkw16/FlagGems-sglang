<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

### PR Category
<!-- [ Operator | OP Test | Model Test | Benchmark | CI/CD | User Experience | Other ] -->

### Type of Change
<!-- [ Bug Fix | New Feature | Performance Optimization | Refactor | Documentation Update | Other ] -->

### Description
<!--
Briefly describe the changes and the purpose of the changes. For a new
operator, name the op and the tier it targets (generic / vendor / arch),
plus the target hardware if applicable.
-->

### Issue
<!--
List any related issues that this PR resolves, for example:
- Resolves #123
- Associated with Feature #456
-->

### Accuracy Tests
<!--
Fill in the reference implementation you compare against, the
tolerances used, and paste the test summary (pytest output or table).
Leave the row blank when a shape/dtype is not covered.

Reference implementation: <e.g. sgl_kernel.silu_and_mul / torch>
Tolerances: atol=<...>, rtol=<...>

| Shape          | dtype    | Max abs err | Max rel err | Pass |
|----------------|----------|-------------|-------------|------|
| (1, 512)       | fp16     |             |             |      |
| (4, 1024)      | bf16     |             |             |      |
| (32, 2048)     | bf16     |             |             |      |
-->

### Speed Tests and Profiling
<!--
Paste benchmark numbers for the shapes representative of your target
workload. Include the hardware, driver, and command used so the
numbers are reproducible.

Hardware: <e.g. NVIDIA H100 80GB, CUDA 12.4, Triton 3.x>
Command:  pytest -q benchmark/test_<op>.py --level core

| Shape          | dtype    | Baseline (us) | This PR (us) | Speedup |
|----------------|----------|---------------|--------------|---------|
| (1, 512)       | fp16     |               |              |         |
| (4, 1024)      | bf16     |               |              |         |
| (32, 2048)     | bf16     |               |              |         |

Optional: attach an Nsight / rocprof / ncu profile screenshot or trace
link if the change is performance-motivated.
-->

### Progress

- [ ] `pre-commit run --all-files` passes.
- [ ] Op file has Apache 2.0 header and defines `__all__`.
- [ ] `import flaggems_sglang; flaggems_sglang.all_registered_ops()` lists the op locally.
- [ ] Unit test at `tests/test_<op>.py` covers the target dtypes and shape regimes.
- [ ] Benchmark at `benchmark/test_<op>.py` with shapes added to `benchmark/attri_util.py`.
- [ ] Accuracy Tests table filled in above.
- [ ] Speed Tests table filled in above (perf-motivated changes).
- [ ] Change reviewed (1 reviewer required, 2 recommended).
