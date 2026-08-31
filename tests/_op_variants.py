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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Load an operator's generic module plus every backend vendor variant.

Vendor overrides under ``src/flaggems_sglang/runtime/backend/_<vendor>/ops/``
are invisible to unittests that hardcode the generic module path, so a
broken vendor file could only be caught on the competition platform.
``load_operator_modules`` returns the generic module together with all
existing vendor modules so a numeric matrix can iterate every variant
(running vendor kernels on the NVIDIA proxy verifies their math and JIT;
target-chip lowering still needs MCP screening or a vendor container).
"""

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[1]
_OPS_DIR = _REPO_ROOT / "src" / "flaggems_sglang" / "ops"
_BACKEND_DIR = _REPO_ROOT / "src" / "flaggems_sglang" / "runtime" / "backend"


def _load(path, alias):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_operator_modules(operator):
    """Return ``[("generic", mod), ("<vendor>", mod), ...]`` for operator."""
    modules = [
        ("generic", _load(_OPS_DIR / f"{operator}.py", f"{operator}_generic"))
    ]
    for path in sorted(_BACKEND_DIR.glob(f"*/ops/{operator}.py")):
        vendor = path.parents[1].name.lstrip("_")
        modules.append((vendor, _load(path, f"{operator}_{vendor}")))
    return modules
