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

"""
Generic (device-agnostic) Triton operator implementations.

Each submodule defines its public API via its own ``__all__``. The
top-level ``flaggems_sglang`` package auto-discovers these entries at
import time through ``runtime.op_registrar.OpRegistrar`` and exposes the
resolved implementation (generic, vendor, or arch) on the package
namespace.
"""
