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
Multi-level operator registrar.

Priority (higher wins on name collision):

    Level 0: ``flaggems_sglang.ops.*``                       (generic)
    Level 1: ``flaggems_sglang.runtime.backend._<vendor>.ops`` (vendor)
    Level 2: ``flaggems_sglang.runtime.backend._<vendor>.<arch>.ops`` (arch)

A vendor or arch module registers an override simply by defining a
top-level function whose name matches a generic op (e.g. ``gemma_rms_norm``).
The registrar collects generic ops from each ``ops/*.py`` module's
``__all__`` and stacks vendor / arch replacements on top.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Callable, Dict, List, Optional

from . import backend
from .backend.device import DeviceDetector


class OpRegistrar:
    """Collect and route operators from generic / vendor / arch sources."""

    def __init__(self, generic_pkg: str = "flaggems_sglang.ops"):
        self._generic_pkg = generic_pkg
        self._device = DeviceDetector()
        self._resolved: Dict[str, Callable] = {}

    def apply(
        self, target_globals: Optional[Dict] = None
    ) -> Dict[str, Callable]:
        """
        Resolve the operator map and (optionally) inject it into a
        target namespace dict (typically the caller's ``globals()``).
        """
        self._resolved.clear()
        self._collect_generic()
        self._collect_vendor()
        self._collect_arch()
        if target_globals is not None:
            for name, fn in self._resolved.items():
                target_globals[name] = fn
        return dict(self._resolved)

    @property
    def resolved(self) -> Dict[str, Callable]:
        return dict(self._resolved)

    def names(self) -> List[str]:
        return sorted(self._resolved.keys())

    def get(self, name: str) -> Optional[Callable]:
        return self._resolved.get(name)

    def _collect_generic(self) -> None:
        try:
            pkg = importlib.import_module(self._generic_pkg)
        except ModuleNotFoundError:
            return
        for mod_info in pkgutil.iter_modules(pkg.__path__):
            if mod_info.ispkg or mod_info.name.startswith("_"):
                continue
            mod = importlib.import_module(
                f"{self._generic_pkg}.{mod_info.name}"
            )
            self._merge_from_module(mod)

    def _collect_vendor(self) -> None:
        for fn_name, fn in backend.get_current_device_extend_op(
            self._device.vendor_name
        ):
            if inspect.isfunction(fn):
                self._resolved[fn_name] = fn

    def _collect_arch(self) -> None:
        event = backend.BackendArchEvent()
        if not event.has_arch:
            return
        for fn_name, fn in event.get_arch_ops():
            if inspect.isfunction(fn):
                self._resolved[fn_name] = fn

    def _merge_from_module(self, module) -> None:
        exported = getattr(module, "__all__", None)
        if exported is None:
            return
        for name in exported:
            fn = getattr(module, name, None)
            if fn is None or not inspect.isfunction(fn):
                continue
            self._resolved[name] = fn


__all__ = ["OpRegistrar"]
