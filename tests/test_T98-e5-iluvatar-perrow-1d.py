# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Round-2 (e5 iluvatar per-row arm) test module for Task 98.

This file did not exist when the round-1 developer started (verified
against the working tree and ``git log --all``); the canonical operator
matrix lives in ``tests/test_get_mla_kv_buffer.py`` and is reused here
verbatim. The e5 regressions for the iluvatar single-row vendor live in
that canonical module - ``test_row_form_*`` (bound to this vendor's
``_upstream_row_form`` since the kunlunxin arm rolled back to the
aef2c1f1 floor) and ``test_iluvatar_int32_domain_guard`` (the
``_use_int32`` truth table) - and are listed in its
``RELEASE_REQUIRED_TESTS`` so the release gate, which loads exactly
``tests/test_<operator>.py``, exercises them; this module is the
e5 review entry point that re-exports the full suite from one place,
mirroring the e4 entry ``tests/test_轮2-研究员-T98（昆仑臂）.py``.
Run from anywhere:

    python3 tests/test_T98-e5-iluvatar-perrow-1d.py
"""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.test_get_mla_kv_buffer import (  # noqa: E402
    GetMlaKvBufferTest,
    MODULES,
    bits,
    reference,
)
from tests.test_get_mla_kv_buffer import RELEASE_REQUIRED_TESTS as _CANONICAL

# the e5 regressions (test_row_form_* for this vendor plus
# test_iluvatar_int32_domain_guard) are listed in the canonical
# module's RELEASE_REQUIRED_TESTS and must stay there; this alias keeps
# the same required set visible from the e5 entry point.
RELEASE_REQUIRED_TESTS = list(_CANONICAL)

__all__ = [
    "GetMlaKvBufferTest",
    "MODULES",
    "RELEASE_REQUIRED_TESTS",
    "bits",
    "reference",
]

if __name__ == "__main__":
    unittest.main()
