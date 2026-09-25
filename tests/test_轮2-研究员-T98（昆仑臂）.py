# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Round-2 (kunlun arm) test module for Task 98 ``get_mla_kv_buffer``.

This file did not exist when the round-2 developer started (verified
against the working tree and ``git log --all``); the canonical operator
matrix lives in ``tests/test_get_mla_kv_buffer.py`` and is reused here
verbatim. The round-2 regressions for the upstream single-row kunlunxin
vendor (``test_row_form_*``) were added to that canonical module and to
its ``RELEASE_REQUIRED_TESTS`` so the release gate - which loads exactly
``tests/test_<operator>.py`` - exercises them; this module is the
round-2 review entry point that re-exports the full suite from one
place. Run from anywhere:

    python3 "tests/test_轮2-研究员-T98（昆仑臂）.py"
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

# the round-2 regressions (test_row_form_*) are listed in the canonical
# module's RELEASE_REQUIRED_TESTS and must stay there; this alias keeps
# the same required set visible from the round-2 entry point.
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
