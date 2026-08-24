#!/usr/bin/env python3
"""Cache pinned vendor Triton backend sources for offline cross-chip review.

The FlagOS season-2 operator race supports eight chips, but this repository
only carries FlagGems-sglang Git refs. Vendor compiler/driver sources live in
external repositories at immutable commits. This script mirrors them into
``docs/competition/data/vendor-backends/`` so that cross-chip constraint
review (warp semantics, ``num_warps`` limits, dtype support, tunable knobs)
does not depend on network access or GitHub API rate limits.

Every source is pinned to an immutable commit. The manifest records the
upstream URL and SHA-256 of each cached file so later reviews can prove the
bytes were not edited locally.

Usage::

    python tools/fetch_vendor_backends.py            # fetch/refresh cache
    python tools/fetch_vendor_backends.py --verify    # read-only verification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "competition"
    / "data"
    / "vendor-backends"
)
MANIFEST = OUT_DIR / "manifest.json"

# Immutable pins. Keep in sync with docs/competition/learning-path.md and
# docs/competition/reference-repositories.md.
FLAGTREE = "c1ea8285a06e97afad9dd2644bc71f2efca072f4"
TRITON = "dff2f7d03532e9ca0598c728c60c204ae7555fc9"
ASCEND = "865691e2e9b656bc58008170207b4108d92e8dd1"

FLAGTREE_RAW = "https://raw.githubusercontent.com/flagos-ai/FlagTree"
TRITON_RAW = "https://raw.githubusercontent.com/triton-lang/triton"
ASCEND_RAW = "https://raw.githubusercontent.com/Ascend/triton-ascend"

# (chip label, vendor suffix, local subdir, [(local name, upstream url)])
SOURCES: tuple[tuple[str, str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "昆仑芯 Kunlunxin XPU",
        "_kunlunxin",
        "kunlunxin",
        (
            ("compiler.py", f"{FLAGTREE_RAW}/{FLAGTREE}/third_party/xpu/backend/compiler.py"),
            ("driver.py", f"{FLAGTREE_RAW}/{FLAGTREE}/third_party/xpu/backend/driver.py"),
        ),
    ),
    (
        "燧原 Enflame GCU",
        "_enflame",
        "enflame",
        (
            ("compiler.py", f"{FLAGTREE_RAW}/{FLAGTREE}/third_party/enflame/backend/compiler.py"),
            ("driver.py", f"{FLAGTREE_RAW}/{FLAGTREE}/third_party/enflame/backend/driver.py"),
        ),
    ),
    (
        "天数智芯 Iluvatar CoreX",
        "_iluvatar",
        "iluvatar",
        (
            ("compiler.py", f"{FLAGTREE_RAW}/{FLAGTREE}/third_party/iluvatar/backend/compiler.py"),
            ("driver.py", f"{FLAGTREE_RAW}/{FLAGTREE}/third_party/iluvatar/backend/driver.py"),
        ),
    ),
    (
        "沐曦 MetaX MACA",
        "_metax",
        "metax",
        (
            ("compiler.py", f"{FLAGTREE_RAW}/{FLAGTREE}/third_party/metax/backend/compiler.py"),
            ("driver.py", f"{FLAGTREE_RAW}/{FLAGTREE}/third_party/metax/backend/driver.py"),
        ),
    ),
    (
        "海光 Hygon HCU/DCU",
        "_hygon",
        "hygon",
        (
            ("compiler_hcu.py", f"{FLAGTREE_RAW}/{FLAGTREE}/third_party/hcu/backend/compiler_hcu.py"),
            ("compiler.py", f"{FLAGTREE_RAW}/{FLAGTREE}/third_party/hcu/backend/compiler.py"),
            ("driver.py", f"{FLAGTREE_RAW}/{FLAGTREE}/third_party/hcu/backend/driver.py"),
        ),
    ),
    (
        "华为昇腾 Ascend NPU",
        "_ascend",
        "ascend",
        (
            (
                "vector_operator.md",
                f"{ASCEND_RAW}/{ASCEND}/docs/en/programming_guide/vector_operator.md",
            ),
        ),
    ),
    (
        "国际通用 AMD 路径",
        "_amd",
        "amd",
        (
            ("compiler.py", f"{TRITON_RAW}/{TRITON}/third_party/amd/backend/compiler.py"),
            ("driver.py", f"{TRITON_RAW}/{TRITON}/third_party/amd/backend/driver.py"),
        ),
    ),
    (
        "国际通用 NVIDIA 路径",
        "_nvidia",
        "nvidia",
        (
            ("compiler.py", f"{TRITON_RAW}/{TRITON}/third_party/nvidia/backend/compiler.py"),
            ("driver.py", f"{TRITON_RAW}/{TRITON}/third_party/nvidia/backend/driver.py"),
        ),
    ),
)


def fetch(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "text/plain",
            "User-Agent": "FlagOS-vendor-backend-cache/1.0",
        },
    )
    with urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {url}")
        return response.read()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def do_fetch() -> int:
    entries = []
    failures = []
    for label, suffix, subdir, files in SOURCES:
        target_dir = OUT_DIR / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, url in files:
            try:
                payload = fetch(url)
            except (HTTPError, URLError, RuntimeError) as exc:
                failures.append(f"{subdir}/{name}: {exc}")
                continue
            path = target_dir / name
            path.write_bytes(payload)
            digest = sha256(payload)
            entries.append(
                {
                    "chip": label,
                    "vendor_suffix": suffix,
                    "path": f"{subdir}/{name}",
                    "upstream_url": url,
                    "bytes": len(payload),
                    "sha256": digest,
                }
            )
            print(f"cached {subdir}/{name}  {len(payload)} bytes  {digest}")

    manifest = {
        "generated_at": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
        "pins": {
            "flagos-ai/FlagTree": FLAGTREE,
            "triton-lang/triton": TRITON,
            "Ascend/triton-ascend": ASCEND,
        },
        "note": (
            "Read-only upstream mirror for cross-chip constraint review. "
            "Do not edit cached bytes; re-run this script to refresh."
        ),
        "files": entries,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"\nmanifest: {MANIFEST} ({len(entries)} files)")
    if failures:
        print("\nFAILED:", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 1
    return 0


def do_verify() -> int:
    if not MANIFEST.exists():
        print(f"missing manifest: {MANIFEST}", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text())
    bad = []
    for entry in manifest["files"]:
        path = OUT_DIR / entry["path"]
        if not path.exists():
            bad.append(f"{entry['path']}: missing")
            continue
        digest = sha256(path.read_bytes())
        if digest != entry["sha256"]:
            bad.append(
                f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}"
            )
    total = len(manifest["files"])
    if bad:
        print(f"verify FAILED ({len(bad)}/{total}):", file=sys.stderr)
        for item in bad:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(f"verify OK: {total} cached files match manifest SHA-256")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify cached bytes against manifest SHA-256 without network access",
    )
    args = parser.parse_args()
    return do_verify() if args.verify else do_fetch()


if __name__ == "__main__":
    raise SystemExit(main())
