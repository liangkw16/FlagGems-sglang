#!/usr/bin/env python3
"""Build an immutable FlagOS submission ZIP from a Git commit."""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

VENDORS = {
    "amd",
    "ascend",
    "enflame",
    "hygon",
    "iluvatar",
    "kunlunxin",
    "metax",
    "nvidia",
}


def git(root: Path, *args: str, text: bool = True):
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=text
    )
    return result.stdout.strip() if text else result.stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_tree(root: Path, commit: str):
    entries = {}
    payload = git(root, "ls-tree", "-r", "-z", commit, text=False)
    for record in payload.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, _ = metadata.split(b" ", 2)
        path = raw_path.decode("utf-8", errors="surrogateescape")
        entries[path] = (
            mode.decode("ascii"),
            object_type.decode("ascii"),
        )
    return entries


def publish(output: Path, payload: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as artifact:
            artifact.write(payload)
            artifact.flush()
            os.fsync(artifact.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise SystemExit(
                f"refusing to overwrite existing artifact: {output}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator", help="operator basename, without .py")
    parser.add_argument(
        "--stage", required=True, help="candidate stage, for example s0"
    )
    parser.add_argument(
        "--commit", default="HEAD", help="source commit (default: HEAD)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate without writing"
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="audit an existing legacy ZIP without rewriting it",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.operator):
        parser.error("operator must match [a-z][a-z0-9_]*")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.stage):
        parser.error(
            "stage must contain only lowercase letters, digits, dot, _ or -"
        )

    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    commit = git(
        root,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{args.commit}^{{commit}}",
    )
    short_commit = git(root, "rev-parse", "--short=7", commit)
    generic = f"src/flaggems_sglang/ops/{args.operator}.py"
    tree = git_tree(root, commit)
    if generic not in tree:
        raise SystemExit(f"missing generic source at {commit}: {generic}")

    pattern = re.compile(
        r"^src/flaggems_sglang/runtime/backend/_([^/]+)/ops/"
        rf"{re.escape(args.operator)}\.py$"
    )
    sources = {f"{args.operator}.py": generic}
    for source in tree:
        match = pattern.fullmatch(source)
        if not match:
            continue
        vendor = match.group(1)
        if vendor not in VENDORS:
            raise SystemExit(
                f"unsupported competition vendor suffix: {vendor}"
            )
        sources[f"{args.operator}_{vendor}.py"] = source

    for source in sources.values():
        mode, object_type = tree[source]
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise SystemExit(
                "submission source must be a regular Git blob "
                f"(100644 or 100755): {source} is {mode} {object_type}"
            )

    members = {}
    for member, source in sorted(sources.items()):
        data = git(root, "show", f"{commit}:{source}", text=False)
        compile(data.decode("utf-8"), source, "exec")
        members[member] = data

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_STORED
    ) as archive:
        for member, data in members.items():
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    canonical_payload = buffer.getvalue()
    payload = canonical_payload
    if len(canonical_payload) >= 10 * 1024 * 1024:
        raise SystemExit(
            f"ZIP exceeds platform limit: {len(canonical_payload)} bytes"
        )

    with zipfile.ZipFile(io.BytesIO(canonical_payload)) as archive:
        if archive.testzip() is not None or archive.namelist() != list(
            members
        ):
            raise SystemExit("ZIP integrity or member validation failed")
        for member, data in members.items():
            if archive.read(member) != data:
                raise SystemExit(
                    f"ZIP member differs from committed source: {member}"
                )

    output = (
        root
        / "artifacts"
        / "competition"
        / args.operator
        / f"{args.stage}-{short_commit}"
        / f"{args.operator}.zip"
    )
    status = "dry-run"
    if output.exists():
        existing = output.read_bytes()
        try:
            with zipfile.ZipFile(io.BytesIO(existing)) as archive:
                valid = (
                    archive.testzip() is None
                    and archive.namelist() == list(members)
                )
                valid = valid and all(
                    archive.read(member) == data
                    for member, data in members.items()
                )
        except zipfile.BadZipFile:
            valid = False
        if not valid:
            raise SystemExit(
                f"refusing to overwrite different artifact: {output}"
            )
        if existing != canonical_payload and not args.verify_existing:
            raise SystemExit(
                "existing artifact bytes are not canonical; use "
                "--verify-existing for a read-only legacy audit"
            )
        payload = existing
        status = (
            "verified-existing"
            if existing == canonical_payload
            else "verified-existing-legacy"
        )
    elif args.verify_existing:
        raise SystemExit(f"artifact does not exist: {output}")
    elif not args.dry_run:
        publish(output, payload)
        status = "created"
    if len(payload) >= 10 * 1024 * 1024:
        raise SystemExit(f"ZIP exceeds platform limit: {len(payload)} bytes")

    print(
        json.dumps(
            {
                "status": status,
                "operator": args.operator,
                "commit": commit,
                "artifact": str(output),
                "size": len(payload),
                "zip_sha256": sha256(payload),
                "canonical_zip_sha256": sha256(canonical_payload),
                "members": [
                    {
                        "name": member,
                        "source": sources[member],
                        "size": len(data),
                        "sha256": sha256(data),
                    }
                    for member, data in members.items()
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
