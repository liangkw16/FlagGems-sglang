#!/usr/bin/env python3
"""Build an immutable FlagOS submission ZIP from a Git commit."""

import argparse
import hashlib
import io
import json
import os
import re
import secrets
import stat
import struct
import subprocess
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
ZIP_LIMIT = 10 * 1024 * 1024
LOCAL_HEADER = struct.Struct("<4s5H3L2H")
LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"


def git(root: Path, *args: str, text: bool = True):
    result = subprocess.run(
        ["git", "--no-replace-objects", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout.strip() if text else result.stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_archive_path(name: str, is_directory: bool) -> bool:
    if not name or "\\" in name or "\0" in name:
        return False
    parts = name.split("/")
    if is_directory:
        if parts[-1]:
            return False
        parts = parts[:-1]
    return bool(parts) and all(part not in {"", ".", ".."} for part in parts)


def safe_extra_fields(extra: bytes) -> bool:
    offset = 0
    while offset < len(extra):
        if len(extra) - offset < 4:
            return False
        field_id = int.from_bytes(extra[offset : offset + 2], "little")
        field_size = int.from_bytes(extra[offset + 2 : offset + 4], "little")
        offset += 4
        if field_id == 0x7075 or field_size > len(extra) - offset:
            return False
        offset += field_size
    return True


def safe_local_header(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bool:
    position = archive.fp.tell()
    try:
        archive.fp.seek(info.header_offset)
        header = archive.fp.read(LOCAL_HEADER.size)
        if len(header) != LOCAL_HEADER.size:
            return False
        fields = LOCAL_HEADER.unpack(header)
        flag_bits = fields[2]
        compress_type = fields[3]
        name_size, extra_size = fields[-2:]
        raw_name = archive.fp.read(name_size)
        extra = archive.fp.read(extra_size)
        if len(raw_name) != name_size or len(extra) != extra_size:
            return False
        encoding = "utf-8" if flag_bits & 0x800 else "cp437"
        return (
            fields[0] == LOCAL_HEADER_SIGNATURE
            and flag_bits == info.flag_bits
            and compress_type == info.compress_type
            and raw_name.decode(encoding) == info.orig_filename
            and safe_extra_fields(extra)
        )
    except (AttributeError, UnicodeDecodeError):
        return False
    finally:
        archive.fp.seek(position)


def matching_archive_members(archive: zipfile.ZipFile, members: dict):
    infos = archive.infolist()
    matched = []
    seen = set()
    file_paths = set()
    directory_paths = set()
    for info in infos:
        is_directory = info.is_dir()
        file_type = stat.S_IFMT((info.external_attr >> 16) & 0xFFFF)
        if (
            info.orig_filename != info.filename
            or not safe_extra_fields(info.extra)
            or not safe_archive_path(info.filename, is_directory)
            or info.compress_type
            not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or (
                is_directory
                and (info.file_size != 0 or file_type not in {0, stat.S_IFDIR})
            )
            or (
                not is_directory
                and (
                    info.external_attr & 0x10
                    or file_type not in {0, stat.S_IFREG}
                )
            )
        ):
            return None
        if is_directory:
            if not safe_local_header(archive, info):
                return None
            directory_path = info.filename[:-1]
            if directory_path in directory_paths:
                return None
            directory_paths.add(directory_path)
            continue
        basename = info.filename.rsplit("/", 1)[-1]
        if (
            basename not in members
            or basename in seen
            or info.filename in file_paths
            or info.file_size != len(members[basename])
        ):
            return None
        if not safe_local_header(archive, info):
            return None
        seen.add(basename)
        file_paths.add(info.filename)
        matched.append(info.filename)
    ancestors = set()
    for file_path in file_paths:
        parts = file_path.split("/")
        ancestors.update(
            "/".join(parts[:index]) for index in range(1, len(parts))
        )
    if (
        file_paths & ancestors
        or file_paths & directory_paths
        or not directory_paths <= ancestors
    ):
        return None
    if seen != set(members) or archive.testzip() is not None:
        return None
    if any(
        archive.read(info) != members[info.filename.rsplit("/", 1)[-1]]
        for info in infos
        if not info.is_dir()
    ):
        return None
    return matched


def archive_matches(archive: zipfile.ZipFile, members: dict) -> bool:
    return matching_archive_members(archive, members) is not None


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


def open_output_directory(root: Path, output: Path, create: bool):
    try:
        relative = output.relative_to(root)
    except ValueError as error:
        raise SystemExit(
            f"output resolves outside repository: {output}"
        ) from error
    if not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise SystemExit(f"refusing unsafe output path: {output}")

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(root, flags)
    try:
        for part in relative.parts[:-1]:
            if create:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=directory)
                except FileExistsError:
                    pass
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        return directory, relative.name
    except Exception:
        os.close(directory)
        raise


def read_existing(root: Path, output: Path):
    try:
        directory, name = open_output_directory(root, output, create=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SystemExit(f"refusing unsafe output path: {output}") from error
    try:
        try:
            descriptor = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory
            )
        except FileNotFoundError:
            return None
        except OSError as error:
            raise SystemExit(
                f"refusing unsafe output path: {output}"
            ) from error
        with os.fdopen(descriptor, "rb") as artifact:
            metadata = os.fstat(artifact.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise SystemExit(
                    f"submission artifact must be a regular file: {output}"
                )
            if metadata.st_size >= ZIP_LIMIT:
                raise SystemExit(
                    f"ZIP exceeds platform limit: {metadata.st_size} bytes"
                )
            return artifact.read()
    finally:
        os.close(directory)


def publish(root: Path, output: Path, payload: bytes) -> None:
    try:
        directory, name = open_output_directory(root, output, create=True)
    except OSError as error:
        raise SystemExit(f"refusing unsafe output path: {output}") from error
    temporary = None
    try:
        for _ in range(128):
            candidate = f".{name}.{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
                temporary = candidate
                break
            except FileExistsError:
                continue
        else:
            raise SystemExit(f"unable to create temporary artifact: {output}")

        with os.fdopen(descriptor, "wb") as artifact:
            artifact.write(payload)
            artifact.flush()
            os.fsync(artifact.fileno())
            os.fchmod(artifact.fileno(), 0o644)
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise SystemExit(
                f"refusing to overwrite existing artifact: {output}"
            ) from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
        os.close(directory)


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
    ).resolve()
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
    if len(canonical_payload) >= ZIP_LIMIT:
        raise SystemExit(
            f"ZIP exceeds platform limit: {len(canonical_payload)} bytes"
        )

    with zipfile.ZipFile(io.BytesIO(canonical_payload)) as archive:
        canonical_members = matching_archive_members(archive, members)
        if canonical_members != list(members):
            raise SystemExit("ZIP integrity or member validation failed")

    output = (
        root
        / "artifacts"
        / "competition"
        / args.operator
        / f"{args.stage}-{short_commit}"
        / f"{args.operator}.zip"
    )
    archive_member_names = canonical_members
    status = "dry-run"
    existing = read_existing(root, output)
    if existing is not None:
        try:
            with zipfile.ZipFile(io.BytesIO(existing)) as archive:
                existing_members = matching_archive_members(archive, members)
        except zipfile.BadZipFile:
            existing_members = None
        if existing_members is None:
            if args.verify_existing:
                raise SystemExit(
                    "existing artifact does not match committed source or "
                    f"safe submission structure: {output}"
                )
            raise SystemExit(
                f"refusing to overwrite different artifact: {output}"
            )
        if existing != canonical_payload and not args.verify_existing:
            raise SystemExit(
                "existing artifact bytes are not canonical; use "
                "--verify-existing for a read-only legacy audit"
            )
        payload = existing
        archive_member_names = existing_members
        status = (
            "verified-existing"
            if existing == canonical_payload
            else "verified-existing-legacy"
        )
    elif args.verify_existing:
        raise SystemExit(f"artifact does not exist: {output}")
    elif not args.dry_run:
        publish(root, output, payload)
        status = "created"
    if len(payload) >= ZIP_LIMIT:
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
                "archive_members": archive_member_names,
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
