import importlib.util
import io
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_submission.py"
SPEC = importlib.util.spec_from_file_location("build_submission", SCRIPT)
BUILD_SUBMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_SUBMISSION)


def commit_all(root: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Skill Test",
            "-c",
            "user.email=skill-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class BuildSubmissionTest(unittest.TestCase):
    def test_archive_rejects_unsafe_metadata_before_reading(self):
        for size, compression in (
            (9, zipfile.ZIP_STORED),
            (8, zipfile.ZIP_BZIP2),
            (8, zipfile.ZIP_LZMA),
        ):
            with self.subTest(size=size, compression=compression):
                info = mock.Mock(
                    filename="demo.py",
                    orig_filename="demo.py",
                    file_size=size,
                    compress_type=compression,
                    external_attr=stat.S_IFREG << 16,
                    extra=b"",
                )
                info.is_dir.return_value = False
                archive = mock.Mock()
                archive.infolist.return_value = [info]

                self.assertFalse(
                    BUILD_SUBMISSION.archive_matches(
                        archive, {"demo.py": b"expected"}
                    )
                )
                archive.testzip.assert_not_called()
                archive.read.assert_not_called()

    def test_archive_rejects_non_regular_metadata_before_reading(self):
        for mode, dos_attributes in (
            (stat.S_IFLNK | 0o777, 0),
            (stat.S_IFCHR | 0o666, 0),
            (stat.S_IFREG | 0o644, 0x10),
        ):
            with self.subTest(mode=mode, dos_attributes=dos_attributes):
                info = mock.Mock(
                    filename="demo.py",
                    orig_filename="demo.py",
                    file_size=8,
                    compress_type=zipfile.ZIP_STORED,
                    external_attr=(mode << 16) | dos_attributes,
                    extra=b"",
                )
                info.is_dir.return_value = False
                archive = mock.Mock()
                archive.infolist.return_value = [info]

                self.assertFalse(
                    BUILD_SUBMISSION.archive_matches(
                        archive, {"demo.py": b"expected"}
                    )
                )
                archive.testzip.assert_not_called()
                archive.read.assert_not_called()

    def test_archive_rejects_ambiguous_unicode_path_extra(self):
        for raw_name, alternate_name in (
            ("../evil.py", "demo.py"),
            ("demo.py", "../evil.py"),
        ):
            with self.subTest(
                raw_name=raw_name, alternate_name=alternate_name
            ):
                raw = raw_name.encode("ascii")
                alternate = alternate_name.encode("utf-8")
                unicode_path = (
                    struct.pack(
                        "<HHBI",
                        0x7075,
                        1 + 4 + len(alternate),
                        1,
                        zlib.crc32(raw) & 0xFFFFFFFF,
                    )
                    + alternate
                )
                info = zipfile.ZipInfo(raw_name)
                info.extra = unicode_path
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as archive:
                    archive.writestr(info, b"expected")
                with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
                    self.assertFalse(
                        BUILD_SUBMISSION.archive_matches(
                            archive, {"demo.py": b"expected"}
                        )
                    )

    def test_archive_rejects_local_only_unicode_path_extra(self):
        name = b"demo.py"
        payload = b"expected"
        alternate = b"../evil.py"
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        unicode_path = (
            struct.pack(
                "<HHBI",
                0x7075,
                1 + 4 + len(alternate),
                1,
                zlib.crc32(name) & 0xFFFFFFFF,
            )
            + alternate
        )
        local = struct.pack(
            "<4s5H3L2H",
            b"PK\x03\x04",
            20,
            0,
            0,
            0,
            0,
            crc,
            len(payload),
            len(payload),
            len(name),
            len(unicode_path),
        ) + name + unicode_path + payload
        central = struct.pack(
            "<4s6H3L5H2L",
            b"PK\x01\x02",
            0x0314,
            20,
            0,
            0,
            0,
            0,
            crc,
            len(payload),
            len(payload),
            len(name),
            0,
            0,
            0,
            0,
            (stat.S_IFREG | 0o644) << 16,
            0,
        ) + name
        end = struct.pack(
            "<4s4H2LH",
            b"PK\x05\x06",
            0,
            0,
            1,
            1,
            len(central),
            len(local),
            0,
        )
        with zipfile.ZipFile(io.BytesIO(local + central + end)) as archive:
            self.assertFalse(
                BUILD_SUBMISSION.archive_matches(
                    archive, {"demo.py": payload}
                )
            )

    def test_archive_rejects_unsafe_paths_and_duplicate_basenames(self):
        for names in (
            ["../demo.py"],
            ["first/demo.py", "second/demo.py"],
            ["demo.py", "extra.py"],
        ):
            with self.subTest(names=names):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as archive:
                    for name in names:
                        archive.writestr(name, b"expected")
                with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
                    self.assertFalse(
                        BUILD_SUBMISSION.archive_matches(
                            archive, {"demo.py": b"expected"}
                        )
                    )

    def test_archive_rejects_file_and_directory_path_conflicts(self):
        for entries, members in (
            (
                [("demo.py/", b""), ("demo.py", b"generic")],
                {"demo.py": b"generic"},
            ),
            (
                [
                    ("demo.py", b"generic"),
                    ("demo.py/demo_amd.py", b"vendor"),
                ],
                {"demo.py": b"generic", "demo_amd.py": b"vendor"},
            ),
            (
                [
                    ("legacy/demo.py/", b""),
                    ("legacy/demo.py", b"generic"),
                ],
                {"demo.py": b"generic"},
            ),
        ):
            with self.subTest(entries=entries):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as archive:
                    for name, payload in entries:
                        archive.writestr(name, payload)
                with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
                    self.assertFalse(
                        BUILD_SUBMISSION.archive_matches(archive, members)
                    )

    def test_publish_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "operator.zip"
            BUILD_SUBMISSION.publish(root, output, b"first")

            with self.assertRaises(SystemExit):
                BUILD_SUBMISSION.publish(root, output, b"second")

            self.assertEqual(output.read_bytes(), b"first")
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_publish_failure_never_creates_final_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "operator.zip"
            with mock.patch.object(
                BUILD_SUBMISSION.os,
                "fsync",
                side_effect=OSError("simulated write failure"),
            ):
                with self.assertRaises(OSError):
                    BUILD_SUBMISSION.publish(root, output, b"payload")

            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_publish_rejects_parent_replaced_by_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            with tempfile.TemporaryDirectory() as outside_directory:
                root = Path(directory)
                output = root / "artifacts" / "competition" / "demo.zip"
                (root / "artifacts").mkdir()
                (root / "artifacts").rmdir()
                os.symlink(outside_directory, root / "artifacts")

                with self.assertRaises(SystemExit):
                    BUILD_SUBMISSION.publish(root, output, b"payload")

                self.assertFalse(
                    (
                        Path(outside_directory) / "competition" / "demo.zip"
                    ).exists()
                )

    def test_rejects_symlink_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "src" / "flaggems_sglang" / "ops"
            source_dir.mkdir(parents=True)
            (source_dir / "target.py").write_text("VALUE = 1\n")
            os.symlink("target.py", source_dir / "demo.py")
            commit_all(root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "demo",
                    "--stage",
                    "s0",
                    "--dry-run",
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("regular Git blob", result.stderr)

    def test_rejects_symlink_output_paths(self):
        for symlink_part in ("parent", "file"):
            with self.subTest(symlink_part=symlink_part):
                with tempfile.TemporaryDirectory() as directory:
                    with tempfile.TemporaryDirectory() as outside_directory:
                        root = Path(directory)
                        source = (
                            root
                            / "src"
                            / "flaggems_sglang"
                            / "ops"
                            / "demo.py"
                        )
                        source.parent.mkdir(parents=True)
                        source.write_text("VALUE = 1\n")
                        commit = commit_all(root)
                        artifact = (
                            root
                            / "artifacts"
                            / "competition"
                            / "demo"
                            / f"s0-{commit[:7]}"
                            / "demo.zip"
                        )
                        outside = Path(outside_directory)
                        sentinel = outside / "sentinel"
                        sentinel.write_bytes(b"unchanged")
                        if symlink_part == "parent":
                            os.symlink(outside, root / "artifacts")
                        else:
                            artifact.parent.mkdir(parents=True)
                            os.symlink(sentinel, artifact)

                        result = subprocess.run(
                            [
                                sys.executable,
                                str(SCRIPT),
                                "demo",
                                "--stage",
                                "s0",
                            ],
                            cwd=root,
                            capture_output=True,
                            text=True,
                        )

                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("unsafe output path", result.stderr)
                        self.assertEqual(sentinel.read_bytes(), b"unchanged")

    def test_cli_uses_commit_bytes_and_never_rewrites_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generic = root / "src" / "flaggems_sglang" / "ops" / "demo.py"
            vendor = (
                root
                / "src"
                / "flaggems_sglang"
                / "runtime"
                / "backend"
                / "_amd"
                / "ops"
                / "demo.py"
            )
            generic.parent.mkdir(parents=True)
            vendor.parent.mkdir(parents=True)
            generic.write_text('VALUE = "committed"\n')
            vendor.write_text('VENDOR = "committed"\n')
            commit = commit_all(root)
            generic.write_text('VALUE = "dirty"\n')

            command = [
                sys.executable,
                str(SCRIPT),
                "demo",
                "--stage",
                "s0",
                "--commit",
                commit,
            ]
            created = subprocess.run(
                command, cwd=root, check=True, capture_output=True, text=True
            )
            manifest = json.loads(created.stdout)
            artifact = Path(manifest["artifact"])
            canonical = artifact.read_bytes()
            with zipfile.ZipFile(io.BytesIO(canonical)) as archive:
                self.assertEqual(
                    archive.namelist(), ["demo.py", "demo_amd.py"]
                )
                self.assertEqual(
                    archive.read("demo.py"), b'VALUE = "committed"\n'
                )
                self.assertEqual(
                    archive.read("demo_amd.py"),
                    b'VENDOR = "committed"\n',
                )

            repeated = subprocess.run(
                command, cwd=root, check=True, capture_output=True, text=True
            )
            self.assertEqual(
                json.loads(repeated.stdout)["status"], "verified-existing"
            )
            self.assertEqual(artifact.read_bytes(), canonical)

            legacy_buffer = io.BytesIO()
            with zipfile.ZipFile(
                legacy_buffer, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr("demo.py", b'VALUE = "committed"\n')
                archive.writestr("demo_amd.py", b'VENDOR = "committed"\n')
            legacy = legacy_buffer.getvalue()
            self.assertNotEqual(legacy, canonical)
            artifact.write_bytes(legacy)

            verified = subprocess.run(
                [*command, "--verify-existing"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(verified.stdout)["status"],
                "verified-existing-legacy",
            )
            self.assertEqual(artifact.read_bytes(), legacy)

            refused = subprocess.run(
                command, cwd=root, capture_output=True, text=True
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertEqual(artifact.read_bytes(), legacy)

            os.truncate(artifact, BUILD_SUBMISSION.ZIP_LIMIT)
            oversized = subprocess.run(
                [*command, "--verify-existing"],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(oversized.returncode, 0)
            self.assertIn("exceeds platform limit", oversized.stderr)

    def test_cli_ignores_git_replace_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generic = root / "src" / "flaggems_sglang" / "ops" / "demo.py"
            generic.parent.mkdir(parents=True)
            generic.write_text('VALUE = "original"\n')
            original = commit_all(root)

            generic.write_text('VALUE = "replacement"\n')
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Skill Test",
                    "-c",
                    "user.email=skill-test@example.invalid",
                    "commit",
                    "-qm",
                    "replacement",
                ],
                cwd=root,
                check=True,
            )
            replacement = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "replace", original, replacement],
                cwd=root,
                check=True,
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "demo",
                    "--stage",
                    "s0",
                    "--commit",
                    original,
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(result.stdout)
            self.assertEqual(manifest["commit"], original)
            with zipfile.ZipFile(manifest["artifact"]) as archive:
                self.assertEqual(
                    archive.read("demo.py"), b'VALUE = "original"\n'
                )

    def test_verify_existing_accepts_safe_legacy_subdirectories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generic = root / "src" / "flaggems_sglang" / "ops" / "demo.py"
            vendor = (
                root
                / "src"
                / "flaggems_sglang"
                / "runtime"
                / "backend"
                / "_amd"
                / "ops"
                / "demo.py"
            )
            generic.parent.mkdir(parents=True)
            vendor.parent.mkdir(parents=True)
            generic.write_text('VALUE = "generic"\n')
            vendor.write_text('VALUE = "amd"\n')
            commit = commit_all(root)
            artifact = (
                root
                / "artifacts"
                / "competition"
                / "demo"
                / f"s0-{commit[:7]}"
                / "demo.zip"
            )
            artifact.parent.mkdir(parents=True)
            with zipfile.ZipFile(
                artifact, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr("legacy/", b"")
                archive.writestr("legacy/demo_amd.py", b'VALUE = "amd"\n')
                archive.writestr("legacy/demo.py", b'VALUE = "generic"\n')

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "demo",
                    "--stage",
                    "s0",
                    "--commit",
                    commit,
                    "--verify-existing",
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(result.stdout)
            self.assertEqual(manifest["status"], "verified-existing-legacy")
            self.assertEqual(
                manifest["archive_members"],
                ["legacy/demo_amd.py", "legacy/demo.py"],
            )


if __name__ == "__main__":
    unittest.main()
