import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
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
                    file_size=size,
                    compress_type=compression,
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

    def test_publish_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "operator.zip"
            BUILD_SUBMISSION.publish(output, b"first")

            with self.assertRaises(SystemExit):
                BUILD_SUBMISSION.publish(output, b"second")

            self.assertEqual(output.read_bytes(), b"first")
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_publish_failure_never_creates_final_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "operator.zip"
            with mock.patch.object(
                BUILD_SUBMISSION.os,
                "fsync",
                side_effect=OSError("simulated write failure"),
            ):
                with self.assertRaises(OSError):
                    BUILD_SUBMISSION.publish(output, b"payload")

            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

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


if __name__ == "__main__":
    unittest.main()
