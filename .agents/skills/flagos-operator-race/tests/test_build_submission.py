import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_submission.py"
SPEC = importlib.util.spec_from_file_location("build_submission", SCRIPT)
BUILD_SUBMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_SUBMISSION)


class BuildSubmissionTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
