import argparse
import copy
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_release.py"
SPEC = importlib.util.spec_from_file_location("verify_release", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class ExecutionReceiptTest(unittest.TestCase):
    def test_real_execution_is_bound_and_incomplete_or_changed_evidence_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            generic = "src/flaggems_sglang/ops/demo_receipt.py"
            vendor = (
                "src/flaggems_sglang/runtime/backend/_demo/ops/demo_receipt.py"
            )
            test = "tests/test_demo_receipt.py"
            sources = {
                generic: "def demo_receipt(x):\n    return x + 1\n",
                vendor: "def demo_receipt(x):\n    return x + 1\n",
                "tests/__init__.py": "",
                VERIFY.RUNNER: SCRIPT.read_text(),
                test: """import importlib.util
from pathlib import Path
import unittest

class DemoTest(unittest.TestCase):
    def test_variants(self):
        root = Path(__file__).parents[1]
        for path in root.glob("src/**/demo_receipt.py"):
            spec = importlib.util.spec_from_file_location("demo", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with self.subTest(variant=str(path)):
                self.assertEqual(module.demo_receipt(2), 3)
""",
            }
            for name, body in sources.items():
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(repo), *args], stderr=subprocess.PIPE
                )

            git("init", "-q")
            git("add", ".")
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "fixture",
            )
            commit = git("rev-parse", "HEAD").decode().strip()
            stage = root / "stage"
            previous = Path.cwd()
            try:
                os.chdir(repo)
                VERIFY.prepare(
                    argparse.Namespace(
                        operator="demo_receipt",
                        source_commit=commit,
                        verification_commit=commit,
                        directory=str(stage),
                        dependency=[],
                        mode="release",
                    )
                )
            finally:
                os.chdir(previous)
            manifest = json.loads(
                (stage / "verification-input.json").read_text()
            )
            log = io.StringIO()
            result = VERIFY.run_suite(stage, manifest, log)
            VERIFY.require_success(result, manifest["sources"])
            self.assertEqual(result["source_calls"], {generic: 1, vendor: 1})
            self.assertEqual(result["tests_run"], 1)
            self.assertEqual(len(result["cases"]), 3)
            (stage / "verification.log").write_text(log.getvalue())
            receipt = {
                **manifest,
                "exit_code": 0,
                "result": result,
                "environment": {
                    k: "fixture-only"
                    for k in ("python", "torch", "triton", "device", "scope")
                },
                "log_file": "verification.log",
                "log_sha256": VERIFY.digest(log.getvalue().encode()),
            }
            spec = {
                "operator": "demo_receipt",
                "source_commit": commit,
                "verification_commit": commit,
                "verification_receipt": str(stage / "verification.json"),
            }

            def bind(value):
                payload = json.dumps(value).encode()
                (stage / "verification.json").write_bytes(payload)
                return {
                    **spec,
                    "verification_receipt_sha256": VERIFY.digest(payload),
                }

            good = bind(receipt)
            VERIFY.verify_receipt(good, repo)
            mutations = [
                ("mode", "screening"),
                ("exit_code", 1),
                ("source_commit", "0" * 40),
                ("files", []),
                ("result", None),
                ("environment", None),
            ]
            for field, value in mutations:
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        VERIFY.verify_receipt(
                            bind({**receipt, field: value}), repo
                        )
            for field in (
                "skipped",
                "expected_failures",
                "failures",
                "errors",
            ):
                bad = copy.deepcopy(receipt)
                bad["result"][field] = 1
                with self.subTest(result_field=field), self.assertRaises(
                    ValueError
                ):
                    VERIFY.verify_receipt(bind(bad), repo)
            for field, value in (
                ("tests_run", 0),
                ("source_calls", {generic: 1, vendor: 0}),
                ("source_calls", []),
            ):
                bad = copy.deepcopy(receipt)
                bad["result"][field] = value
                with self.subTest(result_field=field), self.assertRaises(
                    ValueError
                ):
                    VERIFY.verify_receipt(bind(bad), repo)
            bad = copy.deepcopy(receipt)
            bad["files"][test] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Git blob"):
                VERIFY.verify_receipt(bind(bad), repo)
            good = bind(receipt)
            (stage / "verification.log").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "log SHA"):
                VERIFY.verify_receipt(good, repo)
            (stage / generic).write_text("changed")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                VERIFY.check_files(stage, manifest["files"])


if __name__ == "__main__":
    unittest.main()
