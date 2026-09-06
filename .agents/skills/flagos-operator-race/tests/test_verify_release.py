import argparse
import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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
            # CPU fixture emulates the JIT frame protocol, not GPU execution.
            body = """runtime = {"__name__": "triton.runtime.jit"}
exec("def run(warmup=False): grid_0 = grid_1 = grid_2 = 1; return object()", runtime)
def demo_receipt(x):
    if x.shape[0]:
        runtime["run"](warmup=False)
    return x.value + 1
"""
            sources = {
                generic: body,
                vendor: body,
                "tests/_op_variants.py": (
                    SCRIPT.parents[4] / "tests/_op_variants.py"
                ).read_text(),
                "tests/__init__.py": "",
                VERIFY.RUNNER: SCRIPT.read_text(),
                test: """import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

class DemoTest(unittest.TestCase):
    def test_variants(self):
        root = Path(__file__).parents[1]
        for path in root.glob("src/**/demo_receipt.py"):
            spec = importlib.util.spec_from_file_location("demo", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with self.subTest(variant=str(path)):
                self.assertEqual(module.demo_receipt(SimpleNamespace(shape=(1,), dtype="float32", value=2)), 3)
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
                        device_vendor="nvidia",
                        proxy_vendor=["demo"],
                    )
                )
            finally:
                os.chdir(previous)
            manifest = json.loads(
                (stage / "verification-input.json").read_text()
            )
            log = io.StringIO()
            virtual = types.ModuleType("virtual_ops")
            virtual.__file__ = "_ops.py"
            previous = Path.cwd()
            try:
                os.chdir(stage)
                with mock.patch.dict(sys.modules, {"virtual_ops": virtual}):
                    result = VERIFY.run_suite(stage, manifest, log)
                    # Real local dependencies must still be rejected.
                    extra = stage / "unbound.py"
                    extra.write_text("# unstaged helper\n")
                    virtual.__file__ = str(extra)
                    with self.assertRaisesRegex(
                        ValueError, "unbound imported dependency"
                    ):
                        VERIFY.run_suite(stage, manifest, io.StringIO())
            finally:
                os.chdir(previous)
            VERIFY.require_success(result, manifest["execution_sources"])
            self.assertEqual(result["source_calls"], {generic: 1, vendor: 1})
            self.assertEqual(
                result["kernel_launches"], {generic: 1, vendor: 1}
            )
            self.assertEqual(result["tests_run"], 1)
            self.assertEqual(len(result["cases"]), 3)
            (stage / "verification.log").write_text(log.getvalue())
            receipt = {
                **manifest,
                "exit_code": 0,
                "result": result,
                "environment": {
                    **{
                        k: "fixture-only"
                        for k in (
                            "python",
                            "torch",
                            "triton",
                            "device",
                            "scope",
                        )
                    },
                    "device_vendor": "nvidia",
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
                ("schema_version", 1),
                ("execution_sources", [generic]),
                ("unexecuted_sources", [vendor]),
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
                ("kernel_launches", {generic: 1, vendor: 0}),
                (
                    "tensor_shapes",
                    [(generic, "float32", [0]), (vendor, "float32", [0])],
                ),
                ("passed_tests", []),
                ("expected_tests", []),
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
            (stage / generic).write_text(sources[generic])
            empty_tests = sources[test].replace("shape=(1,)", "shape=(0,)")
            (stage / test).write_text(empty_tests)
            manifest["files"][test] = VERIFY.digest(empty_tests.encode())
            empty = VERIFY.run_suite(stage, manifest, io.StringIO())
            self.assertGreater(empty["tests_run"], 0)
            self.assertTrue(all(n > 0 for n in empty["source_calls"].values()))
            with self.assertRaisesRegex(ValueError, "kernel launch"):
                VERIFY.require_success(empty, manifest["execution_sources"])
            missing = (
                empty_tests
                + '\nRELEASE_REQUIRED_TESTS = ["DemoTest.test_missing_regression"]\n'
            )
            (stage / test).write_text(missing)
            manifest["files"][test] = VERIFY.digest(missing.encode())
            with self.assertRaisesRegex(ValueError, "contract test missing"):
                VERIFY.run_suite(stage, manifest, io.StringIO())

    def test_device_scope_cannot_omit_matching_vendor(self):
        generic = f"{VERIFY.GENERIC}/demo.py"
        nvidia = f"{VERIFY.BACKENDS}/_nvidia/ops/demo.py"
        ascend = f"{VERIFY.BACKENDS}/_ascend/ops/demo.py"
        sources = sorted([generic, nvidia, ascend])
        scope = VERIFY.execution_scope(sources, "demo", "nvidia", [])
        self.assertEqual(scope["execution_sources"], sorted([generic, nvidia]))
        self.assertEqual(scope["unexecuted_sources"], [ascend])
        proxy = VERIFY.execution_scope(sources, "demo", "nvidia", ["ascend"])
        self.assertEqual(proxy["execution_sources"], sources)
        self.assertEqual(proxy["device_vendor"], "nvidia")
        self.assertEqual(proxy["target_unverified_sources"], [ascend])
        with self.assertRaises(ValueError):
            VERIFY.execution_scope(sources, "demo", "nvidia", ["typo"])
        with self.assertRaises(ValueError):
            VERIFY.execution_scope(sources, "demo", "ascend", [])

    def test_variant_selection_does_not_import_unavailable_vendor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generic = root / "src/flaggems_sglang/ops/demo.py"
            vendor = (
                root
                / "src/flaggems_sglang/runtime/backend/_ascend/ops/demo.py"
            )
            for path, body in (
                (generic, "value = 1"),
                (vendor, "raise RuntimeError('target runtime unavailable')"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)
            spec = importlib.util.spec_from_file_location(
                "scope_variants", SCRIPT.parents[4] / "tests/_op_variants.py"
            )
            variants = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(variants)
            with mock.patch.multiple(
                variants,
                _REPO_ROOT=root,
                _OPS_DIR=generic.parent,
                _BACKEND_DIR=vendor.parents[2],
            ):
                with mock.patch.dict(
                    os.environ,
                    {
                        "FLAGOS_TEST_SOURCES": json.dumps(
                            [generic.relative_to(root).as_posix()]
                        )
                    },
                ):
                    self.assertEqual(
                        [
                            name
                            for name, _ in variants.load_operator_modules(
                                "demo"
                            )
                        ],
                        ["generic"],
                    )
                with mock.patch.dict(
                    os.environ, {}, clear=True
                ), self.assertRaisesRegex(RuntimeError, "target runtime"):
                    variants.load_operator_modules("demo")


if __name__ == "__main__":
    unittest.main()
