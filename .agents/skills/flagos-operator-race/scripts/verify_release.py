#!/usr/bin/env python3
"""Stage committed operator tests and record their actual execution (stdlib)."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path

RUNNER = ".agents/skills/flagos-operator-race/scripts/verify_release.py"
GENERIC = "src/flaggems_sglang/ops"
BACKENDS = "src/flaggems_sglang/runtime/backend"


def execution_scope(sources, operator, device_vendor, proxy_vendors):
    """Keep every candidate byte bound, but execute only applicable paths."""
    if device_vendor not in {"nvidia", "amd"}:
        raise ValueError(
            "this runner supports CUDA/HIP; use target evidence elsewhere"
        )
    vendors = {
        Path(p).parts[-3].lstrip("_")
        for p in sources
        if p.startswith(BACKENDS + "/")
    }
    if not isinstance(proxy_vendors, list) or any(
        v not in vendors for v in proxy_vendors
    ):
        raise ValueError("unknown proxy vendor")
    selected = [
        p
        for p in sources
        if p == f"{GENERIC}/{operator}.py"
        or Path(p).parts[-3].lstrip("_") in {device_vendor, *proxy_vendors}
    ]
    return {
        "device_vendor": device_vendor,
        "proxy_vendors": sorted(set(proxy_vendors)),
        "execution_sources": selected,
        "unexecuted_sources": [p for p in sources if p not in selected],
        "target_unverified_sources": [
            p
            for p in sources
            if p.startswith(BACKENDS + "/")
            and Path(p).parts[-3].lstrip("_") != device_vendor
        ],
    }


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.PIPE, timeout=30
    )


def checked_path(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe evidence path: {name}")
    target = root / path
    if target.is_symlink() or not target.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError(f"unsafe evidence path: {name}")
    return target


def source_paths(paths, operator):
    pattern = rf"{BACKENDS}/_[^/]+/ops/{re.escape(operator)}\.py"
    return sorted(
        p
        for p in paths
        if p == f"{GENERIC}/{operator}.py" or re.fullmatch(pattern, p)
    )


def prepare(args):
    root = Path.cwd().resolve()
    source = (
        git(root, "rev-parse", f"{args.source_commit}^{{commit}}")
        .decode()
        .strip()
    )
    verification = (
        git(root, "rev-parse", f"{args.verification_commit}^{{commit}}")
        .decode()
        .strip()
    )
    tree = (
        git(root, "ls-tree", "-r", "--name-only", source).decode().splitlines()
    )
    sources = source_paths(tree, args.operator)
    if args.mode == "screening":
        sources = source_paths(
            [
                p.relative_to(root).as_posix()
                for p in (root / "src").rglob(f"{args.operator}.py")
            ],
            args.operator,
        )
    if f"{GENERIC}/{args.operator}.py" not in sources:
        raise ValueError("generic source missing")
    dependencies = {
        f"tests/test_{args.operator}.py",
        "tests/__init__.py",
        RUNNER,
        *args.dependency,
    }
    test_tree = (
        git(root, "ls-tree", "-r", "--name-only", verification)
        .decode()
        .splitlines()
    )
    if "tests/_op_variants.py" in test_tree or args.mode == "screening":
        dependencies.add("tests/_op_variants.py")
    contents = {}
    for name in sorted(set(sources) | dependencies):
        path = checked_path(root, name)
        commit = source if name in sources else verification
        contents[name] = (
            path.read_bytes()
            if args.mode == "screening"
            else git(root, "show", f"{commit}:{name}")
        )
    manifest = {
        "schema_version": 2,
        "mode": args.mode,
        "operator": args.operator,
        "source_commit": source,
        "verification_commit": verification,
        "sources": sources,
        "files": {p: digest(b) for p, b in contents.items()},
        **execution_scope(
            sources, args.operator, args.device_vendor, args.proxy_vendor
        ),
    }
    stage = Path(args.directory).resolve()
    stage.mkdir(mode=0o700)  # Never mix this run with an existing directory.
    for name, data in contents.items():
        path = checked_path(stage, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (stage / "verification-input.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(stage)


def check_files(root, files):
    for name, expected in files.items():
        if digest(checked_path(root, name).read_bytes()) != expected:
            raise ValueError(f"execution bytes changed: {name}")


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cases = []
        self.passed = []

    def addSuccess(self, test):
        self.passed.append(test.id())
        super().addSuccess(test)

    def startTest(self, test):
        self.cases.append(test.id())
        super().startTest(test)

    def addSubTest(self, test, subtest, error):
        self.cases.append(subtest.id())
        super().addSubTest(test, subtest, error)


def run_suite(root, manifest, stream):
    """Track public wrapper calls, including dynamic importlib vendor modules."""
    root = root.resolve()
    calls = {p: 0 for p in manifest["execution_sources"]}
    launches = {p: 0 for p in calls}
    by_filename = {str((root / p).resolve()): p for p in calls}
    shapes = set()

    def profile(frame, event, arg):
        # A completed, non-warmup JIT run under the wrapper is stronger than
        # entering a wrapper which can return immediately for empty inputs.
        if (
            event == "return"
            and arg is not None
            and frame.f_code.co_name == "run"
            and frame.f_globals.get("__name__") == "triton.runtime.jit"
            and frame.f_locals.get("warmup") is False
            and all(
                frame.f_locals.get(f"grid_{axis}", 0) > 0 for axis in range(3)
            )
        ):
            caller = frame.f_back
            while caller is not None:
                path = by_filename.get(caller.f_code.co_filename)
                if path and caller.f_code.co_name == manifest["operator"]:
                    launches[path] += 1
                    break
                caller = caller.f_back
        if event != "call" or frame.f_code.co_name != manifest["operator"]:
            return
        path = by_filename.get(frame.f_code.co_filename)
        if path is None:
            return
        calls[path] += 1
        for value in frame.f_locals.values():
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                shapes.add((path, str(value.dtype), tuple(value.shape)))

    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    check_files(root, manifest["files"])
    test_path = root / "tests" / f"test_{manifest['operator']}.py"
    spec = importlib.util.spec_from_file_location(
        f"tests.test_{manifest['operator']}", test_path
    )
    module = importlib.util.module_from_spec(spec)
    previous_sources = os.environ.get("FLAGOS_TEST_SOURCES")
    os.environ["FLAGOS_TEST_SOURCES"] = json.dumps(
        manifest["execution_sources"]
    )
    previous = sys.getprofile()
    try:
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)

        def test_ids(tests):
            for test in tests:
                if isinstance(test, unittest.TestSuite):
                    yield from test_ids(test)
                else:
                    yield test.id()

        expected = sorted(test_ids(suite))
        required = getattr(module, "RELEASE_REQUIRED_TESTS", [])
        if any(
            f"{module.__name__}.{name}" not in expected for name in required
        ):
            raise ValueError("required contract test missing from suite")
        sys.setprofile(profile)
        result = unittest.TextTestRunner(
            stream=stream, verbosity=2, resultclass=RecordedResult
        ).run(suite)
    finally:
        sys.setprofile(previous)
        if previous_sources is None:
            os.environ.pop("FLAGOS_TEST_SOURCES", None)
        else:
            os.environ["FLAGOS_TEST_SOURCES"] = previous_sources
        sys.path.remove(str(root))
    check_files(root, manifest["files"])
    # Imported repository helpers must be staged and bound too.
    for loaded in list(sys.modules.values()):
        name = getattr(loaded, "__file__", None)
        # torch.ops/classes advertise virtual _ops.py/_classes.py filenames.
        if (
            name
            and Path(name).is_file()
            and Path(name).resolve().is_relative_to(root)
        ):
            relative = Path(name).resolve().relative_to(root).as_posix()
            if relative.endswith(".py") and relative not in manifest["files"]:
                raise ValueError(
                    f"unbound imported dependency: {relative}; add --dependency"
                )
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "cases": result.cases,
        "expected_tests": expected,
        "passed_tests": result.passed,
        "source_calls": calls,
        "kernel_launches": launches,
        "tensor_shapes": sorted(shapes),
    }


def require_success(result, sources):
    if not isinstance(result, dict):
        raise ValueError("invalid verification result")
    if type(result.get("tests_run")) is not int or result["tests_run"] <= 0:
        raise ValueError("no executed tests")
    for key in (
        "failures",
        "errors",
        "skipped",
        "expected_failures",
        "unexpected_successes",
    ):
        if type(result.get(key)) is not int or result[key] != 0:
            raise ValueError(f"verification contains {key}")
    if not result.get("cases"):
        raise ValueError("missing executed cases")
    expected = result.get("expected_tests")
    if (
        not isinstance(expected, list)
        or not expected
        or len(expected) != result["tests_run"]
        or len(set(expected)) != len(expected)
        or sorted(result.get("passed_tests", [])) != sorted(expected)
    ):
        raise ValueError("required test suite did not fully pass")
    calls = result.get("source_calls", {})
    if (
        not isinstance(calls, dict)
        or set(calls) != set(sources)
        or any(type(n) is not int or n <= 0 for n in calls.values())
    ):
        raise ValueError("applicable source was not exercised")
    launches = result.get("kernel_launches", {})
    if (
        not isinstance(launches, dict)
        or set(launches) != set(sources)
        or any(type(n) is not int or n <= 0 for n in launches.values())
    ):
        raise ValueError("applicable source has no completed kernel launch")
    for source in sources:
        if not any(
            p == source
            and dtype
            and isinstance(shape, (list, tuple))
            and all(type(n) is int and n > 0 for n in shape)
            for p, dtype, shape in result.get("tensor_shapes", [])
        ):
            raise ValueError(
                "applicable source has no nonempty tensor coverage"
            )


def environment():
    import torch
    import triton

    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA/HIP device unavailable; skipped tests cannot pass release"
        )
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(),
        "scope": "amd-device" if torch.version.hip else "nvidia-proxy",
        "device_vendor": "amd" if torch.version.hip else "nvidia",
    }


def run(args):
    root = Path(args.directory).resolve()
    manifest = json.loads((root / "verification-input.json").read_text())
    started = time.time()
    receipt = {**manifest, "started_at": started, "exit_code": 1}
    # Exclusive files prevent a rerun from replacing a previously cited result.
    with (root / "verification.log").open("x") as stream:
        try:
            receipt["environment"] = environment()
            if (
                receipt["environment"]["device_vendor"]
                != manifest["device_vendor"]
            ):
                raise ValueError(
                    "execution device differs from prepared scope"
                )
            stream.write(json.dumps(receipt["environment"]) + "\n")
            receipt["result"] = run_suite(root, manifest, stream)
            import torch

            torch.cuda.synchronize()
            require_success(receipt["result"], manifest["execution_sources"])
            receipt["exit_code"] = 0
        except Exception as error:
            receipt["error"] = str(error)
            stream.write(f"VERIFICATION_FAILED: {error}\n")
    receipt.update(
        finished_at=time.time(),
        log_file="verification.log",
        log_sha256=digest((root / "verification.log").read_bytes()),
    )
    with (root / "verification.json").open("x") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                "exit_code": receipt["exit_code"],
                "receipt": str(root / "verification.json"),
                "error": receipt.get("error"),
            }
        )
    )
    return receipt["exit_code"]


def verify_receipt(spec, root):
    """Check execution evidence against Git, never against the working tree."""
    path = Path(spec.get("verification_receipt", ""))
    expected = spec.get("verification_receipt_sha256", "")
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not re.fullmatch(r"[0-9a-f]{64}", expected)
    ):
        raise ValueError("a bound verification receipt is required")
    payload = path.read_bytes()
    if digest(payload) != expected:
        raise ValueError("verification receipt SHA-256 mismatch")
    receipt = json.loads(payload)
    if not isinstance(receipt, dict):
        raise ValueError("invalid verification receipt")
    if receipt.get("schema_version") != 2 or receipt.get("mode") != "release":
        raise ValueError(
            "a v2 release execution receipt is required; rerun old evidence"
        )
    if type(receipt.get("exit_code")) is not int or receipt["exit_code"] != 0:
        raise ValueError("release execution did not succeed")
    for key in ("operator", "source_commit", "verification_commit"):
        if receipt.get(key) != spec[key]:
            raise ValueError(f"receipt identity mismatch: {key}")
    tree = (
        git(root, "ls-tree", "-r", "--name-only", spec["source_commit"])
        .decode()
        .splitlines()
    )
    sources = source_paths(tree, spec["operator"])
    if receipt.get("sources") != sources:
        raise ValueError("receipt source set differs from candidate")
    files = receipt.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("invalid receipt file hashes")
    required = set(sources) | {
        f"tests/test_{spec['operator']}.py",
        "tests/__init__.py",
        RUNNER,
    }
    if not required.issubset(files):
        raise ValueError("receipt omits source/test/runner dependencies")
    for name, sha in files.items():
        checked_path(root, name)
        commit = (
            spec["source_commit"]
            if name in sources
            else spec["verification_commit"]
        )
        if digest(git(root, "show", f"{commit}:{name}")) != sha:
            raise ValueError(f"receipt Git blob mismatch: {name}")
    scope = execution_scope(
        sources,
        spec["operator"],
        receipt.get("device_vendor"),
        receipt.get("proxy_vendors"),
    )
    if any(receipt.get(key) != value for key, value in scope.items()):
        raise ValueError("receipt execution scope differs from candidate")
    require_success(receipt.get("result", {}), scope["execution_sources"])
    env = receipt.get("environment", {})
    if not isinstance(env, dict) or any(
        not env.get(key)
        for key in ("python", "torch", "triton", "device", "scope")
    ):
        raise ValueError("receipt environment is incomplete")
    if env.get("device_vendor") != scope["device_vendor"]:
        raise ValueError("receipt device differs from execution scope")
    log = checked_path(path.parent, receipt.get("log_file", ""))
    if digest(log.read_bytes()) != receipt.get("log_sha256"):
        raise ValueError("release log SHA-256 mismatch")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("prepare")
    stage.add_argument("operator")
    stage.add_argument("--source-commit", required=True)
    stage.add_argument("--verification-commit", required=True)
    stage.add_argument(
        "--mode", choices=("screening", "release"), default="release"
    )
    stage.add_argument("--dependency", action="append", default=[])
    stage.add_argument(
        "--device-vendor", choices=("nvidia", "amd"), default="nvidia"
    )
    stage.add_argument(
        "--proxy-vendor",
        action="append",
        default=[],
        help="also test this vendor on the proxy; does not establish target-chip correctness",
    )
    stage.add_argument("--directory", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--directory", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            if not re.fullmatch(r"[a-z][a-z0-9_]*", args.operator):
                raise ValueError("invalid operator")
            prepare(args)
            return 0
        return run(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"verification error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
