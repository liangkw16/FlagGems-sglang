#!/usr/bin/env python3
"""Stage committed operator tests and record their actual execution (stdlib)."""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path

RUNNER = ".agents/skills/flagos-operator-race/scripts/verify_release.py"
GENERIC = "src/flaggems_sglang/ops"
BACKENDS = "src/flaggems_sglang/runtime/backend"


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
        "schema_version": 1,
        "mode": args.mode,
        "operator": args.operator,
        "source_commit": source,
        "verification_commit": verification,
        "sources": sources,
        "files": {p: digest(b) for p, b in contents.items()},
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

    def startTest(self, test):
        self.cases.append(test.id())
        super().startTest(test)

    def addSubTest(self, test, subtest, error):
        self.cases.append(subtest.id())
        super().addSubTest(test, subtest, error)


def run_suite(root, manifest, stream):
    """Track public wrapper calls, including dynamic importlib vendor modules."""
    root = root.resolve()
    calls = {p: 0 for p in manifest["sources"]}
    by_filename = {str((root / p).resolve()): p for p in calls}
    shapes = set()

    def profile(frame, event, arg):
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
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = unittest.TextTestRunner(
            stream=stream, verbosity=2, resultclass=RecordedResult
        ).run(suite)
    finally:
        sys.setprofile(previous)
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
        "source_calls": calls,
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
    calls = result.get("source_calls", {})
    if (
        not isinstance(calls, dict)
        or set(calls) != set(sources)
        or any(type(n) is not int or n <= 0 for n in calls.values())
    ):
        raise ValueError("generic/vendor source was not exercised")


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
            stream.write(json.dumps(receipt["environment"]) + "\n")
            receipt["result"] = run_suite(root, manifest, stream)
            require_success(receipt["result"], manifest["sources"])
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
    if receipt.get("schema_version") != 1 or receipt.get("mode") != "release":
        raise ValueError("only a release execution receipt is accepted")
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
    require_success(receipt.get("result", {}), sources)
    env = receipt.get("environment", {})
    if not isinstance(env, dict) or any(
        not env.get(key)
        for key in ("python", "torch", "triton", "device", "scope")
    ):
        raise ValueError("receipt environment is incomplete")
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
