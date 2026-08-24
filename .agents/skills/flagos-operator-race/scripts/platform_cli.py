#!/usr/bin/env python3
"""Read FlagOS scores and make one-shot, verified operator submissions."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import gzip
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import (
    HTTPCookieProcessor,
    HTTPRedirectHandler,
    Request,
    build_opener,
)

API = "https://flagos.io/flagos/api/v1"
IAM = "https://flagos.io/flagos/iam/user/getCurrentUserBaseInfo"
AUTH = "https://flagos.io/flagos/user-srv/auth"
UPLOAD = "https://flagos.net/flagos/api/v1/upload/file"
ZIP_LIMIT = 10 * 1024 * 1024
INTENT_TTL = 10 * 60
TERMINAL = {
    "valid",
    "invalid_correctness",
    "invalid_threshold",
    "completed",
    "failed",
}


class CliError(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        raise HTTPError(request.full_url, code, message, headers, response)


def _json(value: Any, *, compact: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _token() -> str:
    value = os.environ.get("FLAGOS_TOKEN", "").strip()
    if value:
        if "\n" in value or "\r" in value:
            raise CliError("FLAGOS_TOKEN must be one line")
        return value

    filename = os.environ.get("FLAGOS_TOKEN_FILE", "")
    if not filename:
        filename = str(_git_path("flagos-token"))
        if not Path(filename).exists():
            raise CliError(
                "run the auth command or set FLAGOS_TOKEN/FLAGOS_TOKEN_FILE"
            )
    path = Path(filename)
    if not path.is_absolute():
        raise CliError("FLAGOS_TOKEN_FILE must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CliError(
            f"cannot open FLAGOS_TOKEN_FILE: {error.strerror}"
        ) from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise CliError("FLAGOS_TOKEN_FILE must be a regular file")
        if info.st_mode & 0o077:
            raise CliError("FLAGOS_TOKEN_FILE must have mode 0600")
        if info.st_size > 65536:
            raise CliError("FLAGOS_TOKEN_FILE is unexpectedly large")
        try:
            value = os.read(descriptor, 65537).decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise CliError("FLAGOS_TOKEN_FILE must be UTF-8") from error
    finally:
        os.close(descriptor)
    if not value or "\n" in value or "\r" in value:
        raise CliError("FLAGOS_TOKEN_FILE must contain one non-empty line")
    return value


class HttpClient:
    def __init__(self, token: str, *, cookies: bool = False) -> None:
        self._headers = {
            "Accept": "application/json",
            "Lang": "CN",
            "User-Agent": "FlagOS-operator-race-cli/1.0",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        handlers = [_NoRedirect()]
        if cookies:
            handlers.append(HTTPCookieProcessor(CookieJar()))
        self._opener = build_opener(*handlers)

    def _request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        success_codes: tuple[int, ...] = (200,),
    ) -> Any:
        target = urlsplit(url)
        if (
            target.scheme != "https"
            or target.hostname not in {"flagos.io", "flagos.net"}
            or target.username is not None
            or target.password is not None
            or target.port is not None
        ):
            raise CliError("refusing non-production FlagOS URL")
        request_headers = dict(self._headers)
        request_headers.update(headers or {})
        request = Request(
            url,
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=120) as response:
                payload = response.read(2 * 1024 * 1024 + 1)
                encoding = (
                    response.headers.get("Content-Encoding") or ""
                ).lower()
        except HTTPError as error:
            raise CliError(
                f"FlagOS HTTP {error.code} at {url.split('?')[0]}"
            ) from error
        except URLError as error:
            raise CliError(f"FlagOS network error: {error.reason}") from error
        if len(payload) > 2 * 1024 * 1024:
            raise CliError("FlagOS response exceeded 2 MiB")
        if encoding == "gzip":
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
                    payload = stream.read(2 * 1024 * 1024 + 1)
            except (OSError, EOFError) as error:
                raise CliError("FlagOS returned invalid gzip data") from error
            if len(payload) > 2 * 1024 * 1024:
                raise CliError("FlagOS response exceeded 2 MiB after gzip")
        elif encoding:
            raise CliError(f"unsupported FlagOS content encoding: {encoding}")
        try:
            result = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CliError("FlagOS returned non-JSON data") from error
        if not isinstance(result, dict):
            raise CliError("FlagOS returned an invalid API response")
        code = result.get("code")
        if isinstance(code, bool) or code not in success_codes:
            if isinstance(code, bool) or not isinstance(code, int):
                code = "unknown"
            raise CliError(
                f"FlagOS API code {code} at {url.split('?')[0]}"
            )
        return result.get("data")

    def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        if params:
            url = f"{url}?{urlencode(params)}"
        return self._request("GET", url)

    def post_json(
        self,
        url: str,
        value: dict[str, Any],
        *,
        success_codes: tuple[int, ...] = (0, 200),
    ) -> Any:
        return self._request(
            "POST",
            url,
            body=_json(value, compact=True).encode(),
            headers={"Content-Type": "application/json"},
            success_codes=success_codes,
        )

    def post_multipart(
        self,
        url: str,
        *,
        fields: dict[str, str],
        files: dict[str, tuple[str, bytes]],
        headers: dict[str, str] | None = None,
    ) -> Any:
        boundary = f"flagos-{secrets.token_hex(16)}"
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    value.encode(),
                    b"\r\n",
                ]
            )
        for name, (filename, payload) in files.items():
            safe_name = Path(filename).name.replace('"', "")
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    (
                        f'Content-Disposition: form-data; name="{name}"; '
                        f'filename="{safe_name}"\r\n'
                    ).encode(),
                    b"Content-Type: application/zip\r\n\r\n",
                    payload,
                    b"\r\n",
                ]
            )
        chunks.append(f"--{boundary}--\r\n".encode())
        request_headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            **(headers or {}),
        }
        return self._request(
            "POST", url, body=b"".join(chunks), headers=request_headers
        )


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CliError(f"invalid FlagOS timestamp: {value}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed


def _task_matches(
    task: dict[str, Any], batch: int, task_no: int, operator: str
) -> bool:
    return (
        str(task.get("batch_no")) == str(batch)
        and str(task.get("task_no")) == str(task_no)
        and task.get("operator") == operator
    )


def _record_matches(
    record: dict[str, Any], batch: int, task_no: int, operator: str
) -> bool:
    return (
        str(record.get("batch_no")) == str(batch)
        and str(record.get("task_no")) == str(task_no)
        and record.get("operator") == operator
    )


def _fingerprint(records: list[dict[str, Any]]) -> str:
    return _sha256(_json(records, compact=True).encode())


def _live_state(
    client: Any, race: str, batch: int, task_no: int, operator: str
) -> dict[str, Any]:
    race_data = client.get(f"{API}/races/{race}")
    if not isinstance(race_data, dict) or str(race_data.get("rid")) != race:
        raise CliError("race ID did not resolve exactly")

    identity = client.get(IAM)
    user = identity.get("userResponse") if isinstance(identity, dict) else None
    if not isinstance(user, dict) or not user.get("username"):
        raise CliError("authenticated account response is incomplete")

    teams = client.get(f"{API}/user/teams")
    current_teams = [
        team
        for team in teams or []
        if isinstance(team, dict) and str(team.get("rid")) == race
    ]
    if len(current_teams) != 1:
        raise CliError("expected exactly one authenticated team for this race")
    team = current_teams[0]
    team_name = team.get("team_name") or team.get("name")
    if not team_name:
        raise CliError("authenticated team response is incomplete")

    overview = client.get(f"{API}/races/{race}/operator-overview")
    tasks = client.get(
        f"{API}/races/{race}/operator-tasks",
        {"keyword": operator, "batch_no": batch},
    )
    exact_tasks = [
        task
        for task in tasks or []
        if isinstance(task, dict)
        and _task_matches(task, batch, task_no, operator)
    ]
    if len(exact_tasks) != 1:
        raise CliError("task lookup was not an exact unique match")
    task = exact_tasks[0]
    if not task.get("tid"):
        raise CliError("task response omitted tid")

    quota = client.get(f"{API}/races/{race}/operator-submissions/quota")
    try:
        used, total = int(quota["used"]), int(quota["total"])
    except (KeyError, TypeError, ValueError) as error:
        raise CliError("quota response is incomplete") from error

    all_records = client.get(
        f"{API}/races/{race}/operator-submissions",
        {"page": 1, "page_size": 100},
    )
    task_records = client.get(
        f"{API}/races/{race}/operator-submissions",
        {
            "batch_no": batch,
            "tid": task["tid"],
            "page": 1,
            "page_size": 100,
        },
    )
    all_records = [
        item for item in all_records or [] if isinstance(item, dict)
    ]
    task_records = [
        item
        for item in task_records or []
        if isinstance(item, dict)
        and _record_matches(item, batch, task_no, operator)
    ]
    task_records.sort(
        key=lambda item: item.get("created_at") or "", reverse=True
    )

    timestamps = [
        parsed
        for parsed in (
            _parse_time(item.get("created_at")) for item in all_records
        )
        if parsed is not None
    ]
    latest = max(timestamps).isoformat() if timestamps else None
    stats = overview.get("stats") if isinstance(overview, dict) else {}
    try:
        minimum_interval = int(
            (stats or {}).get("min_submission_interval_seconds", 120)
        )
    except (TypeError, ValueError) as error:
        raise CliError("invalid minimum submission interval") from error

    return {
        "race_id": race,
        "account": user["username"],
        "account_id": user.get("userId"),
        "team": team_name,
        "team_id": team.get("team_id") or team.get("id"),
        "batch_no": batch,
        "task_no": task_no,
        "operator": operator,
        "tid": task["tid"],
        "task_status": task.get("competition_status") or task.get("status"),
        "submit_start_at": task.get("submit_start_at"),
        "submit_end_at": task.get("submit_end_at"),
        "quota": {"used": used, "total": total, "remaining": total - used},
        "minimum_interval_seconds": minimum_interval,
        "latest_submission_at": latest,
        "submission_fingerprint": _fingerprint(all_records),
        "submissions": task_records,
    }


def _assert_open(live: dict[str, Any], now: float) -> None:
    if live["task_status"] != "competing":
        raise CliError(f"task is not competing: {live['task_status']}")
    start = _parse_time(live["submit_start_at"])
    end = _parse_time(live["submit_end_at"])
    if start is None or end is None:
        raise CliError("task submission window is missing")
    current = datetime.fromtimestamp(now, tz=start.tzinfo)
    if not start <= current <= end:
        raise CliError("current time is outside the task submission window")
    if live["quota"]["remaining"] <= 0:
        raise CliError("no submission quota remains")
    latest = _parse_time(live["latest_submission_at"])
    if latest is not None:
        elapsed = current.timestamp() - latest.timestamp()
        if elapsed < live["minimum_interval_seconds"]:
            wait = int(live["minimum_interval_seconds"] - elapsed + 0.999)
            raise CliError(f"submission interval has {wait}s remaining")


def _live_binding(live: dict[str, Any]) -> dict[str, Any]:
    return {key: live[key] for key in live if key != "submissions"}


def _same_tuple(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "race_id",
        "account",
        "team",
        "batch_no",
        "task_no",
        "operator",
        "zip_sha256",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def _guard_existing_intents(
    state_dir: Path,
    spec: dict[str, Any],
    live: dict[str, Any],
    now: float,
) -> None:
    for path in sorted(state_dir.glob("*.json")):
        intent = _read_intent(path)
        if not _same_tuple(intent.get("spec") or {}, spec):
            continue
        state = intent.get("state")
        if state == "prepared" and intent.get("expires_at", 0) > now:
            raise CliError("an unexpired intent already exists for this tuple")
        if state in {"submitted", "reconciled_submitted"}:
            raise CliError("this exact tuple is already submitted")
        if state not in {"sending", "uncertain"}:
            continue
        upload_url = (intent.get("upload") or {}).get("url")
        if upload_url and any(
            record.get("file_url") == upload_url
            for record in live["submissions"]
        ):
            intent["state"] = "reconciled_submitted"
            intent["reconciled_at"] = now
            _write_intent(path, intent)
            raise CliError(
                "previous uncertain intent is already submitted; "
                "refusing duplicate"
            )
        raise CliError(
            "an unresolved sending/uncertain intent exists for this tuple"
        )


def _verify_artifact(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["zip_path"])
    if not path.is_absolute():
        raise CliError("ZIP path must be absolute")
    try:
        info = path.lstat()
    except OSError as error:
        raise CliError(f"cannot stat ZIP: {error.strerror}") from error
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise CliError("ZIP must be a regular, non-symlink file")
    if info.st_size >= ZIP_LIMIT:
        raise CliError("ZIP must be smaller than 10 MiB")
    build_script = Path(__file__).with_name("build_submission.py")
    command = [
        sys.executable,
        str(build_script),
        spec["operator"],
        "--stage",
        spec["stage"],
        "--commit",
        spec["source_commit"],
        "--verify-existing",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=60
    )
    if result.returncode:
        message = (result.stderr or result.stdout).strip()[-500:]
        raise CliError(f"artifact verification failed: {message}")
    try:
        verified = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise CliError("artifact verifier returned invalid JSON") from error
    expected = {
        "status": "verified-existing",
        "operator": spec["operator"],
        "commit": spec["source_commit"],
        "artifact": str(path),
        "zip_sha256": spec["zip_sha256"],
    }
    for key, value in expected.items():
        actual = verified.get(key)
        if key == "artifact":
            actual = str(Path(actual).resolve()) if actual else actual
            value = str(path.resolve())
        if actual != value:
            raise CliError(f"artifact verifier mismatch: {key}")
    if sorted(verified.get("archive_members") or []) != sorted(
        spec["members"]
    ):
        raise CliError(
            "artifact member list does not match confirmation tuple"
        )
    return verified


def _read_zip(spec: dict[str, Any]) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(spec["zip_path"], flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size >= ZIP_LIMIT:
            raise CliError("ZIP changed after verification")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(ZIP_LIMIT)
    finally:
        os.close(descriptor)
    if _sha256(payload) != spec["zip_sha256"]:
        raise CliError("ZIP SHA-256 changed after verification")
    return payload


def _remote_zip_fingerprint(url: str, expected_size: int) -> dict[str, Any]:
    try:
        target = urlsplit(url)
        host = target.hostname
        port = target.port
    except ValueError as error:
        raise CliError("remote ZIP URL is invalid") from error
    allowed_host = os.environ.get("FLAGOS_REMOTE_ZIP_HOST", "").strip().lower()
    if not re.fullmatch(
        r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}",
        allowed_host,
    ):
        raise CliError(
            "set FLAGOS_REMOTE_ZIP_HOST to the exact trusted hostname"
        )
    if (
        target.scheme != "https"
        or host != allowed_host
        or target.username is not None
        or target.password is not None
        or port is not None
        or target.fragment
    ):
        raise CliError("refusing unsafe remote ZIP URL")
    if expected_size < 0 or expected_size >= ZIP_LIMIT:
        raise CliError("invalid expected remote ZIP size")

    request = Request(
        url,
        headers={
            "Accept": "application/zip",
            "Accept-Encoding": "identity",
            "User-Agent": "FlagOS-operator-race-cli/1.0",
        },
        method="GET",
    )
    try:
        with build_opener(_NoRedirect()).open(
            request, timeout=120
        ) as response:
            encoding = (
                response.headers.get("Content-Encoding") or ""
            ).lower()
            if encoding not in {"", "identity"}:
                raise CliError(
                    f"unsupported remote ZIP content encoding: {encoding}"
                )
            payload = response.read(expected_size + 1)
    except HTTPError as error:
        raise CliError(
            f"remote ZIP HTTP {error.code} at {url.split('?')[0]}"
        ) from error
    except URLError as error:
        raise CliError(
            f"remote ZIP network error: {error.reason}"
        ) from error
    except OSError as error:
        raise CliError(
            f"remote ZIP network error: {type(error).__name__}"
        ) from error
    return {"size": len(payload), "sha256": _sha256(payload)}


def _git_path(name: str) -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-path", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CliError("current directory is not a Git worktree") from error
    return Path(result.stdout.strip()).resolve()


def _state_dir() -> Path:
    return _git_path("flagos-platform")


def _prepare_state_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


@contextmanager
def _state_lock(path: Path):
    _prepare_state_dir(path)
    descriptor = os.open(path / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        # ponytail: one repo-wide lock; split per team only if local concurrency matters.
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _intent_path(state_dir: Path, nonce: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise CliError(
            "confirmation nonce must be 32 lowercase hex characters"
        )
    return state_dir / f"{nonce}.json"


def _write_intent(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=".intent-", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value, stream, ensure_ascii=False, sort_keys=True, indent=2
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_intent(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or path.is_symlink()
            or info.st_mode & 0o077
        ):
            raise CliError("intent file must be a private regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CliError("confirmation nonce does not exist") from error
    except json.JSONDecodeError as error:
        raise CliError("intent file is corrupt") from error
    if not isinstance(value, dict):
        raise CliError("intent file is corrupt")
    return value


def _preflight(
    args: argparse.Namespace, client: Any, state_dir: Path
) -> dict[str, Any]:
    spec = {
        "season": args.season,
        "race_id": args.race,
        "account": args.account,
        "team": args.team,
        "batch_no": args.batch,
        "task_no": args.task,
        "operator": args.operator,
        "stage": args.stage,
        "source_commit": args.commit,
        "zip_path": args.zip,
        "zip_sha256": args.sha256,
        "members": args.member,
    }
    _verify_artifact(spec)
    with _state_lock(state_dir):
        live = _live_state(
            client, args.race, args.batch, args.task, args.operator
        )
        if live["account"] != args.account:
            raise CliError("authenticated account does not match --account")
        if live["team"] != args.team:
            raise CliError("authenticated team does not match --team")
        created = time.time()
        _assert_open(live, created)
        _guard_existing_intents(state_dir, spec, live, created)
        nonce = secrets.token_hex(16)
        intent = {
            "version": 1,
            "nonce": nonce,
            "state": "prepared",
            "created_at": created,
            "expires_at": created + INTENT_TTL,
            "spec": spec,
            "live_binding": _live_binding(live),
        }
        _write_intent(_intent_path(state_dir, nonce), intent)
    return {
        "state": "prepared",
        "nonce": nonce,
        "expires_at": datetime.fromtimestamp(intent["expires_at"])
        .astimezone()
        .isoformat(),
        "tuple": {**spec, **intent["live_binding"]},
        "confirm_command": (
            f"{sys.executable} {Path(__file__).resolve()} submit --confirm {nonce}"
        ),
    }


def _submit(nonce: str, client: Any, state_dir: Path) -> dict[str, Any]:
    with _state_lock(state_dir):
        path = _intent_path(state_dir, nonce)
        intent = _read_intent(path)
        if intent.get("state") != "prepared":
            raise CliError(f"intent is not reusable: {intent.get('state')}")
        if time.time() > intent.get("expires_at", 0):
            intent["state"] = "expired"
            _write_intent(path, intent)
            raise CliError("intent expired; run a new preflight")

        spec = intent["spec"]
        _verify_artifact(spec)
        payload = _read_zip(spec)
        live = _live_state(
            client,
            spec["race_id"],
            spec["batch_no"],
            spec["task_no"],
            spec["operator"],
        )
        try:
            _assert_open(live, time.time())
            if _live_binding(live) != intent["live_binding"]:
                raise CliError(
                    "live account/team/task/quota/submission state changed"
                )
        except CliError:
            intent["state"] = "stale"
            _write_intent(path, intent)
            raise

        intent["state"] = "sending"
        intent["sending_at"] = time.time()
        _write_intent(path, intent)
        try:
            uploaded = client.post_multipart(
                UPLOAD,
                fields={"white_code": "expected_white_code"},
                files={"file": (Path(spec["zip_path"]).name, payload)},
            )
            if not isinstance(uploaded, dict) or not uploaded.get("url"):
                raise CliError("upload response omitted file URL")
            if not isinstance(uploaded["url"], str):
                raise CliError("upload response returned an invalid file URL")
            intent["upload"] = {
                "filename": uploaded.get("filename"),
                "url": uploaded["url"],
            }
            _write_intent(path, intent)
        except BaseException:
            intent["state"] = "uncertain"
            intent["uncertain_at"] = time.time()
            _write_intent(path, intent)
            raise

        try:
            after_upload = _live_state(
                client,
                spec["race_id"],
                spec["batch_no"],
                spec["task_no"],
                spec["operator"],
            )
            _assert_open(after_upload, time.time())
        except BaseException:
            intent["state"] = "uncertain"
            intent["uncertain_at"] = time.time()
            _write_intent(path, intent)
            raise
        if _live_binding(after_upload) != intent["live_binding"]:
            intent["state"] = "stale_after_upload"
            intent["stale_at"] = time.time()
            _write_intent(path, intent)
            raise CliError(
                "live state changed after upload; final POST skipped"
            )

        try:
            submitted = client.post_multipart(
                f"{API}/races/{spec['race_id']}/operator-submissions",
                fields={
                    "tid": after_upload["tid"],
                    "file_url": uploaded["url"],
                },
                files={"archive": (Path(spec["zip_path"]).name, payload)},
                headers={"Idempotency-Key": f"flagos-{nonce}"},
            )
        except BaseException:
            intent["state"] = "uncertain"
            intent["uncertain_at"] = time.time()
            _write_intent(path, intent)
            raise
        intent["state"] = "submitted"
        intent["submitted_at"] = time.time()
        intent["result"] = submitted
        _write_intent(path, intent)
        expected = {
            "expected_size": len(payload),
            "expected_sha256": spec["zip_sha256"],
        }
        try:
            actual = _remote_zip_fingerprint(uploaded["url"], len(payload))
            remote_verification = {
                "status": (
                    "verified"
                    if actual["size"] == len(payload)
                    and actual["sha256"] == spec["zip_sha256"]
                    else "mismatch"
                ),
                **expected,
                **actual,
            }
        except Exception as error:
            if isinstance(error, CliError):
                message = str(error)
            else:
                message = (
                    "remote ZIP verification failed: "
                    f"{type(error).__name__}"
                )
            remote_verification = {
                "status": "unavailable",
                **expected,
                "error": message,
            }
        intent["remote_verification"] = remote_verification
        _write_intent(path, intent)
        latest_at = intent["live_binding"].get("latest_submission_at")
        after = _parse_time(latest_at)
        after_option = (
            f" --after-epoch {after.timestamp():.6f}" if after else ""
        )
        return {
            "state": "submitted",
            "nonce": nonce,
            "result": submitted,
            "remote_verification": remote_verification,
            "watch_command": (
                f"{sys.executable} {Path(__file__).resolve()} status "
                f"--race {spec['race_id']} --batch {spec['batch_no']} "
                f"--task {spec['task_no']} --operator {spec['operator']} "
                f"--file-url-sha256 {_sha256(uploaded['url'].encode())}"
                f"{after_option} "
                "--watch --interval 15 --timeout 900"
            ),
        }


def _status(args: argparse.Namespace, client: Any) -> int:
    deadline = time.monotonic() + args.timeout
    while True:
        live = _live_state(
            client, args.race, args.batch, args.task, args.operator
        )
        view = {
            key: value
            for key, value in live.items()
            if key != "submission_fingerprint"
        }
        view["submissions"] = []
        for record in live["submissions"]:
            public_record = dict(record)
            file_url = public_record.get("file_url")
            if isinstance(file_url, str):
                public_record["file_url_sha256"] = _sha256(
                    file_url.encode()
                )
                try:
                    target = urlsplit(file_url)
                    if (
                        target.scheme != "https"
                        or not target.hostname
                        or target.username is not None
                        or target.password is not None
                        or target.port is not None
                    ):
                        raise ValueError
                    public_record["file_url"] = target._replace(
                        query="", fragment=""
                    ).geturl()
                except ValueError:
                    public_record["file_url"] = "<invalid>"
            view["submissions"].append(public_record)
        view["observed_at"] = datetime.now().astimezone().isoformat()
        print(_json(view, compact=args.watch), flush=True)
        expected = getattr(args, "file_url_sha256", None)
        after_epoch = getattr(args, "after_epoch", None)
        latest = live["submissions"][0] if live["submissions"] else None
        if expected:
            latest = None
            for record in live["submissions"]:
                file_url = record.get("file_url")
                created = _parse_time(record.get("created_at"))
                if not isinstance(file_url, str):
                    continue
                if _sha256(file_url.encode()) != expected:
                    continue
                if after_epoch is not None and (
                    created is None or created.timestamp() <= after_epoch
                ):
                    continue
                latest = record
                break
        if not args.watch or (latest and latest.get("status") in TERMINAL):
            return 0
        if time.monotonic() >= deadline:
            return 124
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))


def _write_token(path: Path, token: str) -> None:
    token = token.strip()
    if (
        not token
        or len(token.encode()) > 65536
        or "\n" in token
        or "\r" in token
    ):
        raise CliError("FlagOS returned an invalid token")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".flagos-token-", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(token)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _auth(args: argparse.Namespace) -> int:
    if not args.accept_terms:
        raise CliError(
            "--accept-terms is required; the FlagOS endpoint may register "
            "an unknown email or phone"
        )
    try:
        contact = input(f"FlagOS {args.method}: ").strip()
    except EOFError as error:
        raise CliError(
            "authentication requires an interactive terminal"
        ) from error

    if args.method == "email":
        if not re.fullmatch(
            r"[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", contact
        ):
            raise CliError("invalid email address")
        send_url = f"{AUTH}/sendMailVerifyCode"
        login_url = f"{AUTH}/mailLoginRegister"
        send_body = {"mailAddr": contact, "language": 0}
        login_body = {"mailAddr": contact}
    else:
        if not re.fullmatch(r"1[3-9]\d{9}", contact):
            raise CliError("phone must be an 11-digit mainland China number")
        send_url = f"{AUTH}/sendSmsVerifyCode"
        login_url = f"{AUTH}/smsLogin"
        send_body = {"phoneNumber": contact, "language": 0}
        login_body = {"phoneNumber": contact}

    client = HttpClient("", cookies=True)
    client.post_json(send_url, send_body)
    try:
        code = getpass.getpass("Verification code: ").strip()
    except EOFError as error:
        raise CliError(
            "authentication requires an interactive terminal"
        ) from error
    if not re.fullmatch(r"\d{6}", code):
        raise CliError("verification code must contain 6 digits")
    login_body["code"] = code
    data = client.post_json(login_url, login_body)
    token = data.get("token") if isinstance(data, dict) else None
    if not isinstance(token, str):
        raise CliError("FlagOS login response omitted the token")
    token = token.strip()
    if (
        not token
        or len(token.encode()) > 65536
        or "\n" in token
        or "\r" in token
    ):
        raise CliError("FlagOS returned an invalid token")

    identity = HttpClient(token).get(IAM)
    user = identity.get("userResponse") if isinstance(identity, dict) else None
    username = user.get("username") if isinstance(user, dict) else None
    if not isinstance(username, str) or not username:
        raise CliError("authenticated account response is incomplete")

    token_path = _git_path("flagos-token")
    _write_token(token_path, token)
    print(
        _json(
            {
                "status": "authenticated",
                "account": username,
                "token_file": str(token_path),
                "mode": "0600",
            }
        )
    )
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--race", required=True)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--task", required=True, type=int)
    parser.add_argument("--operator", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    auth = commands.add_parser(
        "auth", help="log in with a verification code and save the token"
    )
    auth.add_argument("--method", choices=("email", "phone"), required=True)
    auth.add_argument(
        "--accept-terms",
        action="store_true",
        help="confirm the account and FlagOS terms before login/register",
    )
    status_parser = commands.add_parser(
        "status", help="read submission scores"
    )
    _common(status_parser)
    status_parser.add_argument("--watch", action="store_true")
    status_parser.add_argument("--interval", type=float, default=15)
    status_parser.add_argument("--timeout", type=float, default=600)
    status_parser.add_argument("--file-url-sha256")
    status_parser.add_argument("--after-epoch", type=float)

    preflight = commands.add_parser(
        "preflight", help="create a one-use intent"
    )
    _common(preflight)
    preflight.add_argument("--season", required=True)
    preflight.add_argument("--account", required=True)
    preflight.add_argument("--team", required=True)
    preflight.add_argument("--stage", required=True)
    preflight.add_argument("--commit", required=True)
    preflight.add_argument("--zip", required=True)
    preflight.add_argument("--sha256", required=True)
    preflight.add_argument("--member", action="append", required=True)

    submit = commands.add_parser("submit", help="consume one prepared intent")
    submit.add_argument("--confirm", required=True)
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.command in {"status", "preflight"}:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.race):
            raise CliError("invalid race ID")
        if args.batch <= 0 or args.task <= 0:
            raise CliError("batch and task must be positive")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", args.operator):
            raise CliError("invalid operator basename")
    if args.command == "status":
        if args.interval < 5 or args.timeout < 0:
            raise CliError("watch interval must be >=5s and timeout >=0")
        if args.file_url_sha256 and not re.fullmatch(
            r"[0-9a-f]{64}", args.file_url_sha256
        ):
            raise CliError(
                "--file-url-sha256 must be a full lowercase SHA-256"
            )
        if args.after_epoch is not None and (
            not args.file_url_sha256
            or not math.isfinite(args.after_epoch)
            or args.after_epoch < 0
        ):
            raise CliError(
                "--after-epoch requires a file URL hash and finite epoch"
            )
    if args.command == "preflight":
        if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
            raise CliError("--commit must be a full lowercase Git SHA")
        if not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
            raise CliError("--sha256 must be a full lowercase SHA-256")
        if not Path(args.zip).is_absolute():
            raise CliError("--zip must be absolute")
        if len(set(args.member)) != len(args.member):
            raise CliError("duplicate --member")
        if any(
            Path(item).name != item or not item.endswith(".py")
            for item in args.member
        ):
            raise CliError("each --member must be a .py basename")


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        _validate(args)
        if args.command == "auth":
            return _auth(args)
        client = HttpClient(_token())
        if args.command == "status":
            return _status(args, client)
        if args.command == "preflight":
            print(_json(_preflight(args, client, _state_dir())))
            return 0
        print(_json(_submit(args.confirm, client, _state_dir())))
        return 0
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
