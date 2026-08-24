import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request

SCRIPT = Path(__file__).parents[1] / "scripts" / "platform_cli.py"
SPEC = importlib.util.spec_from_file_location("platform_cli", SCRIPT)
PLATFORM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLATFORM)


class FakeClient:
    def __init__(self):
        self.posts = []
        self.quota = {"used": 0, "total": 15}
        self.records = []
        self.fail_final = False
        self.change_quota_on_upload = False

    def get(self, url, params=None):
        if url == PLATFORM.IAM:
            return {"userResponse": {"username": "alice", "userId": "u1"}}
        if url.endswith("/user/teams"):
            return [{"rid": "race1", "team_name": "team1", "team_id": "t1"}]
        if url.endswith("/operator-overview"):
            return {"stats": {"min_submission_interval_seconds": 120}}
        if url.endswith("/operator-tasks"):
            return [
                {
                    "tid": "tid12",
                    "batch_no": 2,
                    "task_no": 12,
                    "operator": "demo",
                    "competition_status": "competing",
                    "submit_start_at": "2020-01-01T00:00:00+08:00",
                    "submit_end_at": "2100-01-01T00:00:00+08:00",
                }
            ]
        if url.endswith("/operator-submissions/quota"):
            return dict(self.quota)
        if url.endswith("/operator-submissions"):
            return list(self.records)
        if url.endswith("/races/race1"):
            return {"rid": "race1"}
        raise AssertionError((url, params))

    def post_multipart(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if len(self.posts) == 1:
            if self.change_quota_on_upload:
                self.quota["used"] += 1
            return {"filename": "demo.zip", "url": "https://upload/demo.zip"}
        if self.fail_final:
            raise PLATFORM.CliError("simulated timeout")
        return {"file_name": "demo.zip", "status": "queued"}


class PlatformCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "demo.zip"
        self.archive.write_bytes(b"same verified ZIP bytes")
        self.sha = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.args = argparse.Namespace(
            season="2",
            race="race1",
            account="alice",
            team="team1",
            batch=2,
            task=12,
            operator="demo",
            stage="e1",
            commit="a" * 40,
            zip=str(self.archive),
            sha256=self.sha,
            member=["demo.py"],
        )

    def tearDown(self):
        self.temporary.cleanup()

    def preflight(self, client):
        with mock.patch.object(PLATFORM, "_verify_artifact", return_value={}):
            return PLATFORM._preflight(self.args, client, self.root / "state")

    def submit(self, nonce, client):
        with mock.patch.object(PLATFORM, "_verify_artifact", return_value={}):
            return PLATFORM._submit(nonce, client, self.root / "state")

    def intent(self, nonce):
        return json.loads((self.root / "state" / f"{nonce}.json").read_text())

    def test_status_is_get_only(self):
        client = FakeClient()
        args = argparse.Namespace(
            race="race1",
            batch=2,
            task=12,
            operator="demo",
            watch=False,
            timeout=0,
            interval=15,
        )
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(PLATFORM._status(args, client), 0)
        self.assertEqual(client.posts, [])
        self.assertEqual(
            json.loads(output.getvalue())["quota"]["remaining"], 15
        )

    def test_preflight_does_not_post_and_submit_is_one_shot(self):
        client = FakeClient()
        prepared = self.preflight(client)
        self.assertEqual(client.posts, [])

        result = self.submit(prepared["nonce"], client)
        self.assertEqual(result["state"], "submitted")
        self.assertEqual(len(client.posts), 2)
        self.assertEqual(
            client.posts[0][1]["fields"],
            {"white_code": "expected_white_code"},
        )
        self.assertEqual(
            client.posts[1][1]["fields"],
            {"tid": "tid12", "file_url": "https://upload/demo.zip"},
        )
        self.assertEqual(
            client.posts[1][1]["headers"]["Idempotency-Key"],
            f"flagos-{prepared['nonce']}",
        )
        upload_bytes = client.posts[0][1]["files"]["file"][1]
        archive_bytes = client.posts[1][1]["files"]["archive"][1]
        self.assertIs(upload_bytes, archive_bytes)
        self.assertEqual(hashlib.sha256(upload_bytes).hexdigest(), self.sha)

        with self.assertRaisesRegex(PLATFORM.CliError, "not reusable"):
            self.submit(prepared["nonce"], client)
        self.assertEqual(len(client.posts), 2)
        with self.assertRaisesRegex(PLATFORM.CliError, "already submitted"):
            self.preflight(client)
        self.assertEqual(len(client.posts), 2)

    def test_duplicate_preflight_is_rejected(self):
        client = FakeClient()
        self.preflight(client)
        with self.assertRaisesRegex(PLATFORM.CliError, "already exists"):
            self.preflight(client)
        self.assertEqual(client.posts, [])

    def test_live_drift_makes_intent_stale_without_post(self):
        client = FakeClient()
        prepared = self.preflight(client)
        client.quota["used"] = 1
        with self.assertRaisesRegex(PLATFORM.CliError, "state changed"):
            self.submit(prepared["nonce"], client)
        self.assertEqual(client.posts, [])
        self.assertEqual(self.intent(prepared["nonce"])["state"], "stale")

    def test_send_error_is_uncertain_and_never_retried(self):
        client = FakeClient()
        prepared = self.preflight(client)
        client.fail_final = True
        with self.assertRaisesRegex(PLATFORM.CliError, "simulated timeout"):
            self.submit(prepared["nonce"], client)
        self.assertEqual(len(client.posts), 2)
        self.assertEqual(self.intent(prepared["nonce"])["state"], "uncertain")

        with self.assertRaisesRegex(PLATFORM.CliError, "not reusable"):
            self.submit(prepared["nonce"], client)
        self.assertEqual(len(client.posts), 2)

        with self.assertRaisesRegex(PLATFORM.CliError, "unresolved"):
            self.preflight(client)
        self.assertEqual(len(client.posts), 2)

        client.records = [
            {
                "batch_no": 2,
                "task_no": 12,
                "operator": "demo",
                "file_url": "https://upload/demo.zip",
                "status": "queued",
            }
        ]
        with self.assertRaisesRegex(PLATFORM.CliError, "refusing duplicate"):
            self.preflight(client)
        self.assertEqual(
            self.intent(prepared["nonce"])["state"],
            "reconciled_submitted",
        )
        self.assertEqual(len(client.posts), 2)

    def test_drift_after_upload_skips_final_post(self):
        client = FakeClient()
        prepared = self.preflight(client)
        client.change_quota_on_upload = True
        with self.assertRaisesRegex(PLATFORM.CliError, "final POST skipped"):
            self.submit(prepared["nonce"], client)
        self.assertEqual(len(client.posts), 1)
        self.assertEqual(
            self.intent(prepared["nonce"])["state"], "stale_after_upload"
        )

    def test_http_client_decodes_gzip_and_rejects_redirects(self):
        class Response:
            headers = {"Content-Encoding": "gzip"}

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def read(self, unused):
                return gzip.compress(b'{"code":200,"data":{"ok":true}}')

        client = PLATFORM.HttpClient("secret")
        client._opener = mock.Mock()
        client._opener.open.return_value = Response()
        self.assertEqual(
            client.get("https://flagos.io/flagos/api/v1/races/race1"),
            {"ok": True},
        )

        with self.assertRaises(HTTPError):
            PLATFORM._NoRedirect().redirect_request(
                Request("https://flagos.io/flagos/api/v1/races/race1"),
                None,
                302,
                "Found",
                {},
                "https://example.invalid/steal",
            )

    def test_token_file_must_be_private(self):
        token_file = self.root / "token"
        token_file.write_text("secret\n")
        token_file.chmod(0o644)
        with mock.patch.dict(
            os.environ,
            {"FLAGOS_TOKEN": "", "FLAGOS_TOKEN_FILE": str(token_file)},
            clear=False,
        ):
            with self.assertRaisesRegex(PLATFORM.CliError, "0600"):
                PLATFORM._token()
            token_file.chmod(0o600)
            self.assertEqual(PLATFORM._token(), "secret")


if __name__ == "__main__":
    unittest.main()
