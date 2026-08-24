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

    def get(self, url, params=None, **unused):
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
        remote = {"size": len(self.archive.read_bytes()), "sha256": self.sha}
        with mock.patch.object(
            PLATFORM, "_verify_artifact", return_value={}
        ), mock.patch.object(
            PLATFORM, "_remote_zip_fingerprint", return_value=remote
        ):
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
        client.records = [
            {
                "batch_no": 2,
                "task_no": 12,
                "operator": "demo",
                "file_url": "https://upload.example/previous.zip",
                "status": "completed",
                "created_at": "2020-01-01T00:00:00+08:00",
            }
        ]
        prepared = self.preflight(client)
        self.assertEqual(client.posts, [])

        result = self.submit(prepared["nonce"], client)
        self.assertEqual(result["state"], "submitted")
        self.assertEqual(
            result["remote_verification"]["status"], "verified"
        )
        self.assertIn("--file-url-sha256", result["watch_command"])
        self.assertIn("--after-epoch", result["watch_command"])
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

    def test_remote_verification_failure_keeps_submission_one_shot(self):
        client = FakeClient()
        prepared = self.preflight(client)
        with mock.patch.object(
            PLATFORM, "_verify_artifact", return_value={}
        ), mock.patch.object(
            PLATFORM,
            "_remote_zip_fingerprint",
            side_effect=ValueError("simulated secret download failure"),
        ):
            result = PLATFORM._submit(
                prepared["nonce"], client, self.root / "state"
            )

        self.assertEqual(result["state"], "submitted")
        self.assertEqual(
            result["remote_verification"]["status"], "unavailable"
        )
        self.assertNotIn("secret", json.dumps(result))
        self.assertEqual(
            self.intent(prepared["nonce"])["state"], "submitted"
        )
        self.assertEqual(len(client.posts), 2)
        with self.assertRaisesRegex(PLATFORM.CliError, "not reusable"):
            self.submit(prepared["nonce"], client)
        self.assertEqual(len(client.posts), 2)

    def test_duplicate_preflight_is_rejected(self):
        client = FakeClient()
        self.preflight(client)
        with self.assertRaisesRegex(PLATFORM.CliError, "already exists"):
            self.preflight(client)
        self.assertEqual(client.posts, [])

    def test_watch_can_bind_to_exact_file_url(self):
        client = FakeClient()
        client.records = [
            {
                "batch_no": 2,
                "task_no": 12,
                "operator": "demo",
                "file_url": "https://upload.example/new.zip?secret=new",
                "status": "completed",
                "created_at": "2020-01-01T00:00:00+08:00",
            },
            {
                "batch_no": 2,
                "task_no": 12,
                "operator": "demo",
                "file_url": "https://user:password@upload.example/old.zip",
                "status": "completed",
                "created_at": "2019-01-01T00:00:00+08:00",
            },
        ]
        args = argparse.Namespace(
            race="race1",
            batch=2,
            task=12,
            operator="demo",
            watch=True,
            timeout=0,
            interval=15,
            file_url_sha256=hashlib.sha256(
                b"https://upload.example/new.zip?secret=new"
            ).hexdigest(),
            after_epoch=PLATFORM._parse_time(
                "2020-01-01T00:00:00+08:00"
            ).timestamp(),
        )
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(PLATFORM._status(args, client), 124)
        public_record = json.loads(output.getvalue())["submissions"][0]
        self.assertEqual(
            public_record["file_url"], "https://upload.example/new.zip"
        )
        self.assertNotIn("secret=new", output.getvalue())
        self.assertNotIn("password", output.getvalue())
        self.assertEqual(
            json.loads(output.getvalue())["submissions"][1]["file_url"],
            "<invalid>",
        )

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

            def __init__(self, payload=b'{"code":200,"data":{"ok":true}}'):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def read(self, unused):
                return gzip.compress(self.payload)

        client = PLATFORM.HttpClient("secret")
        client._opener = mock.Mock()
        client._opener.open.return_value = Response()
        self.assertEqual(
            client.get("https://flagos.io/flagos/api/v1/races/race1"),
            {"ok": True},
        )
        client._opener.open.return_value = Response(
            b'{"code":401,"data":null}'
        )
        with self.assertRaisesRegex(PLATFORM.CliError, "API code 401"):
            client.get("https://flagos.io/flagos/api/v1/races/race1")

        client._opener.open.return_value = Response(
            b'{"code":0,"data":{"userResponse":{"username":"alice"}}}'
        )
        self.assertEqual(
            client.get(PLATFORM.IAM, success_codes=(0, 200))["userResponse"][
                "username"
            ],
            "alice",
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

    def test_remote_verifier_sends_no_credentials(self):
        payload = self.archive.read_bytes()

        class Response:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def read(self, limit):
                self.limit = limit
                return payload

        response = Response()
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.dict(
            os.environ,
            {"FLAGOS_REMOTE_ZIP_HOST": "objects.example.com"},
        ), mock.patch.object(PLATFORM, "build_opener", return_value=opener):
            result = PLATFORM._remote_zip_fingerprint(
                "https://objects.example.com/demo.zip?signature=secret",
                len(payload),
            )
            with self.assertRaisesRegex(PLATFORM.CliError, "unsafe"):
                PLATFORM._remote_zip_fingerprint(
                    "https://127.1/demo.zip", len(payload)
                )
            with self.assertRaisesRegex(PLATFORM.CliError, "unsafe"):
                PLATFORM._remote_zip_fingerprint(
                    "https://other.example.com/demo.zip", len(payload)
                )

        self.assertEqual(opener.open.call_count, 1)
        request = opener.open.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))
        self.assertIsNone(request.get_header("Cookie"))
        self.assertEqual(request.get_header("Accept-encoding"), "identity")
        self.assertEqual(response.limit, len(payload) + 1)
        self.assertEqual(result, {"size": len(payload), "sha256": self.sha})
        with mock.patch.dict(
            os.environ, {"FLAGOS_REMOTE_ZIP_HOST": ""}
        ), self.assertRaisesRegex(PLATFORM.CliError, "exact trusted"):
            PLATFORM._remote_zip_fingerprint(
                "https://objects.example.com/demo.zip", len(payload)
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
            token_file.write_bytes(b"\xff")
            with self.assertRaisesRegex(PLATFORM.CliError, "UTF-8"):
                PLATFORM._token()

    def test_auth_saves_only_an_iam_validated_token(self):
        login_client = mock.Mock()
        login_client.post_json.side_effect = [None, {"token": "secret-token"}]
        validation_client = mock.Mock()
        validation_client.get.return_value = {
            "userResponse": {"username": "alice"}
        }
        token_file = self.root / "flagos-token"
        args = argparse.Namespace(method="email", accept_terms=True)

        with mock.patch("builtins.input", return_value="alice@example.com"), (
            mock.patch.object(
                PLATFORM.getpass, "getpass", return_value="123456"
            )
        ), mock.patch.object(
            PLATFORM,
            "HttpClient",
            side_effect=[login_client, validation_client],
        ), mock.patch.object(
            PLATFORM, "_git_path", return_value=token_file
        ), redirect_stdout(StringIO()) as output:
            self.assertEqual(PLATFORM._auth(args), 0)

        self.assertEqual(token_file.read_text(), "secret-token\n")
        self.assertEqual(token_file.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("secret-token", output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["account"], "alice")
        self.assertEqual(
            login_client.post_json.call_args_list[0].args,
            (
                f"{PLATFORM.AUTH}/sendMailVerifyCode",
                {"mailAddr": "alice@example.com", "language": 0},
            ),
        )
        self.assertEqual(
            login_client.post_json.call_args_list[1].args,
            (
                f"{PLATFORM.AUTH}/mailLoginRegister",
                {"mailAddr": "alice@example.com", "code": "123456"},
            ),
        )

    def test_auth_failure_preserves_existing_token(self):
        token_file = self.root / "flagos-token"
        token_file.write_text("old-token\n")
        token_file.chmod(0o600)
        login_client = mock.Mock()
        login_client.post_json.side_effect = [None, {"token": "new-token"}]
        validation_client = mock.Mock()
        validation_client.get.return_value = {}

        with mock.patch("builtins.input", return_value="13800138000"), (
            mock.patch.object(
                PLATFORM.getpass, "getpass", return_value="123456"
            )
        ), mock.patch.object(
            PLATFORM,
            "HttpClient",
            side_effect=[login_client, validation_client],
        ), mock.patch.object(
            PLATFORM, "_git_path", return_value=token_file
        ), self.assertRaisesRegex(PLATFORM.CliError, "incomplete"):
            PLATFORM._auth(
                argparse.Namespace(method="phone", accept_terms=True)
            )

        self.assertEqual(token_file.read_text(), "old-token\n")

    def test_auth_requires_explicit_terms_acceptance(self):
        with mock.patch.object(PLATFORM, "HttpClient") as client, (
            self.assertRaisesRegex(PLATFORM.CliError, "accept-terms")
        ):
            PLATFORM._auth(
                argparse.Namespace(method="email", accept_terms=False)
            )
        client.assert_not_called()

    def test_token_defaults_to_git_internal_file(self):
        token_file = self.root / "flagos-token"
        token_file.write_text("secret\n")
        token_file.chmod(0o600)
        with mock.patch.dict(
            os.environ,
            {"FLAGOS_TOKEN": "", "FLAGOS_TOKEN_FILE": ""},
            clear=False,
        ), mock.patch.object(
            PLATFORM, "_git_path", return_value=token_file
        ):
            self.assertEqual(PLATFORM._token(), "secret")


if __name__ == "__main__":
    unittest.main()
