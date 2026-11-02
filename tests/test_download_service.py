"""下载资源与本机 HTTP 集成测试；全部账户和数据均为临时资料。"""
import importlib
import importlib.util
import http.client
import tempfile
import time
import unittest
from pathlib import Path

import auth_store


class DownloadServiceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "accounts.sqlite"
        self.assertIsNotNone(importlib.util.find_spec("download_service"), "尚未实现安全下载服务")
        self.module = importlib.import_module("download_service")
        self.password = "TemporaryTestPassword2026!"
        self.admin = auth_store.bootstrap_admin("administrator", self.password, db_path=self.db)
        self.token = auth_store.authenticate("administrator", self.password, db_path=self.db)
        self.store = self.module.DownloadStore(db_path=self.db)

    def report(self, token=None, **kwargs):
        return self.store.register(token or self.token, b"private-report", "report.md", "text/markdown", **kwargs)

    def test_report_roundtrip_and_unguessable_hex_resource_id(self):
        resource = self.report()
        self.assertRegex(resource, r"\A[0-9a-f]{32,}\Z")
        self.assertEqual(self.store.fetch(self.token, resource), (b"private-report", "report.md", "text/markdown"))
        self.assertEqual(resource, self.report())
        changed = self.store.register(self.token, b"different-content", "report.md", "text/markdown")
        self.assertNotEqual(resource, changed)

    def test_same_user_second_session_cannot_fetch_first_session_resource(self):
        resource = self.report()
        second = auth_store.authenticate("administrator", self.password, db_path=self.db)
        with self.assertRaises(PermissionError):
            self.store.fetch(second, resource)

    def test_other_owner_and_forged_session_are_denied(self):
        resource = self.report()
        auth_store.create_user(self.token, "analyst_one", self.password, db_path=self.db)
        other = auth_store.authenticate("analyst_one", self.password, db_path=self.db)
        for token in (other, "forged-token"):
            with self.subTest(session="other" if token == other else "forged"):
                with self.assertRaises(PermissionError):
                    self.store.fetch(token, resource)
        with self.assertRaises(PermissionError):
            self.report("forged-token")

    def test_logout_and_password_reset_immediately_deny_existing_resource(self):
        resource = self.report()
        auth_store.logout(self.token, db_path=self.db)
        with self.assertRaises(PermissionError):
            self.store.fetch(self.token, resource)
        self.token = auth_store.authenticate("administrator", self.password, db_path=self.db)
        resource = self.report()
        auth_store.reset_password("administrator", "ReplacementTestPassword2026!", db_path=self.db)
        with self.assertRaises(PermissionError):
            self.store.fetch(self.token, resource)

    def test_csv_requires_admin_and_approval(self):
        with self.assertRaises(PermissionError):
            self.store.register(self.token, b"x\n1", "raw.csv", "text/csv", kind="csv")
        resource = self.store.register(self.token, b"x\n1", "raw.csv", "text/csv", kind="csv", approved=True)
        self.assertEqual(self.store.fetch(self.token, resource)[0], b"x\n1")
        auth_store.create_user(self.token, "analyst_one", self.password, db_path=self.db)
        analyst = auth_store.authenticate("analyst_one", self.password, db_path=self.db)
        with self.assertRaises(PermissionError):
            self.store.register(analyst, b"x\n1", "raw.csv", "text/csv", kind="csv", approved=True)

    def test_disabled_account_cannot_download(self):
        user = auth_store.create_user(self.token, "analyst_one", self.password, db_path=self.db)
        analyst = auth_store.authenticate("analyst_one", self.password, db_path=self.db)
        resource = self.report(analyst)
        auth_store.set_user_enabled(self.token, user["id"], False, db_path=self.db)
        with self.assertRaises(PermissionError):
            self.store.fetch(analyst, resource)

    def test_expired_and_unknown_resources_raise_keyerror(self):
        self.store = self.module.DownloadStore(db_path=self.db, ttl_seconds=0.01)
        resource = self.report()
        time.sleep(0.03)
        for resource in (resource, "a" * 32):
            with self.assertRaises(KeyError):
                self.store.fetch(self.token, resource)

    def test_revoke_removes_only_its_session_resources(self):
        first = self.report()
        second_token = auth_store.authenticate("administrator", self.password, db_path=self.db)
        second = self.report(second_token)
        self.store.revoke(self.token)
        with self.assertRaises(KeyError):
            self.store.fetch(self.token, first)
        self.assertEqual(self.store.fetch(second_token, second)[0], b"private-report")

    def test_size_limit_rejects_oversize_and_allows_boundary(self):
        self.store = self.module.DownloadStore(db_path=self.db, max_bytes=3)
        resource = self.store.register(self.token, b"123", "a.md", "text/markdown")
        self.assertEqual(self.store.fetch(self.token, resource)[0], b"123")
        with self.assertRaises(ValueError):
            self.store.register(self.token, b"1234", "a.md", "text/markdown")

    def test_metadata_and_content_validation_rejects_header_injection_and_csv_disguise(self):
        for content, name, mime, kind in [(b"x", "a\r\nInjected: yes", "text/markdown", "report"),
                                          (b"x", "../report.md", "text/markdown", "report"),
                                          ("not bytes", "report.md", "text/markdown", "report"),
                                          (b"x", "raw.csv", "text/csv", "report"),
                                          (b"x", "report.md", "text/html", "report"),
                                          (b"x", "report.md", "text/markdown", "unknown")]:
            with self.subTest(name=name, mime=mime, kind=kind):
                with self.assertRaises(ValueError):
                    self.store.register(self.token, content, name, mime, kind=kind)

    def test_csv_truthy_approval_is_not_explicit_approval(self):
        for approved in (1, "true", [True]):
            with self.subTest(approved=approved):
                with self.assertRaises(PermissionError):
                    self.store.register(self.token, b"x", "raw.csv", "text/csv", kind="csv", approved=approved)

    def test_http_requires_post_bearer_and_exact_origin_and_never_query_token(self):
        resource = self.report()
        server = self.module.start_server(self.store, port=0, origins={"http://localhost:8501"})
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        # start_server owns its serving thread.
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        self.addCleanup(connection.close)
        endpoint = "/download/" + resource
        valid = {"Authorization": "Bearer " + self.token, "Origin": "http://localhost:8501"}
        cases = [("GET", endpoint, valid), ("POST", endpoint, {}),
                 ("POST", endpoint, {"Origin": valid["Origin"]}),
                 ("POST", endpoint, {**valid, "Origin": "http://localhost:8501.attacker.invalid"}),
                 ("POST", endpoint, {**valid, "Origin": "null"}),
                 ("POST", endpoint, {"Authorization": valid["Authorization"]}),
                 ("POST", endpoint + "?token=" + self.token, valid),
                 ("POST", endpoint + "?token=" + self.token, {"Origin": valid["Origin"]})]
        for method, path, headers in cases:
            with self.subTest(method=method, case=cases.index((method, path, headers))):
                connection.request(method, path, headers=headers)
                response = connection.getresponse()
                body = response.read()
                self.assertGreaterEqual(response.status, 400)
                self.assertNotIn(b"private-report", body)
                self.assertNotIn(self.token.encode(), body)
        connection.request("OPTIONS", endpoint, headers={"Origin": valid["Origin"],
                           "Access-Control-Request-Method": "POST",
                           "Access-Control-Request-Headers": "authorization"})
        response = connection.getresponse()
        self.assertEqual(response.status, 204)
        self.assertEqual(response.read(), b"")
        self.assertEqual(response.getheader("Access-Control-Allow-Origin"), valid["Origin"])
        self.assertIn("POST", response.getheader("Access-Control-Allow-Methods", ""))
        self.assertIn("authorization", response.getheader("Access-Control-Allow-Headers", "").lower())
        connection.request("POST", endpoint, headers=valid)
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), b"private-report")
        self.assertIn("no-store", response.getheader("Cache-Control", ""))
        self.assertIn("attachment", response.getheader("Content-Disposition", ""))
        self.assertEqual(response.getheader("Access-Control-Allow-Origin"), valid["Origin"])
        auth_store.logout(self.token, db_path=self.db)
        connection.request("POST", endpoint, headers=valid)
        response = connection.getresponse()
        self.assertGreaterEqual(response.status, 400)
        self.assertNotIn(b"private-report", response.read())


if __name__ == "__main__":
    unittest.main()

