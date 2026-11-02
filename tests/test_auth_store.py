"""账户数据库集成测试；不连接外部服务。"""
import importlib
import importlib.util
import hashlib
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path


PASSWORD = "CorrectPassphrase2026!"


class AuthStoreTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "accounts.sqlite"
        self.assertIsNotNone(importlib.util.find_spec("auth_store"), "尚未实现账户模块")
        self.store = importlib.import_module("auth_store")

    def admin(self):
        user = self.store.bootstrap_admin("administrator", PASSWORD, db_path=self.db)
        token = self.store.authenticate("administrator", PASSWORD, db_path=self.db)
        return user, token

    def test_bootstrap_once_and_safe_user_and_password_storage(self):
        self.assertFalse(self.store.has_users(db_path=self.db))
        user, token = self.admin()
        self.assertTrue(self.store.has_users(db_path=self.db))
        self.assertEqual(set(user), {"id", "username", "role", "enabled"})
        self.assertEqual(user["role"], "admin")
        self.assertEqual(len(user["id"]), 32)
        self.assertIsNotNone(token)
        self.assertNotIn(PASSWORD.encode(), self.db.read_bytes())
        with self.assertRaises((ValueError, PermissionError)):
            self.store.bootstrap_admin("second_admin", PASSWORD, db_path=self.db)

    def test_validation_rejects_invalid_username_and_password_lengths(self):
        for username, password in [("ab", PASSWORD), ("../admin", PASSWORD), ("Admin", PASSWORD),
                                   ("administrator", "short"), ("administrator", "x" * 129)]:
            with self.subTest(username=username, length=len(password)):
                with self.assertRaises(ValueError):
                    self.store.bootstrap_admin(username, password, db_path=self.db)
        self.assertFalse(self.store.has_users(db_path=self.db))

    def test_session_expires_after_eight_hours_and_database_does_not_store_token(self):
        self.store.bootstrap_admin("administrator", PASSWORD, db_path=self.db)
        token = self.store.authenticate("administrator", PASSWORD, db_path=self.db, now=1000)
        self.assertEqual(self.store.resolve_session(token, db_path=self.db, now=1001)["username"], "administrator")
        self.assertIsNone(self.store.resolve_session(token, db_path=self.db, now=1000 + 8 * 3600))
        self.assertNotIn(token.encode(), self.db.read_bytes())

    def test_logout_and_password_reset_revoke_sessions(self):
        _, first = self.admin()
        self.store.logout(first, db_path=self.db)
        self.assertIsNone(self.store.resolve_session(first, db_path=self.db))
        second = self.store.authenticate("administrator", PASSWORD, db_path=self.db)
        self.store.reset_password("administrator", "ReplacementPass2026!", db_path=self.db)
        self.assertIsNone(self.store.resolve_session(second, db_path=self.db))
        self.assertIsNone(self.store.authenticate("administrator", PASSWORD, db_path=self.db))
        self.assertIsNotNone(self.store.authenticate("administrator", "ReplacementPass2026!", db_path=self.db))

    def test_five_failures_lock_for_sixty_seconds(self):
        self.store.bootstrap_admin("administrator", PASSWORD, db_path=self.db)
        for _ in range(5):
            self.assertIsNone(self.store.authenticate("administrator", "IncorrectPass2026!", db_path=self.db, now=1000))
        self.assertIsNone(self.store.authenticate("administrator", PASSWORD, db_path=self.db, now=1059))
        self.assertIsNotNone(self.store.authenticate("administrator", PASSWORD, db_path=self.db, now=1061))

    def test_analyst_cannot_manage_accounts_and_disable_revokes_sessions(self):
        admin, admin_token = self.admin()
        user = self.store.create_user(admin_token, "analyst_one", PASSWORD, db_path=self.db)
        self.assertEqual(user["role"], "analyst")
        analyst_token = self.store.authenticate("analyst_one", PASSWORD, db_path=self.db)
        for operation in [lambda: self.store.list_users(analyst_token, db_path=self.db),
                          lambda: self.store.create_user(analyst_token, "analyst_two", PASSWORD, db_path=self.db),
                          lambda: self.store.set_user_enabled(analyst_token, admin["id"], False, db_path=self.db)]:
            with self.assertRaises(PermissionError):
                operation()
        self.store.set_user_enabled(admin_token, user["id"], False, db_path=self.db)
        self.assertIsNone(self.store.resolve_session(analyst_token, db_path=self.db))
        self.assertIsNone(self.store.authenticate("analyst_one", PASSWORD, db_path=self.db))
        self.store.set_user_enabled(admin_token, user["id"], True, db_path=self.db)
        self.assertIsNotNone(self.store.authenticate("analyst_one", PASSWORD, db_path=self.db))

    def test_last_admin_cannot_be_disabled_and_invalid_roles_rejected(self):
        admin, token = self.admin()
        with self.assertRaises(ValueError):
            self.store.set_user_enabled(token, admin["id"], False, db_path=self.db)
        with self.assertRaises(ValueError):
            self.store.create_user(token, "another_user", PASSWORD, role="superuser", db_path=self.db)
        second = self.store.create_user(token, "second_admin", PASSWORD, role="admin", db_path=self.db)
        self.store.set_user_enabled(token, second["id"], False, db_path=self.db)
        self.assertEqual(len(self.store.list_users(token, db_path=self.db)), 2)

    def test_unknown_credentials_and_forged_tokens_do_not_authenticate(self):
        self.admin()
        self.assertIsNone(self.store.authenticate("missing_user", PASSWORD, db_path=self.db))
        self.assertIsNone(self.store.resolve_session("forged-token", db_path=self.db))
        with self.assertRaises(PermissionError):
            self.store.create_user("forged-token", "another_user", PASSWORD, db_path=self.db)

    def test_same_password_uses_independent_salts_and_pbkdf2_sha256_600000(self):
        _, token = self.admin()
        self.store.create_user(token, "same_password", PASSWORD, db_path=self.db)
        with closing(sqlite3.connect(self.db)) as connection:
            rows = connection.execute("SELECT password_salt, password_hash FROM users ORDER BY username").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0][0], rows[1][0])
        self.assertNotEqual(rows[0][1], rows[1][1])
        for salt, digest in rows:
            self.assertGreaterEqual(len(salt), 16)
            self.assertEqual(digest, hashlib.pbkdf2_hmac("sha256", PASSWORD.encode(), salt, 600000))


if __name__ == "__main__":
    unittest.main()

