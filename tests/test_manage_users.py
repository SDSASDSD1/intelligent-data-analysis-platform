"""可信本机 CLI 与真实账户库/密钥集成测试；隐藏输入用固定测试密码。"""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import auth_store
import manage_users
import secure_storage


PASSWORD = "LocalCliPassword2026!"


class ManageUsersTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.key = self.directory / "outside" / "storage.key"
        for replacement in [patch.object(auth_store, "DEFAULT_DB_PATH", self.directory / "accounts.sqlite"),
                            patch.object(secure_storage, "DEFAULT_KEY_PATH", self.key),
                            patch.object(secure_storage, "DEFAULT_DATA_DIR", self.directory / "private"),
                            patch.dict(os.environ)]:
            replacement.start()
            self.addCleanup(replacement.stop)
        os.environ.pop("SHUXI_STORAGE_KEY", None)

    def call(self, arguments, passwords):
        output = io.StringIO()
        with patch("manage_users.getpass.getpass", side_effect=passwords), contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = manage_users.main(arguments)
        self.assertNotIn(PASSWORD, output.getvalue())
        return result

    def test_init_creates_admin_and_key_once_without_overwriting_key(self):
        secure_storage.initialize_key(self.key)
        original = self.key.read_bytes()
        self.assertEqual(self.call(["init", "--username", "cli_admin"], [PASSWORD, PASSWORD]), 0)
        self.assertEqual(self.key.read_bytes(), original)
        self.assertIsNotNone(auth_store.authenticate("cli_admin", PASSWORD))
        with self.assertRaises(SystemExit):
            self.call(["init", "--username", "another_admin"], [PASSWORD, PASSWORD])
        self.assertEqual(self.key.read_bytes(), original)

    def test_mismatched_passwords_create_neither_account_nor_key(self):
        with self.assertRaises(SystemExit):
            self.call(["init", "--username", "cli_admin"], [PASSWORD, "DifferentPass2026!"])
        self.assertFalse(auth_store.has_users())
        self.assertFalse(self.key.exists())

    def test_reset_password_revokes_existing_session(self):
        self.call(["init", "--username", "cli_admin"], [PASSWORD, PASSWORD])
        token = auth_store.authenticate("cli_admin", PASSWORD)
        new = "ReplacementCliPass2026!"
        self.assertEqual(self.call(["reset-password", "cli_admin"], [new, new]), 0)
        self.assertIsNone(auth_store.resolve_session(token))
        self.assertIsNotNone(auth_store.authenticate("cli_admin", new))

    def test_missing_key_with_existing_user_data_fails_without_creating_replacement(self):
        (secure_storage.DEFAULT_DATA_DIR / "users" / ("a" * 32)).mkdir(parents=True)
        with self.assertRaises(SystemExit):
            self.call(["init", "--username", "cli_admin"], [PASSWORD, PASSWORD])
        self.assertFalse(auth_store.has_users())
        self.assertFalse(self.key.exists())


if __name__ == "__main__":
    unittest.main()

