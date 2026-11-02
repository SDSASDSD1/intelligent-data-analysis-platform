"""临时 SQLite、Fernet、ZIP 和跨进程运行锁集成测试。"""
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch
from typing import TypedDict

import auth_store
import secure_storage
import trace_store
import review_store
from langgraph.graph import StateGraph, START, END


class CheckpointState(TypedDict):
    answer: str


class BackupStoreTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "private"
        self.data.mkdir()
        self.key = self.root / "outside.key"
        secure_storage.initialize_key(self.key)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("SHUXI_STORAGE_KEY", None)
        self.assertIsNotNone(importlib.util.find_spec("backup_store"), "尚未实现备份模块")
        self.backup = importlib.import_module("backup_store")
        self.runtime = importlib.import_module("local_runtime")
        self.db = self.data / "accounts.sqlite"
        password = "BackupTestPassword2026!"
        self.user = auth_store.bootstrap_admin("backup_admin", password, db_path=self.db)
        self.token = auth_store.authenticate("backup_admin", password, db_path=self.db)
        workspace = secure_storage.workspace_for_user(self.user["id"], self.data, self.key)
        self.log = workspace.log_dir / "trace.enc.jsonl"
        trace_store.append_trace({"trace_id": "a" * 32, "event": "temporary-private-log"},
                                 workspace.log_dir, cipher=workspace.cipher)
        self.pending = {"trace_id": "a" * 32, "thread_id": "backup-thread", "fingerprint": "b" * 64,
                        "sales_column": "sales", "date_column": None, "sensitive_columns": [],
                        "question": "temporary-private-pending", "plan": {"steps": []}, "actions": []}
        review_store.save_pending(self.pending, workspace.review_dir, cipher=workspace.cipher)
        self.pending_path = workspace.review_dir / "pending.sqlite"
        self.checkpoint = workspace.checkpoint_path
        self.config = {"configurable": {"thread_id": "backup-checkpoint-thread"}}
        graph = StateGraph(CheckpointState)
        graph.add_node("answer", lambda state: {"answer": "temporary-private-checkpoint"})
        graph.add_edge(START, "answer")
        graph.add_edge("answer", END)
        saver = review_store.open_checkpointer(self.checkpoint, cipher=workspace.cipher)
        try:
            graph.compile(checkpointer=saver).invoke({"answer": "initial"}, self.config)
        finally:
            saver.conn.close()
        self.archive = self.root / "backup.zip"
        self.destination = self.root / "restored"

    def create(self):
        return self.backup.create_backup(self.archive, data_dir=self.data, key_path=self.key)

    def restore(self, archive=None, key=None):
        return self.backup.restore_backup(archive or self.archive, destination=self.destination, key_path=key or self.key)

    def test_authenticated_manifest_hashes_and_roundtrip_without_sessions_or_keys(self):
        (self.data / "storage.key").write_bytes(self.key.read_bytes())
        (self.data / "unrelated.txt").write_text("exclude")
        self.assertEqual(self.create(), self.archive)
        with zipfile.ZipFile(self.archive) as archive:
            names = archive.namelist()
            self.assertIn("manifest.enc", names)
            self.assertNotIn("storage.key", names)
            self.assertNotIn("unrelated.txt", names)
            manifest = json.loads(secure_storage.load_cipher(self.key).decrypt(archive.read("manifest.enc")))
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(set(manifest["files"]), set(names) - {"manifest.enc"})
            for name, digest in manifest["files"].items():
                self.assertEqual(digest, hashlib.sha256(archive.read(name)).hexdigest())
        self.assertEqual(self.restore(), self.destination)
        self.assertEqual((self.destination / self.log.relative_to(self.data)).read_bytes(), self.log.read_bytes())
        with closing(sqlite3.connect(self.destination / "accounts.sqlite")) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 0)
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertIsNotNone(auth_store.resolve_session(self.token, db_path=self.db))
        self.assertFalse((self.destination / "storage.key").exists())
        workspace = secure_storage.workspace_for_user(self.user["id"], self.destination, self.key)
        self.assertEqual(review_store.load_pending("a" * 32, workspace.review_dir, cipher=workspace.cipher), self.pending)
        self.assertEqual(trace_store.read_trace("a" * 32, workspace.log_dir, cipher=workspace.cipher)[0]["event"],
                         "temporary-private-log")
        saver = review_store.open_checkpointer(workspace.checkpoint_path, cipher=workspace.cipher)
        try:
            self.assertEqual(saver.get(self.config)["channel_values"]["answer"], "temporary-private-checkpoint")
        finally:
            saver.conn.close()

    def test_sqlite_wal_committed_rows_are_in_consistent_snapshot(self):
        connection = sqlite3.connect(self.db)
        self.addCleanup(connection.close)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE snapshot_marker (value TEXT)")
        connection.execute("INSERT INTO snapshot_marker VALUES ('committed')")
        connection.commit()
        self.create()
        self.restore()
        with closing(sqlite3.connect(self.destination / "accounts.sqlite")) as restored:
            self.assertEqual(restored.execute("SELECT value FROM snapshot_marker").fetchall(), [("committed",)])

    def test_wrong_key_rejected_without_leaving_destination(self):
        self.create()
        wrong = self.root / "wrong.key"
        secure_storage.initialize_key(wrong)
        with self.assertRaises(self.backup.BackupError):
            self.restore(key=wrong)
        self.assertFalse(self.destination.exists())

    def test_wrong_key_at_create_is_rejected_without_archive(self):
        wrong = self.root / "wrong.key"
        secure_storage.initialize_key(wrong)
        with self.assertRaises(self.backup.BackupError):
            self.backup.create_backup(self.archive, data_dir=self.data, key_path=wrong)
        self.assertFalse(self.archive.exists())

    def test_wrong_key_cannot_back_up_each_encrypted_container_independently(self):
        wrong = self.root / "wrong.key"
        secure_storage.initialize_key(wrong)
        for kind, path in [("log", self.log), ("pending", self.pending_path),
                           ("checkpoint", self.checkpoint), ("writes", self.checkpoint)]:
            with self.subTest(container=kind):
                isolated = self.root / kind
                isolated.mkdir()
                shutil.copyfile(self.db, isolated / "accounts.sqlite")
                target = isolated / path.relative_to(self.data)
                target.parent.mkdir(parents=True)
                shutil.copyfile(path, target)
                if kind == "writes":
                    with closing(sqlite3.connect(target)) as connection:
                        self.assertGreater(connection.execute("SELECT COUNT(*) FROM writes").fetchone()[0], 0)
                        connection.execute("DELETE FROM checkpoints")
                        connection.commit()
                archive = self.root / (kind + ".zip")
                with self.assertRaises(self.backup.BackupError):
                    self.backup.create_backup(archive, data_dir=isolated, key_path=wrong)
                self.assertFalse(archive.exists())

    def test_file_tamper_rejected_without_leaving_destination(self):
        self.create()
        modified = self.root / "tampered.zip"
        with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(modified, "w") as output:
            for name in source.namelist():
                content = source.read(name)
                output.writestr(name, content + b"tamper" if name == "accounts.sqlite" else content)
        with self.assertRaises(self.backup.BackupError):
            self.restore(modified)
        self.assertFalse(self.destination.exists())

    def test_traversal_and_extra_members_rejected_without_writes(self):
        self.create()
        for name in ("../escaped.txt", "users/../../escaped.txt", "C:/escaped.txt", "unlisted.txt"):
            with self.subTest(member=name):
                modified = self.root / "unsafe.zip"
                with zipfile.ZipFile(self.archive) as source, zipfile.ZipFile(modified, "w") as output:
                    for member in source.namelist():
                        output.writestr(member, source.read(member))
                    output.writestr(name, b"attacker")
                with self.assertRaises(self.backup.BackupError):
                    self.restore(modified)
                self.assertFalse(self.destination.exists())
                self.assertFalse((self.root / "escaped.txt").exists())

    def test_existing_nonempty_destination_preserved(self):
        self.create()
        self.destination.mkdir()
        sentinel = self.destination / "keep.txt"
        sentinel.write_bytes(b"preserve")
        with self.assertRaises((self.backup.BackupError, FileExistsError)):
            self.restore()
        self.assertEqual(sentinel.read_bytes(), b"preserve")
        self.assertEqual(list(self.destination.iterdir()), [sentinel])

    def test_active_runtime_denies_backup_and_restore(self):
        self.create()
        with self.runtime.DataLock(self.data):
            with self.assertRaises(RuntimeError):
                self.backup.create_backup(self.root / "active.zip", data_dir=self.data, key_path=self.key)
        self.assertFalse((self.root / "active.zip").exists())
        self.destination.mkdir()
        with self.runtime.DataLock(self.destination):
            with self.assertRaises((RuntimeError, self.backup.BackupError, FileExistsError)):
                self.restore()
        self.assertFalse((self.destination / "accounts.sqlite").exists())

    def test_lock_is_exclusive_across_processes_and_released_on_exit(self):
        script = """import sys
from local_runtime import DataLock
try:
    with DataLock(sys.argv[1]):
        print('acquired')
except RuntimeError:
    print('locked')
"""
        def run():
            result = subprocess.run([sys.executable, "-c", script, str(self.data)], capture_output=True,
                                    text=True, timeout=15, cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout.strip()
        with self.runtime.DataLock(self.data):
            self.assertEqual(run(), "locked")
        self.assertEqual(run(), "acquired")

    def test_app_lock_survives_reruns_and_is_released_when_process_exits(self):
        script = """import sys
from pathlib import Path
import local_runtime
local_runtime.DEFAULT_DATA_DIR = Path(sys.argv[1])
first = local_runtime.ensure_app_lock()
assert local_runtime.ensure_app_lock() is first
try:
    with local_runtime.DataLock(sys.argv[1]):
        raise AssertionError('app lock was not held')
except RuntimeError:
    print('held')
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.data)], capture_output=True,
                                text=True, timeout=15, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "held")
        with self.runtime.DataLock(self.data):
            pass


if __name__ == "__main__":
    unittest.main()

