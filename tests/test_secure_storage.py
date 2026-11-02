"""密钥单元测试与加密文件/SQLite 集成测试。"""
import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import review_store
import trace_store


class SecureStorageTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.key_path = self.directory / "outside" / "storage.key"
        self.assertIsNotNone(importlib.util.find_spec("secure_storage"), "尚未实现加密存储模块")
        self.store = importlib.import_module("secure_storage")
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("SHUXI_STORAGE_KEY", None)

    def cipher(self):
        self.store.initialize_key(self.key_path)
        return self.store.load_cipher(self.key_path)

    def record(self):
        return {"trace_id": "a" * 32, "thread_id": "thread", "fingerprint": "b" * 64,
                "sales_column": "销售额", "date_column": None, "sensitive_columns": [],
                "question": "PRIVATE_STORAGE_MARKER", "plan": {"steps": [{"tool": "full_csv_export", "args": {}}]},
                "actions": [{"name": "full_csv_export", "args": {}}]}

    def test_key_initialization_never_overwrites_existing_key(self):
        self.store.initialize_key(self.key_path)
        original = self.key_path.read_bytes()
        try:
            self.store.initialize_key(self.key_path)
        except (FileExistsError, self.store.StorageError):
            pass
        self.assertEqual(self.key_path.read_bytes(), original)
        cipher = self.store.load_cipher(self.key_path)
        self.assertEqual(cipher.decrypt(cipher.encrypt(b"hello")), b"hello")

    def test_missing_bad_or_wrong_keys_fail_without_plaintext_fallback(self):
        with self.assertRaises(self.store.StorageError):
            self.store.load_cipher(self.key_path)
        self.key_path.parent.mkdir()
        self.key_path.write_bytes(b"not-a-fernet-key")
        with self.assertRaises(self.store.StorageError):
            self.store.load_cipher(self.key_path)
        os.environ["SHUXI_STORAGE_KEY"] = "also-invalid"
        with self.assertRaises(self.store.StorageError):
            self.store.load_cipher(self.key_path)

    def test_environment_key_overrides_file_and_user_paths_are_distinct(self):
        cipher = self.cipher()
        alternate_path = self.directory / "alternate.key"
        self.store.initialize_key(alternate_path)
        os.environ["SHUXI_STORAGE_KEY"] = alternate_path.read_text().strip()
        overridden = self.store.load_cipher(self.key_path)
        encrypted = overridden.encrypt(b"override")
        self.assertEqual(self.store.load_cipher(alternate_path).decrypt(encrypted), b"override")
        with self.assertRaises(Exception):
            cipher.decrypt(encrypted)
        first = self.store.workspace_for_user("a" * 32, self.directory / "users", self.key_path)
        second = self.store.workspace_for_user("b" * 32, self.directory / "users", self.key_path)
        self.assertEqual(first.user_id, "a" * 32)
        for attribute in ["log_dir", "review_dir", "checkpoint_path"]:
            self.assertNotEqual(getattr(first, attribute), getattr(second, attribute))
            self.assertTrue(Path(getattr(first, attribute)).is_relative_to(self.directory / "users"))
        for invalid in ["../outside", "A" * 32, "short"]:
            with self.assertRaises(ValueError):
                self.store.workspace_for_user(invalid, self.directory / "users", self.key_path)

    def test_checkpoint_allowlist_clone_still_rejects_unencrypted_payload(self):
        cipher = self.cipher()
        saver = review_store.open_checkpointer(self.directory / "allowlist.sqlite", cipher=cipher)
        try:
            derived = saver.with_allowlist([("langchain_core", "messages", "ai", "AIMessage")])
            self.assertEqual(derived.serde.loads_typed(derived.serde.dumps_typed({"safe": True})), {"safe": True})
            with self.assertRaises(self.store.StorageError):
                derived.serde.loads_typed(("json", b'{"UNENCRYPTED_MARKER": true}'))
        finally:
            saver.conn.close()

    def test_trace_ciphertext_round_trip_and_wrong_key_and_tamper_rejected(self):
        cipher = self.cipher()
        directory = self.directory / "logs"
        trace_store.append_trace({"trace_id": "a" * 32, "query": "PRIVATE_STORAGE_MARKER"}, directory, cipher=cipher)
        trace_store.append_audit({"trace_id": "a" * 32, "decision": "PRIVATE_STORAGE_MARKER"}, directory, cipher=cipher)
        self.assertEqual(trace_store.read_trace("a" * 32, directory, cipher=cipher)[0]["query"], "PRIVATE_STORAGE_MARKER")
        for path in directory.iterdir():
            self.assertNotIn(b"PRIVATE_STORAGE_MARKER", path.read_bytes())
        other = self.directory / "other.key"
        self.store.initialize_key(other)
        with self.assertRaises(self.store.StorageError):
            trace_store.read_trace("a" * 32, directory, cipher=self.store.load_cipher(other))
        path = directory / "trace.enc.jsonl"
        text = path.read_bytes()
        path.write_bytes(text[:30] + b"X" + text[31:])
        with self.assertRaises(self.store.StorageError):
            trace_store.read_trace("a" * 32, directory, cipher=cipher)

    def test_review_ciphertext_round_trip_owner_isolation_and_wrong_key(self):
        cipher = self.cipher()
        first = self.directory / "user_a"
        second = self.directory / "user_b"
        record = self.record()
        review_store.save_pending(record, first, cipher=cipher)
        self.assertEqual(review_store.load_pending(record["trace_id"], first, cipher=cipher), record)
        self.assertIsNone(review_store.load_pending(record["trace_id"], second, cipher=cipher))
        self.assertIsNone(review_store.take_pending(record["trace_id"], second, cipher=cipher))
        self.assertNotIn(b"PRIVATE_STORAGE_MARKER", (first / "pending.sqlite").read_bytes())
        other = self.directory / "other.key"
        self.store.initialize_key(other)
        with self.assertRaises(self.store.StorageError):
            review_store.load_pending(record["trace_id"], first, cipher=self.store.load_cipher(other))
        self.assertEqual(review_store.take_pending(record["trace_id"], first, cipher=cipher), record)
        self.assertIsNone(review_store.take_pending(record["trace_id"], first, cipher=cipher))

    def test_checkpoint_payload_encrypted_and_resumable_with_original_key(self):
        from langgraph.graph import StateGraph, START, END
        from langgraph.types import interrupt, Command
        from typing import TypedDict

        class State(TypedDict):
            question: str
            answer: str

        def await_review(state):
            approved = interrupt({"question": state["question"]})
            return {"answer": "approved" if approved else "rejected"}

        cipher = self.cipher()
        db = self.directory / "checkpoint.sqlite"
        builder = StateGraph(State)
        builder.add_node("review", await_review)
        builder.add_edge(START, "review")
        builder.add_edge("review", END)
        config = {"configurable": {"thread_id": "persistent-encrypted-thread"}}
        saver = review_store.open_checkpointer(db, cipher=cipher)
        try:
            result = builder.compile(checkpointer=saver).invoke({"question": "PRIVATE_CHECKPOINT_MARKER"}, config)
            self.assertTrue(result.get("__interrupt__"))
        finally:
            saver.conn.close()

        self.assertNotIn(b"PRIVATE_CHECKPOINT_MARKER", db.read_bytes())
        saver = review_store.open_checkpointer(db, cipher=self.store.load_cipher(self.key_path))
        try:
            result = builder.compile(checkpointer=saver).invoke(Command(resume=True), config)
            self.assertEqual(result["answer"], "approved")
        finally:
            saver.conn.close()
        other = self.directory / "other.key"
        self.store.initialize_key(other)
        saver = review_store.open_checkpointer(db, cipher=self.store.load_cipher(other))
        try:
            with self.assertRaises(self.store.StorageError):
                saver.get(config)
        finally:
            saver.conn.close()

    def test_encrypted_agent_review_resumes_in_fresh_process_without_raw_checkpoint_marker(self):
        self.cipher()
        script = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "tests"))
import pandas as pd
from langchain_core.messages import AIMessage
from langgraph.types import Command
from test_analysis_agent import ScriptedChatModel
import analysis_agent, review_store, secure_storage
phase, db_path, key_path = sys.argv[1:]
data = pd.DataFrame({"商品": ["A"], "销售额": [20], "邮箱": ["NEVER_SEND_RAW_ROW"]})
model = ScriptedChatModel(responses=([AIMessage(content="", tool_calls=[
    {"name": "full_csv_export", "args": {}, "id": "export"}])] if phase == "start"
    else [AIMessage(content="approved")]))
artifacts = {}
saver = review_store.open_checkpointer(db_path, cipher=secure_storage.load_cipher(key_path))
try:
    agent = analysis_agent.create_analysis_agent(model, data, data["销售额"], None, ["邮箱"], artifacts,
             enable_review=True, checkpointer=saver)
    config = {"configurable": {"thread_id": "encrypted-agent-thread"}, "recursion_limit": 20}
    payload = ({"messages": [("human", "PRIVATE_PROCESS_CHECKPOINT_MARKER")]} if phase == "start"
               else Command(resume={"decisions": [{"type": "approve"}]}))
    result = agent.invoke(payload, config)
    if phase == "start":
        assert result.get("__interrupt__") and not artifacts
    else:
        assert not result.get("__interrupt__") and "full_csv" in artifacts
        assert result["messages"][-1].content == "approved"
    assert "NEVER_SEND_RAW_ROW" not in str(model.requests)
    print(json.dumps({"ok": True}))
finally:
    saver.conn.close()
'''
        db_path = self.directory / "process.sqlite"
        for phase in ["start", "resume"]:
            result = subprocess.run([sys.executable, "-c", script, phase, str(db_path), str(self.key_path)],
                                    cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                    text=True, encoding="utf-8", timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"ok": true', result.stdout)
            for file in self.directory.glob("process.sqlite*"):
                self.assertNotIn(b"PRIVATE_PROCESS_CHECKPOINT_MARKER", file.read_bytes())
                self.assertNotIn(b"NEVER_SEND_RAW_ROW", file.read_bytes())

if __name__ == "__main__":
    unittest.main()

