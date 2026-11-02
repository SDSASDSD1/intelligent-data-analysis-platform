"""本机待审核元数据与 LangGraph SQLite 检查点。"""

import json
import os
import re
import sqlite3
from copy import copy
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

import trace_store
from secure_storage import StorageError, encrypted_serializer, seal_json, unseal_json


DEFAULT_REVIEW_DIR = Path(__file__).resolve().parent / "logs" / "pending"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "logs" / "checkpoints.sqlite"
RECORD_FIELDS = (
    "trace_id", "thread_id", "fingerprint", "sales_column", "date_column",
    "sensitive_columns", "plan", "question", "actions",
)
TRACE_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


class _EncryptedSqliteSaver(SqliteSaver):
    def with_allowlist(self, extra_allowlist):
        """LangGraph 扩充可信类型时，继续保留禁止明文回退的序列化器。"""
        clone = copy(self)
        clone.serde = type(self.serde)(
            self.serde.cipher, self.serde.serde.with_msgpack_allowlist(extra_allowlist)
        )
        return clone


def _database_path(review_dir=None) -> Path:
    directory = Path(review_dir) if review_dir is not None else DEFAULT_REVIEW_DIR
    return directory / "pending.sqlite"


def _connect_pending(review_dir=None, create=False):
    path = _database_path(review_dir)
    if not create and not path.is_file():
        return None
    if create:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            path.parent.chmod(0o700)
    connection = sqlite3.connect(path, timeout=10)
    if create:
        if os.name == "posix":
            path.chmod(0o600)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS pending ("
            "trace_id TEXT PRIMARY KEY, review_id TEXT, payload TEXT NOT NULL)"
        )
    return connection


def _redact_plan_or_actions(items, sensitive_values, name_key):
    """保留工具名和列名，只遮挡可能来自数据单元格的商品名。"""
    return [
        {
            name_key: item[name_key],
            "args": {
                key: trace_store.redact(value, sensitive_values) if key == "product_name" else value
                for key, value in item["args"].items()
            },
        }
        for item in items
    ]


def save_pending(record: dict, review_dir=None, sensitive_values=(), cipher=None) -> None:
    trace_id = record.get("trace_id")
    if not isinstance(trace_id, str) or not TRACE_ID_PATTERN.fullmatch(trace_id):
        raise ValueError("无效的 trace_id。")
    payload = {key: record[key] for key in RECORD_FIELDS}
    if "review_id" in record:
        payload["review_id"] = record["review_id"]
    if "owner_id" in record:
        payload["owner_id"] = record["owner_id"]
    payload["question"] = trace_store.redact(payload["question"], sensitive_values)
    payload["plan"] = {
        "steps": _redact_plan_or_actions(payload["plan"]["steps"], sensitive_values, "tool")
    }
    payload["actions"] = _redact_plan_or_actions(payload["actions"], sensitive_values, "name")
    stored_payload = seal_json(payload, cipher) if cipher is not None else payload
    connection = _connect_pending(review_dir, create=True)
    try:
        with connection:
            connection.execute(
                "INSERT INTO pending (trace_id, review_id, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(trace_id) DO UPDATE SET "
                "review_id=excluded.review_id, payload=excluded.payload",
                (trace_id, payload.get("review_id"), json.dumps(stored_payload, ensure_ascii=False)),
            )
    finally:
        connection.close()


def _decode_payload(payload, cipher):
    try:
        record = json.loads(payload)
    except (json.JSONDecodeError, UnicodeError) as error:
        if cipher is not None:
            raise StorageError("加密待办格式错误，已停止读取。") from error
        raise
    if cipher is not None:
        return unseal_json(record, cipher)
    if isinstance(record, dict) and record.get("cipher") == "fernet":
        raise StorageError("读取加密待办需要存储密钥。")
    return record


def load_pending(trace_id: str, review_dir=None, cipher=None) -> dict | None:
    if not isinstance(trace_id, str) or not TRACE_ID_PATTERN.fullmatch(trace_id):
        return None
    connection = _connect_pending(review_dir)
    if connection is None:
        return None
    try:
        row = connection.execute("SELECT payload FROM pending WHERE trace_id=?", (trace_id,)).fetchone()
        return _decode_payload(row[0], cipher) if row else None
    finally:
        connection.close()


def take_pending(trace_id: str, review_dir=None, expected_review_id=None, cipher=None) -> dict | None:
    """只消费页面所显示版本的待办；其他版本留在库中。"""
    if not isinstance(trace_id, str) or not TRACE_ID_PATTERN.fullmatch(trace_id):
        return None
    connection = _connect_pending(review_dir)
    if connection is None:
        return None
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT review_id, payload FROM pending WHERE trace_id=?", (trace_id,)
            ).fetchone()
            if row is None or (expected_review_id is not None and row[0] != expected_review_id):
                return None
            record = _decode_payload(row[1], cipher)
            connection.execute("DELETE FROM pending WHERE trace_id=?", (trace_id,))
            return record
    finally:
        connection.close()


def matches_context(record: dict, fingerprint: str, sales_column: str,
                    date_column: str | None, sensitive_columns) -> bool:
    return (
        record.get("fingerprint") == fingerprint
        and record.get("sales_column") == sales_column
        and record.get("date_column") == date_column
        and sorted(record.get("sensitive_columns", [])) == sorted(sensitive_columns)
    )


def open_checkpointer(db_path=None, cipher=None) -> SqliteSaver:
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    path = Path(db_path) if db_path is not None else DEFAULT_CHECKPOINT_PATH
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        path.parent.chmod(0o700)
    connection = sqlite3.connect(path, check_same_thread=False)
    if os.name == "posix":
        path.chmod(0o600)
    if cipher is not None:
        return _EncryptedSqliteSaver(connection, serde=encrypted_serializer(cipher))
    return SqliteSaver(connection)

