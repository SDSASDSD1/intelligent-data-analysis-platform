"""本机账号与会话：数据库只保存密码派生值和会话令牌摘要。"""

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "private_data" / "accounts.sqlite"
PASSWORD_ITERATIONS = 600_000
SESSION_SECONDS = 8 * 3600
USERNAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{2,31}\Z")


@contextmanager
def _database(db_path=None):
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'analyst')),
                enabled INTEGER NOT NULL DEFAULT 1,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at REAL NOT NULL
            );
        """)
        # 写事务先取得锁，使权限检查和随后的修改成为同一个操作。
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _validate_credentials(username, password):
    if not isinstance(username, str) or not USERNAME_PATTERN.fullmatch(username):
        raise ValueError("用户名须为 3 至 32 位小写字母、数字或下划线，并以字母开头。")
    if not isinstance(password, str) or not 12 <= len(password) <= 128:
        raise ValueError("密码须为 12 至 128 个字符。")


def _password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)


def _safe_user(row):
    return {"id": row["id"], "username": row["username"],
            "role": row["role"], "enabled": bool(row["enabled"])}


def _token_hash(token):
    if not isinstance(token, str) or not token or len(token) > 256:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _session_user(connection, token, now):
    digest = _token_hash(token)
    if digest is None:
        return None
    return connection.execute("""
        SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id
        WHERE token_hash = ? AND expires_at > ? AND users.enabled = 1
    """, (digest, now)).fetchone()


def _require_admin(connection, token):
    user = _session_user(connection, token, time.time())
    if user is None or user["role"] != "admin":
        raise PermissionError("只有已登录的管理员可以管理账号。")
    return user


def _insert_user(connection, username, password, role):
    _validate_credentials(username, password)
    if role not in ("admin", "analyst"):
        raise ValueError("角色只能为 admin 或 analyst。")
    salt = secrets.token_bytes(16)
    user_id = uuid.uuid4().hex
    try:
        connection.execute("""
            INSERT INTO users (id, username, password_salt, password_hash, role)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, username, salt, _password_hash(password, salt), role))
    except sqlite3.IntegrityError as error:
        raise ValueError("用户名已存在。") from error
    return {"id": user_id, "username": username, "role": role, "enabled": True}


def has_users(db_path=None):
    with _database(db_path) as connection:
        return connection.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def bootstrap_admin(username, password, db_path=None):
    """仅空数据库允许创建首个管理员；并发初始化也只能成功一次。"""
    _validate_credentials(username, password)
    with _database(db_path) as connection:
        if connection.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None:
            raise ValueError("账号库已经初始化，请登录后管理账号。")
        return _insert_user(connection, username, password, "admin")


def authenticate(username, password, db_path=None, now=None):
    """核对密码；成功时返回随机令牌，失败或暂时锁定时返回 None。"""
    if not isinstance(username, str) or not isinstance(password, str) or len(password) > 128:
        return None
    current = time.time() if now is None else now
    with _database(db_path) as connection:
        user = connection.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user is None:
            # 未知账号也执行一次密码派生，避免直接返回造成明显的耗时差异。
            _password_hash(password, b"\0" * 16)
            return None
        if not user["enabled"] or user["locked_until"] > current:
            return None
        correct = hmac.compare_digest(_password_hash(password, user["password_salt"]), user["password_hash"])
        if not correct:
            failures = user["failed_attempts"] + 1 if user["locked_until"] == 0 else 1
            locked_until = current + 60 if failures >= 5 else 0
            connection.execute("UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                               (failures, locked_until, user["id"]))
            return None
        connection.execute("UPDATE users SET failed_attempts = 0, locked_until = 0 WHERE id = ?", (user["id"],))
        token = secrets.token_urlsafe(32)
        connection.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                           (_token_hash(token), user["id"], current + SESSION_SECONDS))
        return token


def resolve_session(token, db_path=None, now=None):
    with _database(db_path) as connection:
        user = _session_user(connection, token, time.time() if now is None else now)
        return _safe_user(user) if user is not None else None


def logout(token, db_path=None):
    with _database(db_path) as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def create_user(actor_token, username, password, role="analyst", db_path=None):
    with _database(db_path) as connection:
        _require_admin(connection, actor_token)
        return _insert_user(connection, username, password, role)


def list_users(actor_token, db_path=None):
    with _database(db_path) as connection:
        _require_admin(connection, actor_token)
        return [_safe_user(row) for row in connection.execute("SELECT * FROM users ORDER BY username")]


def set_user_enabled(actor_token, target_id, enabled, db_path=None):
    with _database(db_path) as connection:
        _require_admin(connection, actor_token)
        if not isinstance(enabled, bool):
            raise ValueError("启用状态必须为 True 或 False。")
        user = connection.execute("SELECT * FROM users WHERE id = ?", (target_id,)).fetchone()
        if user is None:
            raise ValueError("账号不存在。")
        if not enabled and user["enabled"] and user["role"] == "admin":
            count = connection.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND enabled = 1").fetchone()[0]
            if count <= 1:
                raise ValueError("不能停用最后一个启用的管理员。")
        connection.execute("UPDATE users SET enabled = ? WHERE id = ?", (int(enabled), target_id))
        if not enabled:
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (target_id,))
        return {"id": user["id"], "username": user["username"], "role": user["role"], "enabled": enabled}


def reset_password(username, password, db_path=None):
    """可信的本机终端入口；更新密码时撤销该账号的全部会话。"""
    _validate_credentials(username, password)
    with _database(db_path) as connection:
        user = connection.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user is None:
            raise ValueError("账号不存在。")
        salt = secrets.token_bytes(16)
        connection.execute("""
            UPDATE users SET password_salt = ?, password_hash = ?, failed_attempts = 0, locked_until = 0
            WHERE id = ?
        """, (salt, _password_hash(password, salt), user["id"]))
        connection.execute("DELETE FROM sessions WHERE user_id = ?", (user["id"],))

