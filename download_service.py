"""只在内存保存下载内容，每次 HTTP 下载重新校验原登录会话。"""

import hashlib
import hmac
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote
from uuid import uuid4

import auth_store


@dataclass(frozen=True)
class _Artifact:
    owner_id: str
    session_hash: str
    content: bytes
    filename: str
    mime: str
    kind: str
    expires_at: float


class DownloadStore:
    def __init__(self, db_path=None, ttl_seconds=900, max_bytes=100 * 1024 * 1024):
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self._items = {}
        self._lock = threading.Lock()

    def _user(self, token):
        user = auth_store.resolve_session(token, db_path=self.db_path)
        if user is None:
            raise PermissionError("登录已失效，请重新登录。")
        return user

    def _prune(self):
        now = time.monotonic()
        self._items = {key: item for key, item in self._items.items()
                       if item.expires_at > now}

    def register(self, token, content, filename, mime, kind="report", approved=False):
        user = self._user(token)
        if kind not in ("report", "csv"):
            raise ValueError("不支持的下载类型。")
        if kind == "csv" and (user["role"] != "admin" or approved is not True):
            raise PermissionError("原始数据下载需要管理员审核批准。")
        expected_mime = "text/csv" if kind == "csv" else "text/markdown"
        if (not isinstance(content, bytes) or mime != expected_mime
                or not isinstance(filename, str) or not filename or len(filename) > 180
                or any(ord(char) < 32 or char in '/\\":' for char in filename)):
            raise ValueError("下载内容或文件名不符合要求。")
        session_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            self._prune()
            # 页面重跑时复用同一资源，避免重复占用内存。
            for artifact_id, item in self._items.items():
                if (item.session_hash == session_hash and item.content == content
                        and item.filename == filename and item.kind == kind):
                    return artifact_id
            if (len(self._items) >= 256
                    or sum(len(item.content) for item in self._items.values()) + len(content) > self.max_bytes):
                raise ValueError("下载缓存已满或文件过大，请稍后重新生成；总容量为 100 MiB。")
            artifact_id = uuid4().hex
            self._items[artifact_id] = _Artifact(
                user["id"], session_hash, content, filename, mime, kind,
                time.monotonic() + self.ttl_seconds,
            )
            return artifact_id

    def fetch(self, token, artifact_id):
        user = self._user(token)
        session_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            self._prune()
            item = self._items.get(artifact_id)
            if item is None:
                raise KeyError("下载已过期，请在页面重新生成。")
            if (item.owner_id != user["id"]
                    or not hmac.compare_digest(item.session_hash, session_hash)
                    or (item.kind == "csv" and user["role"] != "admin")):
                raise PermissionError("当前会话不能下载此文件。")
            return item.content, item.filename, item.mime

    def revoke(self, token):
        if not isinstance(token, str):
            return
        session_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            self._items = {key: item for key, item in self._items.items()
                           if item.session_hash != session_hash}

    def purge_expired(self):
        with self._lock:
            self._prune()


def start_server(store, port=0, origins=None):
    """启动只监听回环地址的下载服务；调用者负责 shutdown/server_close。"""
    allowed_origins = frozenset(origins if origins is not None else (
        "http://localhost:8501", "http://127.0.0.1:8501",
    ))

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format, *args):
            # 不记录请求地址、会话令牌或内容。
            pass

        def _respond(self, status, content=b"", filename=None, mime="text/plain; charset=utf-8"):
            self.send_response(status)
            origin = self.headers.get("Origin")
            if origin in allowed_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            if filename:
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(filename, safe=""))
            if status == 204:
                self.send_header("Access-Control-Allow-Methods", "POST")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.end_headers()
            if content:
                self.wfile.write(content)

        def _valid_request(self):
            if self.headers.get("Origin") not in allowed_origins:
                self._respond(403)
                return False
            if not re.fullmatch(r"/download/[0-9a-f]{32}", self.path):
                self._respond(404)
                return False
            return True

        def do_OPTIONS(self):
            if not self._valid_request():
                return
            headers = {value.strip().lower() for value in
                       self.headers.get("Access-Control-Request-Headers", "").split(",") if value.strip()}
            if (self.headers.get("Access-Control-Request-Method") != "POST"
                    or not headers.issubset({"authorization", "content-type"})):
                self._respond(403)
                return
            self._respond(204)

        def do_GET(self):
            self._respond(405)

        def do_POST(self):
            if not self._valid_request():
                return
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Length", "0") != "0":
                self._respond(400)
                return
            authorization = self.headers.get("Authorization", "")
            if not authorization.startswith("Bearer ") or len(authorization) > 263:
                self._respond(401)
                return
            token = authorization[7:]
            try:
                content, filename, mime = store.fetch(token, self.path.rsplit("/", 1)[1])
            except PermissionError:
                self._respond(403)
            except KeyError:
                self._respond(404)
            except (sqlite3.Error, OSError):
                self._respond(503)
            else:
                self._respond(200, content, filename, mime)

    class DownloadServer(ThreadingHTTPServer):
        def service_actions(self):
            store.purge_expired()

    server = DownloadServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="shuxi-download").start()
    return server

