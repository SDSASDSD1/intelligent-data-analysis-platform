"""停机后备份白名单资料，并用原存储密钥验证恢复内容。"""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path

from cryptography.fernet import InvalidToken

import secure_storage
from local_runtime import DataLock


MAX_BACKUP_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MANIFEST_NAME = "manifest.enc"
_USER_FILE = re.compile(
    r"users/[0-9a-f]{32}/(?:logs/(?:trace|audit)\.enc\.jsonl|"
    r"pending/pending\.sqlite|checkpoints\.sqlite)\Z"
)


class BackupError(ValueError):
    """备份内容、路径、密钥或完整性不符合要求。"""


def _allowed_name(name):
    return isinstance(name, str) and (name == "accounts.sqlite" or _USER_FILE.fullmatch(name) is not None)


def _no_links(path):
    """在 resolve 或打开文件之前拒绝符号链接和 Windows 重解析点。"""
    for part in reversed((path, *path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise BackupError("资料路径不能包含符号链接或目录联接。")


def _absolute_path(path):
    path = Path(path).expanduser().absolute()
    _no_links(path)
    return path.resolve()


def _source_files(root):
    account = root / "accounts.sqlite"
    _no_links(account)
    if not account.is_file():
        raise BackupError("账户数据库不存在，请先初始化本机账户。")
    files = [account]
    users = root / "users"
    _no_links(users)
    if users.exists():
        if not users.is_dir():
            raise BackupError("用户资料路径不是目录。")
        for directory in sorted(users.iterdir()):
            if not secure_storage.USER_ID_PATTERN.fullmatch(directory.name):
                continue
            _no_links(directory)
            if not directory.is_dir():
                raise BackupError("用户资料路径不是目录。")
            for relative in ("logs/trace.enc.jsonl", "logs/audit.enc.jsonl",
                             "pending/pending.sqlite", "checkpoints.sqlite"):
                path = directory / relative
                _no_links(path)
                if path.exists():
                    if not path.is_file():
                        raise BackupError("备份白名单路径必须是普通文件。")
                    files.append(path)
    return files


def _snapshot_database(source, target, clear_sessions=False):
    for suffix in ("-wal", "-shm", "-journal"):
        _no_links(Path(str(source) + suffix))
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as reader:
        with closing(sqlite3.connect(target)) as writer:
            reader.backup(writer)
            writer.execute("PRAGMA journal_mode=DELETE")
            if clear_sessions:
                writer.execute("PRAGMA secure_delete=ON")
                writer.execute("DELETE FROM sessions")
                writer.commit()


def _digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _validate_stored_content(path, name, cipher):
    """只在内存验证密文，确保备份使用的密钥能读取原资料。"""
    try:
        if name.endswith(".enc.jsonl"):
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        secure_storage.unseal_json(json.loads(line), cipher)
        elif name.endswith(".sqlite"):
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
                if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise BackupError("数据库完整性检查失败。")
                if name.endswith("/pending.sqlite"):
                    for (payload,) in connection.execute("SELECT payload FROM pending"):
                        secure_storage.unseal_json(json.loads(payload), cipher)
                elif name.endswith("/checkpoints.sqlite"):
                    tables = {row[0] for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )}
                    # 尚未执行 Agent 的检查点库可能还没有表。
                    for table, column in (("checkpoints", "checkpoint"), ("writes", "value")):
                        if table not in tables:
                            continue
                        for kind, payload in connection.execute(f"SELECT type, {column} FROM {table}"):
                            if not isinstance(kind, str) or not kind.endswith("+fernet"):
                                raise BackupError("检查点不是预期的加密格式。")
                            cipher.decrypt(payload)
    except (InvalidToken, secure_storage.StorageError, UnicodeError, ValueError, TypeError, sqlite3.Error) as error:
        raise BackupError("资料验证失败：原存储密钥不匹配或资料已损坏；未写入备份或恢复目标。") from error


def create_backup(output_path, data_dir=None, key_path=None) -> Path:
    """生成私有 ZIP；不修改原库，副本内撤销全部登录会话。"""
    root = _absolute_path(data_dir if data_dir is not None else secure_storage.DEFAULT_DATA_DIR)
    output = _absolute_path(output_path)
    created = False
    completed = False
    try:
        with DataLock(root):
            cipher = secure_storage.load_cipher(key_path)
            if output.exists():
                raise BackupError("备份文件已经存在，请使用新的文件名。")
            files = _source_files(root)
            if sum(path.stat().st_size for path in files) > MAX_BACKUP_BYTES:
                raise BackupError("资料超过 512 MiB，不能使用本机备份工具。")
            with tempfile.TemporaryDirectory(prefix=".shuxi-backup-", dir=output.parent) as temporary:
                staging = Path(temporary)
                hashes = {}
                total = 0
                for source in files:
                    name = source.relative_to(root).as_posix()
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    if source.suffix == ".sqlite":
                        _snapshot_database(source, target, clear_sessions=name == "accounts.sqlite")
                    else:
                        shutil.copyfile(source, target)
                    total += target.stat().st_size
                    if total > MAX_BACKUP_BYTES:
                        raise BackupError("资料超过 512 MiB，不能使用本机备份工具。")
                    _validate_stored_content(target, name, cipher)
                    hashes[name] = _digest(target)
                manifest = cipher.encrypt(json.dumps(
                    {"version": 1, "files": hashes}, ensure_ascii=False, sort_keys=True
                ).encode("utf-8"))
                if len(manifest) > MAX_MANIFEST_BYTES:
                    raise BackupError("备份清单过大。")
                descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                created = True
                with os.fdopen(descriptor, "wb") as sink:
                    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                        archive.writestr(MANIFEST_NAME, manifest)
                        for name in hashes:
                            archive.write(staging / name, name)
        completed = True
        return output
    except (OSError, sqlite3.Error, secure_storage.StorageError, zipfile.BadZipFile) as error:
        raise BackupError(f"备份失败：{error}") from error
    finally:
        if created and not completed:
            output.unlink(missing_ok=True)


def _verified_members(archive, cipher):
    members = archive.infolist()
    names = [member.filename for member in members]
    if (len(names) != len(set(names)) or MANIFEST_NAME not in names
            or "accounts.sqlite" not in names):
        raise BackupError("备份清单缺失、成员重复或账户数据库缺失。")
    total = 0
    for member in members:
        mode = member.external_attr >> 16
        if (member.is_dir() or member.orig_filename != member.filename or stat.S_ISLNK(mode)
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                or (member.filename != MANIFEST_NAME and not _allowed_name(member.filename))
                or member.flag_bits & 1
                or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise BackupError("备份包含不允许的文件或路径。")
        limit = MAX_MANIFEST_BYTES if member.filename == MANIFEST_NAME else MAX_BACKUP_BYTES
        if member.filename != MANIFEST_NAME:
            total += member.file_size
        if member.file_size > limit or total > MAX_BACKUP_BYTES:
            raise BackupError("备份内容超过允许的大小。")
    try:
        manifest = json.loads(cipher.decrypt(archive.read(MANIFEST_NAME)))
    except (InvalidToken, UnicodeError, ValueError, TypeError) as error:
        raise BackupError("备份验证失败：请使用原存储密钥，或检查备份是否损坏。") from error
    if (not isinstance(manifest, dict) or manifest.get("version") != 1
            or not isinstance(manifest.get("files"), dict)
            or set(manifest["files"]) != set(names) - {MANIFEST_NAME}):
        raise BackupError("备份清单与实际文件不一致。")
    for name, expected in manifest["files"].items():
        if not _allowed_name(name) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise BackupError("备份清单包含无效的文件摘要。")
        with archive.open(name) as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise BackupError("备份文件摘要不匹配，资料已损坏。")
    return manifest["files"]


def _require_empty_destination(destination):
    _no_links(destination)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise BackupError("恢复目标必须不存在或为空目录；不能覆盖已有资料。")


def restore_backup(archive_path, destination=None, key_path=None) -> Path:
    """先完整验证，再将同级临时目录原子改名为恢复目标。"""
    target = _absolute_path(destination if destination is not None else secure_storage.DEFAULT_DATA_DIR)
    source = _absolute_path(archive_path)
    try:
        with DataLock(target):
            _require_empty_destination(target)
            cipher = secure_storage.load_cipher(key_path)
            with zipfile.ZipFile(source) as archive:
                files = _verified_members(archive, cipher)
                staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-restore-", dir=target.parent))
                try:
                    for name, expected in files.items():
                        output = staging / name
                        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(descriptor, "wb") as sink, archive.open(name) as reader:
                            shutil.copyfileobj(reader, sink)
                        if _digest(output) != expected:
                            raise BackupError("恢复时备份内容发生变化，已停止写入。")
                        _validate_stored_content(output, name, cipher)
                    _require_empty_destination(target)
                    existed = target.exists()
                    if existed:
                        target.rmdir()
                    try:
                        staging.rename(target)
                    except OSError:
                        if existed and not target.exists():
                            target.mkdir(mode=0o700)
                        raise
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
        return target
    except (OSError, sqlite3.Error, secure_storage.StorageError, zipfile.BadZipFile, EOFError) as error:
        raise BackupError(f"恢复失败：{error}") from error

