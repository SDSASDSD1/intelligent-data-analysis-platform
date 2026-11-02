"""本机密钥、每位用户的资料目录与认证加密。"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "private_data"
if os.name == "nt":
    DEFAULT_KEY_PATH = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ShuxiDataAnalysis" / "storage.key"
else:
    DEFAULT_KEY_PATH = Path.home() / ".local" / "share" / "ShuxiDataAnalysis" / "storage.key"
USER_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


class StorageError(RuntimeError):
    """密钥或密文不可用时停止读取，不退回明文。"""


def seal_json(record: dict, cipher) -> dict:
    plaintext = json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")
    return {"version": 1, "cipher": "fernet", "token": cipher.encrypt(plaintext).decode("ascii")}


def unseal_json(envelope, cipher) -> dict:
    if (not isinstance(envelope, dict) or envelope.get("version") != 1
            or envelope.get("cipher") != "fernet" or not isinstance(envelope.get("token"), str)):
        raise StorageError("资料未使用预期加密格式，已停止读取。")
    try:
        record = json.loads(cipher.decrypt(envelope["token"].encode("ascii")))
    except (InvalidToken, ValueError, TypeError, UnicodeError) as error:
        raise StorageError("资料解密失败：密钥不匹配或资料已损坏。") from error
    if not isinstance(record, dict):
        raise StorageError("加密资料内容格式错误。")
    return record


def _make_cipher(key) -> Fernet:
    try:
        return Fernet(key)
    except (ValueError, TypeError) as error:
        raise StorageError("存储密钥格式错误，请检查原密钥；不要覆盖它。") from error


def initialize_key(key_path=None) -> Path:
    """首次生成文件密钥；已有密钥只校验，绝不覆盖。"""
    path = Path(key_path) if key_path is not None else DEFAULT_KEY_PATH
    if path.exists():
        try:
            _make_cipher(path.read_bytes().strip())
        except OSError as error:
            raise StorageError("无法读取存储密钥文件。") from error
        return path
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            path.parent.chmod(0o700)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return initialize_key(path)
    except OSError as error:
        raise StorageError("无法创建存储密钥文件。") from error
    with os.fdopen(descriptor, "wb") as output:
        output.write(Fernet.generate_key())
    return path


def load_cipher(key_path=None) -> Fernet:
    """优先读取环境变量；缺失或错误密钥均明确失败。"""
    environment_key = os.environ.get("SHUXI_STORAGE_KEY")
    if environment_key is not None:
        return _make_cipher(environment_key.encode("utf-8"))
    path = Path(key_path) if key_path is not None else DEFAULT_KEY_PATH
    try:
        key = path.read_bytes().strip()
    except OSError as error:
        raise StorageError("未找到或无法读取存储密钥，请先初始化本机环境。") from error
    return _make_cipher(key)


@dataclass(frozen=True)
class UserWorkspace:
    user_id: str
    log_dir: Path
    review_dir: Path
    checkpoint_path: Path
    cipher: Fernet = field(repr=False)


def workspace_for_user(user_id: str, data_dir=None, key_path=None) -> UserWorkspace:
    """只用数据库中的随机用户 ID 生成独立路径。"""
    if not isinstance(user_id, str) or not USER_ID_PATTERN.fullmatch(user_id):
        raise ValueError("无效的用户 ID。")
    cipher = load_cipher(key_path)
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    directory = root / "users" / user_id
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        directory.chmod(0o700)
    return UserWorkspace(user_id, directory / "logs", directory / "pending",
                         directory / "checkpoints.sqlite", cipher)


class _FernetAdapter:
    def __init__(self, cipher):
        self.cipher = cipher

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return "fernet", self.cipher.encrypt(plaintext)

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        if ciphername != "fernet":
            raise StorageError("不支持的检查点加密格式。")
        try:
            return self.cipher.decrypt(ciphertext)
        except (InvalidToken, TypeError, ValueError) as error:
            raise StorageError("检查点解密失败：密钥不匹配或资料已损坏。") from error


class _StrictEncryptedSerializer(EncryptedSerializer):
    def loads_typed(self, data):
        if not isinstance(data[0], str) or not data[0].endswith("+fernet"):
            raise StorageError("检查点未使用预期加密格式，已停止读取。")
        return super().loads_typed(data)


def encrypted_serializer(cipher) -> EncryptedSerializer:
    return _StrictEncryptedSerializer(
        _FernetAdapter(cipher),
        JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=None),
    )

