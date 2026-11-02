"""页面与离线维护工具共用的本机进程锁。"""

import atexit
import os
import threading
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "private_data"
_app_lock = None
_app_lock_guard = threading.Lock()


class DataLock:
    """持有系统排它锁；锁文件保留，进程退出后由系统释放锁。"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.path = self.data_dir.parent / f".{self.data_dir.name}.lock"
        self._file = None

    def __enter__(self):
        if self._file is not None:
            raise RuntimeError("资料锁已经持有，不能重复获取。")
        handle = None
        try:
            descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            handle = os.fdopen(descriptor, "r+b")
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if handle is not None:
                handle.close()
            raise RuntimeError("无法获取资料锁：应用或其他维护工具可能正在运行，请先停止后重试。") from error
        self._file = handle
        return self

    def __exit__(self, *_):
        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def ensure_app_lock():
    """Streamlit 多次重跑共用一个锁，直到页面服务进程退出。"""
    global _app_lock
    with _app_lock_guard:
        if _app_lock is None:
            lock = DataLock(DEFAULT_DATA_DIR)
            lock.__enter__()
            atexit.register(lock.__exit__)
            _app_lock = lock
        return _app_lock

