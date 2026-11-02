"""旧分析页面回归测试的认证边界；安全测试使用不替换认证的 secure_download_app。"""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from secure_download_app import AppTest as StreamlitAppTest

import review_store
import trace_store


class AppTest(StreamlitAppTest):
    @classmethod
    def from_file(cls, script_path, *, default_timeout=3):
        return cls(str(Path(script_path).resolve()), default_timeout=default_timeout)

    def _run(self, *args, **kwargs):
        import app_access

        user = {"id": "a" * 32, "username": "regression_admin", "role": "admin", "enabled": True}
        workspace = SimpleNamespace(user_id=user["id"], log_dir=trace_store.DEFAULT_LOG_DIR,
                                    review_dir=review_store.DEFAULT_REVIEW_DIR,
                                    checkpoint_path=review_store.DEFAULT_CHECKPOINT_PATH, cipher=None)
        with ExitStack() as stack:
            stack.enter_context(patch.object(app_access, "require_user", return_value=user))
            stack.enter_context(patch.object(app_access, "workspace", return_value=workspace))
            stack.enter_context(patch.object(app_access, "render_account_management"))
            return super()._run(*args, **kwargs)

