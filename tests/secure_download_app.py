"""真实账户页面测试：仅隔离运行锁及下载渲染边界，不替换认证。"""
from types import SimpleNamespace
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest as StreamlitAppTest


class AppTest(StreamlitAppTest):
    @classmethod
    def from_file(cls, script_path, *, default_timeout=3):
        return cls(str(Path(script_path).resolve()), default_timeout=default_timeout)

    def _run(self, *args, **kwargs):
        import secure_download

        self.downloads = []

        def render(label, content, **options):
            self.downloads.append(SimpleNamespace(label=label, content=content, **options))

        with ExitStack() as stack:
            stack.enter_context(patch("local_runtime.ensure_app_lock"))
            if not isinstance(secure_download.render_download, Mock):
                stack.enter_context(patch("secure_download.render_download", side_effect=render))
            return super()._run(*args, **kwargs)

