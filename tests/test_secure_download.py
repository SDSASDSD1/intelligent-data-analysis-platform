"""下载渲染单元测试：隔离 Streamlit 和 HTTP 服务生命周期。"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import secure_download


class SecureDownloadTest(unittest.TestCase):
    def test_component_receives_resource_reference_without_file_payload(self):
        class Store:
            def register(self, token, content, filename, mime, kind, approved):
                return "a" * 32

        with patch.object(secure_download.st, "session_state", {"auth_token": "temporary-token"}), \
                patch.object(secure_download, "get_service", return_value=(Store(), SimpleNamespace(server_address=("127.0.0.1", 12345)))), \
                patch.object(secure_download.st.components.v2, "component") as component:
            secure_download.render_download("Download", b"PRIVATE_BYTES_MARKER", file_name="report.md",
                                            mime="text/markdown", key="report")
            options = component.return_value.call_args.kwargs
            self.assertEqual(options["key"], "report")
            self.assertEqual(options["data"]["endpoint"], "http://127.0.0.1:12345/download/" + "a" * 32)
            self.assertEqual(options["data"]["filename"], "report.md")
            self.assertNotIn("PRIVATE_BYTES_MARKER", str(component.call_args_list))
            self.assertNotIn("PRIVATE_BYTES_MARKER", str(options))
            self.assertNotIn("temporary-token", options["data"]["endpoint"])

    def test_logged_out_render_does_not_expose_component(self):
        with patch.object(secure_download.st, "session_state", {}), \
                patch.object(secure_download.st, "warning"), \
                patch.object(secure_download.st.components.v2, "component") as component:
            secure_download.render_download("Download", b"private", file_name="report.md",
                                            mime="text/markdown", key="report")
            component.assert_not_called()

    def test_rejected_registration_does_not_expose_component(self):
        class DeniedStore:
            def register(self, *args):
                raise PermissionError("denied")

        with patch.object(secure_download.st, "session_state", {"auth_token": "temporary-token"}), \
                patch.object(secure_download, "get_service", return_value=(DeniedStore(), None)), \
                patch.object(secure_download.st, "warning"), \
                patch.object(secure_download.st.components.v2, "component") as component:
            secure_download.render_download("Download", b"private", file_name="report.md",
                                            mime="text/markdown", key="report")
            component.assert_not_called()


if __name__ == "__main__":
    unittest.main()

