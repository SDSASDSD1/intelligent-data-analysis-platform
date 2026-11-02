"""真实认证、用户存储及 Streamlit 页面集成；模型仅用脚本替身。"""
import importlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage
from secure_download_app import AppTest

from test_analysis_agent import ScriptedChatModel


ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "PrivatePassphrase2026!"


class AccessPageTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("auth_store"), "尚未实现账户模块")
        self.assertIsNotNone(importlib.util.find_spec("app_access"), "尚未实现登录页面")
        self.auth = importlib.import_module("auth_store")
        self.storage = importlib.import_module("secure_storage")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.db = self.directory / "accounts.sqlite"
        key = self.directory / "storage.key"
        self.storage.initialize_key(key)
        for replacement in [patch.object(self.auth, "DEFAULT_DB_PATH", self.db),
                            patch.object(self.storage, "DEFAULT_DATA_DIR", self.directory / "private"),
                            patch.object(self.storage, "DEFAULT_KEY_PATH", key),
                            patch.dict(os.environ, {"SHUXI_STORAGE_KEY": key.read_text(), "DASHSCOPE_API_KEY": "fake-page-key"})]:
            replacement.start()
            self.addCleanup(replacement.stop)
        self.admin = self.auth.bootstrap_admin("admin_owner", PASSWORD)
        self.admin_token = self.auth.authenticate("admin_owner", PASSWORD)
        self.other = self.auth.create_user(self.admin_token, "admin_other", PASSWORD, role="admin")
        self.analyst = self.auth.create_user(self.admin_token, "analyst_user", PASSWORD)
        self.data = pd.DataFrame({"商品": ["A"], "销售额": [20], "邮箱": ["PRIVATE_ROW@example.com"]})

    def page(self):
        app = AppTest.from_file(ROOT / "app.py").run(timeout=15)
        self.assertFalse(app.exception)
        return app

    def login(self, username="admin_owner"):
        app = self.page()
        app.text_input(key="login_username").input(username)
        app.text_input(key="login_password").input(PASSWORD)
        app.button(key="login_submit").click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertTrue(app.file_uploader)
        return app

    def upload(self, app):
        app.file_uploader[0].upload("sales.csv", self.data.to_csv(index=False).encode(), "text/csv").run(timeout=15)
        self.assertFalse(app.exception)
        return app

    def test_logged_out_page_has_no_private_analysis_or_download_controls(self):
        app = self.page()
        self.assertTrue(app.button(key="login_submit"))
        self.assertFalse(app.file_uploader)
        self.assertFalse(app.dataframe)
        self.assertFalse(app.downloads)
        self.assertFalse(app.get("download_button"), "页面不得生成 Streamlit 的未认证下载链接")
        keys = [button.key for button in app.button]
        for key in ["agent_ask", "agent_approve", "trace_search"]:
            self.assertNotIn(key, keys)

    def test_incorrect_password_does_not_open_analysis_page(self):
        app = self.page()
        app.text_input(key="login_username").input("admin_owner")
        app.text_input(key="login_password").input("IncorrectPass2026!")
        app.button(key="login_submit").click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertFalse(app.file_uploader)

    def test_logout_clears_uploaded_data_pending_report_and_model(self):
        app = self.upload(self.login())
        app.session_state["agent_pending"] = {"private": "PENDING_MARKER"}
        app.session_state["agent_result"] = {"private": "REPORT_MARKER"}
        app.session_state["model_instance"] = "MODEL_MARKER"
        app.button(key="logout").click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertFalse(app.file_uploader)
        self.assertFalse(app.downloads)
        self.assertFalse(app.dataframe)
        for key in ["agent_pending", "agent_result", "model_instance"]:
            self.assertNotIn(key, app.session_state)
        app.text_input(key="login_username").input("admin_other")
        app.text_input(key="login_password").input(PASSWORD)
        app.button(key="login_submit").click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertFalse(app.dataframe)
        self.assertFalse(app.downloads)

    def test_disabled_account_is_revalidated_on_every_rerun(self):
        app = self.upload(self.login("analyst_user"))
        self.auth.set_user_enabled(self.admin_token, self.analyst["id"], False)
        app.run(timeout=15)
        self.assertFalse(app.exception)
        self.assertFalse(app.file_uploader)
        self.assertFalse(app.dataframe)
        self.assertFalse(app.downloads)

    def test_trace_lookup_uses_only_authenticated_users_workspace(self):
        import trace_store
        own = self.storage.workspace_for_user(self.admin["id"])
        trace_store.append_trace({"trace_id": "a" * 32, "event": "OWNER_ONLY_MARKER"}, own.log_dir, cipher=own.cipher)
        for username, visible in [("admin_owner", True), ("admin_other", False)]:
            with self.subTest(username=username):
                app = self.upload(self.login(username))
                app.text_input(key="trace_lookup").input("a" * 32).run(timeout=15)
                app.button(key="trace_search").click().run(timeout=15)
                self.assertFalse(app.exception)
                self.assertEqual("OWNER_ONLY_MARKER" in str([item.value for item in app.json]), visible)

    def test_pending_review_can_only_be_recovered_by_its_owner(self):
        model = ScriptedChatModel(responses=[AIMessage(content=json.dumps({"steps": [{"tool": "full_csv_export", "args": {}}]})),
            AIMessage(content="", tool_calls=[{"name": "full_csv_export", "args": {}, "id": "export"}]), AIMessage(content="批准完成")])
        with patch("langchain_openai.ChatOpenAI", return_value=model):
            owner = self.upload(self.login())
            owner.text_input(key="agent_question").input("导出全部数据").run(timeout=15)
            owner.button(key="agent_ask").click().run(timeout=15)
            self.assertFalse(owner.exception)
            trace_id = owner.session_state["agent_pending"]["trace_id"]
            other = self.upload(self.login("admin_other"))
            other.text_input(key="agent_recover_trace").input(trace_id).run(timeout=15)
            other.button(key="agent_recover").click().run(timeout=15)
            self.assertFalse(other.exception)
            self.assertNotIn("agent_approve", [button.key for button in other.button])
            self.assertFalse(other.downloads)
            restored = self.upload(self.login())
            restored.text_input(key="agent_recover_trace").input(trace_id).run(timeout=15)
            restored.button(key="agent_recover").click().run(timeout=15)
            self.assertFalse(restored.exception)
            restored.button(key="agent_approve").click().run(timeout=15)
            self.assertFalse(restored.exception)
            self.assertIn("agent_csv_download", [button.key for button in restored.downloads])
            csv = next(item for item in restored.downloads if item.key == "agent_csv_download")
            self.assertEqual(csv.kind, "csv")
            self.assertIs(csv.approved, True)

    def test_admin_page_creates_and_disables_account_analyst_has_no_management_controls(self):
        app = self.login()
        app.text_input(key="new_username").input("created_in_page")
        app.text_input(key="new_password").input(PASSWORD)
        app.selectbox(key="new_role").select("analyst")
        app.button(key="create_user").click().run(timeout=15)
        self.assertFalse(app.exception)
        token = self.auth.authenticate("created_in_page", PASSWORD)
        self.assertIsNotNone(token)
        created = self.auth.resolve_session(token)
        app.selectbox(key="manage_user_id").select(created["id"])
        app.selectbox(key="manage_enabled").select(False)
        app.button(key="update_user").click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertIsNone(self.auth.resolve_session(token))
        analyst_page = self.login("analyst_user")
        self.assertNotIn("create_user", [button.key for button in analyst_page.button])
        self.assertNotIn("update_user", [button.key for button in analyst_page.button])

    def test_reject_with_wrong_storage_key_shows_error_and_keeps_pending_unconsumed(self):
        import review_store
        from cryptography.fernet import Fernet

        model = ScriptedChatModel(responses=[AIMessage(content=json.dumps({"steps": [{"tool": "full_csv_export", "args": {}}]})),
            AIMessage(content="", tool_calls=[{"name": "full_csv_export", "args": {}, "id": "export"}])])
        with patch("langchain_openai.ChatOpenAI", return_value=model):
            app = self.upload(self.login())
            app.text_input(key="agent_question").input("导出全部数据").run(timeout=15)
            app.button(key="agent_ask").click().run(timeout=15)
            self.assertFalse(app.exception)
            trace_id = app.session_state["agent_pending"]["trace_id"]
            own = self.storage.workspace_for_user(self.admin["id"])
            original = review_store.load_pending(trace_id, own.review_dir, cipher=own.cipher)
            with patch.dict(os.environ, {"SHUXI_STORAGE_KEY": Fernet.generate_key().decode()}):
                app.button(key="agent_reject").click().run(timeout=15)
            self.assertEqual(review_store.load_pending(trace_id, own.review_dir, cipher=own.cipher), original)
            self.assertFalse(app.downloads)
            self.assertEqual(len(model.requests), 2)
            self.assertFalse(app.exception, "错误密钥应显示提示，而非页面 traceback")
            self.assertTrue(app.error)

    def test_analyst_page_rejects_restricted_plan_without_tool_execution_or_review(self):
        model = ScriptedChatModel(responses=[AIMessage(content=json.dumps({"steps": [{"tool": "full_csv_export", "args": {}}]}))])
        with patch("langchain_openai.ChatOpenAI", return_value=model):
            app = self.upload(self.login("analyst_user"))
            app.text_input(key="agent_question").input("导出全部数据").run(timeout=15)
            app.button(key="agent_ask").click().run(timeout=15)
            self.assertFalse(app.exception)
            self.assertEqual(len(model.requests), 1)
            self.assertNotIn("agent_approve", [button.key for button in app.button])
            self.assertNotIn("agent_csv_download", [button.key for button in app.downloads])
            result = app.session_state.get("agent_result") or {}
            self.assertNotIn("full_csv", result.get("artifacts", {}))

    def test_analyst_page_can_run_summary_and_download_markdown_report(self):
        model = ScriptedChatModel(responses=[AIMessage(content=json.dumps({"steps": [{"tool": "sales_summary", "args": {}}]})),
            AIMessage(content="", tool_calls=[{"name": "sales_summary", "args": {}, "id": "summary"}]), AIMessage(content="销售额合计20")])
        with patch("langchain_openai.ChatOpenAI", return_value=model):
            app = self.upload(self.login("analyst_user"))
            app.text_input(key="agent_question").input("汇总销售额").run(timeout=15)
            app.button(key="agent_ask").click().run(timeout=15)
            self.assertFalse(app.exception)
            self.assertEqual(len(model.requests), 3)
            self.assertTrue(app.downloads)
            self.assertFalse(app.get("download_button"), "报告应通过认证下载边界渲染")
            report = next(item for item in app.downloads if item.key == "agent_report_download")
            self.assertEqual(report.mime, "text/markdown")
            self.assertIsInstance(report.content, bytes)
            self.assertNotIn("agent_csv_download", [button.key for button in app.downloads])


if __name__ == "__main__":
    unittest.main()


