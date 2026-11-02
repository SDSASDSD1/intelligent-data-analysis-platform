import importlib
import json
import os
import unittest
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage
from langgraph.types import Command

from test_analysis_agent import APP_PATH, ScriptedChatModel
from authenticated_app import AppTest


class AgentReviewTest(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("analysis_agent")
        self.data = pd.DataFrame({"商品": ["笔记本"], "销售额": [24], "邮箱": ["PRIVATE_EMAIL@example.com"], "备注": ["PRIVATE_NOTE"]})
        self.grouped = pd.DataFrame({"商品": ["笔记本"], "销售额合计": [24]})

    def model_for(self, name, args, with_plan=False):
        messages = [AIMessage(content=json.dumps({"steps": [{"tool": name, "args": args}]}))] if with_plan else []
        return ScriptedChatModel(responses=messages + [
            AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "review-call"}]),
            AIMessage(content="本地内容已准备好。"),
        ])

    def test_sensitive_column_suggestions_match_chinese_and_english_names(self):
        self.assertTrue(hasattr(self.module, "suggest_sensitive_columns"), "尚未实现敏感列识别")
        columns = ["手机号", "身份证号", "联系邮箱", "phone_number", "mobile", "email", "id_card", "商品", "销售额", "备注"]
        self.assertEqual(self.module.suggest_sensitive_columns(columns), columns[:7])

    def test_sensitive_preview_and_full_export_wait_for_approval_then_resume(self):
        for name, args, artifact in [("column_preview", {"column_name": "邮箱"}, "column_preview"), ("full_csv_export", {}, "full_csv")]:
            with self.subTest(tool=name):
                model = self.model_for(name, args)
                artifacts = {}
                agent = self.module.create_analysis_agent(model, self.data, self.data["销售额"], self.grouped, ["邮箱"], artifacts, enable_review=True)
                config = {"configurable": {"thread_id": name}, "recursion_limit": 20}
                result = agent.invoke({"messages": [("human", "准备本地内容")]}, config)
                self.assertTrue(result.get("__interrupt__"))
                self.assertEqual(artifacts, {})
                request = result["__interrupt__"][0].value
                self.assertEqual(request["action_requests"][0]["name"], name)
                result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
                self.assertFalse(result.get("__interrupt__"))
                self.assertIn(artifact, artifacts)
                self.assertNotIn("PRIVATE", str(model.requests))

    def test_non_sensitive_preview_runs_without_review_interrupt(self):
        artifacts = {}
        model = self.model_for("column_preview", {"column_name": "备注"})
        agent = self.module.create_analysis_agent(model, self.data, self.data["销售额"], self.grouped, ["邮箱"], artifacts, enable_review=True)
        result = agent.invoke({"messages": [("human", "预览备注")]}, {"configurable": {"thread_id": "public"}, "recursion_limit": 20})
        self.assertFalse(result.get("__interrupt__"))
        pd.testing.assert_frame_equal(artifacts["column_preview"], self.data[["备注"]])
        self.assertNotIn("PRIVATE_NOTE", str(model.requests))

    def start_page(self, model):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload("sales.csv", self.data.to_csv(index=False).encode("utf-8"), "text/csv").run(timeout=10)
        return app

    def test_page_manual_sensitive_selection_and_approval_show_local_preview(self):
        model = self.model_for("column_preview", {"column_name": "备注"}, True)
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-review-key"}), patch("langchain_openai.ChatOpenAI", return_value=model):
            app = self.start_page(model)
            selector = app.multiselect(key="sensitive_columns")
            self.assertIn("邮箱", selector.value)
            selector.set_value(["邮箱", "备注"]).run(timeout=10)
            app.text_input(key="agent_question").input("预览备注").run(timeout=10)
            app.button(key="agent_ask").click().run(timeout=10)
            self.assertFalse(app.exception)
            self.assertEqual(len(model.requests), 2)
            app.button(key="agent_approve").click().run(timeout=10)
            self.assertFalse(app.exception)
            self.assertEqual(len(model.requests), 3)
            self.assertTrue(any(list(item.value.columns) == ["备注"] for item in app.dataframe))
            self.assertNotIn("PRIVATE_NOTE", str(model.requests))

    def test_page_reject_ends_run_without_artifacts_or_further_model_calls(self):
        model = self.model_for("full_csv_export", {}, True)
        captured = []
        factory = self.module.create_analysis_agent

        def tracked(*args, **kwargs):
            captured.append(args[5])
            return factory(*args, **kwargs)

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-review-key"}), patch("langchain_openai.ChatOpenAI", return_value=model), patch.object(self.module, "create_analysis_agent", side_effect=tracked):
            app = self.start_page(model)
            app.text_input(key="agent_question").input("导出全部数据").run(timeout=10)
            app.button(key="agent_ask").click().run(timeout=10)
            self.assertEqual(captured, [{}])
            app.button(key="agent_reject").click().run(timeout=10)
            self.assertFalse(app.exception)
            self.assertEqual(captured, [{}])
            self.assertEqual(len(model.requests), 2)
            self.assertFalse([button for button in app.button if button.key == "agent_approve"])
            self.assertFalse(app.downloads)

    def test_page_approved_full_export_offers_local_csv_download(self):
        model = self.model_for("full_csv_export", {}, True)
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-review-key"}), patch("langchain_openai.ChatOpenAI", return_value=model):
            app = self.start_page(model)
            app.text_input(key="agent_question").input("导出全部数据").run(timeout=10)
            app.button(key="agent_ask").click().run(timeout=10)
            self.assertFalse(app.downloads)
            app.button(key="agent_approve").click().run(timeout=10)
            self.assertFalse(app.exception)
            self.assertTrue(app.downloads)
            self.assertNotIn("PRIVATE", str(model.requests))

    def test_uploading_new_file_removes_pending_review_without_execution(self):
        model = self.model_for("full_csv_export", {}, True)
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-review-key"}), patch("langchain_openai.ChatOpenAI", return_value=model):
            app = self.start_page(model)
            app.text_input(key="agent_question").input("导出全部数据").run(timeout=10)
            app.button(key="agent_ask").click().run(timeout=10)
            self.assertIsNotNone(app.button(key="agent_approve"))
            app.file_uploader[0].upload("new.csv", "商品,销售额\n鼠标,99\n".encode("utf-8"), "text/csv").run(timeout=10)
            self.assertFalse(app.exception)
            self.assertFalse([button for button in app.button if button.key == "agent_approve"])
            self.assertFalse(app.downloads)
            self.assertEqual(len(model.requests), 2)


if __name__ == "__main__":
    unittest.main()


