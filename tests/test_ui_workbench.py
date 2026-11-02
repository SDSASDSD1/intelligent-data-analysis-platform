"""Streamlit 工作台行为集成测试；外部模型及用户存储均被隔离。"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from authenticated_app import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class WorkbenchPageTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for target, value in [
            ("trace_store.DEFAULT_LOG_DIR", root / "logs"),
            ("review_store.DEFAULT_REVIEW_DIR", root / "reviews"),
            ("review_store.DEFAULT_CHECKPOINT_PATH", root / "checkpoints.sqlite"),
        ]:
            boundary = patch(target, value)
            boundary.start()
            self.addCleanup(boundary.stop)
        self.root = root
        model = patch("langchain_openai.ChatOpenAI")
        self.model = model.start()
        self.addCleanup(model.stop)
        environment = patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-ui-key"})
        environment.start()
        self.addCleanup(environment.stop)

    def start(self, upload=False):
        app = AppTest.from_file(APP_PATH).run(timeout=15)
        if upload:
            app.file_uploader[0].upload(
                "sample_sales.csv", (APP_PATH.parent / "sample_sales.csv").read_bytes(), "text/csv"
            ).run(timeout=15)
        self.assertFalse(app.exception)
        return app

    def test_empty_workspace_explains_the_three_steps_without_model_call(self):
        app = self.start()
        copy = " ".join(str(item.value) for kind in
                        (app.markdown, app.caption, app.info, app.subheader) for item in kind)
        copy += " ".join(item.proto.body for item in app.get("html"))
        for instruction in ["上传数据", "核对字段", "开始分析"]:
            self.assertIn(instruction, copy)
        self.assertFalse(app.metric)
        self.model.assert_not_called()

    def assert_example_fills_only(self, key, expected):
        app = self.start(upload=True)
        app.text_input(key="agent_question").input("旧问题").run(timeout=15)
        app.button(key=key).click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(app.text_input(key="agent_question").value, expected)
        self.assertNotIn("agent_result", app.session_state)
        self.assertNotIn("agent_pending", app.session_state)
        self.assertFalse(app.downloads)
        self.assertFalse(list(self.root.rglob("*.jsonl")))
        self.model.assert_not_called()

    def test_summary_example_replaces_question_without_starting_analysis(self):
        self.assert_example_fills_only("agent_example_summary", "统计销售额总和和平均值。")

    def test_workflow_example_replaces_question_without_starting_analysis(self):
        self.assert_example_fills_only("agent_example_workflow", "先查看字段，再统计销售额，最后选择柱状图。")

    def test_uploaded_sample_shows_four_overview_metrics(self):
        app = self.start(upload=True)
        self.assertEqual([(item.label, item.value) for item in app.metric], [
            ("销售额总和", "400.00"), ("销售额平均值", "133.33"),
            ("记录数量", "3"), ("字段数量", "3"),
        ])
        self.model.assert_not_called()

    def test_sidebar_has_trace_but_no_connection_test(self):
        app = self.start()
        self.assertNotIn("通义千问连接测试", [item.label for item in app.button])
        self.assertNotIn("模型连接", [item.label for item in app.sidebar.expander])
        self.assertEqual(app.sidebar.button(key="trace_search").label, "查看 Trace")
        self.assertIn("编号", app.sidebar.text_input(key="trace_lookup").placeholder)
        self.model.assert_not_called()


if __name__ == "__main__":
    unittest.main()

