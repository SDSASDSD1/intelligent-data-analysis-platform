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
from authenticated_app import AppTest

from test_analysis_agent import APP_PATH, ScriptedChatModel


class ReportExportTest(unittest.TestCase):
    def setUp(self):
        self.sales = pd.Series([120.5, 79.5, 200], name="销售额")
        self.grouped = pd.DataFrame({"商品": ["笔记本", "鼠标", "键盘"], "销售额合计": self.sales})
        self.plan = {"steps": [{"tool": "sales_summary", "args": {}}, {"tool": "sales_chart", "args": {"chart_type": "bar"}}]}

    def build(self, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec("report_export"), "尚未实现 report_export")
        arguments = dict(trace_id="trace-report", question="总额是多少？", plan=self.plan, answer="销售额总和400.00。", sales=self.sales, grouped=self.grouped, chart="bar")
        arguments.update(kwargs)
        return importlib.import_module("report_export").build_markdown_report(**arguments)

    def test_report_contains_trace_question_plan_statistics_answer_and_chart(self):
        text = self.build()
        self.assertIsInstance(text, str)
        self.assertTrue(text.startswith("# "))
        for value in ["trace-report", "总额是多少？", "sales_summary", "sales_chart", "400.00", "133.33", "销售额总和400.00。", "笔记本", "120.50", "柱状图"]:
            self.assertIn(value, text)
        self.assertIn("3", text)

    def test_sensitive_product_grouping_is_omitted(self):
        text = self.build(sensitive_columns=["商品"])
        for product in self.grouped["商品"]:
            self.assertNotIn(product, text)
        self.assertIn("400.00", text)

    def test_empty_sales_report_has_zero_total_and_unavailable_average(self):
        text = self.build(sales=pd.Series([], dtype=float), grouped=None, chart=None)
        self.assertIn("0.00", text)
        self.assertIn("无法计算", text)

    def test_group_table_escapes_pipe_and_embedded_newline(self):
        grouped = pd.DataFrame({"商品": ["A|B\nC"], "销售额合计": [400]})
        text = self.build(grouped=grouped)
        self.assertIn("A\\|B", text)
        self.assertNotIn("A|B\nC", text)
        self.assertTrue("<br>" in text or "<br/>" in text or "<br />" in text or "A\\|B C" in text)

    def downloads(self, app, key):
        return [element for element in app.downloads if key == element.key]

    def test_page_finished_agent_offers_report_and_new_file_removes_it(self):
        self.build()
        data = pd.DataFrame({"商品": ["笔记本"], "销售额": [24], "备注": ["PRIVATE_NOTE"], "邮箱": ["PRIVATE@example.com"]})
        plan = {"steps": [{"tool": "sales_summary", "args": {}}]}
        model = ScriptedChatModel(responses=[AIMessage(content=json.dumps(plan)), AIMessage(content="", tool_calls=[{"name": "sales_summary", "args": {}, "id": "summary"}]), AIMessage(content="销售额24.00。")])
        store = importlib.import_module("trace_store")
        with tempfile.TemporaryDirectory() as directory, patch.object(store, "DEFAULT_LOG_DIR", Path(directory)), patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-report-key"}), patch("langchain_openai.ChatOpenAI", return_value=model):
            app = AppTest.from_file(APP_PATH).run(timeout=10)
            app.file_uploader[0].upload("sales.csv", data.to_csv(index=False).encode("utf-8"), "text/csv").run(timeout=10)
            self.assertFalse(self.downloads(app, "agent_report_download"))
            app.text_input(key="agent_question").input("销售总额是多少").run(timeout=10)
            app.button(key="agent_ask").click().run(timeout=10)
            self.assertFalse(app.exception)
            self.assertTrue(self.downloads(app, "agent_report_download"))
            self.assertFalse(self.downloads(app, "agent_csv_download"))
            report_call = self.build(sales=data["销售额"], grouped=None, answer="销售额24.00。")
            self.assertNotIn("PRIVATE_NOTE", report_call)
            self.assertNotIn("PRIVATE@example.com", report_call)
            app.file_uploader[0].upload("other.csv", "商品,销售额\n鼠标,99\n".encode("utf-8"), "text/csv").run(timeout=10)
            self.assertFalse(self.downloads(app, "agent_report_download"))

    def test_pending_and_rejected_export_have_no_report_or_csv_download(self):
        self.build()
        data = pd.DataFrame({"商品": ["笔记本"], "销售额": [24]})
        plan = {"steps": [{"tool": "full_csv_export", "args": {}}]}
        model = ScriptedChatModel(responses=[AIMessage(content=json.dumps(plan)), AIMessage(content="", tool_calls=[{"name": "full_csv_export", "args": {}, "id": "export"}])])
        store = importlib.import_module("trace_store")
        with tempfile.TemporaryDirectory() as directory, patch.object(store, "DEFAULT_LOG_DIR", Path(directory)), patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-report-key"}), patch("langchain_openai.ChatOpenAI", return_value=model):
            app = AppTest.from_file(APP_PATH).run(timeout=10)
            app.file_uploader[0].upload("sales.csv", data.to_csv(index=False).encode("utf-8"), "text/csv").run(timeout=10)
            app.text_input(key="agent_question").input("导出全部原始数据").run(timeout=10)
            app.button(key="agent_ask").click().run(timeout=10)
            self.assertFalse(app.exception)
            self.assertFalse(app.downloads)
            app.button(key="agent_reject").click().run(timeout=10)
            self.assertFalse(app.exception)
            self.assertFalse(app.downloads)


if __name__ == "__main__":
    unittest.main()


