import unittest
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st
from authenticated_app import AppTest
from langchain_core.messages import AIMessage

import trace_store
from test_analysis_agent import ScriptedChatModel


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class SalesColumnSelectionTest(unittest.TestCase):
    def upload_csv(self, app, content, filename="sales.csv"):
        app.file_uploader[0].upload(filename, content.encode("utf-8"), "text/csv").run(timeout=10)
        self.assertFalse(app.exception)
        return app

    def test_sample_defaults_to_sales_column_and_preserves_total(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        self.upload_csv(app, (APP_PATH.parent / "sample_sales.csv").read_text(encoding="utf-8-sig"))
        self.assertEqual(app.selectbox(key="sales_column").value, "销售额")
        self.assertEqual(app.metric[0].value, "400.00")

    def test_csv_requires_selection_then_all_rows_feed_statistics_and_charts(self):
        data = pd.DataFrame({"商品": ["A", "B"] * 6, "金额": range(1, 13)})
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        self.upload_csv(app, data.to_csv(index=False))
        self.assertFalse(app.metric)
        messages = [item.value for kind in (app.info, app.warning) for item in kind]
        self.assertTrue(any("选择" in message and "销售额" in message for message in messages))
        self.assertEqual(len(app.dataframe[0].value), 10)
        with patch("streamlit.pyplot", wraps=st.pyplot) as pyplot:
            app.selectbox(key="sales_column").select("金额").run(timeout=10)
        self.assertFalse(app.exception)
        self.assertEqual([metric.value for metric in app.metric], ["78.00", "6.50", "12", "2"])
        self.assertEqual(list(app.dataframe[2].value["销售额合计"]), [36, 42])
        self.assertEqual(len(app.image), 2)
        lines = [line for call in pyplot.call_args_list for axis in call.args[0].axes for line in axis.lines]
        self.assertEqual(list(lines[0].get_ydata()), list(range(1, 13)))

    def test_excel_selected_amount_uses_all_rows(self):
        data = pd.DataFrame({"商品": ["A"] * 12, "金额": range(1, 13)})
        excel = BytesIO()
        data.to_excel(excel, index=False, engine="openpyxl")
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload("sales.xlsx", excel.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet").run(timeout=10)
        self.assertFalse(app.exception)
        self.assertFalse(app.metric)
        app.selectbox(key="sales_column").select("金额").run(timeout=10)
        self.assertFalse(app.exception)
        self.assertEqual([metric.value for metric in app.metric], ["78.00", "6.50", "12", "2"])
        self.assertEqual(app.dataframe[2].value["销售额合计"].iloc[0], 78)
        self.assertEqual(len(app.image), 2)

    def test_nonnumeric_selected_column_shows_numeric_error(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        self.upload_csv(app, "商品,金额\nA,10\nB,20\n")
        app.selectbox(key="sales_column").select("商品").run(timeout=10)
        self.assertFalse(app.exception)
        self.assertTrue(any("数字" in error.value for error in app.error))
        self.assertFalse(app.metric)
        self.assertFalse(app.image)

    def test_new_upload_resets_selected_amount_even_when_column_still_exists(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        self.upload_csv(app, "商品,金额\nA,10\nB,20\n")
        app.selectbox(key="sales_column").select("金额").run(timeout=10)
        self.assertEqual(app.metric[0].value, "30.00")
        self.upload_csv(app, "商品,金额\nA,100\nB,200\n", filename="next.csv")
        self.assertIsNone(app.selectbox(key="sales_column").value)
        self.assertFalse(app.metric)
        self.upload_csv(app, "商品,销售额,金额\nA,7,100\nB,8,200\n", filename="default.csv")
        self.assertEqual(app.selectbox(key="sales_column").value, "销售额")
        self.assertEqual(app.metric[0].value, "15.00")

    def test_changing_sales_column_clears_completed_or_pending_agent_run(self):
        for tool_name, state_key in [("sales_summary", "agent_result"), ("full_csv_export", "agent_pending")]:
            with self.subTest(state_key=state_key):
                plan = {"steps": [{"tool": tool_name, "args": {}}]}
                responses = [AIMessage(content=json.dumps(plan)), AIMessage(content="", tool_calls=[
                    {"name": tool_name, "args": {}, "id": "selected-sales-tool"}
                ])]
                if tool_name == "sales_summary":
                    responses.append(AIMessage(content="销售总额30.00。"))
                model = ScriptedChatModel(responses=responses)
                with tempfile.TemporaryDirectory() as directory, patch.object(trace_store, "DEFAULT_LOG_DIR", Path(directory)), patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-field-selection-key"}), patch("langchain_openai.ChatOpenAI", return_value=model):
                    app = AppTest.from_file(APP_PATH).run(timeout=10)
                    self.upload_csv(app, "商品,销售额,金额\nA,10,100\nB,20,200\n")
                    app.text_input(key="agent_question").input("查询汇总" if tool_name == "sales_summary" else "导出全部原始数据").run(timeout=10)
                    app.button(key="agent_ask").click().run(timeout=10)
                    self.assertFalse(app.exception)
                    self.assertIsNotNone(app.session_state[state_key])
                    app.selectbox(key="sales_column").select("金额").run(timeout=10)
                    self.assertFalse(app.exception)
                    self.assertEqual(app.metric[0].value, "300.00")
                    self.assertNotIn("agent_pending", app.session_state)
                    self.assertNotIn("agent_result", app.session_state)
                    self.assertFalse(app.downloads)
                    self.assertFalse(any(button.key == "agent_approve" for button in app.button))


if __name__ == "__main__":
    unittest.main()


