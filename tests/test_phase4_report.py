import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage
from authenticated_app import AppTest

import report_export
import review_store
import trace_store
from test_analysis_agent import APP_PATH, ScriptedChatModel


class ExtendedReportTest(unittest.TestCase):
    def setUp(self):
        self.sales = pd.Series([10, 20, 30], name="金额")
        self.daily = pd.DataFrame({"日期": pd.to_datetime(["2026-10-01", "2026-10-02"]),
                                   "销售额合计": [40, 20]})
        self.grouped = pd.DataFrame({"商品": ["PRIVATE_PRODUCT"], "销售额合计": [60]})
        self.arguments = dict(trace_id="report-phase4", question="分析销售", plan={"steps": []},
                              answer="总额60。", sales=self.sales, grouped=self.grouped)

    def test_report_includes_selected_column_names_and_every_daily_total(self):
        text = report_export.build_markdown_report(**self.arguments, sales_column="金额", date_column="日期",
                                                   daily=self.daily)
        for value in ["金额", "日期", "2026-10-01", "2026-10-02", "40.00", "20.00", "60.00"]:
            self.assertIn(value, text)

    def test_old_signature_still_works_and_sensitive_product_table_is_omitted(self):
        text = report_export.build_markdown_report(**self.arguments, sensitive_columns=["商品"])
        self.assertIn("60.00", text)
        self.assertNotIn("PRIVATE_PRODUCT", text)

    def test_sensitive_selected_date_column_is_not_listed_as_daily_values_in_report(self):
        text = report_export.build_markdown_report(**self.arguments, sales_column="金额", date_column="日期",
                                                   daily=self.daily, sensitive_columns=["日期"])
        self.assertNotIn("2026-10-01", text)
        self.assertNotIn("2026-10-02", text)
        self.assertIn("60.00", text)

    def test_report_redacts_sensitive_values_in_question_and_model_answer(self):
        marker = "PRIVATE_CUSTOM_VALUE"
        text = report_export.build_markdown_report(
            **{**self.arguments, "question": f"分析{marker}", "answer": f"结果涉及{marker}"},
            sensitive_values=(marker,))
        self.assertNotIn(marker, text)
        self.assertIn("60.00", text)

    def test_page_download_uses_selected_columns_and_computed_daily_table(self):
        data = pd.DataFrame({"商品": ["A", "B"], "金额": [10, 20],
                             "日期": ["2026-10-01", "2026-10-02"],
                             "邮箱": ["private@example.com", "other@example.com"]})
        model = ScriptedChatModel(responses=[
            AIMessage(content=json.dumps({"steps": [{"tool": "sales_summary", "args": {}}]})),
            AIMessage(content="", tool_calls=[{"name": "sales_summary", "args": {}, "id": "sum"}]),
            AIMessage(content="销售总和30。"),
        ])
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-report"}), \
                patch("langchain_openai.ChatOpenAI", return_value=model), \
                patch.object(trace_store, "DEFAULT_LOG_DIR", Path(directory) / "logs"), \
                patch.object(review_store, "DEFAULT_CHECKPOINT_PATH", Path(directory) / "checkpoint.sqlite"), \
                patch.object(report_export, "build_markdown_report", wraps=report_export.build_markdown_report) as build, \
                patch("secure_download.render_download") as download:
            app = AppTest.from_file(APP_PATH).run(timeout=15)
            app.file_uploader[0].upload("sales.csv", data.to_csv(index=False).encode("utf-8"), "text/csv").run(timeout=15)
            app.selectbox(key="sales_column").select("金额").run(timeout=15)
            app.text_input(key="agent_question").input("分析销售总额").run(timeout=15)
            app.button(key="agent_ask").click().run(timeout=15)
            self.assertFalse(app.exception)
            self.assertTrue(build.called)
            kwargs = build.call_args.kwargs
            self.assertEqual(kwargs.get("sales_column"), "金额")
            self.assertEqual(kwargs.get("date_column"), "日期")
            self.assertEqual(kwargs["daily"]["销售额合计"].tolist(), [10, 20])
            self.assertIn("private@example.com", kwargs["sensitive_values"])
            reports = [call for call in download.call_args_list if call.kwargs.get("key") == "agent_report_download"]
            self.assertEqual(len(reports), 1)
            self.assertIn("2026-10-02", reports[0].args[1].decode("utf-8"))


class LocalLogProtectionTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "Windows chmod 不等同于 ACL，权限断言仅适用 POSIX")
    def test_new_log_directory_and_jsonl_are_owner_only(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory) / "logs"
            trace_store.append_trace({"trace_id": "permission", "event": "test"}, logs)
            trace_store.append_audit({"trace_id": "permission", "event": "test"}, logs)
            self.assertEqual(stat.S_IMODE(logs.stat().st_mode), 0o700)
            for path in logs.glob("*.jsonl"):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_exception_text_in_allowed_field_does_not_leak_sensitive_values(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = "PRIVATE_EXCEPTION_VALUE"
            trace_store.append_trace({"trace_id": "error", "event": "error", "status": f"ValueError: {marker}"},
                                     Path(directory), sensitive_values=(marker,))
            text = (Path(directory) / "trace.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(marker, text)
            self.assertIn("ValueError", text)


if __name__ == "__main__":
    unittest.main()

