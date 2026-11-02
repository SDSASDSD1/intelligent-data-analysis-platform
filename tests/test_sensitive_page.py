import tempfile
import unittest
from pathlib import Path

import pandas as pd
import streamlit as st
from authenticated_app import AppTest

import trace_store


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class SensitivePageTest(unittest.TestCase):
    def upload(self):
        data = pd.DataFrame({"商品": ["PRIVATE_PRODUCT_A", "PRIVATE_PRODUCT_B"],
                             "销售额": [10, 20], "邮箱": ["private-a@example.com", "private-b@example.com"],
                             "内部备注": ["PRIVATE_NOTE_A", "PRIVATE_NOTE_B"]})
        app = AppTest.from_file(APP_PATH).run(timeout=15)
        app.file_uploader[0].upload("sales.csv", data.to_csv(index=False).encode("utf-8"),
                                  "text/csv").run(timeout=15)
        self.assertFalse(app.exception)
        return app

    def test_automatically_sensitive_values_are_masked_in_plain_preview(self):
        app = self.upload()
        preview = app.dataframe[0].value
        self.assertEqual(preview["邮箱"].tolist(), ["需审核", "需审核"])
        self.assertEqual(preview["商品"].tolist(), ["PRIVATE_PRODUCT_A", "PRIVATE_PRODUCT_B"])
        self.assertEqual(app.metric[0].value, "30.00")

    def test_manually_marked_column_is_masked_on_same_page_rerun(self):
        app = self.upload()
        app.multiselect(key="sensitive_columns").set_value(["邮箱", "内部备注"]).run(timeout=15)
        self.assertFalse(app.exception)
        preview = app.dataframe[0].value
        self.assertEqual(preview["内部备注"].tolist(), ["需审核", "需审核"])
        self.assertEqual(preview["邮箱"].tolist(), ["需审核", "需审核"])
        self.assertNotIn("PRIVATE_NOTE", str([frame.value for frame in app.dataframe]))

    def test_sensitive_product_hides_group_table_and_bar_chart_but_keeps_line_and_metrics(self):
        app = self.upload()
        app.multiselect(key="sensitive_columns").set_value(["邮箱", "商品"]).run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value["商品"].tolist(), ["需审核", "需审核"])
        self.assertFalse(any(list(frame.value.columns) == ["商品", "销售额合计"] for frame in app.dataframe))
        self.assertNotIn("按商品统计销售额", [header.value for header in app.subheader])
        self.assertIn("逐条销售额折线图", [header.value for header in app.subheader])
        self.assertEqual(len(app.image), 1)
        self.assertEqual([metric.value for metric in app.metric], ["30.00", "15.00", "2", "4"])

    def test_builtin_dataframe_export_is_disabled_globally(self):
        old_value = st.get_option("client.disableDataExport")
        self.addCleanup(st.set_option, "client.disableDataExport", old_value)
        st.set_option("client.disableDataExport", False)
        self.upload()
        self.assertIs(st.get_option("client.disableDataExport"), True)


class SensitiveTraceRegressionTest(unittest.TestCase):
    def test_sensitive_values_are_redacted_even_from_status_and_decision_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = "PRIVATE_STATUS_SENSITIVE"
            event = {"trace_id": "status-regression", "event": "review", "status": marker,
                     "decision": marker}
            trace_store.append_trace(event, Path(directory), sensitive_values=(marker,))
            trace_store.append_audit(event, Path(directory), sensitive_values=(marker,))
            for filename in ["trace.jsonl", "audit.jsonl"]:
                self.assertNotIn(marker, (Path(directory) / filename).read_text(encoding="utf-8"))

    def test_manually_sensitive_values_are_redacted_from_query_plan_and_tool_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = "PRIVATE_PRODUCT_INTERNAL"
            event = {"trace_id": "sensitive-regression", "event": "tool_call", "query": f"分析{marker}",
                     "plan": {"steps": [{"tool": "product_sales_total", "args": {"product_name": marker}}]},
                     "tool": "product_sales_total", "args": {"product_name": marker}, "status": "success"}
            trace_store.append_trace(event, Path(directory), sensitive_values=(marker,))
            trace_store.append_audit({**event, "decision": "approve"}, Path(directory), sensitive_values=(marker,))
            for filename in ["trace.jsonl", "audit.jsonl"]:
                text = (Path(directory) / filename).read_text(encoding="utf-8")
                self.assertNotIn(marker, text)
                self.assertIn("product_sales_total", text)


if __name__ == "__main__":
    unittest.main()

