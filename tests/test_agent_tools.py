import importlib
import io
import unittest

import pandas as pd
from langchain_core.tools import BaseTool


class AgentToolsTest(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("sales_tools")
        self.data = pd.DataFrame({
            "商品": ["笔记本"] * 12,
            "销售额": list(range(1, 13)),
            "邮箱": ["PRIVATE_EMAIL@example.com"] * 12,
            "备注": [f"PRIVATE_NOTE_{i}" for i in range(12)],
        })
        self.original = self.data.copy(deep=True)
        self.grouped = pd.DataFrame({"商品": ["笔记本"], "销售额合计": [78]})
        self.artifacts = {}

    def factory(self, name):
        self.assertTrue(hasattr(self.module, name), f"尚未实现 {name}")
        return getattr(self.module, name)

    def tearDown(self):
        pd.testing.assert_frame_equal(self.data, self.original)

    def test_schema_excludes_sensitive_columns_and_all_row_values(self):
        tool = self.factory("make_schema_tool")(self.data, ["邮箱"])
        self.assertIsInstance(tool, BaseTool)
        self.assertEqual(tool.name, "schema_inspect")
        self.assertEqual(tool.get_input_schema().model_json_schema()["properties"], {})
        result = tool.invoke({})
        self.assertEqual(result, {"row_count": 12, "columns": ["商品", "销售额", "备注"]})
        self.assertNotIn("PRIVATE", str(result))

    def test_chart_selects_bar_or_line_without_returning_source_values(self):
        tool = self.factory("make_chart_tool")(self.data["销售额"], self.grouped, self.artifacts)
        self.assertEqual(tool.name, "sales_chart")
        for kind in ["bar", "line"]:
            with self.subTest(chart_type=kind):
                self.assertEqual(tool.invoke({"chart_type": kind}), {"status": "ready", "chart_type": kind})
                self.assertEqual(self.artifacts["chart"], kind)

    def test_chart_rejects_invalid_type_and_unavailable_data(self):
        factory = self.factory("make_chart_tool")
        cases = [
            (self.data["销售额"], self.grouped, "pie"),
            (self.data["销售额"], None, "bar"),
            (pd.Series([], dtype=float), self.grouped, "line"),
        ]
        for sales, grouped, kind in cases:
            with self.subTest(chart_type=kind, missing=grouped is None):
                artifacts = {}
                result = factory(sales, grouped, artifacts).invoke({"chart_type": kind})
                self.assertTrue(result.get("error"))
                self.assertNotIn("chart", artifacts)

    def test_column_preview_keeps_only_first_ten_values_in_local_artifact(self):
        tool = self.factory("make_column_preview_tool")(self.data, ["邮箱"], self.artifacts)
        self.assertEqual(tool.name, "column_preview")
        result = tool.invoke({"column_name": "备注"})
        self.assertEqual(result, {"status": "ready", "column_name": "备注", "row_count": 10})
        self.assertNotIn("PRIVATE", str(result))
        pd.testing.assert_frame_equal(self.artifacts["column_preview"], self.data[["备注"]].head(10))

    def test_column_preview_rejects_partial_or_unknown_column_without_artifact(self):
        tool = self.factory("make_column_preview_tool")(self.data, ["邮箱"], self.artifacts)
        for name in ["", "邮", "不存在", "import os"]:
            with self.subTest(column_name=name):
                self.assertTrue(tool.invoke({"column_name": name}).get("error"))
                self.assertNotIn("column_preview", self.artifacts)

    def test_full_csv_export_includes_every_row_with_utf8_bom_only_locally(self):
        tool = self.factory("make_full_csv_export_tool")(self.data, self.artifacts)
        self.assertEqual(tool.name, "full_csv_export")
        result = tool.invoke({})
        self.assertEqual(result, {"status": "ready", "row_count": 12})
        self.assertNotIn("PRIVATE", str(result))
        content = self.artifacts["full_csv"]
        self.assertIsInstance(content, bytes)
        self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
        pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(content)), self.data)

    def test_validator_accepts_only_six_exact_valid_tool_interfaces(self):
        validate = self.factory("validate_agent_tool_call")
        cases = [
            ("sales_summary", {}), ("product_sales_total", {"product_name": "笔记本"}),
            ("schema_inspect", {}), ("sales_chart", {"chart_type": "bar"}),
            ("sales_chart", {"chart_type": "line"}), ("column_preview", {"column_name": "邮箱"}),
            ("full_csv_export", {}),
        ]
        for name, args in cases:
            with self.subTest(name=name, args=args):
                self.assertIs(validate(name, args, self.data, self.grouped), True)

    def test_validator_rejects_unknown_names_extra_keys_bad_types_and_missing_data(self):
        validate = self.factory("validate_agent_tool_call")
        cases = [
            ("exec", {}), ("sales_summary", {"extra": 1}), ("schema_inspect", {"extra": 1}),
            ("full_csv_export", {"path": "elsewhere.csv"}), ("sales_summary", []),
            ("product_sales_total", {}), ("product_sales_total", {"product_name": "笔记"}),
            ("product_sales_total", {"product_name": 3}),
            ("product_sales_total", {"product_name": "笔记本", "extra": 1}),
            ("sales_chart", {"chart_type": "pie"}), ("sales_chart", {"chart_type": "BAR"}),
            ("sales_chart", {"chart_type": "bar", "extra": 1}),
            ("column_preview", {"column_name": "邮"}),
            ("column_preview", {"column_name": "邮箱", "extra": 1}),
        ]
        for name, args in cases:
            with self.subTest(name=name, args=args):
                self.assertIs(validate(name, args, self.data, self.grouped), False)
        self.assertIs(validate("sales_chart", {"chart_type": "bar"}, self.data, None), False)
        self.assertIs(validate("product_sales_total", {"product_name": "笔记本"}, self.data, None), False)

    def test_validator_rejects_code_strings_even_when_they_are_existing_column_names(self):
        validate = self.factory("validate_agent_tool_call")
        for code in ["import os", "exec('x')", "os.system('whoami')"]:
            with self.subTest(code=code):
                data = self.data.rename(columns={"备注": code})
                grouped = pd.DataFrame({"商品": [code], "销售额合计": [78]})
                self.assertIs(validate("column_preview", {"column_name": code}, data, grouped), False)
                self.assertIs(validate("product_sales_total", {"product_name": code}, data, grouped), False)


if __name__ == "__main__":
    unittest.main()

