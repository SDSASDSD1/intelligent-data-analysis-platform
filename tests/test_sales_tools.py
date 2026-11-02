import importlib
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from langchain_core.tools import BaseTool
from authenticated_app import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class ProductSalesToolTest(unittest.TestCase):
    def factory(self):
        self.assertIsNotNone(
            importlib.util.find_spec("sales_tools"),
            "尚未实现 sales_tools.make_product_sales_tool",
        )
        return importlib.import_module("sales_tools").make_product_sales_tool

    def test_tool_schema_exposes_only_required_product_name(self):
        tool = self.factory()(pd.DataFrame({"商品": ["笔记本"], "销售额合计": [24]}))

        self.assertIsInstance(tool, BaseTool)
        schema = tool.get_input_schema().model_json_schema()
        self.assertEqual(list(schema["properties"]), ["product_name"])
        self.assertEqual(schema["properties"]["product_name"]["type"], "string")
        self.assertEqual(schema["required"], ["product_name"])

    def test_tool_returns_aggregated_total_for_exact_product_name(self):
        data = pd.DataFrame({"商品": ["笔记本", "鼠标", "笔记本", "笔记本"], "销售额": [1, 99, 11, 12]})
        grouped = data.groupby("商品", as_index=False, sort=False)["销售额"].sum().rename(
            columns={"销售额": "销售额合计"}
        )
        original = grouped.copy(deep=True)
        tool = self.factory()(grouped)

        result = tool.invoke({"product_name": "笔记本"})

        self.assertEqual(result, {"product_name": "笔记本", "sales_total": 24.0})
        self.assertIs(type(result["sales_total"]), float)
        pd.testing.assert_frame_equal(grouped, original)

    def test_tool_rejects_blank_unknown_partial_and_code_strings(self):
        grouped = pd.DataFrame({"商品": ["笔记本"], "销售额合计": [24]})
        original = grouped.copy(deep=True)
        tool = self.factory()(grouped)
        for name in ["", "   ", "不存在", "笔记", "__import__('os').getcwd()"]:
            with self.subTest(product_name=name):
                result = tool.invoke({"product_name": name})
                self.assertIsInstance(result, dict)
                self.assertTrue(result.get("error"))
                self.assertNotIn("sales_total", result)
        pd.testing.assert_frame_equal(grouped, original)

    def test_app_uses_real_tool_result_and_preserves_analysis(self):
        factory = self.factory()
        tools = []

        def tracked_factory(grouped):
            tool = Mock(wraps=factory(grouped))
            tools.append(tool)
            return tool

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-tool-test-key"}), patch(
            "sales_tools.make_product_sales_tool", side_effect=tracked_factory
        ), patch("langchain_openai.ChatOpenAI") as model:
            model.return_value.invoke.return_value = SimpleNamespace(content="笔记本合计 24.00。")
            app = AppTest.from_file(APP_PATH).run(timeout=10)
            app.file_uploader[0].upload(
                "sales.csv", "商品,销售额\n笔记本,1\nOTHER_PRIVATE_PRODUCT,99\n笔记本,11\n笔记本,12\n".encode("utf-8"), "text/csv"
            ).run(timeout=10)
            app.text_input(key="analysis_question").input("笔记本的销售额是多少？").run(timeout=10)
            model.assert_not_called()
            app.button(key="analysis_ask").click().run(timeout=10)

            self.assertFalse(app.exception)
            used_tools = [tool for tool in tools if tool.invoke.called]
            self.assertEqual(len(used_tools), 1, "页面应实际调用商品查询工具")
            used_tools[0].invoke.assert_called_once_with({"product_name": "笔记本"})
            model.return_value.invoke.assert_called_once()
            request = str(model.return_value.invoke.call_args.args[0])
            self.assertIn("24.00", request)
            self.assertNotIn("OTHER_PRIVATE_PRODUCT", request)
            self.assertTrue(any("笔记本合计 24.00。" in item.value for item in app.markdown))
            self.assertEqual(app.metric[0].value, "123.00")
            self.assertEqual(len(app.dataframe), 3)
            self.assertEqual(len(app.image), 2)


class SalesSummaryToolTest(unittest.TestCase):
    def factory(self):
        module = importlib.import_module("sales_tools")
        self.assertTrue(
            hasattr(module, "make_sales_summary_tool"),
            "尚未实现 sales_tools.make_sales_summary_tool",
        )
        return module.make_sales_summary_tool

    def test_summary_tool_has_no_inputs_and_counts_all_rows(self):
        sales = pd.Series(range(1, 13), name="销售额")
        original = sales.copy(deep=True)
        tool = self.factory()(sales)

        self.assertIsInstance(tool, BaseTool)
        schema = tool.get_input_schema().model_json_schema()
        self.assertEqual(schema["properties"], {})
        self.assertFalse(schema.get("required"))
        result = tool.invoke({})
        self.assertEqual(result, {"row_count": 12, "sales_total": 78.0, "sales_average": 6.5})
        self.assertIs(type(result["row_count"]), int)
        self.assertIs(type(result["sales_total"]), float)
        self.assertIs(type(result["sales_average"]), float)
        pd.testing.assert_series_equal(sales, original)

    def test_summary_tool_empty_series_has_no_average(self):
        sales = pd.Series([], dtype="float64", name="销售额")
        original = sales.copy(deep=True)
        result = self.factory()(sales).invoke({})

        self.assertEqual(result, {"row_count": 0, "sales_total": 0.0, "sales_average": None})
        self.assertIs(type(result["row_count"]), int)
        self.assertIs(type(result["sales_total"]), float)
        pd.testing.assert_series_equal(sales, original)

    def test_app_summary_question_uses_real_summary_tool(self):
        factory = self.factory()
        tools = []

        def tracked_factory(sales):
            tool = Mock(wraps=factory(sales))
            tools.append(tool)
            return tool

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-summary-test-key"}), patch(
            "sales_tools.make_sales_summary_tool", side_effect=tracked_factory
        ), patch("langchain_openai.ChatOpenAI") as model:
            model.return_value.invoke.return_value = SimpleNamespace(content="总额78.00，平均6.50。")
            rows = "\n".join(f"PRIVATE_PRODUCT,{i},PRIVATE_NOTE" for i in range(1, 13))
            app = AppTest.from_file(APP_PATH).run(timeout=10)
            app.file_uploader[0].upload(
                "sales.csv", f"商品,销售额,备注\n{rows}\n".encode("utf-8"), "text/csv"
            ).run(timeout=10)
            question = "总额和平均值是多少？"
            app.text_input(key="analysis_question").input(question).run(timeout=10)
            model.assert_not_called()
            app.button(key="analysis_ask").click().run(timeout=10)

            self.assertFalse(app.exception)
            used_tools = [tool for tool in tools if tool.invoke.called]
            self.assertEqual(len(used_tools), 1, "页面应实际调用销售摘要工具")
            used_tools[0].invoke.assert_called_once_with({})
            model.return_value.invoke.assert_called_once()
            messages = model.return_value.invoke.call_args.args[0]
            self.assertEqual(messages[1], (
                "human", f"数据行数：12\n销售额总和：78.00\n销售额平均值：6.50\n问题：{question}"
            ))
            request = str(messages)
            for private in ["PRIVATE_PRODUCT", "PRIVATE_NOTE", "fake-summary-test-key"]:
                self.assertNotIn(private, request)
            self.assertTrue(any("总额78.00，平均6.50。" in item.value for item in app.markdown))
            self.assertEqual(app.metric[0].value, "78.00")
            self.assertEqual(len(app.dataframe), 3)
            self.assertEqual(len(app.image), 2)


class ToolSelectionTest(unittest.TestCase):
    def selector(self):
        module = importlib.import_module("sales_tools")
        self.assertTrue(hasattr(module, "select_tool_calls"), "尚未实现 sales_tools.select_tool_calls")
        return module.select_tool_calls

    def test_selection_preserves_matching_semantics_and_tool_order(self):
        select = self.selector()
        grouped = pd.DataFrame({"商品": ["笔记本", "鼠标", None, "   "], "销售额合计": [24, 99, 5, 6]})
        original = grouped.copy(deep=True)
        summary = [("sales_summary", {})]
        cases = [
            ("总额是多少？", grouped, summary),
            ("笔记本是多少？笔记本合计。", grouped, summary + [("product_sales_total", {"product_name": "笔记本"})]),
            ("笔记本与鼠标各多少？", grouped, []),
            ("笔记本合计", None, summary),
            ("总额是多少？", pd.DataFrame({"销售额合计": [24]}), summary),
            ("总额是多少？", pd.DataFrame({"商品": [None, "   "], "销售额合计": [5, 6]}), summary),
            ("笔记本合计", pd.DataFrame({"商品": [" 笔记本 "], "销售额合计": [24]}),
             summary + [("product_sales_total", {"product_name": " 笔记本 "})]),
        ]
        for question, data, expected in cases:
            with self.subTest(question=question, expected=expected):
                self.assertEqual(select(question, data), expected)
        pd.testing.assert_frame_equal(grouped, original)

    def test_app_selection_executes_tools_and_shows_safe_tool_names(self):
        module = importlib.import_module("sales_tools")
        scenarios = [
            ("总额是多少？", ["sales_summary"], ["销售汇总"]),
            ("笔记本合计是多少？", ["sales_summary", "product_sales_total"], ["销售汇总", "指定商品销售额"]),
            ("笔记本和鼠标各多少？", [], []),
        ]
        for question, expected_calls, expected_names in scenarios:
            with self.subTest(question=question):
                calls = []

                def tracked_factory(name, factory):
                    def create(data):
                        real_tool = factory(data)

                        def invoke(arguments):
                            calls.append(name)
                            return real_tool.invoke(arguments)

                        spy = Mock(wraps=real_tool)
                        spy.invoke.side_effect = invoke
                        return spy
                    return create

                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-selection-test-key"}), patch(
                    "sales_tools.make_sales_summary_tool",
                    side_effect=tracked_factory("sales_summary", module.make_sales_summary_tool),
                ), patch(
                    "sales_tools.make_product_sales_tool",
                    side_effect=tracked_factory("product_sales_total", module.make_product_sales_tool),
                ), patch("langchain_openai.ChatOpenAI") as model:
                    model.return_value.invoke.return_value = SimpleNamespace(content="工具分析完成。")
                    app = AppTest.from_file(APP_PATH).run(timeout=10)
                    app.file_uploader[0].upload(
                        "sales.csv", "商品,销售额,备注\n笔记本,24,PRIVATE_ROW_NOTE\n鼠标,99,PRIVATE_ROW_NOTE\n".encode("utf-8"), "text/csv"
                    ).run(timeout=10)
                    app.text_input(key="analysis_question").input(question).run(timeout=10)
                    self.assertEqual(calls, [])
                    model.assert_not_called()
                    app.button(key="analysis_ask").click().run(timeout=10)

                    self.assertFalse(app.exception)
                    self.assertEqual(calls, expected_calls)
                    self.assertEqual(app.metric[0].value, "123.00")
                    self.assertEqual(len(app.dataframe), 3)
                    self.assertEqual(len(app.image), 2)
                    text = [item.value for kind in (app.markdown, app.caption, app.info, app.success) for item in kind]
                    if expected_calls:
                        model.return_value.invoke.assert_called_once()
                        displayed = [value for value in text if "本次工具" in value]
                        self.assertTrue(displayed, "成功回答时应显示本次工具名称")
                        displayed = "\n".join(displayed)
                        for name in expected_names:
                            self.assertIn(name, displayed)
                        if len(expected_names) == 1:
                            self.assertNotIn("指定商品销售额", displayed)
                        for private in ["PRIVATE_ROW_NOTE", "99", "fake-selection-test-key"]:
                            self.assertNotIn(private, displayed)
                        self.assertTrue(any("工具分析完成。" in value for value in text))
                    else:
                        model.assert_not_called()
                        messages = [item.value for kind in (app.info, app.warning, app.error) for item in kind]
                        self.assertTrue(any("一个" in value and "商品" in value for value in messages))


if __name__ == "__main__":
    unittest.main()

