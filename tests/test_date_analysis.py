import importlib
import importlib.util
import json
import os
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage
from authenticated_app import AppTest

import sales_tools
from test_analysis_agent import ScriptedChatModel


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def daily_fixture():
    return pd.DataFrame({
        "日期": ["2026-10-03 18:00:00", "2026-10-01 09:00:00"] * 5
                + ["2026-10-02 12:00:00", "2026-10-01 22:00:00"],
        "商品": ["PRIVATE_PRODUCT"] * 12,
        "销售额": range(1, 13),
    })


class DailySalesUnitTest(unittest.TestCase):
    def builder(self):
        self.assertIsNotNone(importlib.util.find_spec("date_analysis"), "尚未实现日期分析模块")
        return importlib.import_module("date_analysis").build_daily_sales

    def test_all_rows_are_grouped_by_calendar_day_in_ascending_order(self):
        data = daily_fixture()
        original = data.copy(deep=True)
        result = self.builder()(data, data["销售额"], "日期")
        self.assertEqual(list(result.columns), ["日期", "销售额合计"])
        self.assertEqual(pd.to_datetime(result["日期"]).dt.strftime("%Y-%m-%d").tolist(),
                         ["2026-10-01", "2026-10-02", "2026-10-03"])
        self.assertEqual(result["销售额合计"].tolist(), [42, 11, 25])
        pd.testing.assert_frame_equal(data, original)

    def test_missing_and_invalid_dates_do_not_silently_drop_sales(self):
        build = self.builder()
        for values in [["2026-10-01", None], ["2026-10-01", "bad-date"],
                       ["2026-10-01", "2026-02-30"]]:
            with self.subTest(values=values):
                data = pd.DataFrame({"日期": values, "销售额": [10, 20]})
                with self.assertRaises(ValueError):
                    build(data, data["销售额"], "日期")

    def test_date_tool_returns_only_daily_aggregates_and_checks_arguments(self):
        self.assertTrue(hasattr(sales_tools, "make_daily_sales_tool"), "尚未实现日期工具")
        data = daily_fixture()
        daily = pd.DataFrame({"日期": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03"]),
                              "销售额合计": [42, 11, 25]})
        tool = sales_tools.make_daily_sales_tool(daily)
        schema = tool.get_input_schema().model_json_schema()
        self.assertEqual(schema["properties"], {})
        self.assertFalse(schema.get("required"))
        result = tool.invoke({})
        self.assertEqual(result["daily_sales"], [
            {"日期": "2026-10-01", "销售额合计": 42.0},
            {"日期": "2026-10-02", "销售额合计": 11.0},
            {"日期": "2026-10-03", "销售额合计": 25.0},
        ])
        self.assertNotIn("PRIVATE_PRODUCT", str(result))
        self.assertTrue(sales_tools.validate_agent_tool_call(
            "daily_sales_summary", {}, data, None, daily=daily))
        self.assertFalse(sales_tools.validate_agent_tool_call(
            "daily_sales_summary", {}, data, None, daily=None))
        for args in [{"date_column": "日期"}, {"code": "exec('x')"},
                     {"date_column": "exec('x')"}, {"date_column": 123}]:
            with self.subTest(args=args):
                self.assertFalse(sales_tools.validate_agent_tool_call(
                    "daily_sales_summary", args, data, None, daily=daily))


class DailySalesAgentTest(unittest.TestCase):
    def test_agent_runs_selected_daily_summary_and_plan_requires_available_daily_data(self):
        module = importlib.import_module("analysis_agent")
        data = daily_fixture()
        daily = pd.DataFrame({"日期": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03"]),
                              "销售额合计": [42, 11, 25]})
        plan = {"steps": [{"tool": "daily_sales_summary", "args": {}}]}
        self.assertTrue(module.validate_analysis_plan(plan, data, None, daily=daily))
        self.assertFalse(module.validate_analysis_plan(plan, data, None, daily=None))
        model = ScriptedChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "daily_sales_summary", "args": {}, "id": "daily-1"}]),
            AIMessage(content="10月1日销售额最高，为42。"),
        ])
        agent = module.create_analysis_agent(model, data, data["销售额"], None, [], {}, daily=daily)
        result = agent.invoke({"messages": [("human", "按日期汇总销售额")]}, {"recursion_limit": 10})
        self.assertEqual(result["messages"][-1].content, "10月1日销售额最高，为42。")
        messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].tool_call_id, "daily-1")
        self.assertIn("2026-10-01", messages[0].content)
        self.assertNotIn("PRIVATE_PRODUCT", str(model.requests))


class DailySalesPageTest(unittest.TestCase):
    def test_marking_selected_date_sensitive_hides_daily_results_and_disables_date_agent_tool(self):
        model = ScriptedChatModel(responses=[
            AIMessage(content=json.dumps({"steps": [{"tool": "daily_sales_summary", "args": {}}]})),
        ])
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-sensitive-date"}), \
                patch("langchain_openai.ChatOpenAI", return_value=model), \
                patch("analysis_agent.create_analysis_agent") as create:
            app = self.upload(daily_fixture())
            app.multiselect(key="sensitive_columns").set_value(["日期"]).run(timeout=15)
            self.assertFalse(app.exception)
            self.assertFalse(any(list(frame.value.columns) == ["日期", "销售额合计"] for frame in app.dataframe))
            self.assertEqual(len(app.image), 2)
            self.assertEqual([metric.value for metric in app.metric], ["78.00", "6.50", "12", "3"])
            app.text_input(key="agent_question").input("按日期汇总销售额").run(timeout=15)
            app.button(key="agent_ask").click().run(timeout=15)
            self.assertFalse(app.exception)
            create.assert_not_called()
            self.assertEqual(len(model.requests), 1)

    def upload(self, data, excel=False):
        app = AppTest.from_file(APP_PATH).run(timeout=15)
        if excel:
            content = BytesIO()
            data.to_excel(content, index=False, engine="openpyxl")
            app.file_uploader[0].upload("sales.xlsx", content.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet").run(timeout=15)
        else:
            app.file_uploader[0].upload("sales.csv", data.to_csv(index=False).encode("utf-8"),
                                      "text/csv").run(timeout=15)
        self.assertFalse(app.exception)
        return app

    def test_csv_and_excel_default_date_column_show_sorted_daily_table_and_chart(self):
        for excel in [False, True]:
            with self.subTest(excel=excel):
                app = self.upload(daily_fixture(), excel)
                self.assertEqual(app.selectbox(key="date_column").value, "日期")
                tables = [element.value for element in app.dataframe
                          if list(element.value.columns) == ["日期", "销售额合计"]]
                self.assertEqual(len(tables), 1)
                self.assertEqual(tables[0]["销售额合计"].tolist(), [42, 11, 25])
                self.assertEqual(len(app.dataframe[0].value), 10)
                self.assertEqual(app.metric[0].value, "78.00")
                self.assertEqual(app.metric[1].value, "6.50")
                self.assertEqual(len(app.image), 3)

    def test_custom_date_column_can_be_selected(self):
        app = self.upload(daily_fixture().rename(columns={"日期": "下单时间"}))
        self.assertIsNone(app.selectbox(key="date_column").value)
        app.selectbox(key="date_column").select("下单时间").run(timeout=15)
        self.assertFalse(app.exception)
        tables = [element.value for element in app.dataframe
                  if list(element.value.columns) == ["日期", "销售额合计"]]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["销售额合计"].sum(), 78)

    def test_invalid_or_missing_dates_keep_sales_statistics_and_original_charts(self):
        for bad_date in [None, "bad-date"]:
            with self.subTest(bad_date=bad_date):
                data = daily_fixture()
                data.loc[11, "日期"] = bad_date
                app = self.upload(data)
                self.assertEqual(app.metric[0].value, "78.00")
                self.assertEqual(app.metric[1].value, "6.50")
                self.assertEqual(len(app.image), 2)
                self.assertEqual(len(app.dataframe), 3)
                messages = [item.value for kind in [app.info, app.warning, app.error] for item in kind]
                self.assertTrue(any("日期" in text for text in messages))

    def test_no_date_column_preserves_original_page(self):
        app = self.upload(pd.DataFrame({"商品": ["A", "B"], "销售额": [10, 20]}))
        self.assertIsNone(app.selectbox(key="date_column").value)
        self.assertEqual(app.metric[0].value, "30.00")
        self.assertEqual(len(app.dataframe), 3)
        self.assertEqual(len(app.image), 2)
        self.assertIsNotNone(app.button(key="agent_ask"))


if __name__ == "__main__":
    unittest.main()

