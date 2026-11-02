import unittest
import os
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import streamlit as st
from authenticated_app import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class QwenPageInitializationTest(unittest.TestCase):
    def test_initial_and_uploaded_page_have_no_connection_test_or_model_request(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-initial-page-key"}), \
                patch("langchain_openai.ChatOpenAI") as model:
            app = AppTest.from_file(APP_PATH).run(timeout=10)
            self.assertFalse(app.exception)
            self.assertNotIn("通义千问连接测试", [button.label for button in app.button])
            self.assertNotIn("模型连接", [expander.label for expander in app.sidebar.expander])
            model.assert_not_called()
            app.file_uploader[0].upload(
                "sales.csv", "销售额\n10\n20\n".encode("utf-8"), "text/csv"
            ).run(timeout=10)
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "30.00")
            self.assertNotIn("通义千问连接测试", [button.label for button in app.button])
            self.assertNotIn("模型连接", [expander.label for expander in app.sidebar.expander])
            model.assert_not_called()


class QwenAnalysisTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "test-analysis-key-no-real-request",
            "DASHSCOPE_BASE_URL": "https://analysis.example/compatible-mode/v1",
        })
        env.start()
        self.addCleanup(env.stop)
        model = patch("langchain_openai.ChatOpenAI")
        self.model = model.start()
        self.addCleanup(model.stop)
        self.model.return_value.invoke.return_value = SimpleNamespace(content="销售总额为 78.00，平均值为 6.50。")

    def uploaded_app(self):
        rows = "\n".join(f"PRIVATE_PRODUCT,{i},PRIVATE_NOTE_{i}" for i in range(1, 13))
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.csv", f"商品,销售额,备注\n{rows}\n".encode("utf-8"), "text/csv"
        ).run(timeout=10)
        self.assertFalse(app.exception)
        self.assertTrue(app.text_input, "上传有效数据后应显示自然语言问题输入框")
        return app

    def test_analysis_controls_do_not_call_model_before_click(self):
        app = self.uploaded_app()
        app.text_input(key="analysis_question").input("销售额总和是多少？").run(timeout=10)

        self.assertFalse(app.exception)
        self.assertTrue(app.button(key="analysis_ask"))
        self.assertEqual(app.metric[0].value, "78.00")
        self.model.assert_not_called()

    def test_analysis_sends_only_question_and_all_row_summary_and_shows_reply(self):
        app = self.uploaded_app()
        question = "请解释总额和平均值。"
        app.text_input(key="analysis_question").input(question).run(timeout=10)
        app.button(key="analysis_ask").click().run(timeout=10)

        self.assertFalse(app.exception)
        self.model.return_value.invoke.assert_called_once()
        request = str(self.model.return_value.invoke.call_args.args[0])
        for expected in [question, "12", "78.00", "6.50"]:
            self.assertIn(expected, request)
        for private in ["PRIVATE_PRODUCT", "PRIVATE_NOTE", "商品", "备注", "test-analysis-key-no-real-request"]:
            self.assertNotIn(private, request)
        self.assertNotIn("1,2,3", request.replace(" ", ""))
        self.assertTrue(any("销售总额为 78.00，平均值为 6.50。" in item.value for item in app.markdown))
        self.assertEqual(len(app.dataframe[0].value), 10)
        self.assertEqual(self.model.call_args.kwargs["base_url"], "https://analysis.example/compatible-mode/v1")
        self.assertNotIn("test-analysis-key-no-real-request", str(app))

    def test_analysis_empty_question_shows_hint_without_request(self):
        app = self.uploaded_app()
        app.text_input(key="analysis_question").input("   ").run(timeout=10)
        app.button(key="analysis_ask").click().run(timeout=10)

        self.assertFalse(app.exception)
        messages = [item.value for kind in (app.info, app.warning, app.error) for item in kind]
        self.assertTrue(any("问题" in message for message in messages))
        self.model.assert_not_called()

    def test_analysis_missing_key_shows_configuration_hint_without_request(self):
        os.environ["DASHSCOPE_API_KEY"] = ""
        app = self.uploaded_app()
        app.text_input(key="analysis_question").input("总额是多少？").run(timeout=10)
        app.button(key="analysis_ask").click().run(timeout=10)

        self.assertFalse(app.exception)
        messages = [item.value for kind in (app.info, app.warning, app.error) for item in kind]
        self.assertTrue(any("DASHSCOPE_API_KEY" in message for message in messages))
        self.model.assert_not_called()

    def test_analysis_failure_shows_safe_hint_and_preserves_statistics(self):
        self.model.return_value.invoke.side_effect = RuntimeError("PRIVATE_ERROR test-analysis-key-no-real-request")
        app = self.uploaded_app()
        app.text_input(key="analysis_question").input("请分析总额。").run(timeout=10)
        app.button(key="analysis_ask").click().run(timeout=10)

        self.assertFalse(app.exception)
        self.assertTrue(any("失败" in item.value for item in app.error))
        self.assertNotIn("PRIVATE_ERROR", str(app))
        self.assertNotIn("test-analysis-key-no-real-request", str(app))
        self.assertEqual(app.metric[0].value, "78.00")

    def test_product_question_uses_all_matching_rows_without_other_product_details(self):
        data = pd.DataFrame({
            "商品": ["笔记本"] + ["OTHER_SECRET_PRODUCT"] * 9 + ["笔记本", "笔记本", None],
            "销售额": [1] + [123.45] * 9 + [11, 12, 77],
            "备注": ["PRIVATE_PRODUCT_NOTE"] * 13,
        })
        excel = BytesIO()
        data.to_excel(excel, index=False, engine="openpyxl")
        files = [
            ("sales.csv", data.to_csv(index=False).encode("utf-8"), "text/csv"),
            ("sales.xlsx", excel.getvalue(),
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ]
        for filename, content, mime in files:
            with self.subTest(filename=filename):
                self.model.reset_mock()
                self.model.return_value.invoke.return_value = SimpleNamespace(content="笔记本销售额合计为 24.00。")
                app = AppTest.from_file(APP_PATH).run(timeout=10)
                app.file_uploader[0].upload(filename, content, mime).run(timeout=10)
                question = "笔记本的销售额是多少？笔记本合计告诉我。"
                app.text_input(key="analysis_question").input(question).run(timeout=10)
                self.model.assert_not_called()
                app.button(key="analysis_ask").click().run(timeout=10)

                self.assertFalse(app.exception)
                self.model.return_value.invoke.assert_called_once()
                request = str(self.model.return_value.invoke.call_args.args[0])
                self.assertIn(question, request)
                self.assertIn("24.00", request, "应包含全部重复商品的合计，包括第 11、12 行")
                for private in ["OTHER_SECRET_PRODUCT", "123.45", "1111.05", "PRIVATE_PRODUCT_NOTE", "test-analysis-key-no-real-request"]:
                    self.assertNotIn(private, request)
                self.assertTrue(any("笔记本销售额合计为 24.00。" in item.value for item in app.markdown))
                self.assertEqual(len(app.dataframe[0].value), 10)

    def test_multiple_product_question_shows_hint_without_model_request(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.csv", "商品,销售额\n笔记本,10\n鼠标,20\n笔记本,30\n".encode("utf-8"), "text/csv"
        ).run(timeout=10)
        app.text_input(key="analysis_question").input("笔记本和鼠标各卖了多少？").run(timeout=10)
        app.button(key="analysis_ask").click().run(timeout=10)

        self.assertFalse(app.exception)
        messages = [item.value for kind in (app.info, app.warning, app.error) for item in kind]
        self.assertTrue(any("一个" in message and "商品" in message for message in messages))
        self.model.assert_not_called()
        self.assertEqual(app.metric[0].value, "60.00")

    def test_unmatched_or_missing_product_keeps_base_summary_question(self):
        files = [
            "商品,销售额\nOTHER_SECRET_PRODUCT,10\nOTHER_SECRET_PRODUCT,20\n",
            "销售额\n10\n20\n",
            "商品,销售额\n,10\n   ,20\n",
        ]
        for csv in files:
            with self.subTest(csv=csv):
                self.model.reset_mock()
                app = AppTest.from_file(APP_PATH).run(timeout=10)
                app.file_uploader[0].upload("sales.csv", csv.encode("utf-8"), "text/csv").run(timeout=10)
                app.text_input(key="analysis_question").input("请分析 销售总额。").run(timeout=10)
                app.button(key="analysis_ask").click().run(timeout=10)

                self.assertFalse(app.exception)
                self.model.return_value.invoke.assert_called_once()
                request = str(self.model.return_value.invoke.call_args.args[0])
                self.assertIn("30.00", request)
                self.assertIn("15.00", request)
                self.assertNotIn("OTHER_SECRET_PRODUCT", request)
                self.assertNotIn("10.00", request)
                self.assertNotIn("20.00", request)
                self.assertTrue(any("销售总额为 78.00" in item.value for item in app.markdown))


class SalesAppTest(unittest.TestCase):
    def test_line_chart_csv_and_xlsx_use_all_rows(self):
        data = pd.DataFrame({"商品": ["B"] * 12, "销售额": range(1, 13)})
        excel = BytesIO()
        data.to_excel(excel, index=False, engine="openpyxl")
        files = [
            ("sales.csv", data.to_csv(index=False).encode("utf-8"), "text/csv"),
            ("sales.xlsx", excel.getvalue(),
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ]
        for filename, content, mime in files:
            with self.subTest(filename=filename):
                app = AppTest.from_file(APP_PATH).run(timeout=10)
                with patch("streamlit.pyplot", wraps=st.pyplot) as pyplot:
                    app.file_uploader[0].upload(filename, content, mime).run(timeout=10)

                self.assertFalse(app.exception)
                self.assertEqual(len(app.image), 2)
                self.assertTrue(all(image.value[0] for image in app.image))
                self.assertIn("逐条销售额折线图", [header.value for header in app.subheader])
                self.assertEqual(len(app.dataframe), 3)
                self.assertEqual(app.dataframe[2].value["销售额合计"].iloc[0], 78)
                metrics = {metric.label: metric.value for metric in app.metric}
                self.assertEqual(metrics["销售额总和"], "78.00")
                self.assertEqual(metrics["销售额平均值"], "6.50")
                lines = [
                    line for call in pyplot.call_args_list
                    for axis in call.args[0].axes for line in axis.lines
                ]
                self.assertEqual(len(lines), 1)
                self.assertEqual(list(lines[0].get_xdata()), list(range(1, 13)))
                self.assertEqual(list(lines[0].get_ydata()), list(range(1, 13)))

    def test_line_chart_without_product_column_is_displayed(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.csv", "销售额\n10\n20\n".encode("utf-8"), "text/csv"
        ).run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.image), 1)
        self.assertIn("逐条销售额折线图", [header.value for header in app.subheader])
        self.assertEqual(len(app.dataframe), 2)
        self.assertEqual(app.metric[0].value, "30.00")

    def test_line_chart_is_absent_for_empty_or_invalid_sales(self):
        for csv in ["商品,销售额\n", "商品,销售额\nB,abc\n", "商品,金额\nB,10\n"]:
            with self.subTest(csv=csv):
                app = AppTest.from_file(APP_PATH).run(timeout=10)
                app.file_uploader[0].upload(
                    "sales.csv", csv.encode("utf-8"), "text/csv"
                ).run(timeout=10)

                self.assertFalse(app.exception)
                self.assertFalse(app.image)
                self.assertNotIn("逐条销售额折线图", [header.value for header in app.subheader])

    def test_bar_chart_csv_and_xlsx_preserve_analysis_results(self):
        data = pd.DataFrame({"商品": ["B", "A", "B"], "销售额": [10, 20, 30]})
        excel = BytesIO()
        data.to_excel(excel, index=False, engine="openpyxl")
        files = [
            ("sales.csv", data.to_csv(index=False).encode("utf-8"), "text/csv"),
            ("sales.xlsx", excel.getvalue(),
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ]
        for filename, content, mime in files:
            with self.subTest(filename=filename):
                app = AppTest.from_file(APP_PATH).run(timeout=10)
                app.file_uploader[0].upload(filename, content, mime).run(timeout=10)

                self.assertFalse(app.exception)
                self.assertEqual(len(app.dataframe), 3)
                self.assertEqual(list(app.dataframe[2].value["销售额合计"]), [40, 20])
                metrics = {metric.label: metric.value for metric in app.metric}
                self.assertEqual(metrics["销售额总和"], "60.00")
                self.assertEqual(metrics["销售额平均值"], "20.00")
                self.assertEqual(len(app.image), 2)
                self.assertTrue(app.image[0].value[0])

    def test_bar_chart_is_absent_without_groupable_data(self):
        for csv in ["销售额\n10\n20\n", "商品,销售额\n"]:
            with self.subTest(csv=csv):
                app = AppTest.from_file(APP_PATH).run(timeout=10)
                app.file_uploader[0].upload(
                    "sales.csv", csv.encode("utf-8"), "text/csv"
                ).run(timeout=10)

                self.assertFalse(app.exception)
                self.assertNotIn("按商品统计销售额", [header.value for header in app.subheader])
                self.assertEqual(len(app.dataframe), 2)
                self.assertTrue(any("无法分组" in info.value for info in app.info))

    def test_csv_grouping_uses_all_rows_in_first_seen_order(self):
        products = ["B", "A"] * 5 + ["B", "C"]
        csv = "商品,销售额\n" + "\n".join(
            f"{product},{amount}" for amount, product in enumerate(products, 1)
        )
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload("sales.csv", csv.encode("utf-8"), "text/csv").run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 3)
        self.assertEqual(len(app.dataframe[0].value), 10)
        grouped = app.dataframe[2].value
        self.assertEqual(list(grouped.columns), ["商品", "销售额合计"])
        self.assertEqual(list(grouped["商品"]), ["B", "A", "C"])
        self.assertEqual(list(grouped["销售额合计"]), [36, 30, 12])
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["销售额总和"], "78.00")
        self.assertEqual(metrics["销售额平均值"], "6.50")

    def test_grouping_keeps_missing_products_as_separate_group(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.csv", "商品,销售额\nB,10\n,20\nB,30\n,40\nA,50\n".encode("utf-8"), "text/csv"
        ).run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 3)
        grouped = app.dataframe[2].value
        self.assertEqual(list(grouped.columns), ["商品", "销售额合计"])
        self.assertEqual(len(grouped), 3)
        self.assertEqual(grouped["商品"].iloc[0], "B")
        self.assertTrue(pd.isna(grouped["商品"].iloc[1]))
        self.assertEqual(grouped["商品"].iloc[2], "A")
        self.assertEqual(list(grouped["销售额合计"]), [40, 60, 50])
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(float(metrics["销售额总和"]), grouped["销售额合计"].sum())

    def test_xlsx_grouping_is_displayed(self):
        excel = BytesIO()
        pd.DataFrame({"商品": ["B", "A", "B"], "销售额": [10, 20, 30]}).to_excel(
            excel, index=False, engine="openpyxl"
        )
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.xlsx", excel.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ).run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 3)
        grouped = app.dataframe[2].value
        self.assertEqual(list(grouped.columns), ["商品", "销售额合计"])
        self.assertEqual(list(grouped["商品"]), ["B", "A"])
        self.assertEqual(list(grouped["销售额合计"]), [40, 20])

    def test_grouping_without_product_column_shows_explanation(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.csv", "销售额\n10\n20\n".encode("utf-8"), "text/csv"
        ).run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 2)
        self.assertTrue(any("无法分组" in info.value for info in app.info))
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["销售额总和"], "30.00")
        self.assertEqual(metrics["销售额平均值"], "15.00")

    def test_grouping_empty_csv_shows_explanation(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.csv", "商品,销售额\n".encode("utf-8"), "text/csv"
        ).run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 2)
        self.assertTrue(any("无法分组" in info.value for info in app.info))
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["销售额总和"], "0.00")
        self.assertEqual(metrics.get("销售额平均值"), "—")
        self.assertEqual(metrics.get("记录数量"), "0")
        self.assertEqual(metrics.get("字段数量"), "2")
        self.assertTrue(any("无法计算平均值" in info.value for info in app.info))

    def test_csv_average_uses_all_rows(self):
        rows = "\n".join(f"{i},{i}" for i in range(1, 13))
        csv = f"订单号,销售额\n{rows}\n".encode("utf-8")

        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload("sales.csv", csv, "text/csv").run(timeout=10)

        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics.get("销售额总和"), "78.00")
        self.assertEqual(metrics.get("销售额平均值"), "6.50")

    def test_xlsx_average_is_displayed(self):
        excel = BytesIO()
        pd.DataFrame({"销售额": [10, 20]}).to_excel(
            excel, index=False, engine="openpyxl"
        )
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.xlsx",
            excel.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ).run(timeout=10)

        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics.get("销售额总和"), "30.00")
        self.assertEqual(metrics.get("销售额平均值"), "15.00")

    def test_empty_csv_average_shows_explanation(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.csv", "订单号,销售额\n".encode("utf-8"), "text/csv"
        ).run(timeout=10)

        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics.get("销售额总和"), "0.00")
        self.assertEqual(metrics.get("销售额平均值"), "—")
        self.assertEqual(metrics.get("记录数量"), "0")
        self.assertEqual(metrics.get("字段数量"), "2")
        self.assertTrue(any("无法计算平均值" in info.value for info in app.info))

    def test_csv_field_information_counts_all_rows(self):
        rows = "\n".join(
            f"{i},{i},{'正常' if i < 12 else ''}" for i in range(1, 13)
        )
        csv = f"订单号,销售额,备注\n{rows}\n".encode("utf-8")
        data = pd.read_csv(BytesIO(csv))

        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload("sales.csv", csv, "text/csv").run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 2)
        pd.testing.assert_frame_equal(app.dataframe[0].value, data.head(10))
        fields = app.dataframe[1].value
        self.assertEqual(
            list(fields.columns), ["字段名", "数据类型", "非空数量", "缺失数量"]
        )
        self.assertEqual(list(fields["字段名"]), ["订单号", "销售额", "备注"])
        self.assertEqual(list(fields["数据类型"]), [str(dtype) for dtype in data.dtypes])
        self.assertEqual(list(fields["非空数量"]), [12, 12, 11])
        self.assertEqual(list(fields["缺失数量"]), [0, 0, 1])
        self.assertEqual(app.metric[0].value, "78.00")

    def test_xlsx_displays_field_information(self):
        data = pd.DataFrame({"订单号": [1, 2], "销售额": [10, 20]})
        excel = BytesIO()
        data.to_excel(excel, index=False, engine="openpyxl")

        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.xlsx",
            excel.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ).run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 2)
        pd.testing.assert_frame_equal(app.dataframe[0].value, data)
        fields = app.dataframe[1].value
        self.assertEqual(
            list(fields.columns), ["字段名", "数据类型", "非空数量", "缺失数量"]
        )
        self.assertEqual(list(fields["字段名"]), ["订单号", "销售额"])
        self.assertEqual(list(fields["数据类型"]), ["int64", "int64"])
        self.assertEqual(list(fields["非空数量"]), [2, 2])
        self.assertEqual(list(fields["缺失数量"]), [0, 0])
        self.assertEqual(app.metric[0].value, "30.00")

    def test_xlsx_upload_displays_first_ten_rows_and_sums_first_sheet(self):
        first_sheet = pd.DataFrame({"订单号": range(1, 13), "销售额": range(1, 13)})
        second_sheet = pd.DataFrame({"订单号": [99], "销售额": [999]})
        excel = BytesIO()
        with pd.ExcelWriter(excel, engine="openpyxl") as writer:
            first_sheet.to_excel(writer, index=False, sheet_name="销售明细")
            second_sheet.to_excel(writer, index=False, sheet_name="其他数据")

        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.xlsx",
            excel.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ).run(timeout=10)

        self.assertFalse(app.exception)
        pd.testing.assert_frame_equal(app.dataframe[0].value, first_sheet.head(10))
        self.assertEqual(app.metric[0].value, "78.00")

    def test_corrupt_xlsx_shows_message(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "broken.xlsx",
            b"not a valid Excel workbook",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ).run(timeout=10)

        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertIn("Excel", app.error[0].value)
        self.assertFalse(app.metric)
        self.assertFalse(app.dataframe)

    def test_upload_displays_first_ten_rows_and_sums_all_rows(self):
        rows = "\n".join(f"{i},{i}" for i in range(1, 13))
        csv = f"订单号,销售额\n{rows}\n".encode("utf-8")

        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload("sales.csv", csv, "text/csv").run(timeout=10)

        self.assertEqual(len(app.dataframe[0].value), 10)
        self.assertEqual(app.metric[0].value, "78.00")
        self.assertFalse(app.exception)

    def test_missing_sales_column_prompts_selection(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.csv", "订单号,金额\n1,100\n".encode("utf-8"), "text/csv"
        ).run(timeout=10)

        messages = [item.value for kind in (app.info, app.warning) for item in kind]
        self.assertTrue(any("选择" in message and "销售额" in message for message in messages))
        self.assertFalse(app.metric)

    def test_non_numeric_sales_shows_message(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.csv", "订单号,销售额\n1,abc\n".encode("utf-8"), "text/csv"
        ).run(timeout=10)

        self.assertIn("数字", app.error[0].value)
        self.assertFalse(app.metric)


if __name__ == "__main__":
    unittest.main()

