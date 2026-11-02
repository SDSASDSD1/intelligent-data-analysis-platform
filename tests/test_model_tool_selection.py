import importlib
import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage
from authenticated_app import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def proposed(name, args):
    return {"name": name, "args": args, "id": "test-call", "type": "tool_call"}


class ModelToolValidationTest(unittest.TestCase):
    def validator(self):
        module = importlib.import_module("sales_tools")
        self.assertTrue(hasattr(module, "validate_model_tool_calls"), "尚未实现 validate_model_tool_calls")
        return module.validate_model_tool_calls

    def test_valid_model_suggestions_have_safe_order_and_no_suggestion_is_empty(self):
        validate = self.validator()
        grouped = pd.DataFrame({"商品": ["笔记本", "鼠标"], "销售额合计": [24, 99]})
        original = grouped.copy(deep=True)
        summary = [("sales_summary", {})]
        self.assertEqual(validate([], "总额多少？", grouped), [])
        self.assertEqual(validate([proposed("sales_summary", {})], "总额多少？", None), summary)
        self.assertEqual(
            validate([proposed("product_sales_total", {"product_name": "笔记本"})], "笔记本合计？", grouped),
            summary + [("product_sales_total", {"product_name": "笔记本"})],
        )
        pd.testing.assert_frame_equal(grouped, original)

    def test_invalid_model_suggestions_are_rejected(self):
        validate = self.validator()
        grouped = pd.DataFrame({"商品": ["笔记本", None, "   "], "销售额合计": [24, 5, 6]})
        cases = [
            [proposed("unknown_tool", {})],
            [proposed("sales_summary", {"extra": "not allowed"})],
            [proposed("sales_summary", [])],
            [proposed("sales_summary", None)],
            [proposed("product_sales_total", {"product_name": "笔记本", "extra": 1})],
            [proposed("product_sales_total", {})],
            [proposed("product_sales_total", {"product_name": "不存在"})],
            [proposed("product_sales_total", {"product_name": "笔记"})],
            [proposed("product_sales_total", {"product_name": ""})],
            [proposed("product_sales_total", {"product_name": "   "})],
            [proposed("product_sales_total", {"product_name": 24})],
            [proposed("product_sales_total", "笔记本")],
            [proposed("sales_summary", {}), proposed("sales_summary", {})],
        ]
        for calls in cases:
            with self.subTest(calls=calls):
                self.assertIsNone(validate(calls, "笔记本合计？", grouped))
        product_call = [proposed("product_sales_total", {"product_name": "笔记本"})]
        for question, data in [
            ("只问总额", grouped), ("笔记本合计？", None),
            ("笔记本合计？", pd.DataFrame({"销售额合计": [24]})),
        ]:
            with self.subTest(question=question, data=data):
                self.assertIsNone(validate(product_call, question, data))

    def test_missing_or_empty_tool_call_id_is_rejected(self):
        validate = self.validator()
        for call_id in [None, "", "   "]:
            with self.subTest(call_id=call_id):
                call = proposed("sales_summary", {})
                if call_id is None:
                    call.pop("id")
                else:
                    call["id"] = call_id
                self.assertIsNone(validate([call], "总额多少？", None))


class ModelToolSelectionAppTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-model-selection-key"})
        env.start()
        self.addCleanup(env.stop)
        model = patch("langchain_openai.ChatOpenAI")
        self.model = model.start()
        self.addCleanup(model.stop)
        self.model.return_value.invoke.return_value = AIMessage(content="分析回答已完成。")
        self.executed = []
        module = importlib.import_module("sales_tools")
        for name, factory_name in [
            ("sales_summary", "make_sales_summary_tool"),
            ("product_sales_total", "make_product_sales_tool"),
        ]:
            real_factory = getattr(module, factory_name)

            def tracked_factory(data, name=name, real_factory=real_factory):
                real_tool = real_factory(data)

                def invoke(arguments):
                    self.executed.append(name)
                    return real_tool.invoke(arguments)

                spy = Mock(wraps=real_tool)
                spy.invoke.side_effect = invoke
                return spy

            factory_patch = patch(f"sales_tools.{factory_name}", side_effect=tracked_factory)
            factory_patch.start()
            self.addCleanup(factory_patch.stop)

    def uploaded_app(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.csv", "商品,销售额,备注\n笔记本,24,PRIVATE_RAW_NOTE\n鼠标,99,PRIVATE_RAW_NOTE\n".encode("utf-8"), "text/csv"
        ).run(timeout=10)
        self.assertFalse(app.exception)
        self.assertTrue(app.checkbox, "缺少让模型建议工具复选框")
        self.assertEqual(app.checkbox[0].label, "让模型建议工具")
        return app

    def test_default_off_keeps_fixed_selection_and_no_model_before_click(self):
        app = self.uploaded_app()
        self.assertFalse(app.checkbox[0].value)
        app.text_input(key="analysis_question").input("总额多少？").run(timeout=10)
        self.model.assert_not_called()
        app.button(key="analysis_ask").click().run(timeout=10)

        self.assertFalse(app.exception)
        self.model.return_value.bind_tools.assert_not_called()
        self.assertEqual(self.executed, ["sales_summary"])
        self.assertTrue(any("选择方式：固定规则" in item.value for item in app.caption))

    def test_valid_suggestion_or_empty_suggestion_executes_and_answers(self):
        scenarios = [
            ([proposed("sales_summary", {})], "总额多少？", ["sales_summary"], "模型建议（已校验）"),
            ([proposed("product_sales_total", {"product_name": "笔记本"})], "笔记本合计？",
             ["sales_summary", "product_sales_total"], "模型建议（已校验）"),
            ([], "笔记本合计？", ["sales_summary", "product_sales_total"], "固定规则"),
        ]
        for calls, question, expected_tools, source in scenarios:
            with self.subTest(calls=calls):
                self.model.reset_mock()
                self.executed.clear()
                self.model.return_value.bind_tools.return_value.invoke.return_value = AIMessage(content="", tool_calls=calls)
                app = self.uploaded_app()
                app.checkbox[0].check().run(timeout=10)
                app.text_input(key="analysis_question").input(question).run(timeout=10)
                self.model.assert_not_called()
                app.button(key="analysis_ask").click().run(timeout=10)

                self.assertFalse(app.exception)
                self.model.return_value.bind_tools.assert_called_once()
                self.assertEqual(self.model.return_value.bind_tools.call_args.kwargs["tool_choice"], "auto")
                bound_tools = self.model.return_value.bind_tools.call_args.args[0]
                self.assertEqual(len(bound_tools), 2)
                self.model.return_value.bind_tools.return_value.invoke.assert_called_once()
                self.model.return_value.invoke.assert_called_once()
                if not calls:
                    final_messages = self.model.return_value.invoke.call_args.args[0]
                    self.assertFalse(any(isinstance(message, (AIMessage, ToolMessage)) for message in final_messages))
                self.assertEqual(self.executed, expected_tools)
                self.assertTrue(any(f"选择方式：{source}" in item.value for item in app.caption))
                captions = "\n".join(item.value for item in app.caption if "本次工具" in item.value)
                self.assertIn("销售汇总", captions)
                if len(expected_tools) == 2:
                    self.assertIn("指定商品销售额", captions)
                self.assertTrue(any("分析回答已完成。" in item.value for item in app.markdown))
                self.assertEqual(app.metric[0].value, "123.00")
                self.assertEqual(len(app.dataframe), 3)
                self.assertEqual(len(app.image), 2)

    def test_valid_proposal_returns_tool_message_with_original_call_id(self):
        cases = [
            ("sales_summary", {}, "总额多少？", {"row_count": 2, "sales_total": 123.0, "sales_average": 61.5}),
            ("product_sales_total", {"product_name": "笔记本"}, "笔记本合计？",
             {"product_name": "笔记本", "sales_total": 24.0}),
        ]
        for name, arguments, question, expected_result in cases:
            with self.subTest(tool=name):
                self.model.reset_mock()
                self.executed.clear()
                call = proposed(name, arguments)
                call["id"] = f"original-{name}-call-id"
                proposal = AIMessage(content="", tool_calls=[call])
                self.model.return_value.bind_tools.return_value.invoke.return_value = proposal
                app = self.uploaded_app()
                app.checkbox[0].check().run(timeout=10)
                app.text_input(key="analysis_question").input(question).run(timeout=10)
                app.button(key="analysis_ask").click().run(timeout=10)

                self.assertFalse(app.exception)
                self.model.return_value.invoke.assert_called_once()
                messages = self.model.return_value.invoke.call_args.args[0]
                assistant_messages = [(index, message) for index, message in enumerate(messages) if isinstance(message, AIMessage)]
                self.assertEqual(len(assistant_messages), 1, "最终请求应保留原工具建议AIMessage")
                assistant_index, assistant = assistant_messages[0]
                self.assertEqual(assistant, proposal)
                tool_messages = [(index, message) for index, message in enumerate(messages) if isinstance(message, ToolMessage)]
                self.assertEqual(len(tool_messages), 1)
                tool_index, result = tool_messages[0]
                self.assertEqual(tool_index, assistant_index + 1)
                self.assertEqual(result.tool_call_id, call["id"])
                self.assertEqual(json.loads(result.content), expected_result)
                self.assertNotIn("PRIVATE_RAW_NOTE", result.content)
                self.assertTrue(any("分析回答已完成。" in item.value for item in app.markdown))

    def test_invalid_suggestion_never_executes_local_tools_or_final_answer(self):
        self.model.return_value.bind_tools.return_value.invoke.return_value = AIMessage(
            content="", tool_calls=[proposed("unknown_tool", {})]
        )
        app = self.uploaded_app()
        app.checkbox[0].check().run(timeout=10)
        app.text_input(key="analysis_question").input("总额多少？").run(timeout=10)
        app.button(key="analysis_ask").click().run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(self.executed, [])
        self.model.return_value.invoke.assert_not_called()
        messages = [item.value for kind in (app.warning, app.error, app.info) for item in kind]
        self.assertTrue(any("模型建议未通过检查" in value for value in messages))
        self.assertEqual(app.metric[0].value, "123.00")
        self.assertEqual(len(app.image), 2)


if __name__ == "__main__":
    unittest.main()

