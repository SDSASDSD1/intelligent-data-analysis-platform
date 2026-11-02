import importlib
import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from authenticated_app import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class ScriptedChatModel(BaseChatModel):
    responses: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)
    tool_names: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted-test-model"

    def bind_tools(self, tools, **kwargs):
        self.tool_names[:] = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        if not self.responses:
            raise AssertionError("模型调用次数超出测试脚本")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


class AnalysisAgentTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"商品": ["笔记本", "鼠标"], "销售额": [24, 99], "备注": ["PRIVATE_NOTE"] * 2})
        self.grouped = pd.DataFrame({"商品": ["笔记本", "鼠标"], "销售额合计": [24, 99]})
        self.plan = {"steps": [{"tool": "schema_inspect", "args": {}}, {"tool": "sales_summary", "args": {}}]}

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("analysis_agent"), "尚未实现 analysis_agent")
        return importlib.import_module("analysis_agent")

    def test_plan_parser_accepts_direct_fenced_and_embedded_json(self):
        parse = self.module().parse_analysis_plan
        encoded = json.dumps(self.plan)
        for text in [encoded, f"```json\n{encoded}\n```", f"计划如下：{encoded}\n请确认"]:
            with self.subTest(text=text):
                self.assertEqual(parse(text), self.plan)

    def test_plan_parser_returns_none_for_unstructured_or_broken_text(self):
        parse = self.module().parse_analysis_plan
        for text in ["这里只是文字分析", "", '{"steps": [', "[]", "null"]:
            with self.subTest(text=text):
                self.assertIsNone(parse(text))

    def test_plan_validation_accepts_one_to_six_steps(self):
        validate = self.module().validate_analysis_plan
        for count in [1, 6]:
            plan = {"steps": [{"tool": "sales_summary", "args": {}} for _ in range(count)]}
            self.assertIs(validate(plan, self.data, self.grouped), True)

    def test_plan_validation_rejects_bad_structure_and_unauthorized_calls(self):
        validate = self.module().validate_analysis_plan
        cases = [None, {}, {"steps": []}, {"steps": "sales_summary"},
                 {"steps": [{"tool": "sales_summary", "args": {}}] * 7},
                 {"steps": [{"tool": "sales_summary"}]},
                 {"steps": [{"tool": "sales_summary", "args": {}, "code": "exec('x')"}]},
                 {"steps": [{"tool": "exec", "args": {}}]},
                 {"steps": [{"tool": "column_preview", "args": {"column_name": "不存在"}}]},
                 {"steps": [{"tool": "product_sales_total", "args": {"product_name": "不存在"}}]}]
        for plan in cases:
            with self.subTest(plan=plan):
                self.assertIs(validate(plan, self.data, self.grouped), False)

    def test_real_agent_executes_two_tool_rounds_and_returns_answer_without_raw_rows(self):
        module = self.module()
        model = ScriptedChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "schema_inspect", "args": {}, "id": "schema-1"}]),
            AIMessage(content="", tool_calls=[{"name": "sales_summary", "args": {}, "id": "summary-2"}]),
            AIMessage(content="两条销售记录，总额123.00。"),
        ])
        artifacts = {}
        agent = module.create_analysis_agent(model, self.data, self.data["销售额"], self.grouped, [], artifacts)
        result = agent.invoke({"messages": [("human", "先检查字段，再汇总销售额。") ]}, {"recursion_limit": 15})
        self.assertEqual(result["messages"][-1].content, "两条销售记录，总额123.00。")
        tools = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([message.tool_call_id for message in tools], ["schema-1", "summary-2"])
        self.assertEqual(json.loads(tools[1].content)["sales_total"], 123.0)
        self.assertEqual(set(model.tool_names), {"sales_summary", "product_sales_total", "schema_inspect", "sales_chart", "column_preview", "full_csv_export"})
        self.assertNotIn("PRIVATE_NOTE", str(model.requests))

    def test_page_invalid_plan_does_not_start_agent(self):
        module = self.module()
        model = ScriptedChatModel(responses=[AIMessage(content="普通文本，无法形成计划。")])
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-agent-test"}), patch(
            "langchain_openai.ChatOpenAI", return_value=model
        ), patch.object(module, "create_analysis_agent") as create:
            app = self.upload_app()
            app.text_input(key="agent_question").input("先看字段再统计总额").run(timeout=10)
            app.button(key="agent_ask").click().run(timeout=10)
            self.assertFalse(app.exception)
            create.assert_not_called()
            self.assertEqual(len(model.requests), 1)
            self.assertTrue(app.warning or app.error)
            self.assertTrue(any("普通文本，无法形成计划。" in item.value for items in [app.markdown, app.text] for item in items))

    def test_page_planner_exposes_all_tool_contracts_without_raw_rows(self):
        self.module()
        model = ScriptedChatModel(responses=[AIMessage(content="暂不执行")])
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-planner-key"}), patch(
            "langchain_openai.ChatOpenAI", return_value=model
        ):
            app = self.upload_app()
            app.text_input(key="agent_question").input("先检查字段再分析销售情况").run(timeout=10)
            app.button(key="agent_ask").click().run(timeout=10)
            self.assertFalse(app.exception)
            self.assertEqual(len(model.requests), 1)
            request = "\n".join(str(message.content) for message in model.requests[0])
            for name in ["sales_summary", "product_sales_total", "schema_inspect", "sales_chart", "column_preview", "full_csv_export"]:
                with self.subTest(tool=name):
                    self.assertIn(name, request)
            for parameter in ["product_name", "column_name", "chart_type", "bar", "line", "商品", "销售额", "备注"]:
                with self.subTest(parameter=parameter):
                    self.assertIn(parameter, request)
            for raw_value in ["PRIVATE_NOTE", "笔记本", "鼠标", "fake-planner-key"]:
                self.assertNotIn(raw_value, request)

    def test_page_valid_plan_shows_steps_and_final_answer(self):
        self.module()
        model = ScriptedChatModel(responses=[
            AIMessage(content=json.dumps(self.plan)),
            AIMessage(content="", tool_calls=[{"name": "schema_inspect", "args": {}, "id": "schema-1"}]),
            AIMessage(content="", tool_calls=[{"name": "sales_summary", "args": {}, "id": "summary-2"}]),
            AIMessage(content="多步分析完成：销售额123.00。"),
        ])
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-agent-test"}), patch(
            "langchain_openai.ChatOpenAI", return_value=model
        ):
            app = self.upload_app()
            app.text_input(key="agent_question").input("先看字段再统计总额").run(timeout=10)
            app.button(key="agent_ask").click().run(timeout=10)
            self.assertFalse(app.exception)
            visible = str([item.value for items in [app.markdown, app.caption, app.json] for item in items])
            self.assertIn("schema_inspect", visible)
            self.assertIn("sales_summary", visible)
            self.assertIn("多步分析完成：销售额123.00。", visible)
            self.assertEqual(app.metric[0].value, "123.00")
            self.assertNotIn("PRIVATE_NOTE", str(model.requests))

    def upload_app(self):
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload("sales.csv", self.data.to_csv(index=False).encode("utf-8"), "text/csv").run(timeout=10)
        self.assertIsNotNone(app.text_input(key="analysis_question"))
        self.assertIsNotNone(app.button(key="analysis_ask"))
        return app


if __name__ == "__main__":
    unittest.main()

