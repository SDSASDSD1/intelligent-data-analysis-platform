"""真实 LangChain 图与页面集成：正常工具预算、执行上限和审核恢复。"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langgraph.errors import GraphRecursionError

from authenticated_app import AppTest
from test_analysis_agent import APP_PATH, ScriptedChatModel
import analysis_agent
import review_store
import trace_store


def tool_reply(name, index, arguments=None):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": arguments or {}, "id": f"call-{index}",
    }])


class AgentExecutionLimitsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        stack = ExitStack()
        self.stack = stack
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(trace_store, "DEFAULT_LOG_DIR", self.directory / "logs"))
        stack.enter_context(patch.object(review_store, "DEFAULT_REVIEW_DIR", self.directory / "pending"))
        stack.enter_context(patch.object(review_store, "DEFAULT_CHECKPOINT_PATH", self.directory / "checkpoints.sqlite"))
        stack.enter_context(patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-limit-test-key"}))
        # 本组验证 Agent 与审核流程，图表像素渲染由已有页面测试负责。
        stack.enter_context(patch("streamlit.pyplot"))
        stack.enter_context(patch("matplotlib.figure.Figure.tight_layout"))

    def start_page(self, steps, replies):
        model = ScriptedChatModel(responses=[
            AIMessage(content=json.dumps({"steps": steps})), *replies,
        ])
        self.stack.enter_context(patch("langchain_openai.ChatOpenAI", return_value=model))
        app = AppTest.from_file(APP_PATH).run(timeout=10)
        app.file_uploader[0].upload(
            "sales.csv", "商品,销售额\n笔记本,120.50\n鼠标,79.50\n键盘,200.00\n".encode(), "text/csv"
        ).run(timeout=10)
        app.text_input(key="agent_question").input("先检查字段，计算销售总额和平均值，最后选择柱状图").run(timeout=10)
        app.button(key="agent_ask").click().run(timeout=15)
        self.assertFalse(app.exception)
        return app, model

    def events(self):
        path = self.directory / "logs" / "trace.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_tutorial_three_tool_rounds_finish_and_offer_report(self):
        steps = [{"tool": "schema_inspect", "args": {}},
                 {"tool": "sales_summary", "args": {}},
                 {"tool": "sales_chart", "args": {"chart_type": "bar"}}]
        app, model = self.start_page(steps, [
            *(tool_reply(step["tool"], index, step["args"]) for index, step in enumerate(steps)),
            AIMessage(content="共3条记录，总额400.00，平均133.33，已选择柱状图。"),
        ])
        self.assertFalse(app.error, [item.value for item in app.error])
        self.assertIn("总额400.00", str([item.value for item in app.markdown]))
        self.assertEqual(len(model.requests), 5)
        self.assertTrue(any(item.file_name.endswith(".md") for item in app.downloads))
        tools = [event["tool"] for event in self.events() if event["event"] == "tool_call"]
        self.assertEqual(tools, [step["tool"] for step in steps])

    def test_six_allowed_tool_rounds_can_finish_with_seventh_model_answer(self):
        steps = [{"tool": "sales_summary", "args": {}}] * 6
        app, model = self.start_page(steps, [
            *(tool_reply("sales_summary", index) for index in range(6)),
            AIMessage(content="六次合法调用完成，总额400.00。"),
        ])
        self.assertFalse(app.error, [item.value for item in app.error])
        self.assertIn("六次合法调用完成", str([item.value for item in app.markdown]))
        self.assertEqual(len(model.requests), 8)  # 计划一次，Agent 七次。
        self.assertEqual(sum(event["event"] == "tool_call" for event in self.events()), 6)

    def test_repeated_tool_calls_stop_at_six_without_report_or_csv(self):
        app, model = self.start_page(
            [{"tool": "sales_summary", "args": {}}],
            [tool_reply("sales_summary", index) for index in range(8)],
        )
        errors = "\n".join(item.value for item in app.error)
        self.assertIn("执行上限", errors)
        self.assertNotIn("检查模型连接", errors)
        self.assertIn("trace_id", errors)
        self.assertLessEqual(len(model.requests), 8)
        events = self.events()
        self.assertEqual(sum(event["event"] == "tool_call" for event in events), 6)
        self.assertEqual(events[-1]["status"], "limit_reached")
        self.assertIn(events[-1]["trace_id"], errors)
        self.assertFalse(app.downloads)

    def test_graph_limit_has_clear_message_trace_and_no_download(self):
        with patch.object(analysis_agent, "create_analysis_agent") as factory:
            factory.return_value.invoke.side_effect = GraphRecursionError("graph budget exhausted")
            app, _ = self.start_page([{"tool": "sales_summary", "args": {}}], [])
        errors = "\n".join(item.value for item in app.error)
        self.assertIn("执行上限", errors)
        self.assertNotIn("检查模型连接", errors)
        event = self.events()[-1]
        self.assertEqual(event["status"], "limit_reached")
        self.assertIn(event["trace_id"], errors)
        self.assertFalse(app.downloads)

    def test_model_limit_has_clear_message_trace_and_no_download(self):
        with patch.object(analysis_agent, "create_analysis_agent") as factory:
            factory.return_value.invoke.side_effect = ModelCallLimitExceededError(7, 7, None, 7)
            app, _ = self.start_page([{"tool": "sales_summary", "args": {}}], [])
        errors = "\n".join(item.value for item in app.error)
        self.assertIn("执行上限", errors)
        self.assertNotIn("检查模型连接", errors)
        event = self.events()[-1]
        self.assertEqual(event["status"], "limit_reached")
        self.assertIn(event["trace_id"], errors)
        self.assertFalse(app.downloads)

    def test_review_resumes_remaining_five_tools_with_old_session_graph_limit(self):
        steps = [{"tool": "full_csv_export", "args": {}}] + [
            {"tool": "sales_summary", "args": {}},
        ] * 5
        app, model = self.start_page(steps, [
            tool_reply("full_csv_export", 0),
            *(tool_reply("sales_summary", index) for index in range(1, 6)),
            AIMessage(content="审核批准，六步已完成，原始 CSV 已在本地准备。"),
        ])
        self.assertFalse(app.error, [item.value for item in app.error])
        self.assertFalse(app.downloads)
        self.assertIsNotNone(app.button(key="agent_approve"))
        # 热更新前的同一页面会话可能仍保存旧预算。
        app.session_state["agent_pending"]["config"]["recursion_limit"] = 20
        app.button(key="agent_approve").click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertFalse(app.error, [item.value for item in app.error])
        self.assertIn("六步已完成", str([item.value for item in app.markdown]))
        self.assertEqual(len(model.requests), 8)
        self.assertTrue(any(item.file_name.endswith(".csv") and item.approved for item in app.downloads))

    def test_limit_during_approval_clears_pending_and_produces_no_download(self):
        app, model = self.start_page(
            [{"tool": "full_csv_export", "args": {}}],
            [tool_reply("full_csv_export", 0)],
        )
        self.assertIsNotNone(app.button(key="agent_approve"))
        with patch.object(analysis_agent, "create_analysis_agent") as factory:
            factory.return_value.invoke.side_effect = GraphRecursionError("resume budget exhausted")
            app.button(key="agent_approve").click().run(timeout=10)
        self.assertFalse(app.exception)
        self.assertIn("执行上限", "\n".join(item.value for item in app.error))
        self.assertEqual(self.events()[-1]["status"], "limit_reached")
        self.assertFalse(app.downloads)
        self.assertEqual(len(model.requests), 2)
        self.assertNotIn("agent_pending", app.session_state)
        self.assertNotIn("agent_result", app.session_state)
        app.run(timeout=10)
        self.assertFalse([button for button in app.button if button.key == "agent_approve"])
        self.assertFalse(app.downloads)


if __name__ == "__main__":
    unittest.main()

