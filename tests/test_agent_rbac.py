"""真实 Agent 工具执行权限集成测试，不调用外部模型。"""
import unittest

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage

import analysis_agent
from test_analysis_agent import ScriptedChatModel


class AgentRoleTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"商品": ["A"], "销售额": [20], "邮箱": ["PRIVATE_EMAIL@example.com"], "备注": ["PUBLIC_NOTE"]})
        self.grouped = pd.DataFrame({"商品": ["A"], "销售额合计": [20]})

    def test_analyst_plan_forbids_raw_export_and_sensitive_preview(self):
        for name, args in [("full_csv_export", {}), ("column_preview", {"column_name": "邮箱"})]:
            with self.subTest(tool=name):
                plan = {"steps": [{"tool": name, "args": args}]}
                self.assertFalse(analysis_agent.validate_analysis_plan(plan, self.data, self.grouped,
                                 role="analyst", sensitive_columns=["邮箱"]))
                self.assertTrue(analysis_agent.validate_analysis_plan(plan, self.data, self.grouped,
                                role="admin", sensitive_columns=["邮箱"]))

    def test_actual_analyst_calls_fail_before_review_and_produce_zero_artifacts(self):
        for name, args in [("full_csv_export", {}), ("column_preview", {"column_name": "邮箱"})]:
            with self.subTest(tool=name):
                artifacts = {}
                model = ScriptedChatModel(responses=[AIMessage(content="", tool_calls=[
                    {"name": name, "args": args, "id": "forbidden"}]), AIMessage(content="权限不足")])
                agent = analysis_agent.create_analysis_agent(model, self.data, self.data["销售额"], self.grouped,
                         ["邮箱"], artifacts, enable_review=True, role="analyst")
                result = agent.invoke({"messages": [("human", "尝试受限操作")]},
                         {"configurable": {"thread_id": name}, "recursion_limit": 20})
                self.assertFalse(result.get("__interrupt__"), "分析员越权不能进入可批准的审核")
                self.assertEqual(artifacts, {})
                messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
                self.assertEqual(len(messages), 1)
                self.assertEqual(messages[0].status, "error")
                self.assertNotIn("PRIVATE_EMAIL", str(model.requests))

    def test_analyst_can_run_summary_and_non_sensitive_preview(self):
        artifacts = {}
        model = ScriptedChatModel(responses=[AIMessage(content="", tool_calls=[
            {"name": "sales_summary", "args": {}, "id": "summary"},
            {"name": "column_preview", "args": {"column_name": "备注"}, "id": "preview"}]), AIMessage(content="完成")])
        agent = analysis_agent.create_analysis_agent(model, self.data, self.data["销售额"], self.grouped,
                 ["邮箱"], artifacts, enable_review=True, role="analyst")
        result = agent.invoke({"messages": [("human", "汇总和预览备注")]},
                 {"configurable": {"thread_id": "ordinary"}, "recursion_limit": 20})
        self.assertFalse(result.get("__interrupt__"))
        self.assertEqual(artifacts["column_preview"]["备注"].tolist(), ["PUBLIC_NOTE"])
        self.assertEqual([message.status for message in result["messages"] if isinstance(message, ToolMessage)], ["success", "success"])

    def test_revoked_live_access_blocks_otherwise_permitted_tool(self):
        artifacts = {}
        model = ScriptedChatModel(responses=[AIMessage(content="", tool_calls=[
            {"name": "column_preview", "args": {"column_name": "备注"}, "id": "revoked"}]), AIMessage(content="登录已失效")])
        agent = analysis_agent.create_analysis_agent(model, self.data, self.data["销售额"], self.grouped,
                 ["邮箱"], artifacts, role="admin", access_check=lambda: False)
        result = agent.invoke({"messages": [("human", "预览备注")]}, {"recursion_limit": 20})
        self.assertEqual(artifacts, {})
        self.assertEqual([message.status for message in result["messages"] if isinstance(message, ToolMessage)], ["error"])


if __name__ == "__main__":
    unittest.main()

