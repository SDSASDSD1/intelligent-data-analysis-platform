"""收尾策略单元测试及真实 Agent / Streamlit 集成测试；全部使用脚本模型。"""
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import analysis_agent
import report_export
import review_store
import trace_store
from authenticated_app import AppTest
from test_analysis_agent import APP_PATH, ScriptedChatModel


class FinalPolicyTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"商品": ["PRIVATE_PRODUCT"], "销售额": [1234.56],
                                  "日期": ["2026-10-01"], "邮箱": ["private@example.com"]})
        self.grouped = pd.DataFrame({"商品": ["PRIVATE_PRODUCT"], "销售额合计": [1234.56]})
        self.daily = pd.DataFrame({"日期": pd.to_datetime(["2026-10-01"]), "销售额合计": [1234.56]})

    def test_sensitive_product_plan_and_actual_call_cannot_bypass_policy(self):
        call = {"name": "product_sales_total", "args": {"product_name": "PRIVATE_PRODUCT"}, "id": "blocked"}
        plan = {"steps": [{"tool": call["name"], "args": call["args"]}]}
        self.assertFalse(analysis_agent.validate_analysis_plan(plan, self.data, self.grouped,
                         sensitive_columns=["商品"]))
        self.assert_blocked(call, ["商品"])

    def assert_blocked(self, call, sensitive, **kwargs):
        model = ScriptedChatModel(responses=[AIMessage(content="", tool_calls=[call]), AIMessage(content="已拒绝")])
        artifacts = {}
        agent = analysis_agent.create_analysis_agent(model, self.data, self.data["销售额"],
                 self.grouped, sensitive, artifacts, daily=self.daily, **kwargs)
        result = agent.invoke({"messages": [("human", "分析")]}, {"recursion_limit": 20})
        messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([message.status for message in messages], ["error"])
        self.assertEqual(artifacts, {})
        self.assertNotIn("1234.56", str(model.requests))

    def test_sensitive_amount_blocks_every_derived_tool_even_without_plan(self):
        for name, args in [("sales_summary", {}), ("product_sales_total", {"product_name": "PRIVATE_PRODUCT"}),
                           ("daily_sales_summary", {}), ("sales_chart", {"chart_type": "line"})]:
            with self.subTest(tool=name):
                self.assert_blocked({"name": name, "args": args, "id": "blocked"}, ["销售额"],
                                    sales_column="销售额", date_column="日期")

    def test_sensitive_date_blocks_daily_tool(self):
        self.assert_blocked({"name": "daily_sales_summary", "args": {}, "id": "blocked"},
                            ["日期"], date_column="日期")

    def test_sensitive_product_actual_call_is_guarded_independently_of_plan(self):
        self.assert_blocked({"name": "product_sales_total", "args": {"product_name": "PRIVATE_PRODUCT"},
                             "id": "blocked"}, ["商品"])

    def test_sensitive_access_without_review_middleware_is_blocked(self):
        for name, args in [("full_csv_export", {}), ("column_preview", {"column_name": "邮箱"})]:
            with self.subTest(tool=name):
                self.assert_blocked({"name": name, "args": args, "id": "blocked"}, ["邮箱"])

    def test_risk_has_reason_and_review_is_only_for_admin_sensitive_access(self):
        policy = importlib.import_module("risk_policy")
        cases = [("sales_summary", {}, "admin", [], "low", False),
                 ("column_preview", {"column_name": "邮箱"}, "admin", ["邮箱"], "high", True),
                 ("full_csv_export", {}, "admin", [], "high", True),
                 ("column_preview", {"column_name": "邮箱"}, "analyst", ["邮箱"], "blocked", False),
                 ("full_csv_export", {}, "analyst", [], "blocked", False),
                 ("product_sales_total", {"product_name": "PRIVATE_PRODUCT"}, "admin", ["商品"], "blocked", False)]
        for name, args, role, sensitive, level, review in cases:
            with self.subTest(tool=name, role=role, sensitive=sensitive):
                result = policy.evaluate_tool_risk(name, args, role=role, sensitive_columns=sensitive)
                self.assertEqual(result["level"], level)
                self.assertIs(result["requires_review"], review)
                self.assertTrue(result["reason"])
        plan = {"steps": [{"tool": "sales_summary", "args": {}}, {"tool": "full_csv_export", "args": {}}]}
        risk = policy.evaluate_plan_risk(plan)
        self.assertEqual(risk["level"], "high")
        self.assertEqual(len(risk["steps"]), 2)

    def test_sensitive_amount_report_contains_no_derived_statistics_or_chart(self):
        text = report_export.build_markdown_report("safe-report", "分析", {"steps": []}, "无法分析敏感金额",
               self.data["销售额"], self.grouped, chart="line", sensitive_columns=["销售额"],
               sales_column="销售额", date_column="日期", daily=self.daily)
        for value in ["1234.56", "PRIVATE_PRODUCT", "销售额总和：", "销售额平均值：", "折线图", "2026-10-01"]:
            self.assertNotIn(value, text)

    def test_custom_amount_column_uses_selected_field_policy(self):
        renamed = self.data.rename(columns={"销售额": "成交金额"})
        plan = {"steps": [{"tool": "sales_summary", "args": {}}]}
        self.assertFalse(analysis_agent.validate_analysis_plan(plan, renamed, self.grouped,
                         sensitive_columns=["成交金额"], sales_column="成交金额"))
        self.assertTrue(analysis_agent.validate_analysis_plan(plan, renamed, self.grouped,
                        sensitive_columns=["邮箱"], sales_column="成交金额"))


class FinalTraceTest(unittest.TestCase):
    def test_resumed_legacy_checkpoint_redacts_history_without_changing_call_identity(self):
        data = pd.DataFrame({"商品": ["PRIVATE_PRODUCT"], "销售额": [20],
                             "内部备注": ["PRIVATE_NOTE"], "邮箱": ["private@example.com"]})
        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "legacy-private-history"}, "recursion_limit": 20}
        legacy_model = ScriptedChatModel(responses=[AIMessage(content="", tool_calls=[
            {"name": "column_preview", "args": {"column_name": "内部备注"}, "id": "review-original-id"}])])
        legacy_agent = analysis_agent.create_analysis_agent(legacy_model, data, data["销售额"], None,
                       ["商品", "内部备注", "邮箱"], {}, enable_review=True, checkpointer=saver)
        history = [HumanMessage(content="PRIVATE_NOTE private@example.com"),
                   HumanMessage(content="13800000000"),
                   AIMessage(content="旧分析 PRIVATE_NOTE", tool_calls=[{"name": "product_sales_total",
                       "args": {"product_name": "PRIVATE_PRODUCT"}, "id": "legacy-original-id"}]),
                   ToolMessage(content='{"product_name":"PRIVATE_PRODUCT","sales_total":20}',
                               tool_call_id="legacy-original-id"),
                   HumanMessage(content="请预览内部备注")]
        paused = legacy_agent.invoke({"messages": history}, config)
        self.assertTrue(paused.get("__interrupt__"))
        self.assertIn("PRIVATE_PRODUCT", str(legacy_agent.get_state(config).values["messages"]))
        model = ScriptedChatModel(responses=[AIMessage(content="已完成")])
        artifacts = {}
        resumed_agent = analysis_agent.create_analysis_agent(model, data, data["销售额"], None,
                        ["商品", "内部备注", "邮箱"], artifacts, enable_review=True, checkpointer=saver,
                        sensitive_values=["PRIVATE_NOTE", "PRIVATE_PRODUCT", "private@example.com"])
        result = resumed_agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
        self.assertFalse(result.get("__interrupt__"))
        self.assertTrue(model.requests)
        for secret in ["PRIVATE_NOTE", "PRIVATE_PRODUCT", "private@example.com", "13800000000"]:
            self.assertNotIn(secret, str(model.requests))
        calls = [call for message in model.requests[0] if isinstance(message, AIMessage)
                 for call in message.tool_calls]
        self.assertEqual([call["id"] for call in calls], ["legacy-original-id", "review-original-id"])
        self.assertEqual(calls[-1]["args"], {"column_name": "内部备注"})
        self.assertEqual([message.tool_call_id for message in model.requests[0]
                          if isinstance(message, ToolMessage)], ["legacy-original-id", "review-original-id"])
        self.assertEqual(artifacts["column_preview"]["内部备注"].tolist(), ["PRIVATE_NOTE"])

    def test_agent_tool_event_includes_risk_and_safe_computed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            data = pd.DataFrame({"销售额": [10, 20], "备注": ["PRIVATE_ROW_A", "PRIVATE_ROW_B"]})
            model = ScriptedChatModel(responses=[AIMessage(content="", tool_calls=[
                {"name": "sales_summary", "args": {}, "id": "summary"}]), AIMessage(content="总额30")])
            agent = analysis_agent.create_analysis_agent(model, data, data["销售额"], None,
                     [], {}, trace_id="safe-tool", log_dir=Path(directory))
            agent.invoke({"messages": [("human", "统计")]}, {"recursion_limit": 20})
            events = trace_store.read_trace("safe-tool", Path(directory))
            event = next(item for item in events if item.get("tool") == "sales_summary")
            self.assertEqual(event.get("risk_level"), "low")
            self.assertTrue(event.get("risk_reason"))
            self.assertEqual(event.get("result_summary", {}).get("sales_total"), 30)
            self.assertNotIn("PRIVATE_ROW", str(events))

    def test_tool_summaries_drop_raw_and_unknown_fields(self):
        summarize = trace_store.summarize_tool_result
        for name, body, expected in [
            ("sales_summary", {"row_count": 2, "sales_total": 30, "sales_average": 15}, "sales_total"),
            ("column_preview", {"status": "ready", "column_name": "邮箱", "row_count": 1}, "status"),
            ("full_csv_export", {"status": "ready", "row_count": 1}, "status")]:
            with self.subTest(tool=name):
                message = ToolMessage(content=json.dumps({**body, "raw_rows": ["PRIVATE_ROW"],
                            "unknown": "PRIVATE_UNKNOWN", "content": "PRIVATE_CONTENT"}), tool_call_id="safe")
                result = summarize(name, message)
                self.assertIsInstance(result, dict)
                self.assertIn(expected, result)
                self.assertNotIn("PRIVATE_", str(result))
        self.assertEqual(summarize("unknown", ToolMessage(content='{"raw_rows":["PRIVATE"]}', tool_call_id="x")), {})

    def test_trace_keeps_safe_results_and_redacted_answer_for_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_store.append_trace({"trace_id": "safe", "event": "answer", "risk_level": "low",
                "risk_reason": "正常统计", "result_summary": {"sales_total": 30},
                "answer": "完成 PRIVATE_NAME private@example.com sk-test-SECRET_KEY_123456",
                "raw_rows": ["PRIVATE_ROW"]}, Path(directory), sensitive_values=["PRIVATE_NAME"])
            events = trace_store.read_trace("safe", Path(directory))
            self.assertEqual(events[0].get("result_summary"), {"sales_total": 30})
            self.assertEqual(events[0].get("risk_level"), "low")
            self.assertIn("完成", events[0].get("answer", ""))
            for secret in ["PRIVATE_NAME", "private@example.com", "sk-test-SECRET_KEY_123456", "PRIVATE_ROW"]:
                self.assertNotIn(secret, str(events))

    def test_step_builder_preserves_event_order_and_legacy_missing_fields(self):
        viewer = importlib.import_module("trace_viewer")
        events = [{"event": "query", "query": "分析"}, {"event": "tool_call", "tool": "sales_summary",
                   "status": "success", "result_summary": {"sales_total": 30}},
                  {"event": "answer", "answer": "总额30", "status": "complete"}]
        steps = viewer.build_trace_steps(events)
        self.assertEqual(len(steps), 3)
        self.assertTrue(all(isinstance(step, dict) for step in steps))
        self.assertIn("sales_summary", str(steps[1]))
        self.assertIn("总额30", str(steps[2]))
        self.assertEqual(viewer.build_trace_steps([]), [])


class FinalPageTest(unittest.TestCase):
    def test_review_trace_records_actual_high_risk_action_when_plan_was_low_risk(self):
        for decision in ["approve", "reject"]:
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = ScriptedChatModel(responses=[
                    AIMessage(content=json.dumps({"steps": [{"tool": "sales_summary", "args": {}}]})),
                    AIMessage(content="", tool_calls=[{"name": "full_csv_export", "args": {}, "id": "actual-export"}]),
                    AIMessage(content="已准备下载")])
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-review-test-only"}), patch(
                     "langchain_openai.ChatOpenAI", return_value=model), patch.object(trace_store, "DEFAULT_LOG_DIR", root / "logs"), patch.object(
                     review_store, "DEFAULT_REVIEW_DIR", root / "reviews"), patch.object(
                     review_store, "DEFAULT_CHECKPOINT_PATH", root / "checkpoints.sqlite"):
                    app = self.upload()
                    app.text_input(key="agent_question").input("分析销售额").run(timeout=15)
                    app.button(key="agent_ask").click().run(timeout=15)
                    self.assertFalse(app.exception)
                    events = [json.loads(line) for line in (root / "logs" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
                    pending = [event for event in events if event.get("event") == "review" and event.get("status") == "pending"]
                    self.assertTrue(pending)
                    self.assertTrue(any(event.get("tool") == "full_csv_export" and event.get("args") == {}
                                        and event.get("risk_level") == "high" and event.get("risk_reason") for event in pending))
                    self.assertFalse(app.downloads)
                    app.button(key=f"agent_{decision}").click().run(timeout=15)
                    self.assertFalse(app.exception)
                    for filename in ["trace.jsonl", "audit.jsonl"]:
                        records = [json.loads(line) for line in (root / "logs" / filename).read_text(encoding="utf-8").splitlines()]
                        self.assertTrue(any(event.get("decision") == decision and event.get("tool") == "full_csv_export"
                                            and event.get("args") == {} and event.get("risk_level") == "high"
                                            and event.get("risk_reason") for event in records), filename)
                    if decision == "reject":
                        self.assertFalse(app.downloads)
                        self.assertEqual(len(model.requests), 2)
                    else:
                        self.assertTrue(app.downloads)

    def upload(self):
        data = pd.DataFrame({"商品": ["公开商品"], "销售额": [345.67],
                             "内部备注": ["PRIVATE_NOTE_MARKER"], "邮箱": ["private@example.com"]})
        app = AppTest.from_file(APP_PATH).run(timeout=15)
        app.file_uploader[0].upload("safe.csv", data.to_csv(index=False).encode(), "text/csv").run(timeout=15)
        app.multiselect(key="sensitive_columns").set_value(["邮箱", "内部备注"]).run(timeout=15)
        self.assertFalse(app.exception)
        return app

    def test_all_question_modes_redact_sensitive_values_before_model(self):
        for mode in ["fixed", "suggested", "agent"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                responses = [AIMessage(content=json.dumps({"steps": [{"tool": "sales_summary", "args": {}}]})),
                    AIMessage(content="", tool_calls=[{"name": "sales_summary", "args": {}, "id": "summary"}]),
                    AIMessage(content="总额345.67") ] if mode == "agent" else (
                    [AIMessage(content="", tool_calls=[]), AIMessage(content="总额345.67")] if mode == "suggested"
                    else [AIMessage(content="总额345.67")])
                model = ScriptedChatModel(responses=responses)
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-test-only"}), patch(
                     "langchain_openai.ChatOpenAI", return_value=model), patch.object(trace_store, "DEFAULT_LOG_DIR", Path(directory)):
                    app = self.upload()
                    if mode == "suggested":
                        app.checkbox[0].check().run(timeout=15)
                    key = "agent_question" if mode == "agent" else "analysis_question"
                    button = "agent_ask" if mode == "agent" else "analysis_ask"
                    app.text_input(key=key).input("PRIVATE_NOTE_MARKER private@example.com 的总额是多少").run(timeout=15)
                    app.button(key=button).click().run(timeout=15)
                    self.assertFalse(app.exception)
                    self.assertTrue(model.requests)
                    self.assertNotIn("PRIVATE_NOTE_MARKER", str(model.requests))
                    self.assertNotIn("private@example.com", str(model.requests))
                    if mode == "agent":
                        events = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
                        self.assertTrue(any(item.get("event") == "answer" and "345.67" in str(item.get("answer", "")) for item in events))

    def test_sensitive_amount_has_no_metrics_charts_or_downloads(self):
        app = self.upload()
        app.multiselect(key="sensitive_columns").set_value(["销售额", "邮箱"]).run(timeout=15)
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value["销售额"].tolist(), ["需审核"])
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.image), 0)
        self.assertFalse(app.downloads)
        self.assertTrue(app.info or app.warning)


if __name__ == "__main__":
    unittest.main()

