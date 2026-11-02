import importlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage
from authenticated_app import AppTest

from test_analysis_agent import APP_PATH, ScriptedChatModel


class TraceStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.log_dir = Path(self.temporary.name)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("trace_store"), "尚未实现 trace_store")
        return importlib.import_module("trace_store")

    def test_trace_and_audit_write_utf8_json_lines_with_automatic_timestamp(self):
        module = self.module()
        event = {"trace_id": "trace-a", "event": "开始", "status": "ready"}
        module.append_trace(event, self.log_dir)
        module.append_trace({**event, "event": "结束"}, self.log_dir)
        module.append_audit({**event, "decision": "reject"}, self.log_dir)
        traces = [json.loads(line) for line in (self.log_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        audits = [json.loads(line) for line in (self.log_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0]["event"], "开始")
        self.assertTrue(traces[0]["timestamp"])
        self.assertEqual(audits[0]["decision"], "reject")
        self.assertNotIn("timestamp", event)

    def test_log_allowlist_excludes_raw_results_and_redacts_nested_secrets(self):
        module = self.module()
        private = "sk-test-PRIVATE_SECRET_1234567890 private@example.com 13812345678"
        event = {"trace_id": "safe", "event": "plan", "query": private,
                 "plan": {"steps": [{"args": {"column_name": private}}]}, "args": {"product_name": private},
                 "api_key": "RAW_API_KEY", "raw_rows": ["PRIVATE_ROW"], "result": "PRIVATE_RESULT"}
        for method, filename in [(module.append_trace, "trace.jsonl"), (module.append_audit, "audit.jsonl")]:
            with self.subTest(filename=filename):
                method(event, self.log_dir)
                text = (self.log_dir / filename).read_text(encoding="utf-8")
                parsed = json.loads(text)
                self.assertNotIn("api_key", parsed)
                self.assertNotIn("raw_rows", parsed)
                self.assertNotIn("result", parsed)
                for value in ["sk-test-PRIVATE_SECRET_1234567890", "private@example.com", "13812345678", "PRIVATE_ROW", "PRIVATE_RESULT", "RAW_API_KEY"]:
                    self.assertNotIn(value, text)

    def test_read_trace_filters_id_and_skips_corrupt_lines(self):
        module = self.module()
        self.assertEqual(module.read_trace("missing", self.log_dir), [])
        module.append_trace({"trace_id": "a", "event": "first"}, self.log_dir)
        with (self.log_dir / "trace.jsonl").open("a", encoding="utf-8") as output:
            output.write("broken JSON\n[]\nnull\n")
        module.append_trace({"trace_id": "b", "event": "other"}, self.log_dir)
        module.append_trace({"trace_id": "a", "event": "second"}, self.log_dir)
        self.assertEqual([event["event"] for event in module.read_trace("a", self.log_dir)], ["first", "second"])

    def test_hex_trace_id_containing_eighteen_digits_is_preserved_for_replay(self):
        module = self.module()
        trace_id = "7f481621311049789448ba0872baaee2"
        module.append_trace({"trace_id": trace_id, "event": "created", "query": "证件号481621311049789448"}, self.log_dir)
        module.append_audit({"trace_id": trace_id, "event": "review", "decision": "reject"}, self.log_dir)
        events = module.read_trace(trace_id, self.log_dir)
        self.assertEqual(len(events), 1, "合法随机ID不能被证件号脱敏破坏")
        self.assertEqual(events[0]["trace_id"], trace_id)
        self.assertNotIn("481621311049789448", events[0]["query"])
        audit = json.loads((self.log_dir / "audit.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(audit["trace_id"], trace_id)

    def test_real_agent_records_tool_name_checked_args_status_and_duration(self):
        store = self.module()
        module = importlib.import_module("analysis_agent")
        data = pd.DataFrame({"商品": ["笔记本"], "销售额": [24], "备注": ["PRIVATE_ROW_VALUE"]})
        grouped = pd.DataFrame({"商品": ["笔记本"], "销售额合计": [24]})
        model = ScriptedChatModel(responses=[AIMessage(content="", tool_calls=[{"name": "sales_summary", "args": {}, "id": "summary"}]), AIMessage(content="已完成")])
        agent = module.create_analysis_agent(model, data, data["销售额"], grouped, [], {}, trace_id="tool-trace", log_dir=self.log_dir)
        agent.invoke({"messages": [("human", "汇总销售额")]}, {"recursion_limit": 20})
        events = [event for event in store.read_trace("tool-trace", self.log_dir) if event.get("tool") == "sales_summary"]
        self.assertTrue(events)
        event = events[-1]
        self.assertEqual(event["args"], {})
        self.assertTrue(event["status"])
        self.assertGreaterEqual(event["duration_ms"], 0)
        self.assertNotIn("PRIVATE_ROW_VALUE", (self.log_dir / "trace.jsonl").read_text(encoding="utf-8"))

    def test_manually_sensitive_product_values_never_enter_trace_or_audit(self):
        store = self.module()
        data = pd.DataFrame({"商品": ["笔记本"], "销售额": [24]})
        plan = {"steps": [
            {"tool": "product_sales_total", "args": {"product_name": "笔记本"}},
            {"tool": "full_csv_export", "args": {}},
        ]}
        model = ScriptedChatModel(responses=[
            AIMessage(content=json.dumps(plan)),
            AIMessage(content=json.dumps({"steps": [{"tool": "full_csv_export", "args": {}}]})),
            AIMessage(content="", tool_calls=[{"name": "full_csv_export", "args": {}, "id": "export"}]),
            AIMessage(content="已准备下载"),
        ])
        with patch.object(store, "DEFAULT_LOG_DIR", self.log_dir), patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-sensitive-product-key"}), patch("langchain_openai.ChatOpenAI", return_value=model):
            app = AppTest.from_file(APP_PATH).run(timeout=10)
            app.file_uploader[0].upload("sales.csv", data.to_csv(index=False).encode("utf-8"), "text/csv").run(timeout=10)
            app.multiselect(key="sensitive_columns").set_value(["商品"]).run(timeout=10)
            app.text_input(key="agent_question").input("先查笔记本的销售额，再导出全部数据").run(timeout=10)
            app.button(key="agent_ask").click().run(timeout=10)
            self.assertFalse(app.exception)
            self.assertNotIn("agent_approve", [button.key for button in app.button])
            self.assertFalse(app.downloads, "敏感商品计划被拒绝后不能生成任何下载")
            app.text_input(key="agent_question").input("导出全部原始数据").run(timeout=10)
            app.button(key="agent_ask").click().run(timeout=10)
            self.assertFalse(app.exception)
            app.button(key="agent_approve").click().run(timeout=10)
            self.assertFalse(app.exception)
        for filename in ["trace.jsonl", "audit.jsonl"]:
            with self.subTest(filename=filename):
                text = (self.log_dir / filename).read_text(encoding="utf-8")
                events = [json.loads(line) for line in text.splitlines()]
                self.assertTrue(all(event.get("event") for event in events))
                self.assertTrue(any(event.get("status") or event.get("decision") for event in events))
                self.assertNotIn("笔记本", text)
        events = store.read_trace(json.loads((self.log_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0])["trace_id"], self.log_dir)
        self.assertTrue(any(event.get("event") == "plan" and event.get("status") == "rejected" for event in events))
        self.assertFalse(any(event.get("tool") == "product_sales_total" for event in events),
                         "敏感商品计划在工具执行前必须被拒绝")

    def test_page_review_decisions_are_audited_and_trace_can_be_replayed(self):
        store = self.module()
        for decision in ["approve", "reject"]:
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as directory:
                data = pd.DataFrame({"商品": ["笔记本"], "销售额": [24], "邮箱": ["PRIVATE_DATA@example.com"]})
                plan = {"steps": [{"tool": "full_csv_export", "args": {}}]}
                model = ScriptedChatModel(responses=[AIMessage(content=json.dumps(plan)), AIMessage(content="", tool_calls=[{"name": "full_csv_export", "args": {}, "id": "export"}]), AIMessage(content="已准备下载")])
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test-SECRET_MODEL_KEY"}), patch("langchain_openai.ChatOpenAI", return_value=model), patch.object(store, "DEFAULT_LOG_DIR", Path(directory)):
                    app = AppTest.from_file(APP_PATH).run(timeout=10)
                    app.file_uploader[0].upload("sales.csv", data.to_csv(index=False).encode("utf-8"), "text/csv").run(timeout=10)
                    app.text_input(key="agent_question").input("导出全部数据").run(timeout=10)
                    app.button(key="agent_ask").click().run(timeout=10)
                    self.assertFalse(app.exception)
                    events = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
                    trace_id = events[0]["trace_id"]
                    displayed = str([item.value for items in [app.caption, app.markdown, app.text] for item in items])
                    self.assertIn(trace_id, displayed)
                    app.button(key=f"agent_{decision}").click().run(timeout=10)
                    self.assertFalse(app.exception)
                    audits = [json.loads(line) for line in (Path(directory) / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertTrue(any(event.get("decision") == decision and event["trace_id"] == trace_id for event in audits))
                    for file in Path(directory).glob("*.jsonl"):
                        text = file.read_text(encoding="utf-8")
                        self.assertNotIn("PRIVATE_DATA@example.com", text)
                        self.assertNotIn("sk-test-SECRET_MODEL_KEY", text)
                    app.text_input(key="trace_lookup").input(trace_id).run(timeout=10)
                    app.button(key="trace_search").click().run(timeout=10)
                    self.assertFalse(app.exception)
                    self.assertIn(trace_id, str([item.value for item in app.json]))


if __name__ == "__main__":
    unittest.main()

