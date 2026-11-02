import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage
from authenticated_app import AppTest

from test_analysis_agent import ScriptedChatModel


ROOT = Path(__file__).resolve().parents[1]
TRACE_ID = "a" * 32


class ReviewStoreTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.record = {"trace_id": TRACE_ID, "thread_id": "thread-review", "fingerprint": "b" * 64,
                       "sales_column": "销售额", "date_column": "日期", "sensitive_columns": ["邮箱"],
                       "plan": {"steps": [{"tool": "full_csv_export", "args": {}}]},
                       "question": "导出全部数据", "actions": [{"name": "full_csv_export", "args": {}}]}

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("review_store"), "尚未实现审核持久化模块")
        return importlib.import_module("review_store")

    def test_pending_metadata_round_trip_ignores_original_data_rows(self):
        store = self.module()
        store.save_pending({**self.record, "raw_rows": [{"邮箱": "PRIVATE_ORIGINAL_ROW"}]}, self.directory)
        self.assertEqual(store.load_pending(TRACE_ID, self.directory), self.record)
        for path in self.directory.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"PRIVATE_ORIGINAL_ROW", path.read_bytes())

    def test_context_requires_matching_file_sales_date_and_sensitive_columns(self):
        store = self.module()
        context = {"fingerprint": "b" * 64, "sales_column": "销售额", "date_column": "日期",
                   "sensitive_columns": ["邮箱"]}
        self.assertTrue(store.matches_context(self.record, **context))
        for key, wrong in [("fingerprint", "c" * 64), ("sales_column", "金额"),
                           ("date_column", None), ("sensitive_columns", [])]:
            with self.subTest(key=key):
                self.assertFalse(store.matches_context(self.record, **{**context, key: wrong}))

    def test_pending_metadata_redacts_marked_values_in_question_plan_and_actions(self):
        store = self.module()
        marker = "PRIVATE_PENDING_VALUE"
        record = {**self.record, "question": f"预览{marker}",
                  "plan": {"steps": [{"tool": "product_sales", "args": {"product_name": marker}}]},
                  "actions": [{"name": "product_sales", "args": {"product_name": marker}}]}
        store.save_pending(record, self.directory, sensitive_values=(marker,))
        loaded = store.load_pending(TRACE_ID, self.directory)
        self.assertIsNotNone(loaded)
        self.assertNotIn(marker, json.dumps(loaded))
        for path in self.directory.rglob("*"):
            if path.is_file():
                self.assertNotIn(marker.encode(), path.read_bytes())

    def test_approval_or_rejection_can_claim_pending_record_only_once(self):
        store = self.module()
        for decision in ["approve", "reject"]:
            with self.subTest(decision=decision):
                store.save_pending(self.record, self.directory)
                self.assertEqual(store.take_pending(TRACE_ID, self.directory), self.record)
                self.assertIsNone(store.take_pending(TRACE_ID, self.directory))
                self.assertIsNone(store.load_pending(TRACE_ID, self.directory))

    def test_redaction_preserves_structural_tool_names_and_redacts_argument_values(self):
        store = self.module()
        marker = "full_csv_export"
        record = {**self.record, "question": f"请分析{marker}",
                  "plan": {"steps": [
                      {"tool": marker, "args": {}},
                      {"tool": "column_preview", "args": {"column_name": marker}},
                      {"tool": "product_sales", "args": {"product_name": marker}}]},
                  "actions": [{"name": marker, "args": {}},
                              {"name": "column_preview", "args": {"column_name": marker}},
                              {"name": "product_sales", "args": {"product_name": marker}}]}
        store.save_pending(record, self.directory, sensitive_values=(marker,))
        loaded = store.load_pending(TRACE_ID, self.directory)
        self.assertEqual(loaded["actions"][0]["name"], marker)
        self.assertEqual(loaded["plan"]["steps"][0]["tool"], marker)
        self.assertNotIn(marker, loaded["question"])
        self.assertEqual(loaded["plan"]["steps"][1]["args"]["column_name"], marker)
        self.assertEqual(loaded["actions"][1]["args"]["column_name"], marker)
        self.assertNotIn(marker, loaded["plan"]["steps"][2]["args"]["product_name"])
        self.assertNotIn(marker, loaded["actions"][2]["args"]["product_name"])

    def test_concurrent_claims_allow_exactly_one_consumer(self):
        store = self.module()
        store.save_pending(self.record, self.directory)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: store.take_pending(TRACE_ID, self.directory), range(2)))
        self.assertEqual(sum(result is not None for result in results), 1)

    def test_wrong_trace_id_cannot_read_or_consume_another_pending_record(self):
        store = self.module()
        store.save_pending(self.record, self.directory)
        self.assertIsNone(store.load_pending("c" * 32, self.directory))
        self.assertIsNone(store.take_pending("c" * 32, self.directory))
        self.assertEqual(store.load_pending(TRACE_ID, self.directory), self.record)
        for invalid in ["../outside", "not-a-trace"]:
            with self.subTest(invalid=invalid):
                try:
                    result = store.load_pending(invalid, self.directory)
                except ValueError:
                    continue
                self.assertIsNone(result)

    def test_sqlite_checkpoint_resumes_review_in_fresh_python_process(self):
        self.module()
        script = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "tests"))
import pandas as pd
from langchain_core.messages import AIMessage
from langgraph.types import Command
from test_analysis_agent import ScriptedChatModel
import analysis_agent, review_store
phase, db_path = sys.argv[1:]
data = pd.DataFrame({"商品": ["A", "B"], "销售额": [10, 20]})
artifacts = {}
responses = ([AIMessage(content="", tool_calls=[{"name": "full_csv_export", "args": {}, "id": "export-1"}])]
             if phase == "start" else [AIMessage(content="审核通过，下载已准备")])
model = ScriptedChatModel(responses=responses)
saver = review_store.open_checkpointer(db_path)
try:
    agent = analysis_agent.create_analysis_agent(model, data, data["销售额"], None, [], artifacts,
            enable_review=True, checkpointer=saver)
    config = {"configurable": {"thread_id": "same-persistent-thread"}, "recursion_limit": 20}
    payload = ({"messages": [("human", "导出全部数据")]} if phase == "start"
               else Command(resume={"decisions": [{"type": "approve"}]}))
    result = agent.invoke(payload, config)
    if phase == "start":
        assert result.get("__interrupt__"), "首次调用没有暂停审核"
        assert "full_csv" not in artifacts, "批准之前已导出"
    else:
        assert not result.get("__interrupt__"), "批准后仍待审核"
        assert len(pd.read_csv(__import__('io').BytesIO(artifacts["full_csv"]))) == 2
        assert result["messages"][-1].content == "审核通过，下载已准备"
    print(json.dumps({"phase": phase, "ok": True}))
finally:
    saver.conn.close()
'''
        db_path = str(self.directory / "review.sqlite")
        for phase in ["start", "resume"]:
            result = subprocess.run([sys.executable, "-c", script, phase, db_path], cwd=ROOT,
                                    capture_output=True, text=True, encoding="utf-8", timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout.strip().splitlines()[-1])["ok"])


class ReviewRecoveryPageTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("review_store"), "尚未实现审核恢复模块")
        store = importlib.import_module("review_store")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.model = ScriptedChatModel(responses=[
            AIMessage(content=json.dumps({"steps": [{"tool": "full_csv_export", "args": {}}]})),
            AIMessage(content="", tool_calls=[{"name": "full_csv_export", "args": {}, "id": "export-1"}]),
            AIMessage(content="恢复审核后已准备下载"),
        ])
        for replacement in [patch.object(store, "DEFAULT_REVIEW_DIR", directory / "pending"),
                            patch.object(store, "DEFAULT_CHECKPOINT_PATH", directory / "checkpoint.sqlite"),
                            patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-recovery-key"}),
                            patch("langchain_openai.ChatOpenAI", return_value=self.model),
                            patch("trace_store.DEFAULT_LOG_DIR", directory / "logs")]:
            replacement.start()
            self.addCleanup(replacement.stop)
        self.data = pd.DataFrame({"商品": ["A", "B"], "销售额": [10, 20], "金额": [100, 200],
                                  "日期": ["2026-10-01", "2026-10-02"],
                                  "日期2": ["2026-10-03", "2026-10-04"],
                                  "邮箱": ["a@example.com", "b@example.com"]})

    def upload(self, data=None):
        app = AppTest.from_file(ROOT / "app.py").run(timeout=15)
        app.file_uploader[0].upload("sales.csv", (self.data if data is None else data).to_csv(
            index=False).encode("utf-8"), "text/csv").run(timeout=15)
        self.assertFalse(app.exception)
        return app

    def start_pending(self):
        app = self.upload()
        app.text_input(key="agent_question").input("导出全部数据").run(timeout=15)
        app.button(key="agent_ask").click().run(timeout=15)
        self.assertFalse(app.exception)
        self.assertTrue(app.button(key="agent_approve"))
        self.assertFalse(any(button.key == "agent_csv_download" for button in app.downloads))
        return app.session_state["agent_pending"]["trace_id"]

    def recover(self, app, trace_id):
        app.text_input(key="agent_recover_trace").input(trace_id).run(timeout=15)
        app.button(key="agent_recover").click().run(timeout=15)
        self.assertFalse(app.exception)
        return app

    def test_new_session_recovers_pending_approval_and_consumes_trace_once(self):
        trace_id = self.start_pending()
        restored = self.recover(self.upload(), trace_id)
        self.assertTrue(restored.button(key="agent_approve"))
        self.assertTrue(restored.button(key="agent_reject"))
        self.assertFalse(any(button.key == "agent_csv_download" for button in restored.downloads))
        restored.button(key="agent_approve").click().run(timeout=15)
        self.assertFalse(restored.exception)
        self.assertTrue(any(button.key == "agent_csv_download" for button in restored.downloads))
        third = self.recover(self.upload(), trace_id)
        self.assertFalse(any(button.key == "agent_approve" for button in third.button))
        self.assertFalse(any(button.key == "agent_csv_download" for button in third.downloads))

    def test_changed_file_or_selected_context_cannot_recover_review(self):
        trace_id = self.start_pending()
        for mismatch in ["file", "sales", "date", "sensitive"]:
            with self.subTest(mismatch=mismatch):
                changed = self.data.copy()
                if mismatch == "file":
                    changed.loc[0, "销售额"] = 99
                app = self.upload(changed)
                if mismatch == "sales":
                    app.selectbox(key="sales_column").select("金额").run(timeout=15)
                elif mismatch == "date":
                    app.selectbox(key="date_column").select("日期2").run(timeout=15)
                elif mismatch == "sensitive":
                    app.multiselect(key="sensitive_columns").set_value(["邮箱", "商品"]).run(timeout=15)
                app = self.recover(app, trace_id)
                self.assertFalse(any(button.key == "agent_approve" for button in app.button))
                self.assertFalse(any(button.key == "agent_csv_download" for button in app.downloads))

    def check_stale_first_review_cannot_decide_second_review(self, decision):
        store = importlib.import_module("review_store")
        self.model.responses[:] = [
            AIMessage(content=json.dumps({"steps": [
                {"tool": "full_csv_export", "args": {}},
                {"tool": "column_preview", "args": {"column_name": "邮箱"}}]})),
            AIMessage(content="", tool_calls=[{"name": "full_csv_export", "args": {}, "id": "export-first"}]),
            AIMessage(content="", tool_calls=[{"name": "column_preview", "args": {"column_name": "邮箱"}, "id": "preview-second"}]),
            AIMessage(content="两次审核完成"),
        ]
        owner = self.upload()
        owner.text_input(key="agent_question").input("导出全部数据，然后预览邮箱").run(timeout=15)
        owner.button(key="agent_ask").click().run(timeout=15)
        self.assertFalse(owner.exception)
        trace_id = owner.session_state["agent_pending"]["trace_id"]
        stale = self.recover(self.upload(), trace_id)
        self.assertEqual(stale.session_state["agent_pending"]["interrupt"]["action_requests"][0]["name"], "full_csv_export")

        owner.button(key="agent_approve").click().run(timeout=15)
        self.assertFalse(owner.exception)
        second = store.load_pending(trace_id)
        self.assertEqual(second["actions"][0]["name"], "column_preview")
        requests_before_stale_click = len(self.model.requests)

        stale.button(key=decision).click().run(timeout=15)
        self.assertFalse(stale.exception)
        self.assertEqual(store.load_pending(trace_id), second,
                         "第一次审核的旧页面不得消费同 trace 的第二次审核")
        self.assertEqual(len(self.model.requests), requests_before_stale_click)
        self.assertFalse(any(button.key == "agent_csv_download" for button in stale.downloads))
        self.assertNotIn("column_preview", owner.session_state["agent_pending"]["artifacts"])
        recovered = self.recover(self.upload(), trace_id)
        self.assertTrue(recovered.button(key="agent_approve"))
        self.assertEqual(recovered.session_state["agent_pending"]["interrupt"]["action_requests"][0]["name"], "column_preview")

    def test_stale_first_review_approval_cannot_approve_second_interrupt(self):
        self.check_stale_first_review_cannot_decide_second_review("agent_approve")

    def test_stale_first_review_rejection_cannot_reject_second_interrupt(self):
        self.check_stale_first_review_cannot_decide_second_review("agent_reject")


if __name__ == "__main__":
    unittest.main()


