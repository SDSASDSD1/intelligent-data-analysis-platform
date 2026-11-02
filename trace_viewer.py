"""按记录顺序展示 Trace；查看过程不会重新执行工具。"""

import streamlit as st

from trace_store import ALLOWED_FIELDS


EVENT_LABELS = {
    "query": "提出问题",
    "plan": "生成计划",
    "tool_call": "调用工具",
    "review_pending": "等待审核",
    "review": "人工审核",
    "answer": "分析回答",
}
STATUS_LABELS = {
    "accepted": "已接受", "rejected": "已拒绝", "pending": "待审核",
    "success": "成功", "ok": "成功", "complete": "已完成", "error": "失败",
    "blocked": "已拦截", "approved": "已批准", "ready": "已准备",
    "limit_reached": "达到执行上限，已停止",
}
RISK_LABELS = {"low": "低风险", "high": "高风险", "blocked": "禁止执行"}


def build_trace_steps(events) -> list[dict]:
    """兼容旧日志没有风险、结果摘要和回答字段的情况。"""
    steps = []
    for event in events:
        if not isinstance(event, dict):
            continue
        step = {key: value for key, value in event.items() if key in ALLOWED_FIELDS or key == "timestamp"}
        step["step"] = len(steps) + 1
        step["label"] = EVENT_LABELS.get(event.get("event"), "过程记录")
        step.setdefault("status", "")
        steps.append(step)
    return steps


def render_trace(events):
    steps = build_trace_steps(events)
    if not steps:
        st.info("没有找到这次分析的过程记录。")
        return
    trace_id = next((step["trace_id"] for step in steps if step.get("trace_id")), "旧记录未保存")
    st.caption(f"Trace ID：{trace_id} · 按记录顺序查看，不会再次执行工具。")
    st.dataframe([
        {
            "步骤": step["step"],
            "阶段": step["label"],
            "工具": step.get("tool", "—"),
            "风险": RISK_LABELS.get(step.get("risk_level"), "未记录"),
            "状态": STATUS_LABELS.get(step["status"], step["status"] or "未记录"),
            "耗时（毫秒）": step.get("duration_ms"),
        }
        for step in steps
    ], hide_index=True)
    for step in steps:
        title = f'{step["step"]}. {step["label"]}'
        if step.get("tool"):
            title += f' · {step["tool"]}'
        with st.expander(title, expanded=step.get("event") == "answer"):
            if step.get("timestamp"):
                st.caption(step["timestamp"])
            if "query" in step:
                st.write("问题")
                st.text(step["query"])
            if "plan" in step:
                st.write("计划")
                st.json(step["plan"])
            if "args" in step:
                st.write("工具参数")
                st.json(step["args"])
            if step.get("risk_level"):
                st.write(f'风险：{RISK_LABELS.get(step["risk_level"], step["risk_level"])}')
            if step.get("risk_reason"):
                st.text(step["risk_reason"])
            if "result_summary" in step:
                st.write("结果摘要")
                st.json(step["result_summary"])
            if step.get("decision"):
                st.write("审核决定：" + {"approve": "批准", "reject": "拒绝"}.get(
                    step["decision"], step["decision"]))
            st.write("状态：" + STATUS_LABELS.get(step["status"], step["status"] or "旧记录未保存"))
            if "duration_ms" in step:
                st.caption(f'耗时：{step["duration_ms"]} 毫秒')
            if "answer" in step:
                st.write("最终回答")
                st.text(step["answer"])
            elif step.get("event") == "answer" and step.get("status") == "complete":
                st.info("这条旧记录未保存最终回答正文。")
    with st.expander("查看原始事件（已脱敏）"):
        st.json(events)

