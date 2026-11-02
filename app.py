import json
import hashlib
import os
from uuid import uuid4
from time import perf_counter
from zipfile import BadZipFile
from functools import partial

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from openpyxl.utils.exceptions import InvalidFileException

import sales_tools
import analysis_agent
import date_analysis
import trace_store
import report_export
import review_store
import auth_store
import app_access
import secure_storage
import secure_download
import local_runtime
import risk_policy
import trace_viewer
import ui_theme


def save_review_pending(run, upload_fingerprint, sales_column, date_column, sensitive_values, storage):
    run["review_id"] = uuid4().hex
    review_store.save_pending({
        "trace_id": run["trace_id"],
        "review_id": run["review_id"],
        "owner_id": run["owner_id"],
        "thread_id": run["config"]["configurable"]["thread_id"],
        "fingerprint": upload_fingerprint,
        "sales_column": sales_column,
        "date_column": date_column,
        "sensitive_columns": list(run["sensitive_columns"]),
        "plan": run["plan"],
        "question": run["question"],
        "actions": run["interrupt"]["action_requests"],
    }, review_dir=storage.review_dir, sensitive_values=sensitive_values, cipher=storage.cipher)


def review_matches_pending(record, pending, upload_fingerprint, sales_column,
                           date_column, sensitive_columns, owner_id):
    pending_actions = pending["interrupt"]["action_requests"]
    return (
        record is not None
        and record.get("owner_id") == pending.get("owner_id") == owner_id
        and isinstance(pending.get("review_id"), str)
        and record.get("review_id") == pending["review_id"]
        and [(action.get("name"), action.get("args")) for action in record.get("actions", [])]
        == [(action.get("name"), action.get("args")) for action in pending_actions]
        and review_store.matches_context(
            record, upload_fingerprint, sales_column, date_column, sensitive_columns
        )
    )


def show_plan_risk(plan, policy_context):
    """把代码计算的风险翻译成页面说明；风险不由模型自行决定。"""
    risk = risk_policy.evaluate_plan_risk(plan, **policy_context)
    labels = {"low": "低风险", "high": "高风险，执行前需审核", "blocked": "禁止执行"}
    st.info(f"计划风险：{labels[risk['level']]}。{risk['reason']}")
    for step in risk.get("steps", []):
        item = step["risk"]
        st.caption(f"{step['tool']}：{labels[item['level']]}；{item['reason']}")
    return risk


def append_review_trace(append_trace, trace_id, actions, status, policy_context, duration_ms=None):
    """记录本次实际待审动作，不能仅根据原计划推测审核了什么。"""
    for action in actions:
        risk = risk_policy.evaluate_tool_risk(action["name"], action["args"], **policy_context)
        event = {
            "trace_id": trace_id, "event": "review", "status": status,
            "tool": action["name"], "args": action["args"],
            "risk_level": risk["level"], "risk_reason": risk["reason"],
        }
        if status in {"approved", "rejected"}:
            event["decision"] = "approve" if status == "approved" else "reject"
        if duration_ms is not None:
            event["duration_ms"] = duration_ms
        append_trace(event)


def show_agent_error(error, trace_id):
    """区分执行上限与其他故障，并返回写入 Trace 的状态。"""
    if isinstance(error, analysis_agent.EXECUTION_LIMIT_ERRORS):
        st.error(
            "Agent 达到本次分析的执行上限，已停止。请缩小问题范围后重新分析；"
            f"可用 trace_id 查看已完成的步骤。trace_id：{trace_id}"
        )
        return "limit_reached"
    st.error(
        f"Agent 分析失败（{type(error).__name__}）。请检查模型连接后重试。"
        f"trace_id：{trace_id}"
    )
    return "error"


def main():
    ui_theme.configure_page()
    st.set_option("client.disableDataExport", True)
    ui_theme.render_header()
    try:
        local_runtime.ensure_app_lock()
    except (RuntimeError, OSError):
        st.error("资料正在被另一实例或备份恢复操作使用，请关闭其他实例后重试。")
        return
    user = app_access.require_user()
    if user is None:
        return
    try:
        storage = app_access.workspace(user)
    except secure_storage.StorageError:
        st.error("本机存储密钥无法读取，请检查原密钥后重启；不要重新生成密钥覆盖旧资料。")
        return
    app_access.render_account_management(user)
    for state_key in ("agent_pending", "agent_result"):
        run = st.session_state.get(state_key)
        if run is not None and run.get("owner_id") != user["id"]:
            st.session_state.pop(state_key, None)
    append_trace = partial(trace_store.append_trace, log_dir=storage.log_dir, cipher=storage.cipher)
    append_audit = partial(trace_store.append_audit, log_dir=storage.log_dir, cipher=storage.cipher)
    load_pending = partial(review_store.load_pending, review_dir=storage.review_dir, cipher=storage.cipher)
    take_pending = partial(review_store.take_pending, review_dir=storage.review_dir, cipher=storage.cipher)
    open_checkpointer = partial(review_store.open_checkpointer, db_path=storage.checkpoint_path, cipher=storage.cipher)
    session_token = st.session_state.get("auth_token")
    access_check = (lambda: auth_store.resolve_session(session_token) is not None) if session_token else None
    with st.sidebar.expander("历史分析 · Trace", icon=":material/history:"):
        st.caption("这里查询已有分析记录；新问题请在主页面的“多步 Agent”中填写。")
        lookup_id = st.text_input(
            "按 trace_id 查看分析过程", key="trace_lookup",
            placeholder="填写分析结果中的编号",
            help="trace_id 是一串字母和数字，可从分析结果或错误提示中复制。",
        )
        trace_requested = st.button("查看 Trace", key="trace_search", width="stretch")
    if trace_requested:
        try:
            events = trace_store.read_trace(lookup_id.strip(), storage.log_dir, cipher=storage.cipher)
        except secure_storage.StorageError:
            st.error("分析过程无法解密，请检查原密钥或资料完整性。")
            events = []
        if events:
            trace_viewer.render_trace(events)
        else:
            st.info("未找到该 trace_id 的记录。")

    with st.container(key="panel_upload"):
        st.subheader("上传数据")
        ui_theme.render_steps()
        uploaded_file = st.file_uploader(
            "上传销售数据 CSV 或 Excel（.xlsx）文件", type=["csv", "xlsx"]
        )
        st.caption("支持 UTF-8 编码的 CSV，以及 .xlsx 的第一个工作表。")
        if uploaded_file is not None:
            st.caption("当前文件：" + uploaded_file.name)
    if uploaded_file is None:
        ui_theme.render_empty_state()
        return

    upload_fingerprint = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("agent_upload_fingerprint") != upload_fingerprint:
        st.session_state.pop("agent_pending", None)
        st.session_state.pop("agent_result", None)
        st.session_state.pop("sales_column", None)
        st.session_state.pop("agent_sales_column", None)
        st.session_state.pop("date_column", None)
        st.session_state.pop("agent_date_column", None)
        st.session_state.pop("sensitive_columns", None)
        st.session_state.pop("agent_sensitive_columns", None)
        st.session_state["agent_upload_fingerprint"] = upload_fingerprint

    if uploaded_file.name.lower().endswith(".xlsx"):
        try:
            data = pd.read_excel(uploaded_file, engine="openpyxl", sheet_name=0)
        except (BadZipFile, InvalidFileException, OSError, ValueError):
            st.error("无法读取 Excel 文件，请检查 .xlsx 文件格式。")
            return
    else:
        try:
            data = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            st.error("无法读取 CSV，请检查文件内容和 UTF-8 编码。")
            return

    with st.container(key="panel_privacy"):
        st.subheader("字段保护")
        required_sensitive = analysis_agent.suggest_sensitive_columns(data.columns) if user["role"] == "analyst" else []
        if required_sensitive:
            st.caption("分析员账户保留自动识别的敏感字段：" + "、".join(required_sensitive))
        selected_sensitive = st.multiselect(
            "敏感字段（可补充勾选）",
            options=[column for column in data.columns if column not in required_sensitive],
            default=[] if user["role"] == "analyst" else analysis_agent.suggest_sensitive_columns(data.columns),
            key="sensitive_columns",
        )
        sensitive_columns = list(dict.fromkeys(required_sensitive + selected_sensitive))
        if user["role"] == "analyst":
            st.caption("分析员可进行统计和下载报告；敏感字段预览与原始数据全量导出需要管理员角色。")
        sensitive_values = tuple(
            str(value) for column in sensitive_columns
            for value in data[column].dropna().unique() if str(value)
        )
        append_trace = partial(append_trace, sensitive_values=sensitive_values)
        append_audit = partial(append_audit, sensitive_values=sensitive_values)
        if st.session_state.get("agent_sensitive_columns") != tuple(sensitive_columns):
            st.session_state.pop("agent_pending", None)
            st.session_state.pop("agent_result", None)
            st.session_state["agent_sensitive_columns"] = tuple(sensitive_columns)

    preview_column, fields_column = st.columns([3, 2])
    with preview_column, st.container(key="panel_preview"):
        preview = data.head(10).copy()
        for column in sensitive_columns:
            preview[column] = "需审核"
        st.subheader("数据预览（前 10 行）")
        st.dataframe(preview, hide_index=True)

    with fields_column, st.container(key="panel_fields"):
        field_info = pd.DataFrame({
            "字段名": data.columns.tolist(),
            "数据类型": data.dtypes.astype(str).tolist(),
            "非空数量": data.count().tolist(),
            "缺失数量": data.isna().sum().tolist(),
        })
        st.subheader("字段信息")
        st.dataframe(field_info, hide_index=True)

    with st.container(key="panel_settings"):
        st.subheader("分析字段设置")
        columns = data.columns.tolist()
        sales_column = st.selectbox(
            "选择销售额字段",
            options=columns,
            index=columns.index("销售额") if "销售额" in columns else None,
            placeholder="请选择用于计算销售额的列",
            key="sales_column",
        )
        if st.session_state.get("agent_sales_column") != sales_column:
            st.session_state.pop("agent_pending", None)
            st.session_state.pop("agent_result", None)
            st.session_state["agent_sales_column"] = sales_column

        if sales_column is None:
            st.info("文件缺少“销售额”列，请选择用于计算销售额的字段。")
            return

        if sales_column in sensitive_columns:
            st.info("销售额字段已标记为敏感，自动统计、图表和模型分析已停止。请选择非敏感的金额字段后继续。")
            return

        sales = pd.to_numeric(data[sales_column], errors="coerce")
        if sales.isna().any():
            st.error(f"“{sales_column}”列必须全部填写数字。")
            return

        date_column = st.selectbox(
            "选择日期字段（可选）",
            options=columns,
            index=columns.index("日期") if "日期" in columns else None,
            placeholder="不按日期分析",
            key="date_column",
        )
        if st.session_state.get("agent_date_column") != date_column:
            st.session_state.pop("agent_pending", None)
            st.session_state.pop("agent_result", None)
            st.session_state["agent_date_column"] = date_column
        daily = None
        if date_column is not None and date_column not in sensitive_columns:
            try:
                daily = date_analysis.build_daily_sales(data, sales, date_column)
            except ValueError as error:
                st.error(str(error))
        if date_column in sensitive_columns:
            daily = None
            st.info("日期字段已标记为敏感，按日期统计暂不显示。")

    grouped = None
    if "商品" in data.columns and "商品" not in sensitive_columns and not sales.empty:
        grouped = (
            pd.DataFrame({"商品": data["商品"], "销售额合计": sales})
            .groupby("商品", as_index=False, sort=False, dropna=False)["销售额合计"]
            .sum()
        )

    policy_context = {
        "role": user["role"], "sensitive_columns": sensitive_columns,
        "sales_column": sales_column, "date_column": date_column,
    }
    with st.container(key="workspace_metrics"):
        total_card, average_card, rows_card, fields_card = st.columns(4)
        total_card.metric("销售额总和", f"{sales.sum():,.2f}")
        average_card.metric("销售额平均值", "—" if sales.empty else f"{sales.mean():,.2f}")
        rows_card.metric("记录数量", len(data))
        fields_card.metric("字段数量", len(data.columns))
    if sales.empty:
        st.info("没有销售数据，无法计算平均值。")

    with st.expander("普通问答 · 固定规则或模型建议工具", icon=":material/chat:"):
        st.subheader("向数据提问")
        st.caption("问题先脱敏，只发送必要的统计值；已标记敏感的商品不参与商品查询。")
        question = st.text_input("请输入关于销售额的问题", key="analysis_question")
        safe_question = trace_store.redact(question.strip(), sensitive_values)
        use_model_selection = st.checkbox("让模型建议工具", value=False)
        if st.button("提问", key="analysis_ask"):
            if not question.strip():
                st.warning("请先输入问题。")
            else:
                tool_calls = sales_tools.select_tool_calls(safe_question, grouped)
                if not tool_calls:
                    st.warning("一次请只问一个商品的销售额。")
                else:
                    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
                    if not api_key:
                        st.warning("请先设置 DASHSCOPE_API_KEY 环境变量。")
                    else:
                        base_url = os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                        try:
                            available_tools = {"sales_summary": sales_tools.make_sales_summary_tool(sales)}
                            if grouped is not None:
                                available_tools["product_sales_total"] = sales_tools.make_product_sales_tool(grouped)
                            model = ChatOpenAI(
                                model="qwen-plus",
                                api_key=api_key,
                                base_url=base_url,
                                max_tokens=256,
                                timeout=30,
                                max_retries=0,
                                use_responses_api=False,
                            )
                            choice_source = "固定规则"
                            if use_model_selection:
                                proposal = model.bind_tools(
                                    list(available_tools.values()), tool_choice="auto"
                                ).invoke([
                                    ("system", "只建议一个已提供的统计工具，不要回答问题。商品名称必须来自用户的问题。"),
                                    ("human", safe_question),
                                ])
                                validated_calls = sales_tools.validate_model_tool_calls(
                                    proposal.tool_calls, safe_question, grouped
                                )
                                if validated_calls is None:
                                    tool_calls = None
                                elif validated_calls:
                                    tool_calls = validated_calls
                                    choice_source = "模型建议（已校验）"
                                else:
                                    choice_source = "固定规则（模型未建议）"

                            if tool_calls is not None and any(
                                risk_policy.evaluate_tool_risk(name, arguments, **policy_context)["level"] != "low"
                                for name, arguments in tool_calls
                            ):
                                tool_calls = None
                            if tool_calls is not None:
                                results = {}
                                for tool_name, arguments in tool_calls:
                                    results[tool_name] = available_tools[tool_name].invoke(arguments)

                                summary_result = results["sales_summary"]
                                average = (
                                    f"{summary_result['sales_average']:.2f}"
                                    if summary_result["sales_average"] is not None else "无数据，无法计算"
                                )
                                summary = (
                                    f"数据行数：{summary_result['row_count']}\n"
                                    f"销售额总和：{summary_result['sales_total']:.2f}\n"
                                    f"销售额平均值：{average}"
                                )
                                if "product_sales_total" in results:
                                    product_result = results["product_sales_total"]
                                    summary += f"\n所问商品销售额合计：{product_result['sales_total']:.2f}"
                                if choice_source == "模型建议（已校验）":
                                    proposed_call = proposal.tool_calls[0]
                                    reply = model.invoke([
                                        ("system", f"只能根据给定的销售汇总和工具结果回答问题，不要编造数据。\n{summary}"),
                                        ("human", safe_question),
                                        proposal,
                                        ToolMessage(
                                            content=json.dumps(results[proposed_call["name"]], ensure_ascii=False),
                                            tool_call_id=proposed_call["id"],
                                        ),
                                    ])
                                else:
                                    reply = model.invoke([
                                        ("system", "只能根据给定的销售汇总回答问题。如果摘要无法回答，就说明无法从当前摘要判断，不要编造数据。"),
                                        ("human", f"{summary}\n问题：{safe_question}"),
                                    ])
                        except Exception as error:
                            st.error(f"提问失败（{type(error).__name__}）。请检查模型连接后重试。")
                        else:
                            if tool_calls is None:
                                st.warning("模型建议未通过检查，本次未执行工具。")
                            else:
                                tool_labels = {
                                    "sales_summary": "销售汇总",
                                    "product_sales_total": "指定商品销售额",
                                }
                                st.caption("选择方式：" + choice_source)
                                st.caption("本次工具：" + "、".join(tool_labels[name] for name, _ in tool_calls))
                                st.write(trace_store.redact(reply.content, sensitive_values))

    with st.container(key="panel_agent"):
        st.subheader("多步 Agent")
        st.caption("先展示结构化计划，再由受控工具逐步分析；模型不会收到上传文件的原始数据行。")
        ui_theme.render_question_examples()
        agent_question = st.text_input(
            "请输入多步分析问题", key="agent_question",
            placeholder="例如：先查看字段，再统计销售额，最后选择柱状图。",
        )
        safe_agent_question = trace_store.redact(agent_question.strip(), sensitive_values)
        if st.button("开始 Agent 分析", key="agent_ask", type="primary", icon=":material/auto_awesome:"):
            st.session_state.pop("agent_pending", None)
            st.session_state.pop("agent_result", None)
            if not agent_question.strip():
                st.warning("请先输入多步分析问题。")
            else:
                api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
                if not api_key:
                    st.warning("请先设置 DASHSCOPE_API_KEY 环境变量。")
                else:
                    base_url = os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    trace_id = uuid4().hex
                    started_at = perf_counter()
                    append_trace({"trace_id": trace_id, "event": "query", "query": agent_question.strip(), "status": "started"}, sensitive_values=sensitive_values)
                    try:
                        model = ChatOpenAI(
                            model="qwen-plus", api_key=api_key, base_url=base_url,
                            max_tokens=512, timeout=30, max_retries=0, use_responses_api=False,
                        )
                        public_columns = data.columns.tolist()
                        plan_reply = model.invoke([
                            ("system", "你是销售数据分析规划员。根据当前文件的字段和用户问题制定计划；不要猜测字段值。只输出一个 JSON 对象，顶层仅有 steps，最多六步；每步仅有 tool 和 args。允许的工具及参数：schema_inspect {} 查看字段；sales_summary {} 计算销售汇总；product_sales_total {\"product_name\":\"用户问题中的完整商品名\"}；sales_chart {\"chart_type\":\"bar 或 line\"}；column_preview {\"column_name\":\"当前文件的字段名\"}；full_csv_export {}。" + ("已选择日期字段时还可用 daily_sales_summary {} 查看按日汇总。" if daily is not None and not daily.empty else "") + "示例：问题‘先看字段再计算总额’，输出 {\"steps\":[{\"tool\":\"schema_inspect\",\"args\":{}},{\"tool\":\"sales_summary\",\"args\":{}}]}。不要输出数组、Markdown 或代码。"),
                            ("human", f"当前字段：{public_columns}\n敏感字段：{sensitive_columns}；禁止使用敏感字段作商品、金额或日期统计。\n问题：{safe_agent_question}"),
                        ])
                        plan = analysis_agent.parse_analysis_plan(plan_reply.content)
                        plan_valid = analysis_agent.validate_analysis_plan(
                            plan, data, grouped, daily, **policy_context
                        )
                        plan_risk = show_plan_risk(plan, policy_context)
                        append_trace({"trace_id": trace_id, "event": "plan", "plan": plan,
                                      "risk_level": plan_risk["level"], "risk_reason": plan_risk["reason"],
                                      "status": "accepted" if plan_valid else "rejected"}, sensitive_values=sensitive_values)
                        if plan_valid:
                            artifacts = {}
                            checkpointer = open_checkpointer()
                            config = {"configurable": {"thread_id": str(uuid4())},
                                      "recursion_limit": analysis_agent.GRAPH_RECURSION_LIMIT}
                            try:
                                agent = analysis_agent.create_analysis_agent(
                                    model, data, sales, grouped, sensitive_columns, artifacts,
                                    enable_review=True, trace_id=trace_id, sensitive_values=sensitive_values,
                                    daily=daily, checkpointer=checkpointer,
                                    role=user["role"], access_check=access_check,
                                    log_dir=storage.log_dir, log_cipher=storage.cipher,
                                    sales_column=sales_column, date_column=date_column,
                                )
                                result = agent.invoke({"messages": [("human", f"问题：{safe_agent_question}\n已校验计划：{json.dumps(plan, ensure_ascii=False)}")]}, config)
                            finally:
                                checkpointer.conn.close()
                            run = {"model": model, "config": config, "plan": plan, "artifacts": artifacts,
                                   "owner_id": user["id"],
                                   "fingerprint": upload_fingerprint, "trace_id": trace_id,
                                   "started_at": started_at, "question": safe_agent_question,
                                   "sensitive_columns": tuple(sensitive_columns)}
                            if result.get("__interrupt__"):
                                run["interrupt"] = result["__interrupt__"][0].value
                                save_review_pending(run, upload_fingerprint, sales_column, date_column, sensitive_values, storage)
                                st.session_state["agent_pending"] = run
                                append_review_trace(append_trace, trace_id, run["interrupt"]["action_requests"],
                                                    "pending", policy_context)
                            else:
                                run["answer"] = trace_store.redact(result["messages"][-1].content, sensitive_values)
                                st.session_state["agent_result"] = run
                                append_trace({"trace_id": trace_id, "event": "answer", "status": "complete", "answer": run["answer"],
                                                          "duration_ms": round((perf_counter() - started_at) * 1000, 3)})
                        else:
                            st.warning("模型计划未通过结构或权限检查，本次未运行 Agent。")
                            st.text(str(trace_store.redact(plan_reply.content, sensitive_values)))
                    except Exception as error:
                        status = show_agent_error(error, trace_id)
                        append_trace({"trace_id": trace_id, "event": "answer", "status": status,
                                                  "duration_ms": round((perf_counter() - started_at) * 1000, 3)})

        with st.expander("恢复历史审核", icon=":material/restore:"):
            recover_trace = st.text_input("恢复待审核 trace_id", key="agent_recover_trace")
            if st.button("恢复待审核", key="agent_recover"):
                try:
                    record = load_pending(recover_trace.strip())
                except secure_storage.StorageError:
                    st.error("待审核资料无法解密，请检查原密钥或资料完整性。")
                    record = None
                if record is None:
                    st.warning("未找到可恢复的待审核请求。")
                elif record.get("owner_id") != user["id"] or user["role"] != "admin":
                    st.warning("当前账户无权恢复此审核。")
                elif not isinstance(record.get("review_id"), str):
                    st.warning("待审核记录缺少版本信息，请重新发起分析。")
                elif not review_store.matches_context(
                    record, upload_fingerprint, sales_column, date_column, sensitive_columns
                ):
                    st.warning("当前文件或字段设置与待审核请求不一致，无法恢复。")
                else:
                    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
                    if not api_key:
                        st.warning("请先设置 DASHSCOPE_API_KEY 环境变量。")
                    else:
                        base_url = os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                        try:
                            model = ChatOpenAI(
                                model="qwen-plus", api_key=api_key, base_url=base_url,
                                max_tokens=512, timeout=30, max_retries=0, use_responses_api=False,
                            )
                            checkpointer = open_checkpointer()
                            config = {"configurable": {"thread_id": record["thread_id"]},
                                      "recursion_limit": analysis_agent.GRAPH_RECURSION_LIMIT}
                            try:
                                checkpoint_exists = checkpointer.get(config) is not None
                            finally:
                                checkpointer.conn.close()
                            if not checkpoint_exists:
                                st.warning("审核检查点不存在，无法恢复。")
                            else:
                                artifacts = {}
                                st.session_state["agent_pending"] = {
                                    "model": model, "config": config, "plan": record["plan"],
                                    "review_id": record["review_id"],
                                    "owner_id": user["id"],
                                    "artifacts": artifacts, "fingerprint": upload_fingerprint,
                                    "trace_id": record["trace_id"], "started_at": perf_counter(),
                                    "question": record["question"],
                                    "sensitive_columns": tuple(sensitive_columns),
                                    "interrupt": {"action_requests": record["actions"]},
                                }
                                st.rerun()
                        except Exception as error:
                            st.error(f"恢复审核失败（{type(error).__name__}）。")

        pending = st.session_state.get("agent_pending")
        if st.session_state.pop("agent_review_rejected", False):
            st.warning("已拒绝，本次工具链未继续执行。")
        if pending is not None:
            with st.container(key="panel_review"):
                st.subheader("等待你的审核")
                if user["role"] != "admin":
                    st.session_state.pop("agent_pending", None)
                    st.warning("当前角色不能审核敏感预览或原始数据导出。")
                    return
                st.caption("trace_id：" + pending["trace_id"])
                st.code(pending["trace_id"], language=None)
                st.json(pending["plan"])
                show_plan_risk(pending["plan"], policy_context)
                actions = pending["interrupt"]["action_requests"]
                st.warning("以下操作需要人工审核，批准前不会执行。")
                for action in actions:
                    st.write({"工具": action["name"], "参数": action["args"]})
                    risk = risk_policy.evaluate_tool_risk(action["name"], action["args"], **policy_context)
                    st.caption(f"本次操作风险：{risk['level']}；{risk['reason']}")
                if st.button("批准并继续", key="agent_approve", type="primary", icon=":material/check:"):
                    try:
                        if any(risk_policy.evaluate_tool_risk(action["name"], action["args"], **policy_context)["level"] == "blocked"
                               for action in actions):
                            st.error("当前策略禁止这项操作，不能通过审核放行。请拒绝并重新发起分析。")
                            return
                        record = load_pending(pending["trace_id"])
                        if not review_matches_pending(
                            record, pending, upload_fingerprint, sales_column, date_column, sensitive_columns, user["id"]
                        ):
                            st.session_state.pop("agent_pending", None)
                            st.error("待审核请求已失效，请重新发起分析。")
                            return
                        claimed = take_pending(
                            pending["trace_id"], expected_review_id=pending["review_id"]
                        )
                        if not review_matches_pending(
                            claimed, pending, upload_fingerprint, sales_column, date_column, sensitive_columns, user["id"]
                        ):
                            st.session_state.pop("agent_pending", None)
                            st.error("待审核请求已更新，请按 trace_id 重新恢复。")
                            return
                        for action in actions:
                            risk = risk_policy.evaluate_tool_risk(action["name"], action["args"], **policy_context)
                            append_audit({"trace_id": pending["trace_id"], "event": "review",
                                          "tool": action["name"], "args": action["args"], "decision": "approve",
                                          "risk_level": risk["level"], "risk_reason": risk["reason"]})
                        append_review_trace(append_trace, pending["trace_id"], actions, "approved", policy_context)
                        decisions = [{"type": "approve"} for _ in actions]
                        checkpointer = open_checkpointer()
                        try:
                            agent = analysis_agent.create_analysis_agent(
                                pending["model"], data, sales, grouped, sensitive_columns,
                                pending["artifacts"], enable_review=True,
                                trace_id=pending["trace_id"], sensitive_values=sensitive_values,
                                daily=daily, checkpointer=checkpointer,
                                role=user["role"], access_check=access_check,
                                log_dir=storage.log_dir, log_cipher=storage.cipher,
                                sales_column=sales_column, date_column=date_column,
                            )
                            # 页面热更新前留下的待审核会话也使用当前图执行预算。
                            pending["config"]["recursion_limit"] = analysis_agent.GRAPH_RECURSION_LIMIT
                            result = agent.invoke(Command(resume={"decisions": decisions}), pending["config"])
                        finally:
                            checkpointer.conn.close()
                        if result.get("__interrupt__"):
                            pending["interrupt"] = result["__interrupt__"][0].value
                            save_review_pending(pending, upload_fingerprint, sales_column, date_column, sensitive_values, storage)
                            append_review_trace(append_trace, pending["trace_id"], pending["interrupt"]["action_requests"],
                                                "pending", policy_context)
                        else:
                            pending["answer"] = trace_store.redact(result["messages"][-1].content, sensitive_values)
                            pending.pop("model", None)
                            st.session_state["agent_result"] = pending
                            st.session_state.pop("agent_pending", None)
                            append_trace({"trace_id": pending["trace_id"], "event": "answer", "status": "complete", "answer": pending["answer"],
                                                      "duration_ms": round((perf_counter() - pending["started_at"]) * 1000, 3)})
                        st.rerun()
                    except Exception as error:
                        status = show_agent_error(error, pending["trace_id"])
                        if status == "limit_reached":
                            st.session_state.pop("agent_pending", None)
                            st.session_state.pop("agent_result", None)
                        append_trace({"trace_id": pending["trace_id"], "event": "answer", "status": status,
                                      "duration_ms": round((perf_counter() - pending["started_at"]) * 1000, 3)})
                if st.button("拒绝并结束", key="agent_reject", icon=":material/close:"):
                    try:
                        record = load_pending(pending["trace_id"])
                    except secure_storage.StorageError:
                        st.error("待审核资料无法解密，请检查原密钥或资料完整性；本次未处理待办。")
                        return
                    if not review_matches_pending(
                        record, pending, upload_fingerprint, sales_column, date_column, sensitive_columns, user["id"]
                    ):
                        st.session_state.pop("agent_pending", None)
                        st.warning("待审核请求已更新，请按 trace_id 重新恢复。")
                        return
                    try:
                        claimed = take_pending(
                            pending["trace_id"], expected_review_id=pending["review_id"]
                        )
                    except secure_storage.StorageError:
                        st.error("待审核资料无法解密，请检查原密钥或资料完整性；本次未处理待办。")
                        return
                    if not review_matches_pending(
                        claimed, pending, upload_fingerprint, sales_column, date_column, sensitive_columns, user["id"]
                    ):
                        st.session_state.pop("agent_pending", None)
                        st.warning("待审核请求已更新，请按 trace_id 重新恢复。")
                        return
                    for action in actions:
                        risk = risk_policy.evaluate_tool_risk(action["name"], action["args"], **policy_context)
                        append_audit({"trace_id": pending["trace_id"], "event": "review",
                                      "tool": action["name"], "args": action["args"], "decision": "reject",
                                      "risk_level": risk["level"], "risk_reason": risk["reason"]})
                    append_review_trace(append_trace, pending["trace_id"], actions, "rejected", policy_context,
                                        duration_ms=round((perf_counter() - pending["started_at"]) * 1000, 3))
                    st.session_state.pop("agent_pending", None)
                    st.session_state["agent_review_rejected"] = True
                    st.rerun()

        finished = st.session_state.get("agent_result")
        if finished is not None:
            with st.container(key="panel_result"):
                st.subheader("分析结果")
                st.caption("trace_id：" + finished["trace_id"])
                st.code(finished["trace_id"], language=None)
                st.json(finished["plan"])
                show_plan_risk(finished["plan"], policy_context)
                st.write(trace_store.redact(finished["answer"], sensitive_values))
                if "column_preview" in finished["artifacts"]:
                    st.dataframe(finished["artifacts"]["column_preview"], hide_index=True)
                if "chart" in finished["artifacts"]:
                    st.caption("Agent 选择图表：" + finished["artifacts"]["chart"])
                report = report_export.build_markdown_report(
                    finished["trace_id"], finished["question"], finished["plan"],
                    finished["answer"], sales, grouped, finished["artifacts"].get("chart"),
                    finished["sensitive_columns"],
                    sales_column=sales_column, date_column=date_column, daily=daily,
                    sensitive_values=sensitive_values,
                )
                secure_download.render_download(
                    "下载分析报告 Markdown", report.encode("utf-8"),
                    file_name=f"analysis-{finished['trace_id']}.md", mime="text/markdown",
                    key="agent_report_download",
                )
                if "full_csv" in finished["artifacts"] and user["role"] == "admin":
                    secure_download.render_download(
                        "下载全部原始数据 CSV", finished["artifacts"]["full_csv"],
                        file_name="sales_full.csv", mime="text/csv", key="agent_csv_download",
                        kind="csv", approved=True,
                    )

    chinese_font = FontProperties(family=["Microsoft YaHei", "Noto Sans SC", "SimHei"])
    st.header("统计与图表")
    with st.container(key="panel_products"):
        if "商品" in sensitive_columns:
            st.info("商品列已标记为敏感，按商品分组和柱状图暂不显示。")
        elif "商品" not in data.columns:
            st.info("文件缺少“商品”列，无法分组。")
        elif sales.empty:
            st.info("没有销售数据，无法分组。")
        else:
            st.subheader("按商品统计销售额")
            st.dataframe(
                grouped,
                hide_index=True,
                column_config={
                    "销售额合计": st.column_config.NumberColumn(format="%.2f")
                },
            )

            fig = Figure(figsize=(8, 4))
            ax = fig.subplots()
            positions = range(len(grouped))
            ax.bar(positions, grouped["销售额合计"], color=ui_theme.PRIMARY, width=0.6)
            ax.set_xticks(
                positions,
                labels=grouped["商品"].fillna("未填写商品").astype(str),
                fontproperties=chinese_font,
            )
            ax.set_ylabel("销售额合计", fontproperties=chinese_font)
            ax.set_title("按商品统计销售额", fontproperties=chinese_font)
            ui_theme.style_chart(fig, ax)
            fig.tight_layout()
            st.pyplot(fig)

    if not sales.empty:
        with st.container(key="panel_line_chart"):
            st.subheader("逐条销售额折线图")
            st.caption("横轴是文件中的数据行序号，不代表日期。")
            line_fig = Figure(figsize=(8, 4))
            line_ax = line_fig.subplots()
            line_ax.plot(range(1, len(sales) + 1), sales.to_numpy(), marker="o", color=ui_theme.PRIMARY, linewidth=2)
            line_ax.set_xlim(0.5, len(sales) + 0.5)
            line_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            line_ax.set_xlabel("数据行序号", fontproperties=chinese_font)
            line_ax.set_ylabel("销售额", fontproperties=chinese_font)
            line_ax.set_title("逐条销售额", fontproperties=chinese_font)
            ui_theme.style_chart(line_fig, line_ax)
            line_fig.tight_layout()
            st.pyplot(line_fig)

    if daily is not None and not daily.empty:
        with st.container(key="panel_daily_chart"):
            st.subheader("按日期统计销售额")
            st.dataframe(daily, hide_index=True)
            daily_fig = Figure(figsize=(8, 4))
            daily_ax = daily_fig.subplots()
            daily_ax.plot(daily["日期"], daily["销售额合计"], marker="o", color=ui_theme.PRIMARY, linewidth=2)
            daily_ax.set_xlabel("日期", fontproperties=chinese_font)
            daily_ax.set_ylabel("销售额合计", fontproperties=chinese_font)
            daily_ax.set_title("每日销售额趋势", fontproperties=chinese_font)
            daily_fig.autofmt_xdate()
            ui_theme.style_chart(daily_fig, daily_ax)
            daily_fig.tight_layout()
            st.pyplot(daily_fig)


if __name__ == "__main__":
    main()

