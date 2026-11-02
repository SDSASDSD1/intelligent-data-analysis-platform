"""受控的销售分析计划与 LangChain Agent。"""

import json
import re
from time import perf_counter

import pandas as pd
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware, wrap_tool_call, wrap_model_call
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

import sales_tools
import trace_store
from risk_policy import evaluate_tool_risk


# 图步数还包括限流、审核等中间件节点：6 次工具调用加最终回答需约 41 步。
# 为内部检查留出空间；实际模型、工具调用仍由下面的 7 次、6 次限制控制。
GRAPH_RECURSION_LIMIT = 64
EXECUTION_LIMIT_ERRORS = (
    GraphRecursionError, ModelCallLimitExceededError, ToolCallLimitExceededError,
)


def suggest_sensitive_columns(columns) -> list[str]:
    keywords = ("手机号", "手机", "电话", "身份证", "邮箱", "email", "phone", "mobile", "id_card", "address", "地址")
    return [column for column in columns if any(word in str(column).lower() for word in keywords)]


def parse_analysis_plan(text: str) -> dict | None:
    """先解析完整 JSON，再提取文字中的 JSON；解析失败时返回 None。"""
    if not isinstance(text, str):
        return None
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group())
    for candidate in candidates:
        try:
            plan = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(plan, dict):
            return plan
    return None


def _redact_message_content(content, sensitive_values):
    """保留工具 JSON 的数据类型，再遮挡其中的敏感文本。"""
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, (dict, list)):
                return json.dumps(trace_store.redact(parsed, sensitive_values), ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    return trace_store.redact(content, sensitive_values)


def tool_permitted(name, arguments, role, sensitive_columns,
                   sales_column="销售额", date_column=None):
    return evaluate_tool_risk(
        name, arguments, role=role, sensitive_columns=sensitive_columns,
        sales_column=sales_column, date_column=date_column,
    )["level"] != "blocked"


def validate_analysis_plan(plan, data: pd.DataFrame, grouped: pd.DataFrame | None,
                           daily: pd.DataFrame | None = None, role="admin",
                           sensitive_columns=(), sales_column="销售额",
                           date_column=None) -> bool:
    if not isinstance(plan, dict) or set(plan) != {"steps"}:
        return False
    steps = plan["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
        return False
    return all(
        isinstance(step, dict) and set(step) == {"tool", "args"}
        and sales_tools.validate_agent_tool_call(step["tool"], step["args"], data, grouped, daily)
        and tool_permitted(step["tool"], step["args"], role, sensitive_columns,
                           sales_column, date_column)
        for step in steps
    )


def create_analysis_agent(model, data: pd.DataFrame, sales: pd.Series,
                          grouped: pd.DataFrame | None, sensitive_columns, artifacts: dict,
                          enable_review: bool = False, trace_id: str | None = None, log_dir=None,
                          sensitive_values=(), daily: pd.DataFrame | None = None,
                          checkpointer=None, role="admin", access_check=None,
                          log_cipher=None, sales_column=None, date_column=None):
    sales_column = sales_column if sales_column is not None else (
        sales.name if sales.name is not None else "销售额"
    )
    tools = [
        sales_tools.make_sales_summary_tool(sales),
        sales_tools.make_schema_tool(data, sensitive_columns),
        sales_tools.make_chart_tool(sales, grouped, artifacts),
        sales_tools.make_column_preview_tool(data, sensitive_columns, artifacts),
        sales_tools.make_full_csv_export_tool(data, artifacts),
    ]
    if grouped is not None:
        tools.insert(1, sales_tools.make_product_sales_tool(grouped))
    else:
        # 保持六个明确工具接口；缺少商品列时此工具只返回无法查询。
        empty_grouped = pd.DataFrame({"商品": [], "销售额合计": []})
        tools.insert(1, sales_tools.make_product_sales_tool(empty_grouped))
    if daily is not None and not daily.empty:
        tools.append(sales_tools.make_daily_sales_tool(daily))

    @wrap_model_call
    def redact_model_history(request, handler):
        # 旧检查点也可能包含未脱敏的问题；只修改发给模型的副本。
        messages = []
        for message in request.messages:
            content = (_redact_message_content(message.content, sensitive_values)
                       if isinstance(message, ToolMessage)
                       else trace_store.redact(message.content, sensitive_values))
            updates = {"content": content}
            if isinstance(message, AIMessage) and message.tool_calls:
                updates["tool_calls"] = [
                    {**call, "args": {
                        key: value if key == "column_name" and isinstance(value, str) and value in data.columns
                        else trace_store.redact(value, sensitive_values)
                        for key, value in call["args"].items()
                    }}
                    for call in message.tool_calls
                ]
            messages.append(message.model_copy(update=updates))
        return handler(request.override(messages=messages))

    @wrap_tool_call
    def guard_tool_call(request, handler):
        call = request.tool_call
        started = perf_counter()
        risk = evaluate_tool_risk(
            call.get("name"), call.get("args"), role=role, sensitive_columns=sensitive_columns,
            sales_column=sales_column, date_column=date_column,
        )
        if not sales_tools.validate_agent_tool_call(call.get("name"), call.get("args"), data, grouped, daily):
            risk = {"level": "blocked", "reason": "工具名称或参数未通过检查，本次未执行。", "requires_review": False}
        elif access_check is not None and not access_check():
            risk = {"level": "blocked", "reason": "登录状态已失效，本次未执行。", "requires_review": False}
        elif risk["requires_review"] and not enable_review:
            risk = {"level": "blocked", "reason": "此工具需要人工审核，当前未启用审核，本次未执行。", "requires_review": False}
        if risk["level"] == "blocked":
            result = ToolMessage(content=risk["reason"], tool_call_id=call["id"], status="error")
        else:
            result = handler(request)
        # 工具返回的 JSON 先解析再脱敏，保留 JSON 类型与 tool_call_id。
        content = _redact_message_content(result.content, sensitive_values)
        result = result.model_copy(update={"content": content})
        if trace_id is not None:
            trace_store.append_trace({
                "trace_id": trace_id, "event": "tool_call", "tool": call.get("name"),
                "args": call.get("args"), "status": result.status,
                "risk_level": risk["level"], "risk_reason": risk["reason"],
                "result_summary": (
                    trace_store.summarize_tool_result(call.get("name"), result)
                    if result.status == "success" else {}
                ),
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            }, log_dir, sensitive_values=sensitive_values, cipher=log_cipher)
        return result

    middleware = [
        redact_model_history,
        ModelCallLimitMiddleware(run_limit=7, exit_behavior="error"),
        ToolCallLimitMiddleware(run_limit=6, exit_behavior="error"),
        guard_tool_call,
    ]
    if enable_review and role == "admin":
        def requires_review(request):
            return evaluate_tool_risk(
                request.tool_call.get("name"), request.tool_call.get("args"), role=role,
                sensitive_columns=sensitive_columns, sales_column=sales_column,
                date_column=date_column,
            )["requires_review"]

        middleware.append(HumanInTheLoopMiddleware(interrupt_on={
            "column_preview": {
                "allowed_decisions": ["approve", "reject"],
                "when": requires_review,
            },
            "full_csv_export": {"allowed_decisions": ["approve", "reject"], "when": requires_review},
        }))

    return create_agent(
        model=model,
        tools=tools,
        system_prompt="你是销售数据分析助手。仅用提供的工具完成用户问题；每次观察工具结果后再决定下一步。不要猜测数据，不要请求执行代码。工具返回的状态不代表原始字段值已发送给你。",
        middleware=middleware,
        checkpointer=(checkpointer or InMemorySaver()) if enable_review else None,
    )

