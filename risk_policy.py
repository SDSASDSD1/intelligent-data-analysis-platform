"""页面、分析计划和实际工具调用共用的本机数据访问规则。"""


def _risk(level, reason):
    return {"level": level, "reason": reason, "requires_review": level == "high"}


def evaluate_tool_risk(name, arguments, *, role="admin", sensitive_columns=(),
                       sales_column="销售额", date_column=None):
    """敏感字段的统计禁止执行；原始内容仅允许管理员审核后在本地查看。"""
    tool_parameters = {
        "sales_summary": set(),
        "product_sales_total": {"product_name"},
        "schema_inspect": set(),
        "sales_chart": {"chart_type"},
        "column_preview": {"column_name"},
        "full_csv_export": set(),
        "daily_sales_summary": set(),
    }
    if role not in {"admin", "analyst"}:
        return _risk("blocked", "当前账号角色无权执行分析工具。")
    if (not isinstance(name, str) or name not in tool_parameters
            or not isinstance(arguments, dict) or set(arguments) != tool_parameters[name]):
        return _risk("blocked", "工具名称或参数结构不在允许范围内。")

    sensitive = set(sensitive_columns)
    if sales_column in sensitive and name in {
        "sales_summary", "product_sales_total", "daily_sales_summary", "sales_chart"
    }:
        return _risk("blocked", "金额字段已标记为敏感，禁止统计及生成相关图表。")
    if "商品" in sensitive and (
        name == "product_sales_total"
        or (name == "sales_chart" and arguments.get("chart_type") == "bar")
    ):
        return _risk("blocked", "商品字段已标记为敏感，禁止按商品统计及生成柱状图。")
    if date_column in sensitive and name == "daily_sales_summary":
        return _risk("blocked", "日期字段已标记为敏感，禁止按日期统计。")

    sensitive_preview = (
        name == "column_preview"
        and isinstance(arguments.get("column_name"), str)
        and arguments["column_name"] in sensitive
    )
    if name == "full_csv_export" or sensitive_preview:
        if role != "admin":
            return _risk("blocked", "分析员无权预览敏感字段或导出全部原始数据。")
        reason = (
            "导出包含全部原始数据，必须由管理员审核后下载。"
            if name == "full_csv_export"
            else "预览字段包含敏感数据，必须由管理员审核后在本地查看。"
        )
        return _risk("high", reason)
    return _risk("low", "仅使用公开字段的统计、字段信息或本地预览。")


def evaluate_plan_risk(plan, *, role="admin", sensitive_columns=(),
                       sales_column="销售额", date_column=None):
    """计划的风险取所有步骤中的最高等级；执行前仍会逐次检查。"""
    if (not isinstance(plan, dict) or set(plan) != {"steps"}
            or not isinstance(plan["steps"], list) or not 1 <= len(plan["steps"]) <= 6):
        return {**_risk("blocked", "分析计划必须包含一至六个工具步骤。"), "steps": []}
    steps = []
    for step in plan["steps"]:
        if not isinstance(step, dict) or set(step) != {"tool", "args"}:
            return {**_risk("blocked", "分析计划的步骤格式不正确。"), "steps": steps}
        risk = evaluate_tool_risk(
            step["tool"], step["args"], role=role, sensitive_columns=sensitive_columns,
            sales_column=sales_column, date_column=date_column,
        )
        steps.append({"tool": step["tool"], "args": step["args"], "risk": risk})
    highest = max((step["risk"] for step in steps),
                  key=lambda risk: {"low": 0, "high": 1, "blocked": 2}[risk["level"]])
    return {**highest, "steps": steps}

