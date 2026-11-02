"""Generate a short Markdown report from a completed local analysis."""

import trace_store

def _table_cell(value):
    """Keep a data value inside one Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")


def build_markdown_report(trace_id, question, plan, answer, sales, grouped, chart=None,
                          sensitive_columns=(), sales_column=None, date_column=None,
                          daily=None, sensitive_values=()):
    """Summarize computed results without including uploaded raw rows."""
    selected_sales = sales_column if sales_column is not None else (
        sales.name if sales.name is not None else "销售额"
    )
    amount_is_sensitive = selected_sales in sensitive_columns
    lines = [
        "# 数析数据分析报告",
        "",
        f"Trace ID：{trace_id}",
        f"问题：{trace_store.redact(question, sensitive_values)}",
        "",
        "## 统计结果",
        "",
    ]
    if sales_column is not None:
        lines.append(f"金额字段：{sales_column}")
    if date_column is not None:
        lines.append(f"日期字段：{date_column}")
    lines.append(f"记录数：{len(sales)}")
    if amount_is_sensitive:
        lines.append("金额字段已标记为敏感，报告不提供金额统计及相关图表。")
    else:
        average = f"{sales.mean():.2f}" if not sales.empty else "无法计算"
        lines.extend([f"销售额总和：{sales.sum():.2f}", f"销售额平均值：{average}"])
    lines.extend([
        "",
        "## 分析计划",
        "",
    ])
    for step in plan.get("steps", []):
        lines.append(f"- {step['tool']}")
    if not amount_is_sensitive and grouped is not None and "商品" not in sensitive_columns:
        lines.extend(["", "## 按商品统计", "", "| 商品 | 销售额合计 |", "| --- | ---: |"])
        for _, row in grouped.iterrows():
            lines.append(f"| {_table_cell(row['商品'])} | {row['销售额合计']:.2f} |")
    if not amount_is_sensitive and daily is not None and not daily.empty and date_column not in sensitive_columns:
        lines.extend(["", "## 按日期统计", "", "| 日期 | 销售额合计 |", "| --- | ---: |"])
        for _, row in daily.iterrows():
            date_text = row["日期"].strftime("%Y-%m-%d") if hasattr(row["日期"], "strftime") else str(row["日期"])
            lines.append(f"| {_table_cell(date_text)} | {row['销售额合计']:.2f} |")
    if not amount_is_sensitive and chart in {"bar", "line"} and not (
        chart == "bar" and "商品" in sensitive_columns
    ):
        lines.extend(["", "## 图表", "", "已选择" + {"bar": "柱状图", "line": "折线图"}[chart] + "。"])
    lines.extend(["", "## Agent 回答", "", str(trace_store.redact(answer, sensitive_values)), ""])
    return "\n".join(lines)

