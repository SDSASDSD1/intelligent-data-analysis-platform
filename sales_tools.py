import re

import pandas as pd
from langchain_core.tools import tool


def make_product_sales_tool(grouped: pd.DataFrame):
    totals = {
        str(product): float(amount)
        for product, amount in zip(grouped["商品"], grouped["销售额合计"])
        if pd.notna(product) and str(product).strip()
    }

    @tool
    def product_sales_total(product_name: str) -> dict:
        """按完整商品名称查询当前上传数据中的销售额合计。"""
        if not product_name.strip():
            return {"error": "商品名称不能为空。"}
        if product_name not in totals:
            return {"error": "没有找到这个商品。"}
        return {"product_name": product_name, "sales_total": totals[product_name]}

    return product_sales_total


def make_sales_summary_tool(sales: pd.Series):
    row_count = len(sales)
    sales_total = float(sales.sum())
    sales_average = float(sales.mean()) if row_count else None

    @tool
    def sales_summary() -> dict:
        """查询当前上传数据的销售记录数、销售额总和与平均值。"""
        return {
            "row_count": row_count,
            "sales_total": sales_total,
            "sales_average": sales_average,
        }

    return sales_summary


def make_daily_sales_tool(daily: pd.DataFrame):
    rows = [
        {"日期": pd.Timestamp(row["日期"]).strftime("%Y-%m-%d"),
         "销售额合计": float(row["销售额合计"])}
        for _, row in daily.iterrows()
    ]

    @tool
    def daily_sales_summary() -> dict:
        """查询已选择日期字段的每日销售额合计，不包含原始数据行。"""
        return {"daily_sales": rows}

    return daily_sales_summary


def select_tool_calls(question: str, grouped: pd.DataFrame | None) -> list[tuple[str, dict]]:
    products = []
    if grouped is not None and "商品" in grouped.columns:
        products = [
            str(product) for product in grouped["商品"]
            if pd.notna(product) and str(product).strip() and str(product).strip() in question
        ]
    if len(products) > 1:
        return []

    calls = [("sales_summary", {})]
    if products:
        calls.append(("product_sales_total", {"product_name": products[0]}))
    return calls


def validate_model_tool_calls(proposed_calls, question: str, grouped: pd.DataFrame | None):
    if not isinstance(proposed_calls, list):
        return None
    if not proposed_calls:
        return []
    if len(proposed_calls) != 1 or not isinstance(proposed_calls[0], dict):
        return None
    if not isinstance(proposed_calls[0].get("id"), str) or not proposed_calls[0]["id"].strip():
        return None

    name = proposed_calls[0].get("name")
    arguments = proposed_calls[0].get("args")
    if name == "sales_summary" and isinstance(arguments, dict) and not arguments:
        return [("sales_summary", {})]

    allowed_calls = select_tool_calls(question, grouped)
    if (
        name == "product_sales_total"
        and isinstance(arguments, dict)
        and len(arguments) == 1
        and len(allowed_calls) == 2
        and arguments == allowed_calls[1][1]
    ):
        return allowed_calls
    return None


def make_schema_tool(data: pd.DataFrame, sensitive_columns):
    columns = [str(column) for column in data.columns if column not in sensitive_columns]

    @tool
    def schema_inspect() -> dict:
        """查看当前数据的行数和非敏感字段名称，不读取字段值。"""
        return {"row_count": len(data), "columns": columns}

    return schema_inspect


def make_chart_tool(sales: pd.Series, grouped: pd.DataFrame | None, artifacts: dict):
    @tool
    def sales_chart(chart_type: str) -> dict:
        """选择销售额图表，chart_type 只能是 bar 或 line。"""
        if chart_type == "bar" and grouped is not None and not grouped.empty:
            artifacts["chart"] = "bar"
        elif chart_type == "line" and not sales.empty:
            artifacts["chart"] = "line"
        else:
            return {"error": "图表类型无效或缺少作图数据。"}
        return {"status": "ready", "chart_type": chart_type}

    return sales_chart


def make_column_preview_tool(data: pd.DataFrame, sensitive_columns, artifacts: dict):
    @tool
    def column_preview(column_name: str) -> dict:
        """在页面预览指定字段的前 10 行；字段值不会交给模型。"""
        if column_name not in data.columns or _looks_like_code(column_name):
            return {"error": "字段不存在或名称不安全。"}
        artifacts["column_preview"] = data[[column_name]].head(10).copy()
        return {"status": "ready", "column_name": column_name, "row_count": min(10, len(data))}

    return column_preview


def make_full_csv_export_tool(data: pd.DataFrame, artifacts: dict):
    @tool
    def full_csv_export() -> dict:
        """准备当前上传数据的完整 CSV 下载；内容仅保留在本地页面。"""
        artifacts["full_csv"] = data.to_csv(index=False).encode("utf-8-sig")
        return {"status": "ready", "row_count": len(data)}

    return full_csv_export


def _looks_like_code(value: str) -> bool:
    return bool(re.search(r"(?i)(?:\bimport\b|\bexec\s*\(|\beval\s*\(|os\.system\s*\(|\bsubprocess\b)", value))


def validate_agent_tool_call(name, arguments, data: pd.DataFrame, grouped: pd.DataFrame | None,
                             daily: pd.DataFrame | None = None) -> bool:
    if not isinstance(arguments, dict) or not isinstance(name, str):
        return False
    if name in {"sales_summary", "schema_inspect", "full_csv_export"}:
        return not arguments
    if name == "daily_sales_summary":
        return not arguments and daily is not None and not daily.empty
    if name == "product_sales_total" and set(arguments) == {"product_name"}:
        product = arguments["product_name"]
        return (
            isinstance(product, str) and bool(product.strip()) and not _looks_like_code(product)
            and grouped is not None and "商品" in grouped.columns
            and product in grouped["商品"].dropna().astype(str).tolist()
        )
    if name == "sales_chart" and set(arguments) == {"chart_type"}:
        kind = arguments["chart_type"]
        return (kind == "bar" and grouped is not None and not grouped.empty) or (
            kind == "line" and not data.empty
        )
    if name == "column_preview" and set(arguments) == {"column_name"}:
        column = arguments["column_name"]
        return isinstance(column, str) and column in data.columns and not _looks_like_code(column)
    return False

