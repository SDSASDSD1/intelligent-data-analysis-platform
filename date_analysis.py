"""按自然日汇总已校验的销售额。"""

import pandas as pd


def build_daily_sales(data: pd.DataFrame, sales: pd.Series, date_column: str) -> pd.DataFrame:
    """将全部行按日期合计；不忽略无效或空白日期。"""
    if date_column not in data.columns or pd.api.types.is_numeric_dtype(data[date_column]):
        raise ValueError("日期字段不存在或不是日期格式。")
    dates = pd.to_datetime(data[date_column], errors="coerce", format="mixed")
    if dates.isna().any():
        raise ValueError("日期字段包含空白或无法识别的日期。")
    return (
        pd.DataFrame({"日期": dates.dt.normalize(), "销售额合计": sales})
        .groupby("日期", as_index=False, sort=True)["销售额合计"]
        .sum()
    )

