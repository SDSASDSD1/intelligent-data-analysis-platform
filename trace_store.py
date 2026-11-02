"""本地 JSONL 过程记录；不保存上传数据的原始行。"""

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from secure_storage import StorageError, seal_json, unseal_json


DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"
ALLOWED_FIELDS = {
    "trace_id", "event", "query", "plan", "tool", "args", "status", "duration_ms", "decision",
    "risk_level", "risk_reason", "result_summary", "answer",
}
RESULT_FIELDS = {
    "sales_summary": {"row_count", "sales_total", "sales_average"},
    "product_sales_total": {"product_name", "sales_total"},
    "schema_inspect": {"row_count", "columns"},
    "sales_chart": {"status", "chart_type"},
    "column_preview": {"status", "column_name", "row_count"},
    "full_csv_export": {"status", "row_count"},
    "daily_sales_summary": {"day_count", "sales_total"},
}


def _safe_result_summary(summary, name=None):
    """再次检查摘要形状，避免调用方误把原始结果写入日志。"""
    if not isinstance(summary, dict):
        return {}
    allowed = set().union(*RESULT_FIELDS.values()) if name is None else RESULT_FIELDS.get(name, set())
    result = {}
    for key in allowed:
        if key not in summary:
            continue
        value = summary[key]
        if key in {"row_count", "day_count"}:
            valid = type(value) is int and value >= 0
        elif key in {"sales_total", "sales_average"}:
            valid = (key == "sales_average" and value is None) or (
                type(value) in {int, float} and math.isfinite(value)
            )
        elif key == "columns":
            valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
        elif key == "status":
            valid = value == "ready"
        elif key == "chart_type":
            valid = value in ("bar", "line")
        else:
            valid = isinstance(value, str)
        if valid:
            result[key] = value
    return result


def summarize_tool_result(name, result) -> dict:
    """只提取受控工具的统计值和执行状态，不记录预览行或 CSV 内容。"""
    if name not in RESULT_FIELDS or getattr(result, "status", None) == "error":
        return {}
    content = getattr(result, "content", result)
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return {}
    if not isinstance(content, dict) or "error" in content or content.get("status") in (
        "error", "blocked", "rejected"
    ):
        return {}
    if name == "daily_sales_summary":
        days = content.get("daily_sales")
        if not isinstance(days, list) or any(
            not isinstance(day, dict) or type(day.get("销售额合计")) not in {int, float}
            or not math.isfinite(day["销售额合计"]) for day in days
        ):
            return {}
        content = {"day_count": len(days), "sales_total": sum(day["销售额合计"] for day in days)}
    return _safe_result_summary(content, name)


def _clean(value, sensitive_values=()):
    if isinstance(value, str):
        for private_value in sensitive_values:
            value = value.replace(private_value, "[已隐藏敏感值]")
        value = re.sub(r"sk-[A-Za-z0-9._-]+", "[已隐藏密钥]", value)
        value = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[已隐藏邮箱]", value)
        value = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[已隐藏手机号]", value)
        value = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[已隐藏证件号]", value)
        return value
    if isinstance(value, dict):
        return {str(key): _clean(item, sensitive_values) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item, sensitive_values) for item in value]
    return value


def redact(value, sensitive_values=()):
    """隐藏常见私人信息和当前文件中已标记的敏感取值。"""
    replacements = sorted({str(item) for item in sensitive_values if str(item)}, key=len, reverse=True)
    return _clean(value, replacements)


def _append(filename: str, event: dict, log_dir, sensitive_values, cipher=None):
    directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        directory.chmod(0o700)
    event = dict(event)
    if "result_summary" in event:
        event["result_summary"] = {} if (
            event.get("status") in {"error", "blocked", "rejected"} or event.get("decision") == "reject"
        ) else _safe_result_summary(event["result_summary"], event.get("tool"))
    record = {
        key: value if key == "trace_id" and isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{32}", value) else redact(value, sensitive_values)
        for key, value in event.items() if key in ALLOWED_FIELDS
    }
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    if cipher is not None:
        record = seal_json(record, cipher)
    path = directory / filename
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    descriptor = os.open(path, flags, 0o600)
    if os.name == "posix":
        os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def append_trace(event: dict, log_dir=None, sensitive_values=(), cipher=None):
    filename = "trace.enc.jsonl" if cipher is not None else "trace.jsonl"
    _append(filename, event, log_dir, sensitive_values, cipher)


def append_audit(event: dict, log_dir=None, sensitive_values=(), cipher=None):
    filename = "audit.enc.jsonl" if cipher is not None else "audit.jsonl"
    _append(filename, event, log_dir, sensitive_values, cipher)


def read_trace(trace_id: str, log_dir=None, cipher=None) -> list[dict]:
    directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    path = directory / ("trace.enc.jsonl" if cipher is not None else "trace.jsonl")
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                if cipher is not None:
                    raise StorageError("加密日志格式错误，已停止读取。") from error
                continue
            if cipher is not None:
                event = unseal_json(event, cipher)
            if isinstance(event, dict) and event.get("trace_id") == trace_id:
                events.append(event)
    return events

