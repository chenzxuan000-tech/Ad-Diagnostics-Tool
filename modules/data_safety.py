from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from modules.basic_data_audit import (
    AccountSummarySource,
    PERFORMANCE_REPORT_TYPES,
    SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS,
    SP_BULK_FILE,
    UNKNOWN,
    build_file_audit,
    report_type_display,
)
from modules.field_mapping import CANONICAL_FIELDS, detect_field_mapping, normalize_column_name


STRONG_ACTIONS = {"暂停", "否定精准", "否定词组", "降低竞价"}
FEEDBACK_OPTIONS = ["", "准确", "不准确", "太激进", "太保守", "已执行", "暂不执行", "需要复核"]


@dataclass(frozen=True)
class DataTrustResult:
    data_trust_score: int
    data_trust_level: str
    data_quality_warnings: list[str]
    blocking_errors: list[str]
    date_range_status: str
    date_ranges: dict[str, str]


@dataclass(frozen=True)
class ReconciliationInput:
    external_spend: float | None = None
    external_sales: float | None = None
    external_orders: float | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    tool_spend: float
    external_spend: float | None
    spend_diff_amount: float | None
    spend_diff_percent: float | None
    tool_sales: float
    external_sales: float | None
    sales_diff_amount: float | None
    sales_diff_percent: float | None
    tool_orders: float
    external_orders: float | None
    orders_diff_amount: float | None
    orders_diff_percent: float | None
    reconciliation_status: str
    warnings: list[str]
    blocks_strong_actions: bool


@dataclass(frozen=True)
class DiagnosisSafetyGateResult:
    can_diagnose: bool
    can_generate_p0: bool
    safety_level: str
    blocking_reasons: list[str]
    warning_reasons: list[str]


def calculate_data_trust_score(
    report_frames: list[dict[str, object]],
    account_summary_source: AccountSummarySource | None,
    file_audit: pd.DataFrame,
    overview: dict[str, float],
) -> DataTrustResult:
    score = 100
    warnings: list[str] = []
    blocking: list[str] = []

    report_types = [str(report.get("report_type", UNKNOWN)) for report in report_frames]
    unknown_count = report_types.count(UNKNOWN)
    if unknown_count:
        score -= min(unknown_count * 12, 30)
        warnings.append(f"{unknown_count} 个文件未明确识别 report_type。")

    missing_core = _missing_core_field_notes(report_frames)
    if missing_core:
        score -= min(len(missing_core) * 6, 30)
        warnings.extend(missing_core)

    included_count = int(file_audit["是否参与账户总览"].eq("是").sum()) if not file_audit.empty else 0
    if account_summary_source is None:
        score -= 35
        blocking.append("未找到明确的账户总览权威数据源。")
    if included_count > 1:
        score = min(score, 30)
        blocking.append("多个报表同时参与账户总览，存在重复汇总风险。")

    parse_warnings = _metric_parse_warnings(report_frames)
    if parse_warnings:
        score -= min(len(parse_warnings) * 4, 20)
        warnings.extend(parse_warnings)

    metric_warnings, metric_blocking = _metric_reasonableness_warnings(report_frames, overview)
    if metric_warnings:
        score -= min(len(metric_warnings) * 5, 30)
        warnings.extend(metric_warnings)
    blocking.extend(metric_blocking)

    date_status, date_ranges, date_warnings = _date_range_audit(report_frames)
    if date_status != "一致":
        score -= 6 if date_status == "无法校验" else 15
        warnings.extend(date_warnings)

    if any(report_type in report_types for report_type in (SP_BULK_FILE, SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS)):
        wrongly_included = file_audit[
            file_audit["report_type"].isin([SP_BULK_FILE, SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS])
            & file_audit["是否参与账户总览"].eq("是")
        ] if not file_audit.empty else pd.DataFrame()
        if not wrongly_included.empty:
            score = min(score, 25)
            blocking.append("Bulk 或热门搜索词报告被错误纳入广告表现统计。")

    score = int(max(0, min(score, 100)))
    level = "高" if score >= 85 and not blocking else "中" if score >= 70 and not blocking else "低"
    return DataTrustResult(score, level, _unique(warnings), _unique(blocking), date_status, date_ranges)


def reconcile_external_totals(overview: dict[str, float], external: ReconciliationInput | None) -> ReconciliationResult:
    external = external or ReconciliationInput()
    tool_spend = float(overview.get("总花费", 0) or 0)
    tool_sales = float(overview.get("总销售额", 0) or 0)
    tool_orders = float(overview.get("总订单", 0) or 0)
    spend_diff = _diff(tool_spend, external.external_spend)
    sales_diff = _diff(tool_sales, external.external_sales)
    orders_diff = _diff(tool_orders, external.external_orders)
    max_percent = max(
        [value for value in [spend_diff[1], sales_diff[1], orders_diff[1]] if value is not None],
        default=0.0,
    )
    warnings: list[str] = []
    if external.external_spend is not None and spend_diff[1] is not None and spend_diff[1] > 0.03:
        warnings.append(f"工具总花费与外部系统差异 {spend_diff[1]:.2%}。")
    if external.external_sales is not None and sales_diff[1] is not None and sales_diff[1] > 0.03:
        warnings.append(f"工具总销售额与外部系统差异 {sales_diff[1]:.2%}。")
    if external.external_orders is not None and orders_diff[1] is not None and orders_diff[1] > 0.03:
        warnings.append(f"工具总订单与外部系统差异 {orders_diff[1]:.2%}。")

    if max_percent > 0.15:
        status = "阻止诊断"
    elif max_percent > 0.08:
        status = "严重警告"
    elif max_percent > 0.03:
        status = "警告"
    elif any(value is not None for value in [external.external_spend, external.external_sales, external.external_orders]):
        status = "通过"
    else:
        status = "未填写"

    return ReconciliationResult(
        tool_spend=tool_spend,
        external_spend=external.external_spend,
        spend_diff_amount=spend_diff[0],
        spend_diff_percent=spend_diff[1],
        tool_sales=tool_sales,
        external_sales=external.external_sales,
        sales_diff_amount=sales_diff[0],
        sales_diff_percent=sales_diff[1],
        tool_orders=tool_orders,
        external_orders=external.external_orders,
        orders_diff_amount=orders_diff[0],
        orders_diff_percent=orders_diff[1],
        reconciliation_status=status,
        warnings=_unique(warnings),
        blocks_strong_actions=status == "阻止诊断",
    )


def run_diagnosis_safety_gate(
    uploaded_reports: list[dict[str, object]],
    account_summary: dict[str, float],
    data_trust_result: DataTrustResult,
    account_summary_source: AccountSummarySource | None = None,
    file_audit: pd.DataFrame | None = None,
    reconciliation_result: ReconciliationResult | None = None,
) -> DiagnosisSafetyGateResult:
    blocking = list(data_trust_result.blocking_errors)
    warnings = list(data_trust_result.data_quality_warnings)

    if account_summary_source is None:
        blocking.append("账户总览数据源不明确。")

    if file_audit is not None and not file_audit.empty and file_audit["是否参与账户总览"].eq("是").sum() > 1:
        blocking.append("检测到多个报表同时参与账户总览汇总。")

    spend = float(account_summary.get("总花费", 0) or 0)
    sales = float(account_summary.get("总销售额", 0) or 0)
    acos = float(account_summary.get("ACOS", 0) or 0)
    roas = float(account_summary.get("ROAS", 0) or 0)
    if spend < 0 or sales < 0:
        blocking.append("账户总览出现负数花费或销售额。")
    if spend > 0 and sales == 0:
        warnings.append("账户有花费但总销售额为 0，强动作需要人工复核。")
    if acos > 5 or (spend > 0 and roas == 0):
        warnings.append("账户 ACOS / ROAS 极端异常，建议先核对销售额归因窗口。")

    can_diagnose = not blocking
    can_generate_p0 = can_diagnose
    if data_trust_result.data_trust_score < 70:
        can_generate_p0 = False
        warnings.append("data_trust_score 低于 70，系统关闭 P0 今日必做动作。")
    if reconciliation_result and reconciliation_result.blocks_strong_actions:
        can_generate_p0 = False
        warnings.append("外部对账差异超过 15%，系统关闭强动作和 P0。")

    if not can_diagnose:
        safety_level = "阻断"
    elif not can_generate_p0:
        safety_level = "限制强动作"
    elif warnings:
        safety_level = "可诊断需复核"
    else:
        safety_level = "通过"

    return DiagnosisSafetyGateResult(can_diagnose, can_generate_p0, safety_level, _unique(blocking), _unique(warnings))


def apply_diagnosis_safety_to_actions(
    actions: pd.DataFrame,
    safety_gate: DiagnosisSafetyGateResult,
    data_trust_result: DataTrustResult,
    reconciliation_result: ReconciliationResult,
) -> pd.DataFrame:
    actions = ensure_feedback_columns(actions.copy())
    if actions.empty:
        return actions

    actions["需要人工复核"] = actions.apply(
        lambda row: "是" if _is_high_risk_action(row, data_trust_result, reconciliation_result) else "否",
        axis=1,
    )
    actions["高风险动作原因"] = actions.apply(
        lambda row: _high_risk_action_reason(row, data_trust_result, reconciliation_result),
        axis=1,
    )

    if not safety_gate.can_generate_p0 and "execution_tier" in actions.columns:
        actions.loc[actions["execution_tier"].eq("P0"), "execution_tier"] = "P1"
        actions["is_today_action"] = False
        actions["downgrade_reason"] = actions.get("downgrade_reason", "").astype(str).apply(
            lambda value: _join_reason(value, "诊断安全阀关闭 P0 今日必做")
        )

    strong_blocked = data_trust_result.data_trust_level == "低" or reconciliation_result.blocks_strong_actions
    if strong_blocked:
        strong_mask = actions["建议动作"].isin(STRONG_ACTIONS) | actions["合并动作"].astype(str).apply(
            lambda value: any(action in value for action in STRONG_ACTIONS)
        )
        actions.loc[strong_mask, "建议动作"] = "继续观察"
        actions.loc[strong_mask, "合并动作"] = "继续观察"
        actions.loc[strong_mask, "优先级"] = "低"
        actions.loc[strong_mask, "优先级评分"] = actions.loc[strong_mask, "优先级评分"].clip(upper=40)
        if "priority_score" in actions.columns:
            actions.loc[strong_mask, "priority_score"] = actions.loc[strong_mask, "priority_score"].clip(upper=40)
        if "execution_tier" in actions.columns:
            actions.loc[strong_mask, "execution_tier"] = "P3"
            actions.loc[strong_mask, "is_today_action"] = False
        actions.loc[strong_mask, "执行建议"] = "当前数据口径需复核，暂不执行否定、暂停或降价。"
        actions.loc[strong_mask, "人工复核原因"] = actions.loc[strong_mask, "人工复核原因"].astype(str).apply(
            lambda value: _join_reason(value, "数据可信度或外部对账未通过，强动作已降级")
        )
        actions.loc[strong_mask, "需要人工复核"] = "是"

    return actions


def ensure_feedback_columns(actions: pd.DataFrame) -> pd.DataFrame:
    for column in ["operator_feedback", "feedback_reason", "reviewed_by", "reviewed_at"]:
        if column not in actions.columns:
            actions[column] = ""
    if "需要人工复核" not in actions.columns:
        actions["需要人工复核"] = "否"
    if "高风险动作原因" not in actions.columns:
        actions["高风险动作原因"] = ""
    return actions


def data_trust_dataframe(result: DataTrustResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("data_trust_score", result.data_trust_score),
            ("data_trust_level", result.data_trust_level),
            ("date_range_status", result.date_range_status),
            ("warnings", "；".join(result.data_quality_warnings) or "无"),
            ("blocking_errors", "；".join(result.blocking_errors) or "无"),
        ],
        columns=["项目", "内容"],
    )


def reconciliation_dataframe(result: ReconciliationResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "工具总花费": result.tool_spend,
                "外部总花费": result.external_spend,
                "花费差异金额": result.spend_diff_amount,
                "花费差异比例": result.spend_diff_percent,
                "工具总销售额": result.tool_sales,
                "外部总销售额": result.external_sales,
                "销售额差异金额": result.sales_diff_amount,
                "销售额差异比例": result.sales_diff_percent,
                "工具总订单": result.tool_orders,
                "外部总订单": result.external_orders,
                "订单差异金额": result.orders_diff_amount,
                "订单差异比例": result.orders_diff_percent,
                "对账状态": result.reconciliation_status,
                "提示": "；".join(result.warnings) or "无",
            }
        ]
    )


def safety_gate_dataframe(result: DiagnosisSafetyGateResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("can_diagnose", "是" if result.can_diagnose else "否"),
            ("can_generate_p0", "是" if result.can_generate_p0 else "否"),
            ("safety_level", result.safety_level),
            ("blocking_reasons", "；".join(result.blocking_reasons) or "无"),
            ("warning_reasons", "；".join(result.warning_reasons) or "无"),
        ],
        columns=["项目", "内容"],
    )


def rules_version_dataframe(engine_version: str, config_version: str, generated_at: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("诊断引擎版本", engine_version),
            ("规则配置版本", config_version),
            ("生成时间", generated_at.strftime("%Y-%m-%d %H:%M:%S")),
        ],
        columns=["项目", "内容"],
    )


def operator_feedback_dataframe(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "诊断对象",
        "建议动作",
        "operator_feedback",
        "feedback_reason",
        "reviewed_by",
        "reviewed_at",
        "Campaign Name",
        "Ad Group Name",
        "Customer Search Term",
        "Targeting",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)
    available = [column for column in columns if column in actions.columns]
    return actions[available].copy()


def write_diagnosis_audit_report(
    path: str | Path,
    file_audit: pd.DataFrame,
    overview: dict[str, float],
    source: AccountSummarySource | None,
    data_trust: DataTrustResult,
    safety_gate: DiagnosisSafetyGateResult,
    actions: pd.DataFrame,
    excel_filename: str,
    engine_version: str,
    config_version: str,
    generated_at: datetime | None = None,
) -> None:
    generated_at = generated_at or datetime.now()
    tier_counts = actions.get("execution_tier", pd.Series(dtype=str)).value_counts().to_dict() if not actions.empty else {}
    high_risk = int(actions.get("需要人工复核", pd.Series(dtype=str)).astype(str).eq("是").sum()) if not actions.empty else 0
    ordered_negative = bool(
        not actions.empty
        and ((actions.get("Orders", pd.Series(dtype=float)).fillna(0).astype(float) > 0) & actions.get("合并动作", pd.Series(dtype=str)).astype(str).str.contains("否定", na=False)).any()
    )
    low_data_strong = bool(
        not actions.empty
        and (actions.get("数据充分性", pd.Series(dtype=str)).astype(str).eq("不足") & actions.get("合并动作", pd.Series(dtype=str)).astype(str).str.contains("暂停|否定|降低竞价", na=False)).any()
    )
    source_text = f"{report_type_display(source.report_type)} | {source.filename}" if source else "未选择"

    lines = [
        "# 诊断结果审计报告",
        "",
        f"- 生成时间：{generated_at:%Y-%m-%d %H:%M:%S}",
        f"- 诊断引擎版本：{engine_version}",
        f"- 规则配置版本：{config_version}",
        f"- 账户总览数据源：{source_text}",
        f"- 总花费：{overview.get('总花费', 0):,.2f}",
        f"- 总销售额：{overview.get('总销售额', 0):,.2f}",
        f"- 总订单：{overview.get('总订单', 0):,.0f}",
        f"- 数据可信度：{data_trust.data_trust_score} / {data_trust.data_trust_level}",
        f"- 数据质量警告：{'；'.join(data_trust.data_quality_warnings) or '无'}",
        f"- 阻断错误：{'；'.join(data_trust.blocking_errors) or '无'}",
        f"- 是否通过诊断安全阀：{'是' if safety_gate.can_diagnose else '否'}",
        f"- 安全阀级别：{safety_gate.safety_level}",
        f"- 总诊断信号数：{len(actions):,}",
        f"- P0 / P1 / P2 / P3 数量：{tier_counts.get('P0', 0)} / {tier_counts.get('P1', 0)} / {tier_counts.get('P2', 0)} / {tier_counts.get('P3', 0)}",
        f"- 是否存在动作过载：{'是' if tier_counts.get('P0', 0) > 10 else '否'}",
        f"- 高风险需复核动作数：{high_risk}",
        f"- 是否存在有订单对象被否定：{'是' if ordered_negative else '否'}",
        f"- 是否存在数据不足对象被强动作处理：{'是' if low_data_strong else '否'}",
        f"- 导出 Excel 文件名和时间：{excel_filename or '尚未生成'}",
        "",
        "## 上传文件审计",
        "",
        _dataframe_to_markdown(file_audit),
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _missing_core_field_notes(report_frames: list[dict[str, object]]) -> list[str]:
    notes: list[str] = []
    required = ["impressions", "clicks", "spend", "sales", "orders", "campaign_name"]
    for report in report_frames:
        raw = report.get("raw_data", pd.DataFrame())
        if not isinstance(raw, pd.DataFrame):
            continue
        detected = detect_field_mapping(raw.columns)
        missing = [CANONICAL_FIELDS[field] for field in required if field not in detected]
        report_type = str(report.get("report_type", UNKNOWN))
        if report_type in {SP_BULK_FILE, SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS}:
            continue
        if missing:
            notes.append(f"{report.get('filename', '')} 缺少核心字段：{'、'.join(missing)}。")
    return notes


def _metric_parse_warnings(report_frames: list[dict[str, object]]) -> list[str]:
    notes: list[str] = []
    for report in report_frames:
        raw = report.get("raw_data", pd.DataFrame())
        enriched = report.get("enriched_data", pd.DataFrame())
        if not isinstance(raw, pd.DataFrame) or not isinstance(enriched, pd.DataFrame):
            continue
        for field in ["spend", "sales", "orders", "clicks", "impressions"]:
            column = CANONICAL_FIELDS[field]
            if column not in enriched.columns:
                continue
            parsed_na = int(enriched[column].isna().sum())
            if parsed_na:
                notes.append(f"{report.get('filename', '')} 的 {column} 有 {parsed_na} 个空值或无法解析值，已从汇总中忽略。")
    return notes


def _metric_reasonableness_warnings(report_frames: list[dict[str, object]], overview: dict[str, float]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    blocking: list[str] = []
    for report in report_frames:
        df = report.get("enriched_data", pd.DataFrame())
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        filename = report.get("filename", "")
        spend = df.get("Spend", pd.Series(dtype=float)).fillna(0).astype(float)
        sales = df.get("Sales", pd.Series(dtype=float)).fillna(0).astype(float)
        orders = df.get("Orders", pd.Series(dtype=float)).fillna(0).astype(float)
        clicks = df.get("Clicks", pd.Series(dtype=float)).fillna(0).astype(float)
        impressions = df.get("Impressions", pd.Series(dtype=float)).fillna(0).astype(float)

        if (spend < 0).any():
            blocking.append(f"{filename} 存在 Spend < 0。")
        if (sales < 0).any():
            blocking.append(f"{filename} 存在 Sales < 0。")

        # Trust scoring should focus on account-level impossibilities. Row-level
        # high ACOS/CVR can be a valid Amazon attribution or optimization signal,
        # not proof that the uploaded dataset is unreliable.
        if clicks.sum() > impressions.sum() and impressions.sum() > 0:
            warnings.append(f"{filename} 存在 Clicks > Impressions。")
        if orders.sum() > clicks.sum() and clicks.sum() > 0:
            warnings.append(f"{filename} 存在 Orders > Clicks。")
        if clicks.sum() > 0 and orders.sum() / clicks.sum() > 1:
            warnings.append(f"{filename} 存在 CVR > 100%。")

    if float(overview.get("ACOS", 0) or 0) > 5:
        warnings.append("账户总览 ACOS 极端异常。")
    return warnings, blocking


def _date_range_audit(report_frames: list[dict[str, object]]) -> tuple[str, dict[str, str], list[str]]:
    ranges: dict[str, str] = {}
    for report in report_frames:
        raw = report.get("raw_data", pd.DataFrame())
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            continue
        start_col, end_col = _date_columns(raw.columns)
        if not start_col and not end_col:
            continue
        dates = []
        for column in [start_col, end_col]:
            if column:
                parsed = pd.to_datetime(raw[column], errors="coerce")
                dates.extend(parsed.dropna().tolist())
        if dates:
            ranges[str(report.get("filename", ""))] = f"{min(dates).date()} 至 {max(dates).date()}"

    if not ranges:
        return "无法校验", {}, ["报表中未识别到日期字段，无法校验多个报表时间范围是否一致。"]
    if len(set(ranges.values())) > 1:
        return "不一致", ranges, ["多个报表时间范围不一致，请先确认上传文件周期一致。"]
    return "一致", ranges, []


def _date_columns(columns: Iterable[object]) -> tuple[str | None, str | None]:
    start_col = None
    end_col = None
    for column in columns:
        normalized = normalize_column_name(column)
        if normalized in {"startdate", "开始日期", "date"} and start_col is None:
            start_col = str(column)
        if normalized in {"enddate", "结束日期"} and end_col is None:
            end_col = str(column)
    return start_col, end_col


def _diff(tool_value: float, external_value: float | None) -> tuple[float | None, float | None]:
    if external_value is None:
        return None, None
    amount = tool_value - external_value
    baseline = max(abs(external_value), 1.0)
    return amount, abs(amount) / baseline


def _is_high_risk_action(row: pd.Series, data_trust: DataTrustResult, reconciliation: ReconciliationResult) -> bool:
    action = str(row.get("合并动作", row.get("建议动作", "")))
    level = str(row.get("诊断层级", ""))
    orders = float(row.get("Orders", 0) or 0)
    protected = str(row.get("是否保护词", "否")) == "是"
    if "暂停" in action and level in {"广告活动", "广告组"}:
        return True
    if "否定" in action:
        return True
    if "降低竞价" in action and orders > 0:
        return True
    if protected:
        return True
    if str(row.get("置信度", "")) == "低" and action not in {"继续观察", ""}:
        return True
    if data_trust.data_trust_level == "中" and action not in {"继续观察", ""}:
        return True
    if reconciliation.reconciliation_status in {"警告", "严重警告", "阻止诊断"} and action not in {"继续观察", ""}:
        return True
    return False


def _high_risk_action_reason(row: pd.Series, data_trust: DataTrustResult, reconciliation: ReconciliationResult) -> str:
    reasons: list[str] = []
    action = str(row.get("合并动作", row.get("建议动作", "")))
    level = str(row.get("诊断层级", ""))
    orders = float(row.get("Orders", 0) or 0)
    if "暂停" in action and level in {"广告活动", "广告组"}:
        reasons.append("暂停 Campaign / Ad Group 前必须人工复核结构、库存和目标")
    if "否定" in action:
        reasons.append("否定搜索词前必须人工判断相关性")
    if "降低竞价" in action and orders > 0:
        reasons.append("已有订单对象降价需确认利润和放量目标")
    if str(row.get("是否保护词", "否")) == "是":
        reasons.append("命中保护词")
    if str(row.get("置信度", "")) == "低" and action not in {"继续观察", ""}:
        reasons.append("低置信度动作需复核")
    if data_trust.data_trust_level == "中" and action not in {"继续观察", ""}:
        reasons.append("数据可信度中等")
    if reconciliation.reconciliation_status in {"警告", "严重警告", "阻止诊断"} and action not in {"继续观察", ""}:
        reasons.append("外部对账存在差异")
    return "；".join(reasons)


def _join_reason(existing: object, addition: str) -> str:
    parts = [part for part in str(existing or "").split("；") if part]
    if addition not in parts:
        parts.append(addition)
    return "；".join(parts)


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_无数据_"
    columns = [str(column) for column in dataframe.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in dataframe.iterrows():
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in dataframe.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
