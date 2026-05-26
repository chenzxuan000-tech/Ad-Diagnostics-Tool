from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from modules.field_mapping import CANONICAL_FIELDS, detect_field_mapping, normalize_column_name


SP_SEARCH_TERM_REPORT = "SP_SEARCH_TERM_REPORT"
SP_TARGETING_REPORT = "SP_TARGETING_REPORT"
SP_CAMPAIGN_REPORT = "SP_CAMPAIGN_REPORT"
SP_BULK_FILE = "SP_BULK_FILE"
SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS = "SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS"
UNKNOWN = "UNKNOWN"

ACCOUNT_SUMMARY_PRIORITY = [
    SP_CAMPAIGN_REPORT,
    SP_TARGETING_REPORT,
    SP_SEARCH_TERM_REPORT,
    UNKNOWN,
]

REPORT_TYPE_DISPLAY = {
    SP_SEARCH_TERM_REPORT: "商品推广搜索词报表",
    SP_TARGETING_REPORT: "商品推广投放报表",
    SP_CAMPAIGN_REPORT: "商品推广广告活动报表",
    SP_BULK_FILE: "Bulk 文件",
    SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS: "热门搜索词 / Search Query 报告",
    UNKNOWN: "未识别报表",
}

REPORT_USAGE = {
    SP_SEARCH_TERM_REPORT: "搜索词诊断、否定词建议、精准投放机会",
    SP_TARGETING_REPORT: "Targeting 诊断、关键词 / ASIN 投放诊断、调价建议",
    SP_CAMPAIGN_REPORT: "账户总览 + Campaign 层级表现和结构诊断",
    SP_BULK_FILE: "广告结构、Campaign / Ad Group / Targeting 状态辅助诊断",
    SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS: "关键词机会分析、市场搜索趋势参考",
    UNKNOWN: "字段可识别时用于基础诊断；不建议作为长期标准口径",
}

PERFORMANCE_REPORT_TYPES = {
    SP_SEARCH_TERM_REPORT,
    SP_TARGETING_REPORT,
    SP_CAMPAIGN_REPORT,
}


@dataclass(frozen=True)
class AccountSummarySource:
    filename: str
    report_type: str
    dataframe: pd.DataFrame
    source_report: str
    reason: str
    currency: str = "CAD"
    duplicate_guard_enabled: bool = True


def infer_report_type(columns: Iterable[object], filename: str = "") -> str:
    detected = detect_field_mapping(columns)
    normalized_filename = normalize_column_name(filename)
    normalized_columns = {normalize_column_name(column) for column in columns}
    has_metrics = _has_performance_metrics(detected)

    if _looks_like_bulk_file(normalized_filename, normalized_columns):
        return SP_BULK_FILE

    if _looks_like_search_query_or_top_terms(normalized_filename, normalized_columns, has_metrics):
        return SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS

    if detected.get("customer_search_term"):
        return SP_SEARCH_TERM_REPORT

    if "searchterm" in normalized_filename or "搜索词" in normalized_filename:
        return SP_SEARCH_TERM_REPORT if has_metrics else SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS

    if detected.get("targeting") and not detected.get("customer_search_term"):
        return SP_TARGETING_REPORT

    if any(token in normalized_filename for token in ("targeting", "keyword", "投放", "定向", "关键词")):
        return SP_TARGETING_REPORT if has_metrics else UNKNOWN

    if _looks_like_campaign_report(detected, normalized_filename):
        return SP_CAMPAIGN_REPORT

    return UNKNOWN


def select_account_summary_source(report_frames: list[dict[str, object]]) -> AccountSummarySource | None:
    candidates = [
        report
        for report in report_frames
        if str(report.get("report_type")) in ACCOUNT_SUMMARY_PRIORITY
        and _frame_has_metric_signal(report.get("enriched_data"))
    ]
    if not candidates:
        return None

    for report_type in ACCOUNT_SUMMARY_PRIORITY:
        for report in candidates:
            if report.get("report_type") == report_type:
                reason = _summary_source_reason(report_type)
                return AccountSummarySource(
                    filename=str(report.get("filename", "")),
                    report_type=report_type,
                    dataframe=report.get("enriched_data", pd.DataFrame()),
                    source_report=str(report.get("source_report", "")),
                    reason=reason,
                    currency=_infer_currency(report.get("raw_data")),
                )
    return None


def build_file_audit(report_frames: list[dict[str, object]], source: AccountSummarySource | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_report = source.source_report if source else ""
    for report in report_frames:
        dataframe = report.get("enriched_data", pd.DataFrame())
        report_type = str(report.get("report_type", UNKNOWN))
        participates = bool(source_report and str(report.get("source_report", "")) == source_report)
        rows.append(
            {
                "文件名": str(report.get("filename", "")),
                "report_type": report_type,
                "报表类型": report_type_display(report_type),
                "行数": int(len(dataframe)) if isinstance(dataframe, pd.DataFrame) else 0,
                "Spend 合计": _sum_metric(dataframe, "spend"),
                "Sales 合计": _sum_metric(dataframe, "sales"),
                "Orders 合计": _sum_metric(dataframe, "orders"),
                "Clicks 合计": _sum_metric(dataframe, "clicks"),
                "Impressions 合计": _sum_metric(dataframe, "impressions"),
                "是否参与账户总览": "是" if participates else "否",
                "是否只用于诊断辅助": "否" if participates else "是",
                "是否不应参与广告花费 / 销售额汇总": "否" if participates else "是",
                "用途说明": REPORT_USAGE.get(report_type, REPORT_USAGE[UNKNOWN]),
                "排除原因": "" if participates else exclusion_reason(report_type, source.report_type if source else ""),
            }
        )
    return pd.DataFrame(rows)


def duplicate_metric_guard_messages(report_frames: list[dict[str, object]], source: AccountSummarySource | None) -> list[str]:
    present = {str(report.get("report_type")) for report in report_frames}
    if {SP_SEARCH_TERM_REPORT, SP_TARGETING_REPORT, SP_CAMPAIGN_REPORT}.issubset(present):
        return [
            "检测到多个广告表现报表，它们属于不同分析维度。账户总览将优先使用 Campaign Report，其他报表仅用于维度诊断，避免重复计算花费和销售额。"
        ]
    if SP_TARGETING_REPORT in present and SP_SEARCH_TERM_REPORT in present and SP_CAMPAIGN_REPORT not in present:
        return ["账户总览将使用 Targeting Report；Search Term Report 仅用于搜索词诊断。"]
    if source and len(present & PERFORMANCE_REPORT_TYPES) > 1:
        return [
            f"检测到多个广告表现报表。账户总览当前使用 {report_type_display(source.report_type)}，其他报表仅用于对应维度诊断。"
        ]
    return []


def run_basic_data_audit(
    report_frames: list[dict[str, object]],
    source: AccountSummarySource | None,
    overview: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    file_audit = build_file_audit(report_frames, source)
    included_count = int(file_audit["是否参与账户总览"].eq("是").sum()) if not file_audit.empty else 0
    rows.append(_audit_row("多个报表同时参与总览求和", included_count <= 1, "严重错误" if included_count > 1 else "通过", f"当前参与账户总览的文件数：{included_count}"))

    rows.extend(_close_spend_checks(report_frames))

    source_spend = _sum_metric(source.dataframe, "spend") if source else 0.0
    overview_spend = float(overview.get("总花费", 0) or 0)
    rows.append(
        _audit_row(
            "总览花费是否超过权威来源 1.5 倍",
            source_spend <= 0 or overview_spend <= source_spend * 1.5,
            "高风险" if source_spend > 0 and overview_spend > source_spend * 1.5 else "通过",
            f"总览花费 {overview_spend:,.2f}；权威来源花费 {source_spend:,.2f}",
        )
    )

    bulk_included = _included_type(file_audit, SP_BULK_FILE)
    rows.append(_audit_row("Bulk 文件是否被纳入总览", not bulk_included, "严重错误" if bulk_included else "通过", "Bulk 默认只用于广告结构辅助"))

    top_terms_included = _included_type(file_audit, SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS)
    rows.append(
        _audit_row(
            "热门搜索词 / Search Query 是否被纳入广告表现统计",
            not top_terms_included,
            "严重错误" if top_terms_included else "通过",
            "该类报告不是广告表现报表，不能参与 Spend / Sales / Orders / ACOS / ROAS",
        )
    )
    return pd.DataFrame(rows)


def account_summary_source_note(source: AccountSummarySource | None) -> pd.DataFrame:
    if source is None:
        return pd.DataFrame(
            [
                ("当前账户总览数据源", "未选择"),
                ("总览口径说明", "没有找到可用于账户总览的广告表现报表。"),
                ("当前统计货币", "CAD"),
                ("重复计算防护", "是"),
            ],
            columns=["项目", "内容"],
        )

    return pd.DataFrame(
        [
            ("当前账户总览数据源", f"{report_type_display(source.report_type)} | {source.filename}"),
            ("总览口径说明", f"当前账户总览基于：{report_type_display(source.report_type)}。其他已上传报表用于搜索词、投放和机会分析，不参与总花费 / 总销售额重复汇总。"),
            ("选择原因", source.reason),
            ("当前统计货币", source.currency),
            ("重复计算防护", "是" if source.duplicate_guard_enabled else "否"),
        ],
        columns=["项目", "内容"],
    )


def report_type_display(report_type: str) -> str:
    return REPORT_TYPE_DISPLAY.get(str(report_type), str(report_type))


def exclusion_reason(report_type: str, selected_report_type: str = "") -> str:
    if report_type == SP_BULK_FILE:
        return "Bulk 包含多实体层级，直接求和会重复"
    if report_type == SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS:
        return "不是广告表现报表"
    if selected_report_type and report_type in PERFORMANCE_REPORT_TYPES:
        return f"避免与 {report_type_display(selected_report_type)} 重复计算"
    if report_type == UNKNOWN:
        return "报表类型未识别，默认不作为优先总览口径"
    return "不是当前权威账户总览来源"


def _looks_like_bulk_file(normalized_filename: str, normalized_columns: set[str]) -> bool:
    filename_hit = any(token in normalized_filename for token in ("bulk", "批量", "批量操作", "bulksheet"))
    column_hit = bool(normalized_columns & {"entity", "recordtype", "实体", "实体层级", "operation", "操作"})
    return filename_hit or (column_hit and bool(normalized_columns & {"campaignid", "adgroupid", "广告活动编号", "广告组编号"}))


def _looks_like_search_query_or_top_terms(normalized_filename: str, normalized_columns: set[str], has_metrics: bool) -> bool:
    filename_hit = any(
        token in normalized_filename
        for token in ("searchqueryperformance", "topsearchterms", "searchquery", "热门搜索", "搜索查询表现", "搜索查询")
    )
    column_hit = bool(normalized_columns & {"searchquery", "搜索查询", "searchfrequencyrank", "搜索频率排名", "clickshare", "点击份额", "conversionshare", "转化份额"})
    return (filename_hit or column_hit) and not has_metrics


def _looks_like_campaign_report(detected: dict[str, str], normalized_filename: str) -> bool:
    filename_hit = any(token in normalized_filename for token in ("campaign", "广告活动"))
    has_campaign = bool(detected.get("campaign_name"))
    has_detail_dimension = bool(detected.get("customer_search_term") or detected.get("targeting") or detected.get("ad_group_name"))
    return (filename_hit or has_campaign) and _has_performance_metrics(detected) and not has_detail_dimension


def _has_performance_metrics(detected: dict[str, str]) -> bool:
    return bool(detected.get("spend") and (detected.get("sales") or detected.get("orders") or detected.get("clicks")))


def _frame_has_metric_signal(value: object) -> bool:
    if not isinstance(value, pd.DataFrame) or value.empty:
        return False
    return any(_sum_metric(value, key) > 0 for key in ("spend", "sales", "orders", "clicks", "impressions"))


def _summary_source_reason(report_type: str) -> str:
    if report_type == SP_CAMPAIGN_REPORT:
        return "Campaign Report 是账户 / Campaign 层级汇总，最适合作为账户总览权威来源。"
    if report_type == SP_TARGETING_REPORT:
        return "未上传 Campaign Report，使用 Targeting Report 作为账户总览来源。"
    if report_type == SP_SEARCH_TERM_REPORT:
        return "未上传 Campaign / Targeting Report，使用 Search Term Report 作为账户总览来源。"
    return "未识别到标准广告表现报表，临时使用可识别指标的报表作为总览来源。"


def _sum_metric(dataframe: object, field_key: str) -> float:
    if not isinstance(dataframe, pd.DataFrame):
        return 0.0
    column = CANONICAL_FIELDS[field_key]
    if column not in dataframe.columns:
        return 0.0
    return float(pd.to_numeric(dataframe[column], errors="coerce").sum(skipna=True))


def _included_type(file_audit: pd.DataFrame, report_type: str) -> bool:
    if file_audit.empty:
        return False
    return bool(file_audit["report_type"].eq(report_type).any() and file_audit.loc[file_audit["report_type"].eq(report_type), "是否参与账户总览"].eq("是").any())


def _close_spend_checks(report_frames: list[dict[str, object]]) -> list[dict[str, object]]:
    spend_by_type = {
        str(report.get("report_type")): _sum_metric(report.get("enriched_data"), "spend")
        for report in report_frames
        if str(report.get("report_type")) in PERFORMANCE_REPORT_TYPES
    }
    checks: list[dict[str, object]] = []
    pairs = [
        (SP_CAMPAIGN_REPORT, SP_TARGETING_REPORT),
        (SP_CAMPAIGN_REPORT, SP_SEARCH_TERM_REPORT),
        (SP_TARGETING_REPORT, SP_SEARCH_TERM_REPORT),
    ]
    for left, right in pairs:
        if left not in spend_by_type or right not in spend_by_type:
            continue
        left_spend = spend_by_type[left]
        right_spend = spend_by_type[right]
        close = _values_are_close(left_spend, right_spend)
        checks.append(
            _audit_row(
                f"{report_type_display(left)} 与 {report_type_display(right)} 花费是否接近",
                True,
                "提示" if close else "通过",
                (
                    f"两者花费接近（{left_spend:,.2f} vs {right_spend:,.2f}），通常说明它们是同一批广告数据的不同维度，不能累加。"
                    if close
                    else f"两者花费差异较大（{left_spend:,.2f} vs {right_spend:,.2f}）。"
                ),
            )
        )
    if not checks:
        checks.append(_audit_row("广告表现报表花费相近性检查", True, "通过", "当前没有足够的 Campaign / Targeting / Search Term 组合可比较。"))
    return checks


def _values_are_close(left: float, right: float) -> bool:
    baseline = max(abs(left), abs(right), 1.0)
    return abs(left - right) / baseline <= 0.08


def _audit_row(item: str, passed: bool, level: str, detail: str) -> dict[str, object]:
    return {"检查项": item, "是否通过": "是" if passed else "否", "风险级别": level, "说明": detail}


def _infer_currency(raw_data: object) -> str:
    if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
        return "CAD"
    for column in raw_data.columns:
        if normalize_column_name(column) in {"currency", "货币"}:
            values = raw_data[column].dropna().astype(str).str.strip()
            values = values[values.ne("")]
            if not values.empty:
                return values.iloc[0]
    return "CAD"
