from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


CANONICAL_FIELDS = {
    "campaign_name": "Campaign Name",
    "ad_group_name": "Ad Group Name",
    "customer_search_term": "Customer Search Term",
    "targeting": "Targeting",
    "match_type": "Match Type",
    "impressions": "Impressions",
    "clicks": "Clicks",
    "spend": "Spend",
    "sales": "Sales",
    "orders": "Orders",
    "ad_product": "Ad Product",
    "advertised_asin": "Advertised ASIN",
    "purchased_asin": "Purchased ASIN",
    "budget": "Budget",
    "campaign_status": "Campaign Status",
    "ad_group_status": "Ad Group Status",
    "source_report": "Source Report",
}


OPTIONAL_FIELDS = {
    "advertised_asin",
    "purchased_asin",
    "ad_product",
    "budget",
    "campaign_status",
    "ad_group_status",
}


COMMON_REQUIRED_FIELDS = {
    "impressions",
    "clicks",
    "spend",
    "sales",
    "orders",
}


REPORT_REQUIRED_FIELDS = {
    "SP_SEARCH_TERM_REPORT": COMMON_REQUIRED_FIELDS | {"customer_search_term"},
    "SP_TARGETING_REPORT": COMMON_REQUIRED_FIELDS | {"targeting"},
    "SP_CAMPAIGN_REPORT": COMMON_REQUIRED_FIELDS | {"campaign_name"},
    "SP_BULK_FILE": set(),
    "SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS": set(),
    "UNKNOWN": COMMON_REQUIRED_FIELDS,
    # Backward-compatible labels used by older tests and saved workbooks.
    "Search Term Report": COMMON_REQUIRED_FIELDS | {"customer_search_term"},
    "Targeting Report": COMMON_REQUIRED_FIELDS | {"targeting"},
    "Unknown Report": COMMON_REQUIRED_FIELDS,
}


FIELD_ALIASES = {
    "campaign_name": [
        "Campaign Name",
        "Campaign",
        "Campaigns",
        "Campaign Name (Informational only)",
        "广告活动名称",
        "广告活动名称（仅供参考）",
        "广告活动",
    ],
    "ad_group_name": [
        "Ad Group Name",
        "Ad group",
        "Ad Group",
        "Ad Groups",
        "Ad Group Name (Informational only)",
        "广告组名称",
        "广告组名称（仅供参考）",
        "广告组",
    ],
    "customer_search_term": [
        "Customer Search Term",
        "Customer Search Term / Search Term",
        "Search Term",
        "Search term",
        "Customer Query",
        "搜索词",
        "客户搜索词",
        "顾客搜索词",
    ],
    "targeting": [
        "Targeting",
        "Target",
        "Keyword",
        "Keyword or Product Targeting",
        "Product Targeting",
        "Product Targeting Expression",
        "Keyword Text",
        "Keyword text",
        "投放",
        "投放表达式",
        "已解析的投放表达式（仅供参考）",
        "拓展商品投放名称（仅供参考）",
        "拓展商品投放编号",
        "关键词文本",
        "关键词",
        "目标",
    ],
    "match_type": [
        "Match Type",
        "Match type",
        "Match",
        "匹配类型",
    ],
    "impressions": [
        "Impressions",
        "Impression",
        "Impr.",
        "展示量",
        "曝光量",
        "曝光",
    ],
    "clicks": [
        "Clicks",
        "Click",
        "点击量",
        "点击",
    ],
    "spend": [
        "Spend",
        "Cost",
        "Costs",
        "Total Spend",
        "花费",
        "支出",
        "广告花费",
    ],
    "sales": [
        "Sales",
        "Total Sales",
        "7 Day Total Sales",
        "7 Day Total Sales ",
        "14 Day Total Sales",
        "14 Day Total Sales ",
        "Total Advertising Sales",
        "销售额",
        "总销售额",
        "广告销售额",
        "7 天总销售额",
        "7 天总销售额 ",
        "14 天总销售额",
        "14 天总销售额 ",
    ],
    "orders": [
        "Orders",
        "Total Orders",
        "7 Day Total Orders (#)",
        "7 Day Total Orders",
        "14 Day Total Orders (#)",
        "14 Day Total Orders",
        "Purchases",
        "订单",
        "订单量",
        "订单数量",
        "订单量（浏览次数和点击量）",
        "购买量",
        "销量",
        "销量（浏览次数和点击量）",
        "7 天总订单量",
        "7 天总订单数",
        "14 天总订单量",
        "14 天总订单数",
        "7 天总订单量 ",
        "14 天总订单量 ",
        "7 天总销售量",
        "14 天总销售量",
        "7 天总销售量 ",
        "14 天总销售量 ",
    ],
    "ad_product": [
        "Ad Product",
        "Advertising Product",
        "Product",
        "Campaign Type",
        "Ad Type",
        "广告产品",
        "广告产品类型",
        "产品",
        "广告类型",
        "推广类型",
        "Sponsored Products",
        "Sponsored Brands",
        "Sponsored Display",
    ],
    "advertised_asin": [
        "Advertised ASIN",
        "Advertised Asin",
        "Ad ASIN",
        "Advertised Product ASIN",
        "ASIN",
        "ASIN (Informational only)",
        "广告 ASIN",
        "ASIN（仅供参考）",
        "落地页 ASIN",
        "创意素材 ASIN",
        "广告商品 ASIN",
        "推广 ASIN",
    ],
    "purchased_asin": [
        "Purchased ASIN",
        "Purchased Asin",
        "Purchased Product ASIN",
        "Purchased Product",
        "Purchased SKU ASIN",
        "购买 ASIN",
        "成交 ASIN",
    ],
    "budget": [
        "Budget",
        "Campaign Budget",
        "Daily Budget",
        "预算",
        "每日预算",
        "广告活动预算",
        "每日预算",
    ],
    "campaign_status": [
        "Campaign Status",
        "Campaign State",
        "Campaign Serving Status",
        "广告活动状态",
        "广告活动状态（仅供参考）",
        "广告活动开展状态（仅供参考）",
    ],
    "ad_group_status": [
        "Ad Group Status",
        "Ad Group State",
        "广告组状态",
        "广告组状态（仅供参考）",
        "广告组投放状态（仅供参考）",
    ],
}


NUMERIC_FIELDS = {"impressions", "clicks", "spend", "sales", "orders", "budget"}


@dataclass(frozen=True)
class MappingResult:
    report_name: str
    field_key: str
    expected_field: str
    matched_column: str | None
    status: str


def normalize_column_name(name: object) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\ufeff", "")
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
    return text


def _build_alias_lookup() -> dict[str, list[str]]:
    return {
        key: [normalize_column_name(alias) for alias in aliases]
        for key, aliases in FIELD_ALIASES.items()
    }


ALIAS_LOOKUP = _build_alias_lookup()


def detect_field_mapping(
    columns: Iterable[object],
    manual_mapping: dict[str, str] | None = None,
) -> dict[str, str]:
    normalized_columns = _normalized_columns(columns)

    mapping: dict[str, str] = {}
    for field_key, aliases in ALIAS_LOOKUP.items():
        for alias in aliases:
            if alias in normalized_columns:
                mapping[field_key] = normalized_columns[alias][0]
                break

    mapping.update(_valid_manual_mapping(columns, manual_mapping))
    return mapping


def detect_field_candidates(columns: Iterable[object]) -> dict[str, list[str]]:
    normalized_columns = _normalized_columns(columns)
    column_keys = list(normalized_columns.keys())
    candidates: dict[str, list[str]] = {}

    for field_key, aliases in ALIAS_LOOKUP.items():
        for alias in aliases:
            if alias not in normalized_columns:
                continue
            candidates.setdefault(field_key, [])
            for column in normalized_columns[alias]:
                if column not in candidates[field_key]:
                    candidates[field_key].append(column)

    # Fallback: partial (substring) matching for fields with zero exact matches
    for field_key, aliases in ALIAS_LOOKUP.items():
        if field_key in candidates:
            continue
        for alias in aliases:
            for column_key in column_keys:
                if alias in column_key or column_key in alias:
                    candidates.setdefault(field_key, [])
                    for column in normalized_columns[column_key]:
                        if column not in candidates[field_key]:
                            candidates[field_key].append(column)
                    break
            if field_key in candidates:
                break

    return candidates


def mapping_results(
    report_name: str,
    columns: Iterable[object],
    manual_mapping: dict[str, str] | None = None,
) -> list[MappingResult]:
    detected = detect_field_mapping(columns, manual_mapping)
    manual = _valid_manual_mapping(columns, manual_mapping)
    results = []
    for field_key, label in CANONICAL_FIELDS.items():
        if field_key == "source_report":
            continue
        matched = detected.get(field_key)
        results.append(
            MappingResult(
                report_name=report_name,
                field_key=field_key,
                expected_field=label,
                matched_column=matched,
                status=_field_status(field_key, matched, field_key in manual),
            )
        )
    return results


def infer_report_type(columns: Iterable[object], filename: str = "") -> str:
    from modules.basic_data_audit import infer_report_type as infer_standard_report_type

    return infer_standard_report_type(columns, filename)


def missing_required_fields(
    columns: Iterable[object],
    report_type: str,
    manual_mapping: dict[str, str] | None = None,
) -> list[str]:
    detected = detect_field_mapping(columns, manual_mapping)
    required_fields = REPORT_REQUIRED_FIELDS.get(report_type, REPORT_REQUIRED_FIELDS["UNKNOWN"])
    missing = [
        CANONICAL_FIELDS[field_key]
        for field_key in sorted(required_fields)
        if field_key not in detected
    ]
    return missing


def mapping_results_dataframe(results: list[MappingResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Report": item.report_name,
                "标准字段": item.expected_field,
                "识别到的列名": item.matched_column or "",
                "状态": item.status,
            }
            for item in results
        ]
    )


def apply_field_mapping(
    df: pd.DataFrame,
    report_name: str,
    manual_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    detected = detect_field_candidates(df.columns)
    manual = _valid_manual_mapping(df.columns, manual_mapping)
    cleaned = pd.DataFrame(index=df.index)

    for field_key, label in CANONICAL_FIELDS.items():
        if field_key == "source_report":
            continue
        source_columns = [manual[field_key]] if field_key in manual else detected.get(field_key, [])
        if source_columns:
            cleaned[label] = _coalesce_columns(df, source_columns)
        else:
            cleaned[label] = 0 if field_key in NUMERIC_FIELDS else ""

    cleaned[CANONICAL_FIELDS["source_report"]] = report_name
    return cleaned


def _normalized_columns(columns: Iterable[object]) -> dict[str, list[str]]:
    normalized_columns: dict[str, list[str]] = {}
    for column in columns:
        normalized = normalize_column_name(column)
        normalized_columns.setdefault(normalized, []).append(str(column).strip())
    return normalized_columns


def _coalesce_columns(df: pd.DataFrame, source_columns: list[str]) -> pd.Series:
    result = _series_for_column(df, source_columns[0]).astype("object")
    result = result.replace("", pd.NA)
    for source_column in source_columns[1:]:
        next_series = _series_for_column(df, source_column).astype("object").replace("", pd.NA)
        result = result.combine_first(next_series)
    result = result.copy()
    result[pd.isna(result)] = ""
    return result


def _valid_manual_mapping(
    columns: Iterable[object],
    manual_mapping: dict[str, str] | None,
) -> dict[str, str]:
    if not manual_mapping:
        return {}

    available = {str(column).strip(): str(column).strip() for column in columns}
    valid = {}
    for field_key, column in manual_mapping.items():
        column_name = str(column or "").strip()
        if field_key in CANONICAL_FIELDS and column_name in available:
            valid[field_key] = available[column_name]
    return valid


def _series_for_column(df: pd.DataFrame, column: str) -> pd.Series:
    values = df[column]
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0]
    return values


def _field_status(field_key: str, matched_column: str | None, manual: bool = False) -> str:
    if matched_column:
        return "手动指定" if manual else "识别成功"
    if field_key in OPTIONAL_FIELDS:
        return "可选字段未识别"
    return "未识别"
