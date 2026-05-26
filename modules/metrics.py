from __future__ import annotations

import re

import numpy as np
import pandas as pd

from modules.field_mapping import CANONICAL_FIELDS


METRIC_COLUMNS = ["CTR", "CPC", "CVR", "ACOS", "ROAS"]


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in [
        CANONICAL_FIELDS["impressions"],
        CANONICAL_FIELDS["clicks"],
        CANONICAL_FIELDS["spend"],
        CANONICAL_FIELDS["sales"],
        CANONICAL_FIELDS["orders"],
    ]:
        if column not in cleaned.columns:
            cleaned[column] = 0
        cleaned[column] = cleaned[column].apply(parse_currency_value if column in {CANONICAL_FIELDS["spend"], CANONICAL_FIELDS["sales"]} else parse_numeric_value)

    budget_column = CANONICAL_FIELDS.get("budget")
    if budget_column and budget_column in cleaned.columns:
        cleaned[budget_column] = cleaned[budget_column].apply(parse_currency_value)
    return cleaned


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    enriched = clean_numeric_columns(df)
    impressions = enriched[CANONICAL_FIELDS["impressions"]]
    clicks = enriched[CANONICAL_FIELDS["clicks"]]
    spend = enriched[CANONICAL_FIELDS["spend"]]
    sales = enriched[CANONICAL_FIELDS["sales"]]
    orders = enriched[CANONICAL_FIELDS["orders"]]

    enriched["CTR"] = _safe_divide(clicks, impressions)
    enriched["CPC"] = _safe_divide(spend, clicks)
    enriched["CVR"] = _safe_divide(orders, clicks)
    enriched["ACOS"] = _safe_divide(spend, sales, infinite_when_numerator=True)
    enriched["ROAS"] = _safe_divide(sales, spend)
    return enriched


def calculate_account_overview(df: pd.DataFrame) -> dict[str, float]:
    impressions = float(df[CANONICAL_FIELDS["impressions"]].sum())
    clicks = float(df[CANONICAL_FIELDS["clicks"]].sum())
    spend = float(df[CANONICAL_FIELDS["spend"]].sum())
    sales = float(df[CANONICAL_FIELDS["sales"]].sum())
    orders = float(df[CANONICAL_FIELDS["orders"]].sum())

    return {
        "总曝光": impressions,
        "总点击": clicks,
        "总花费": spend,
        "总销售额": sales,
        "总订单": orders,
        "CTR": _safe_scalar_divide(clicks, impressions),
        "CPC": _safe_scalar_divide(spend, clicks),
        "CVR": _safe_scalar_divide(orders, clicks),
        "ACOS": _safe_scalar_divide(spend, sales, infinite_when_numerator=True),
        "ROAS": _safe_scalar_divide(sales, spend),
    }


def overview_dataframe(overview: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"指标": "总曝光", "数值": overview["总曝光"], "展示": f"{overview['总曝光']:,.0f}"},
            {"指标": "总点击", "数值": overview["总点击"], "展示": f"{overview['总点击']:,.0f}"},
            {"指标": "总花费", "数值": overview["总花费"], "展示": f"{overview['总花费']:,.2f}"},
            {"指标": "总销售额", "数值": overview["总销售额"], "展示": f"{overview['总销售额']:,.2f}"},
            {"指标": "总订单", "数值": overview["总订单"], "展示": f"{overview['总订单']:,.0f}"},
            {"指标": "CTR", "数值": overview["CTR"], "展示": format_percent(overview["CTR"])},
            {"指标": "CPC", "数值": overview["CPC"], "展示": f"{overview['CPC']:,.2f}"},
            {"指标": "CVR", "数值": overview["CVR"], "展示": format_percent(overview["CVR"])},
            {"指标": "ACOS", "数值": overview["ACOS"], "展示": format_percent(overview["ACOS"])},
            {"指标": "ROAS", "数值": overview["ROAS"], "展示": f"{overview['ROAS']:,.2f}"},
        ]
    )


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "0.00%"
    if np.isinf(value):
        return "∞"
    return f"{value:.2%}"


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    infinite_when_numerator: bool = False,
) -> pd.Series:
    numerator_float = numerator.astype(float)
    denominator_float = denominator.astype(float)
    result = numerator_float.divide(denominator_float.replace(0, np.nan))
    if infinite_when_numerator:
        result = result.mask((denominator_float == 0) & (numerator_float > 0), np.inf)
        return result.replace([-np.inf], np.nan).fillna(0)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def _safe_scalar_divide(numerator: float, denominator: float, infinite_when_numerator: bool = False) -> float:
    if denominator == 0:
        if infinite_when_numerator and numerator > 0:
            return float("inf")
        return 0.0
    return numerator / denominator


def parse_currency_value(value: object) -> float:
    return parse_numeric_value(value)


def parse_percent_value(value: object) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        numeric = float(value)
        return numeric / 100 if numeric > 1 else numeric

    text = str(value).strip()
    if not text or text in ("—", "–", "-", "—", "N/A", "n/a", "NA", "暂无", "无"):
        return np.nan

    number = _parse_number_text(text)
    if pd.isna(number):
        return np.nan
    return float(number) / 100 if "%" in text or float(number) > 1 else float(number)


def parse_numeric_value(value: object) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text or text in ("—", "–", "-", "—", "N/A", "n/a", "NA", "暂无", "无"):
        return np.nan

    number = _parse_number_text(text)
    return float(number) if not pd.isna(number) else np.nan


def _parse_number_text(text: str) -> float:
    negative = text.startswith("(") and text.endswith(")")
    text = (
        text.replace("$", "")
        .replace("CA$", "")
        .replace("C$", "")
        .replace("CAD", "")
        .replace("USD", "")
        .replace("US$", "")
        .replace("¥", "")
        .replace("￥", "")
        .replace(",", "")
        .replace("%", "")
        .replace("(", "-")
        .replace(")", "")
        .replace("\xa0", "")   # non-breaking space
        .replace(" ", "")  # thin space
        .replace(" ", "")  # narrow no-break space
        .replace(" ", "")       # regular space (thousands sep in some locales)
    )
    text = re.sub(r"^[A-Za-z]+", "", text)
    text = re.sub(r"[A-Za-z]+$", "", text)
    if negative and not text.startswith("-"):
        text = f"-{text}"

    try:
        return float(text)
    except ValueError:
        return np.nan


def _to_number(value: object) -> float:
    number = parse_numeric_value(value)
    return 0.0 if pd.isna(number) else float(number)
