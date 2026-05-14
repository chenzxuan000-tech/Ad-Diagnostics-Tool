from __future__ import annotations

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
        cleaned[column] = cleaned[column].apply(_to_number)

    budget_column = CANONICAL_FIELDS.get("budget")
    if budget_column and budget_column in cleaned.columns:
        cleaned[budget_column] = cleaned[budget_column].apply(_to_number)
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
    enriched["ACOS"] = _safe_divide(spend, sales)
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
        "ACOS": _safe_scalar_divide(spend, sales),
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
    if pd.isna(value) or np.isinf(value):
        return "0.00%"
    return f"{value:.2%}"


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float).divide(denominator.astype(float).replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def _safe_scalar_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _to_number(value: object) -> float:
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text or text in ("—", "–", "-", "—", "N/A", "n/a", "NA", "暂无", "无"):
        return 0.0

    text = (
        text.replace("$", "")
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

    try:
        return float(text)
    except ValueError:
        return 0.0
