from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
import warnings

import pandas as pd

from modules.field_mapping import infer_report_type, missing_required_fields


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
DETAIL_ENTITY_LEVELS = {"关键词", "商品定向", "受众投放"}
SEARCH_TERM_SHEET_HINTS = ("搜索词", "search term", "search terms")


def read_report(uploaded_file: BinaryIO, filename: str) -> pd.DataFrame:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式：{extension}。请上传 CSV、XLSX 或 XLS。")

    if extension == ".csv":
        return _read_csv(uploaded_file)

    return _read_excel_workbook(uploaded_file, filename)


def _read_csv(uploaded_file: BinaryIO) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "latin1"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"CSV 编码无法识别，请另存为 UTF-8 后重试。错误：{last_error}")


def _read_excel_workbook(uploaded_file: BinaryIO, filename: str) -> pd.DataFrame:
    uploaded_file.seek(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style.*")
        workbook = pd.ExcelFile(uploaded_file)
    analyzable_frames = []

    for sheet_name in workbook.sheet_names:
        if sheet_name.lower() == "config":
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Workbook contains no default style.*")
            frame = pd.read_excel(workbook, sheet_name=sheet_name)
        frame = _drop_empty_rows_and_columns(frame)
        if frame.empty:
            continue

        report_type = infer_report_type(frame.columns, f"{filename} {sheet_name}")
        missing = missing_required_fields(frame.columns, report_type)
        frame = _filter_bulk_detail_rows(frame, sheet_name)
        if frame.empty:
            continue

        if _has_required_metrics(frame) and len(missing) <= 2:
            frame["__source_sheet"] = sheet_name
            analyzable_frames.append(frame)

    if analyzable_frames:
        return pd.concat(analyzable_frames, ignore_index=True, sort=False)

    uploaded_file.seek(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style.*")
        fallback = pd.read_excel(uploaded_file)
    return _drop_empty_rows_and_columns(fallback)


def _drop_empty_rows_and_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all").dropna(axis=1, how="all")


def _has_required_metrics(df: pd.DataFrame) -> bool:
    normalized_columns = {str(column).strip() for column in df.columns}
    impressions_aliases = {"Impressions", "Impression", "展示量", "曝光量"}
    clicks_aliases = {"Clicks", "Click", "点击量", "点击"}
    spend_aliases = {"Spend", "Cost", "Costs", "Total Spend", "花费", "支出", "广告花费"}
    return (
        bool(normalized_columns & impressions_aliases)
        and bool(normalized_columns & clicks_aliases)
        and bool(normalized_columns & spend_aliases)
    )


def _filter_bulk_detail_rows(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    if _is_search_term_sheet(sheet_name, df):
        return df

    if "实体层级" not in df.columns:
        return df

    entity_values = df["实体层级"].astype(str).str.strip()
    detail = df[entity_values.isin(DETAIL_ENTITY_LEVELS)].copy()
    if detail.empty:
        return df
    return detail


def _is_search_term_sheet(sheet_name: str, df: pd.DataFrame) -> bool:
    sheet_lower = sheet_name.lower()
    if any(hint in sheet_lower for hint in SEARCH_TERM_SHEET_HINTS):
        return True
    return "顾客搜索词" in df.columns or "Customer Search Term" in df.columns
