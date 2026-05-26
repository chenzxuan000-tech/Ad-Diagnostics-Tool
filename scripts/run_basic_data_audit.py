from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.basic_data_audit import (  # noqa: E402
    account_summary_source_note,
    build_file_audit,
    duplicate_metric_guard_messages,
    run_basic_data_audit,
    select_account_summary_source,
)
from modules.data_loader import read_report  # noqa: E402
from modules.field_mapping import apply_field_mapping, infer_report_type  # noqa: E402
from modules.metrics import add_metrics, calculate_account_overview, overview_dataframe  # noqa: E402


SAMPLE_DIR = ROOT / "sample_data"
OUTPUT_DIR = ROOT / "outputs"
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}


def main() -> None:
    report_frames = load_sample_reports()
    source = select_account_summary_source(report_frames)
    overview = calculate_account_overview(source.dataframe if source else pd.DataFrame())
    file_audit = build_file_audit(report_frames, source)
    basic_audit = run_basic_data_audit(report_frames, source, overview)
    source_note = account_summary_source_note(source)
    guard_messages = duplicate_metric_guard_messages(report_frames, source)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_audit.to_csv(OUTPUT_DIR / "basic_data_audit_result.csv", index=False, encoding="utf-8-sig")
    write_source_check(source_note, overview, basic_audit, guard_messages)

    print("基础数据审计完成")
    print(f"- files: {len(report_frames)}")
    print(f"- source: {source.filename if source else '未选择'}")
    print(f"- spend: {overview.get('总花费', 0):,.2f}")
    print(f"- sales: {overview.get('总销售额', 0):,.2f}")
    print(f"- output: {OUTPUT_DIR / 'basic_data_audit_result.csv'}")
    print(f"- output: {OUTPUT_DIR / 'account_summary_source_check.md'}")


def load_sample_reports() -> list[dict[str, object]]:
    report_frames: list[dict[str, object]] = []
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        with path.open("rb") as file:
            raw = read_report(file, path.name)
        report_type = infer_report_type(raw.columns, path.name)
        source_report = f"{report_type} | {path.name}"
        cleaned = apply_field_mapping(raw, source_report)
        report_frames.append(
            {
                "filename": path.name,
                "report_type": report_type,
                "source_report": source_report,
                "raw_data": raw,
                "cleaned_data": cleaned,
                "enriched_data": add_metrics(cleaned),
            }
        )
    return report_frames


def write_source_check(
    source_note: pd.DataFrame,
    overview: dict[str, float],
    basic_audit: pd.DataFrame,
    guard_messages: list[str],
) -> None:
    lines = [
        "# Account Summary Source Check",
        "",
        "## 总览口径",
        "",
        dataframe_to_markdown(source_note),
        "",
        "## 账户总览",
        "",
        dataframe_to_markdown(overview_dataframe(overview)),
        "",
        "## 重复计算防护提示",
        "",
    ]
    if guard_messages:
        lines.extend(f"- {message}" for message in guard_messages)
    else:
        lines.append("- 未检测到需要提示的多维度重复计算风险。")
    lines.extend(["", "## 基础数据自检", "", dataframe_to_markdown(basic_audit), ""])
    (OUTPUT_DIR / "account_summary_source_check.md").write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
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


if __name__ == "__main__":
    main()
