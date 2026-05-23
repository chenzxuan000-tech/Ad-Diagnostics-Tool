from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.action_overload import (
    build_action_overload_summary,
    enrich_action_prioritization,
    write_action_audit,
    write_overload_summary,
)
from modules.data_loader import read_report
from modules.diagnosis import DiagnosisConfig, run_diagnosis
from modules.field_mapping import apply_field_mapping, infer_report_type
from modules.metrics import add_metrics


REAL_CASE = ROOT / "sample_data" / "real_case_ads_report.xlsx"
OUTPUT_DIR = ROOT / "outputs"


def main() -> int:
    config = DiagnosisConfig()
    data = _load_real_case()
    actions_before = run_diagnosis(data, config, "完整版")
    object_count = _analysis_object_count(data)

    write_action_audit(actions_before, OUTPUT_DIR / "real_case_action_audit_before.csv")
    before_summary = build_action_overload_summary(actions_before, config, object_count)
    write_overload_summary(before_summary, OUTPUT_DIR / "real_case_action_overload_summary.md")

    before_stats = {
        "修改前总建议动作数": len(actions_before),
        "修改前高优先级动作数": int(actions_before["优先级"].eq("高").sum()) if not actions_before.empty else 0,
    }
    actions_after = enrich_action_prioritization(actions_before, config)
    write_action_audit(actions_after, OUTPUT_DIR / "real_case_action_audit_after.csv")
    after_summary = build_action_overload_summary(actions_after, config, object_count, before_stats=before_stats)
    write_overload_summary(after_summary, OUTPUT_DIR / "real_case_action_overload_summary_after.md")

    print("真实报表动作审计完成")
    print(f"- 分析对象数: {object_count}")
    print(f"- before actions: {len(actions_before)}")
    print(f"- before high priority: {before_stats['修改前高优先级动作数']}")
    print(f"- after P0: {after_summary['P0 数量']}")
    print(f"- after P1: {after_summary['P1 数量']}")
    print(f"- after P2: {after_summary['P2 数量']}")
    print(f"- after P3: {after_summary['P3 数量']}")
    print(f"- overload after: {'是' if after_summary['是否存在动作过载'] else '否'}")
    return 0


def _load_real_case() -> pd.DataFrame:
    with REAL_CASE.open("rb") as file:
        raw = read_report(file, REAL_CASE.name)
    report_type = infer_report_type(raw.columns, REAL_CASE.name)
    return add_metrics(apply_field_mapping(raw, f"{report_type} | {REAL_CASE.name}"))


def _analysis_object_count(data: pd.DataFrame) -> int:
    object_columns = [
        column
        for column in ["Customer Search Term", "Targeting", "Campaign Name", "Ad Group Name"]
        if column in data.columns
    ]
    if not object_columns:
        return len(data)
    counts = []
    for column in object_columns:
        counts.append(int(data[column].fillna("").astype(str).str.strip().ne("").sum()))
    return max(counts) if counts else len(data)


if __name__ == "__main__":
    raise SystemExit(main())
