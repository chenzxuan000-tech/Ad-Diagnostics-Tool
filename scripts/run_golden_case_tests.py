from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.action_overload import enrich_action_prioritization  # noqa: E402
from modules.aggregation import build_dimension_aggregations  # noqa: E402
from modules.basic_data_audit import build_file_audit, select_account_summary_source  # noqa: E402
from modules.data_loader import read_report  # noqa: E402
from modules.data_safety import (  # noqa: E402
    ReconciliationInput,
    apply_diagnosis_safety_to_actions,
    calculate_data_trust_score,
    data_trust_dataframe,
    operator_feedback_dataframe,
    reconcile_external_totals,
    reconciliation_dataframe,
    rules_version_dataframe,
    run_diagnosis_safety_gate,
    safety_gate_dataframe,
)
from modules.diagnosis import (  # noqa: E402
    DiagnosisConfig,
    build_bid_adjustments,
    build_exact_targeting_opportunities,
    build_negative_keywords,
    build_pause_list,
    build_priority_list,
    run_diagnosis,
)
from modules.exporter import build_excel_report  # noqa: E402
from modules.field_mapping import apply_field_mapping, infer_report_type  # noqa: E402
from modules.metrics import add_metrics, calculate_account_overview, overview_dataframe  # noqa: E402
from modules.pivot import build_export_pivots  # noqa: E402
from modules.rules_config import DIAGNOSIS_ENGINE_VERSION, RULE_CONFIG_VERSION  # noqa: E402


CONFIG_PATH = ROOT / "tests" / "golden_cases" / "real_case_expected.json"
OUTPUT_DIR = ROOT / "outputs"
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}


def main() -> None:
    expected = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    input_dir = ROOT / expected.get("input_dir", "sample_data")
    report_frames = load_reports(input_dir)
    source = select_account_summary_source(report_frames)
    overview = calculate_account_overview(source.dataframe if source else pd.DataFrame())
    file_audit = build_file_audit(report_frames, source)
    data_trust = calculate_data_trust_score(report_frames, source, file_audit, overview)
    reconciliation = reconcile_external_totals(overview, ReconciliationInput())
    safety_gate = run_diagnosis_safety_gate(
        report_frames,
        overview,
        data_trust,
        account_summary_source=source,
        file_audit=file_audit,
        reconciliation_result=reconciliation,
    )

    enriched_data = pd.concat([frame["enriched_data"] for frame in report_frames], ignore_index=True) if report_frames else pd.DataFrame()
    config = DiagnosisConfig()
    actions = run_diagnosis(enriched_data, config, "完整版") if safety_gate.can_diagnose else pd.DataFrame()
    actions = enrich_action_prioritization(actions, config) if not actions.empty else actions
    actions = apply_diagnosis_safety_to_actions(actions, safety_gate, data_trust, reconciliation)

    failures = run_assertions(expected, source, overview, file_audit, data_trust, actions)
    excel_bytes = build_excel_report(
        overview_dataframe(overview),
        pd.DataFrame(),
        actions,
        build_negative_keywords(actions),
        build_pause_list(actions),
        build_bid_adjustments(actions),
        build_exact_targeting_opportunities(actions),
        enriched_data,
        build_priority_list(actions),
        {**build_export_pivots(actions), **build_dimension_aggregations(enriched_data)},
        file_audit=file_audit,
        data_trust=data_trust_dataframe(data_trust),
        reconciliation=reconciliation_dataframe(reconciliation),
        safety_gate=safety_gate_dataframe(safety_gate),
        operator_feedback=operator_feedback_dataframe(actions),
        rules_version=rules_version_dataframe(DIAGNOSIS_ENGINE_VERSION, RULE_CONFIG_VERSION, datetime.now()),
    )
    if len(excel_bytes) < 10_000:
        failures.append("Excel 导出文件过小，可能未正常生成。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "golden_case_test_result.md"
    result_path.write_text(build_result_markdown(expected, source, overview, data_trust, safety_gate, actions, failures), encoding="utf-8")

    if failures:
        print("Golden case 测试失败：")
        for failure in failures:
            print(f"- {failure}")
        print(f"结果文件：{result_path}")
        raise SystemExit(1)

    print("Golden case 测试通过")
    print(f"- source: {source.report_type if source else '未选择'}")
    print(f"- spend: {overview.get('总花费', 0):,.2f}")
    print(f"- sales: {overview.get('总销售额', 0):,.2f}")
    print(f"- trust: {data_trust.data_trust_score} / {data_trust.data_trust_level}")
    print(f"- output: {result_path}")


def load_reports(input_dir: Path) -> list[dict[str, object]]:
    report_frames: list[dict[str, object]] = []
    for path in sorted(input_dir.iterdir()):
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


def run_assertions(
    expected: dict[str, object],
    source,
    overview: dict[str, float],
    file_audit: pd.DataFrame,
    data_trust,
    actions: pd.DataFrame,
) -> list[str]:
    failures: list[str] = []
    spend = float(overview.get("总花费", 0) or 0)
    sales = float(overview.get("总销售额", 0) or 0)
    spend_min, spend_max = expected["expected_spend_range"]
    sales_min, sales_max = expected["expected_sales_range"]
    if not (spend_min <= spend <= spend_max):
        failures.append(f"总花费 {spend:,.2f} 不在预期范围 {spend_min:,.2f}-{spend_max:,.2f}。")
    if not (sales_min <= sales <= sales_max):
        failures.append(f"总销售额 {sales:,.2f} 不在预期范围 {sales_min:,.2f}-{sales_max:,.2f}。")
    if source is None:
        failures.append("未选择账户总览权威来源。")
    elif source.report_type != expected["expected_summary_source"]:
        failures.append(f"账户总览来源 {source.report_type} 不等于预期 {expected['expected_summary_source']}。")
    if not file_audit.empty and file_audit["是否参与账户总览"].eq("是").sum() != 1:
        failures.append("参与账户总览的文件数量不是 1。")
    for forbidden in expected.get("forbidden_summary_sources", []):
        mask = file_audit["report_type"].eq(forbidden) & file_audit["是否参与账户总览"].eq("是") if not file_audit.empty else pd.Series(dtype=bool)
        if bool(mask.any()):
            failures.append(f"{forbidden} 被错误纳入账户总览。")
    if not actions.empty and "execution_tier" in actions.columns:
        p0_count = int(actions["execution_tier"].eq("P0").sum())
        if p0_count > int(expected.get("max_p0_actions", 10)):
            failures.append(f"P0 数量 {p0_count} 超过上限。")
    negative = build_negative_keywords(actions)
    if not negative.empty:
        ordered_objects = actions[actions.get("Orders", pd.Series(dtype=float)).fillna(0).astype(float) > 0]
        if not ordered_objects.empty and ordered_objects.get("合并动作", pd.Series(dtype=str)).astype(str).str.contains("否定", na=False).any():
            failures.append("有订单对象进入了否定词清单或否定动作。")
    if data_trust.data_trust_score is None:
        failures.append("未输出数据可信度评分。")
    return failures


def build_result_markdown(expected, source, overview, data_trust, safety_gate, actions, failures: list[str]) -> str:
    return "\n".join(
        [
            "# Golden Case Test Result",
            "",
            f"- case_name: {expected.get('case_name')}",
            f"- status: {'FAILED' if failures else 'PASSED'}",
            f"- source: {source.report_type if source else '未选择'}",
            f"- spend: {overview.get('总花费', 0):,.2f}",
            f"- sales: {overview.get('总销售额', 0):,.2f}",
            f"- data_trust: {data_trust.data_trust_score} / {data_trust.data_trust_level}",
            f"- safety_gate: {safety_gate.safety_level}",
            f"- actions: {len(actions):,}",
            "",
            "## Failures",
            "",
            "\n".join(f"- {failure}" for failure in failures) if failures else "- 无",
            "",
        ]
    )


if __name__ == "__main__":
    main()
