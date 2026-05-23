from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.data_loader import read_report
from modules.diagnosis import DiagnosisConfig, run_diagnosis, run_diagnosis_self_check
from modules.field_mapping import apply_field_mapping, infer_report_type
from modules.metrics import add_metrics


SAMPLE_FILE = ROOT / "sample_data" / "sample_professional_diagnosis_cases.csv"
REQUIRED_COLUMNS = ["数据充分性", "置信度", "操作风险", "证据说明"]


def main() -> int:
    with SAMPLE_FILE.open("rb") as file:
        raw = read_report(file, SAMPLE_FILE.name)
    report_type = infer_report_type(raw.columns, SAMPLE_FILE.name)
    data = add_metrics(apply_field_mapping(raw, f"{report_type} | {SAMPLE_FILE.name}"))
    config = DiagnosisConfig(protected_terms=("brand",))
    actions = run_diagnosis(data, config, "完整版")
    self_check = run_diagnosis_self_check(actions, config)

    checks = [
        ("有订单对象没有被建议否定", not _ordered_negative(actions)),
        ("低数据量对象没有被建议暂停", not _low_data_pause(actions, config)),
        ("高花费无订单对象被识别", _object_action_contains(actions, "free replacement parts", ["否定", "降低竞价"])),
        ("低 ACOS 对象被识别为机会", _object_action_contains(actions, "best exact term", ["提高竞价", "提取精准投放"])),
        ("protected term 没有被否定", not _object_action_contains(actions, "brand free parts", ["否定", "暂停"])),
        ("每条建议都有专业字段", all(column in actions.columns and actions[column].astype(str).str.len().gt(0).all() for column in REQUIRED_COLUMNS)),
        ("优先级分数都在 0-100", actions["优先级评分"].between(0, 100).all()),
        ("诊断自检无高风险异常", int(self_check.get("高风险异常数量", 0)) == 0),
    ]

    failed = [name for name, ok in checks if not ok]
    print("专业诊断自检结果")
    print(f"- 样例行数: {len(data)}")
    print(f"- 动作建议: {len(actions)}")
    print(f"- 自检通过项: {self_check.get('通过项数量', 0)}")
    print(f"- 自检警告项: {self_check.get('警告项数量', 0)}")
    print(f"- 高风险异常: {self_check.get('高风险异常数量', 0)}")
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} - {name}")
    if failed:
        detail = self_check.get("异常明细")
        if isinstance(detail, pd.DataFrame) and not detail.empty:
            print("\n异常明细:")
            print(detail.to_string(index=False))
        return 1
    return 0


def _ordered_negative(actions: pd.DataFrame) -> bool:
    return bool(((actions["Orders"] > 0) & actions["合并动作"].astype(str).str.contains("否定", na=False)).any())


def _low_data_pause(actions: pd.DataFrame, config: DiagnosisConfig) -> bool:
    low_data = (actions["Clicks"] < 5) | (actions["Spend"] < config.min_waste_spend)
    return bool((low_data & actions["合并动作"].astype(str).str.contains("暂停", na=False)).any())


def _object_action_contains(actions: pd.DataFrame, object_name: str, keywords: list[str]) -> bool:
    rows = actions[actions["诊断对象"].astype(str).str.contains(object_name, na=False)]
    if rows.empty:
        return False
    action_text = " ".join(rows["合并动作"].astype(str).tolist())
    return any(keyword in action_text for keyword in keywords)


if __name__ == "__main__":
    raise SystemExit(main())
