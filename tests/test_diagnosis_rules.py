from __future__ import annotations

import unittest

import pandas as pd

from modules.diagnosis import DiagnosisConfig, build_negative_keywords, build_pause_list, run_diagnosis
from modules.metrics import add_metrics


def _run(rows: list[dict[str, object]], config: DiagnosisConfig | None = None) -> pd.DataFrame:
    prepared_rows = []
    for row in rows:
        prepared = {
            "Campaign Name": "campaign",
            "Ad Group Name": "group",
            "Customer Search Term": "",
            "Targeting": "",
            "Match Type": "Broad",
            "Impressions": 1000,
            "Clicks": 0,
            "Spend": 0.0,
            "Sales": 0.0,
            "Orders": 0,
        }
        prepared.update(row)
        prepared_rows.append(prepared)
    return run_diagnosis(add_metrics(pd.DataFrame(prepared_rows)), config or DiagnosisConfig(), "完整版")


class DiagnosisRuleTests(unittest.TestCase):
    def test_ordered_object_is_never_negative_keyword(self) -> None:
        actions = _run(
            [
                {
                    "Customer Search Term": "ordered high acos",
                    "Clicks": 25,
                    "Spend": 45,
                    "Sales": 30,
                    "Orders": 1,
                },
                {
                    "Customer Search Term": "healthy term",
                    "Clicks": 30,
                    "Spend": 10,
                    "Sales": 100,
                    "Orders": 3,
                },
            ]
        )

        self.assertFalse(actions[actions["Orders"] > 0]["合并动作"].astype(str).str.contains("否定").any())
        self.assertTrue(build_negative_keywords(actions).empty)

    def test_low_data_object_does_not_pause_or_negative(self) -> None:
        actions = _run(
            [
                {
                    "Customer Search Term": "low data term",
                    "Clicks": 3,
                    "Spend": 2.5,
                    "Sales": 0,
                    "Orders": 0,
                }
            ]
        )

        merged = " ".join(actions["合并动作"].astype(str))
        self.assertNotIn("暂停", merged)
        self.assertNotIn("否定", merged)
        self.assertIn("不足", set(actions["数据充分性"]))

    def test_high_spend_no_order_is_identified(self) -> None:
        actions = _run(
            [
                {
                    "Customer Search Term": "free replacement parts",
                    "Clicks": 30,
                    "Spend": 45,
                    "Sales": 0,
                    "Orders": 0,
                },
                {
                    "Customer Search Term": "healthy term",
                    "Clicks": 20,
                    "Spend": 10,
                    "Sales": 100,
                    "Orders": 3,
                },
            ]
        )

        bad = actions[actions["诊断对象"] == "free replacement parts"]
        self.assertFalse(bad.empty)
        self.assertTrue(bad["合并动作"].astype(str).str.contains("否定|降低竞价").any())
        self.assertTrue(bad["证据说明"].astype(str).str.len().gt(0).all())

    def test_low_acos_object_is_growth_opportunity(self) -> None:
        actions = _run(
            [
                {
                    "Customer Search Term": "best exact term",
                    "Clicks": 30,
                    "Spend": 12,
                    "Sales": 120,
                    "Orders": 4,
                },
                {
                    "Customer Search Term": "average term",
                    "Clicks": 30,
                    "Spend": 30,
                    "Sales": 100,
                    "Orders": 2,
                },
            ]
        )

        opportunity = actions[actions["诊断对象"] == "best exact term"]
        self.assertFalse(opportunity.empty)
        self.assertTrue(opportunity["合并动作"].astype(str).str.contains("提高竞价|提取精准投放").any())

    def test_protected_term_is_not_negative(self) -> None:
        actions = _run(
            [
                {
                    "Customer Search Term": "brand free parts",
                    "Clicks": 35,
                    "Spend": 50,
                    "Sales": 0,
                    "Orders": 0,
                },
                {
                    "Customer Search Term": "healthy term",
                    "Clicks": 20,
                    "Spend": 10,
                    "Sales": 100,
                    "Orders": 3,
                },
            ],
            DiagnosisConfig(protected_terms=("brand",)),
        )

        protected = actions[actions["诊断对象"] == "brand free parts"]
        self.assertFalse(protected.empty)
        self.assertFalse(protected["合并动作"].astype(str).str.contains("否定|暂停").any())
        self.assertIn("高", set(protected["操作风险"]))

    def test_professional_fields_and_score_range_exist(self) -> None:
        actions = _run(
            [
                {
                    "Customer Search Term": "free manual",
                    "Clicks": 30,
                    "Spend": 40,
                    "Sales": 0,
                    "Orders": 0,
                },
                {
                    "Customer Search Term": "healthy term",
                    "Clicks": 20,
                    "Spend": 10,
                    "Sales": 100,
                    "Orders": 3,
                },
            ]
        )

        required_columns = ["置信度", "操作风险", "数据充分性", "证据说明", "运营解释", "执行建议", "复核提醒"]
        for column in required_columns:
            self.assertIn(column, actions.columns)
            self.assertTrue(actions[column].astype(str).str.len().gt(0).all())
        self.assertTrue(actions["优先级评分"].between(0, 100).all())
        pause_list = build_pause_list(actions)
        self.assertTrue(pause_list.empty or pause_list["Orders"].le(0).all())


if __name__ == "__main__":
    unittest.main()
