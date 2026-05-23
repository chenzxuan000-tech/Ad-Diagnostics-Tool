from __future__ import annotations

from io import BytesIO
from pathlib import Path
import unittest

import pandas as pd

from modules.aggregation import build_dimension_aggregations
from modules.ai_report import generate_ai_report, report_to_dataframe
from modules.data_loader import read_report
from modules.deepseek_client import DEEPSEEK_FLASH_MODEL, DEEPSEEK_MODELS, DEEPSEEK_PRO_MODEL
from modules.diagnosis import (
    DiagnosisConfig,
    build_bid_adjustments,
    build_exact_targeting_opportunities,
    build_negative_keywords,
    build_pause_list,
    build_priority_list,
    run_diagnosis,
)
from modules.exporter import build_excel_report
from modules.field_mapping import apply_field_mapping, infer_report_type
from modules.metrics import add_metrics, calculate_account_overview, overview_dataframe
from modules.pivot import build_export_pivots


ROOT = Path(__file__).resolve().parents[1]


class SmokeTests(unittest.TestCase):
    def test_sample_reports_complete_analysis_and_export(self) -> None:
        frames = []
        for path in sorted((ROOT / "sample_data").glob("*.csv")):
            with path.open("rb") as file:
                dataframe = read_report(file, path.name)
            report_type = infer_report_type(dataframe.columns, path.name)
            frames.append(apply_field_mapping(dataframe, f"{report_type} | {path.name}"))

        cleaned_data = pd.concat(frames, ignore_index=True)
        enriched_data = add_metrics(cleaned_data)
        overview = calculate_account_overview(enriched_data)
        aggregations = build_dimension_aggregations(enriched_data)
        config = DiagnosisConfig()
        actions = run_diagnosis(enriched_data, config, "完整版")

        self.assertGreaterEqual(len(cleaned_data), 14)
        self.assertGreater(overview["总花费"], 0)
        self.assertGreater(len(actions), 0)
        self.assertFalse(enriched_data[["CTR", "CPC", "CVR", "ACOS", "ROAS"]].isna().any().any())

        ai_report = report_to_dataframe(generate_ai_report(overview, actions, aggregations, config.target_acos))
        export_tables = {**build_export_pivots(actions), **aggregations}
        excel_bytes = build_excel_report(
            overview_dataframe(overview),
            ai_report,
            actions,
            build_negative_keywords(actions),
            build_pause_list(actions),
            build_bid_adjustments(actions),
            build_exact_targeting_opportunities(actions),
            enriched_data,
            build_priority_list(actions),
            export_tables,
        )

        self.assertGreater(len(excel_bytes), 10_000)

    def test_deepseek_model_names_match_api_models(self) -> None:
        self.assertEqual(DEEPSEEK_PRO_MODEL, "deepseek-v4-pro")
        self.assertEqual(DEEPSEEK_FLASH_MODEL, "deepseek-v4-flash")
        self.assertIn("deepseek-v4-pro", DEEPSEEK_MODELS)
        self.assertIn("deepseek-v4-flash", DEEPSEEK_MODELS)

    def test_csv_utf16_upload_can_be_read(self) -> None:
        csv_text = "Campaign Name,Ad Group Name,Customer Search Term,Impressions,Clicks,Spend,Sales,Orders\ncamp,group,term,10,1,0.5,5,1\n"
        buffer = BytesIO(csv_text.encode("utf-16"))

        dataframe = read_report(buffer, "search_term.csv")

        self.assertEqual(len(dataframe), 1)
        self.assertIn("Customer Search Term", dataframe.columns)


if __name__ == "__main__":
    unittest.main()
