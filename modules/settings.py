from __future__ import annotations

from dataclasses import dataclass

from modules.diagnosis import DiagnosisConfig
from modules.data_safety import ReconciliationInput


@dataclass(frozen=True)
class AppSettings:
    mode: str
    rule_preset: str
    manual_mapping_enabled: bool
    ai_report_enabled: bool
    diagnosis_config: DiagnosisConfig
    reconciliation_input: ReconciliationInput
