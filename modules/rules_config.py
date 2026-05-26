from __future__ import annotations

from dataclasses import dataclass


DEFAULT_TARGET_ACOS = 0.30
DEFAULT_PROTECTED_TERMS: tuple[str, ...] = ()
STRICTNESS_OPTIONS = ("保守", "标准", "激进")
DEFAULT_DIAGNOSIS_STRICTNESS = "标准"
DIAGNOSIS_ENGINE_VERSION = "v2.1-data-safety"
RULE_CONFIG_VERSION = "2026-05-data-safety"


@dataclass(frozen=True)
class RuleThresholds:
    min_clicks_for_negative: int = 15
    min_spend_for_negative: float = 5.0
    min_clicks_for_pause: int = 30
    min_orders_for_scale: int = 2
    low_acos_multiplier: float = 0.70
    high_acos_multiplier: float = 1.25
    min_waste_clicks: int = 8
    hard_waste_clicks: int = 15
    min_waste_spend: float = 5.0
    pause_spend_multiplier: float = 1.50


RULE_THRESHOLDS_BY_STRICTNESS = {
    "保守": RuleThresholds(
        min_clicks_for_negative=20,
        min_spend_for_negative=8.0,
        min_clicks_for_pause=40,
        min_orders_for_scale=3,
        low_acos_multiplier=0.65,
        high_acos_multiplier=1.60,
        min_waste_clicks=10,
        hard_waste_clicks=20,
        min_waste_spend=8.0,
        pause_spend_multiplier=3.0,
    ),
    "标准": RuleThresholds(),
    "激进": RuleThresholds(
        min_clicks_for_negative=12,
        min_spend_for_negative=3.0,
        min_clicks_for_pause=25,
        min_orders_for_scale=2,
        low_acos_multiplier=0.80,
        high_acos_multiplier=1.15,
        min_waste_clicks=6,
        hard_waste_clicks=12,
        min_waste_spend=3.0,
        pause_spend_multiplier=1.25,
    ),
}


def thresholds_for_strictness(strictness: str) -> RuleThresholds:
    return RULE_THRESHOLDS_BY_STRICTNESS.get(strictness, RULE_THRESHOLDS_BY_STRICTNESS[DEFAULT_DIAGNOSIS_STRICTNESS])
