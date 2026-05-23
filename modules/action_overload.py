from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from modules.diagnosis import DiagnosisConfig


@dataclass(frozen=True)
class ActionLimitConfig:
    max_p0_actions: int = 10
    max_p1_actions: int = 20
    max_actions_per_type_p0: int = 5
    max_actions_per_campaign_p0: int = 3


AUDIT_COLUMNS = [
    "execution_tier",
    "is_today_action",
    "action_rank",
    "action_type",
    "priority",
    "priority_score",
    "Campaign Name",
    "Ad Group Name",
    "diagnosis_object",
    "diagnosis_level",
    "Spend",
    "Sales",
    "Orders",
    "ACOS",
    "Clicks",
    "data_sufficiency",
    "confidence",
    "operation_risk",
    "estimated_savings",
    "estimated_opportunity",
    "spend_share",
    "materiality_level",
    "priority_reason",
    "downgrade_reason",
    "recommendation_reason",
]


def enrich_action_prioritization(
    actions: pd.DataFrame,
    config: DiagnosisConfig,
    limit_config: ActionLimitConfig | None = None,
) -> pd.DataFrame:
    limit_config = limit_config or ActionLimitConfig()
    if actions.empty:
        return _ensure_priority_columns(actions.copy())

    enriched = actions.copy()
    total_spend = float(enriched["Spend"].fillna(0).sum())
    enriched["estimated_savings"] = enriched.apply(lambda row: _estimated_savings(row, config), axis=1)
    enriched["estimated_opportunity"] = enriched.apply(lambda row: _estimated_opportunity(row), axis=1)
    enriched["spend_share"] = enriched["Spend"].fillna(0).astype(float) / total_spend if total_spend > 0 else 0.0
    enriched["materiality_level"] = enriched.apply(lambda row: _materiality_level(row, config), axis=1)
    enriched["priority_score"] = enriched.apply(lambda row: _overload_priority_score(row, config), axis=1)
    enriched["优先级评分"] = enriched["priority_score"]
    enriched["priority_reason"] = enriched.apply(lambda row: _priority_reason(row, config), axis=1)
    enriched["execution_tier"] = enriched["priority_score"].apply(_initial_tier)
    enriched["downgrade_reason"] = enriched.apply(lambda row: _base_downgrade_reason(row, config), axis=1)

    enriched = _apply_top_n_limits(enriched, limit_config)
    enriched["is_today_action"] = enriched["execution_tier"].eq("P0")
    enriched = enriched.sort_values(
        ["execution_tier", "action_rank", "priority_score", "Spend"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    return enriched


def build_action_overload_audit(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    audit = pd.DataFrame(
        {
            "action_type": actions.get("建议动作", ""),
            "execution_tier": actions.get("execution_tier", ""),
            "is_today_action": actions.get("is_today_action", False),
            "action_rank": actions.get("action_rank", ""),
            "priority": actions.get("优先级", ""),
            "priority_score": actions.get("优先级评分", actions.get("priority_score", 0)),
            "Campaign Name": actions.get("Campaign Name", ""),
            "Ad Group Name": actions.get("Ad Group Name", ""),
            "diagnosis_object": actions.get("诊断对象", ""),
            "diagnosis_level": actions.get("诊断层级", ""),
            "Spend": actions.get("Spend", 0),
            "Sales": actions.get("Sales", 0),
            "Orders": actions.get("Orders", 0),
            "ACOS": actions.get("ACOS", 0),
            "Clicks": actions.get("Clicks", 0),
            "data_sufficiency": actions.get("数据充分性", ""),
            "confidence": actions.get("置信度", ""),
            "operation_risk": actions.get("操作风险", ""),
            "estimated_savings": actions.get("estimated_savings", 0),
            "estimated_opportunity": actions.get("estimated_opportunity", 0),
            "spend_share": actions.get("spend_share", 0),
            "materiality_level": actions.get("materiality_level", ""),
            "priority_reason": actions.get("priority_reason", ""),
            "downgrade_reason": actions.get("downgrade_reason", ""),
            "recommendation_reason": actions.get("原因", ""),
        }
    )
    return audit[AUDIT_COLUMNS]


def build_action_overload_summary(
    actions: pd.DataFrame,
    config: DiagnosisConfig,
    analysis_object_count: int,
    limit_config: ActionLimitConfig | None = None,
    before_stats: dict[str, int] | None = None,
) -> dict[str, object]:
    limit_config = limit_config or ActionLimitConfig()
    total_actions = int(len(actions))
    p0_count = int((actions.get("execution_tier", pd.Series(dtype=str)) == "P0").sum()) if "execution_tier" in actions else 0
    p1_count = int((actions.get("execution_tier", pd.Series(dtype=str)) == "P1").sum()) if "execution_tier" in actions else 0
    p2_count = int((actions.get("execution_tier", pd.Series(dtype=str)) == "P2").sum()) if "execution_tier" in actions else 0
    p3_count = int((actions.get("execution_tier", pd.Series(dtype=str)) == "P3").sum()) if "execution_tier" in actions else 0
    has_tiers = "execution_tier" in actions.columns and (p0_count + p1_count + p2_count + p3_count) > 0
    high_actions = p0_count if has_tiers else int((actions.get("优先级", pd.Series(dtype=str)) == "高").sum()) if not actions.empty else 0
    focus_actions = p0_count + p1_count if has_tiers else total_actions

    campaign_high = _campaign_high_share(actions)
    overload_flags = {
        "总建议动作数是否过多": focus_actions > 50 or (analysis_object_count > 0 and focus_actions > analysis_object_count * 0.2),
        "高优先级动作是否过多": high_actions > limit_config.max_p0_actions or (focus_actions > 0 and high_actions > focus_actions * 0.4),
        "单个 Campaign 是否占比过高": any(value > 0.3 for value in campaign_high.values()),
        "低影响动作是否进入高优先级": _low_impact_high_priority_count(actions, config) > 0,
        "重复或冲突动作是否过多": _duplicate_object_count(actions) > 0,
    }
    return {
        "是否存在动作过载": any(overload_flags.values()),
        "总建议动作数": total_actions,
        "高优先级动作数": high_actions,
        "重点展示动作数": focus_actions,
        "P0 数量": p0_count,
        "P1 数量": p1_count,
        "P2 数量": p2_count,
        "P3 数量": p3_count,
        "动作类型数量": actions.get("建议动作", pd.Series(dtype=str)).value_counts().to_dict() if not actions.empty else {},
        "P0 动作类型数量": _tier_value_counts(actions, "P0", "建议动作"),
        "Campaign 高优先级占比": campaign_high,
        "低影响高优先级数量": _low_impact_high_priority_count(actions, config),
        "重复对象数量": _duplicate_object_count(actions),
        "overload_flags": overload_flags,
        "before_stats": before_stats or {},
        "limit_config": limit_config,
    }


def write_action_audit(actions: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    build_action_overload_audit(actions).to_csv(path, index=False, encoding="utf-8-sig")


def write_overload_summary(summary: dict[str, object], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 真实报表动作过载审计",
        "",
        f"- 当前是否存在动作过载：{'是' if summary['是否存在动作过载'] else '否'}",
        f"- 总建议动作数：{summary['总建议动作数']}",
        f"- 重点展示动作数（P0+P1）：{summary.get('重点展示动作数', summary['总建议动作数'])}",
        f"- 高优先级动作数：{summary['高优先级动作数']}",
        f"- P0 今日必做数量：{summary['P0 数量']}",
        f"- P1 本周重点数量：{summary['P1 数量']}",
        f"- P2 待观察数量：{summary['P2 数量']}",
        f"- P3 仅记录数量：{summary['P3 数量']}",
        "",
        "## 动作过载检查",
    ]
    flags = summary.get("overload_flags", {})
    for name, value in flags.items():
        lines.append(f"- {name}：{'是' if value else '否'}")
    lines.extend(["", "## 动作类型数量"])
    for action, count in summary.get("动作类型数量", {}).items():
        lines.append(f"- {action}: {count}")
    lines.extend(["", "## P0 今日必做动作类型"])
    p0_action_counts = summary.get("P0 动作类型数量", {})
    if p0_action_counts:
        for action, count in p0_action_counts.items():
            lines.append(f"- {action}: {count}")
    else:
        lines.append("- 无")
    lines.extend(["", "## Campaign 高优先级占比"])
    for campaign, share in summary.get("Campaign 高优先级占比", {}).items():
        lines.append(f"- {campaign}: {share:.1%}")
    before_stats = summary.get("before_stats", {})
    if before_stats:
        lines.extend(["", "## 修改前后对比"])
        for key, value in before_stats.items():
            lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## 建议如何收敛",
            "- P0 今日必做最多保留 10 条。",
            "- 同一动作类型最多占 5 条 P0。",
            "- 同一 Campaign 最多占 3 条 P0。",
            "- 低影响、低置信度、高风险或数据不足对象降级到 P2/P3。",
            "- Campaign 层级只作为结构提醒，优先处理 Search Term / Targeting 具体对象。",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _ensure_priority_columns(actions: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "execution_tier",
        "is_today_action",
        "action_rank",
        "estimated_savings",
        "estimated_opportunity",
        "spend_share",
        "materiality_level",
        "priority_reason",
        "downgrade_reason",
    ]:
        if column not in actions.columns:
            actions[column] = []
    return actions


def _estimated_savings(row: pd.Series, config: DiagnosisConfig) -> float:
    spend = float(row.get("Spend", 0) or 0)
    sales = float(row.get("Sales", 0) or 0)
    orders = float(row.get("Orders", 0) or 0)
    action_text = f"{row.get('建议动作', '')} {row.get('合并动作', '')} {row.get('问题类型', '')}"
    if orders <= 0 and ("无转化" in action_text or "否定" in action_text or "降低竞价" in action_text or "暂停" in action_text):
        return spend
    if orders > 0 and float(row.get("ACOS", 0) or 0) > config.target_acos:
        return max(0.0, spend - sales * config.target_acos)
    return 0.0


def _estimated_opportunity(row: pd.Series) -> float:
    action_text = f"{row.get('建议动作', '')} {row.get('合并动作', '')}"
    if any(action in action_text for action in ["提高竞价", "增加预算", "提取精准投放"]):
        return max(float(row.get("Sales", 0) or 0), float(row.get("Orders", 0) or 0))
    return 0.0


def _materiality_level(row: pd.Series, config: DiagnosisConfig) -> str:
    savings = float(row.get("estimated_savings", 0) or 0)
    opportunity = float(row.get("estimated_opportunity", 0) or 0)
    spend = float(row.get("Spend", 0) or 0)
    share = float(row.get("spend_share", 0) or 0)
    if savings >= config.min_waste_spend * 3 or opportunity >= config.min_waste_spend * 5 or share >= 0.02:
        return "高"
    if savings >= config.min_waste_spend or opportunity > 0 or spend >= config.min_waste_spend:
        return "中"
    return "低"


def _overload_priority_score(row: pd.Series, config: DiagnosisConfig) -> int:
    spend = float(row.get("Spend", 0) or 0)
    clicks = float(row.get("Clicks", 0) or 0)
    acos = float(row.get("ACOS", 0) or 0)
    orders = float(row.get("Orders", 0) or 0)
    savings = float(row.get("estimated_savings", 0) or 0)
    opportunity = float(row.get("estimated_opportunity", 0) or 0)
    spend_share = float(row.get("spend_share", 0) or 0)

    impact_score = min(max(spend, savings, opportunity * 0.25) / max(config.min_waste_spend, 1) * 4, 32)
    impact_score += min(spend_share * 400, 8)
    data_score = {"充分": 20, "一般": 10, "不足": 0}.get(str(row.get("数据充分性", "")), 0)
    severity_score = 0.0
    if orders <= 0 and spend >= config.min_waste_spend:
        severity_score = min(spend / max(config.min_waste_spend, 1) * 3, 20)
    elif orders > 0 and acos > config.target_acos:
        severity_score = min((acos / max(config.target_acos, 0.01) - 1) * 8, 20)
    elif opportunity > 0:
        severity_score = min(orders * 4, 20)
    confidence_score = {"高": 10, "中": 5, "低": 0}.get(str(row.get("置信度", "")), 0)
    actionability_score = _actionability_score(str(row.get("建议动作", "")))
    risk_penalty = {"高": 15, "中": 5, "低": 0}.get(str(row.get("操作风险", "")), 5)
    insufficiency_penalty = 30 if row.get("数据充分性") == "不足" else 0
    if orders > 0 and any(action in str(row.get("合并动作", "")) for action in ["否定", "暂停"]):
        risk_penalty += 20
    score = impact_score + data_score + severity_score + confidence_score + actionability_score - risk_penalty - insufficiency_penalty
    return int(max(0, min(round(score), 100)))


def _actionability_score(action: str) -> int:
    if action in {"否定精准", "降低竞价", "提高竞价", "提取精准投放"}:
        return 10
    if action in {"增加预算", "暂停"}:
        return 7
    if action == "检查 Listing":
        return 3
    return 1


def _initial_tier(score: int) -> str:
    if score >= 80:
        return "P0"
    if score >= 65:
        return "P1"
    if score >= 50:
        return "P2"
    return "P3"


def _base_downgrade_reason(row: pd.Series, config: DiagnosisConfig) -> str:
    reasons = []
    level = str(row.get("诊断层级", ""))
    action = str(row.get("建议动作", ""))
    if level in {"广告活动", "Campaign", "广告组", "Ad Group"} and action != "暂停":
        reasons.append("Campaign/Ad Group 层级仅作为结构提醒，优先处理具体搜索词或 Targeting")
    if row.get("数据充分性") == "不足":
        reasons.append("数据充分性不足，暂不进入今日必做")
    if row.get("置信度") == "低":
        reasons.append("置信度不足，不进入高优先级")
    if row.get("操作风险") == "高":
        reasons.append("操作风险较高，需要人工复核")
    if row.get("materiality_level") == "低":
        reasons.append("影响金额不足，降级为待观察")
    if float(row.get("Spend", 0) or 0) < config.min_waste_spend and float(row.get("Orders", 0) or 0) <= 0:
        reasons.append("花费低于最低阈值")
    return "；".join(reasons)


def _apply_top_n_limits(actions: pd.DataFrame, config: ActionLimitConfig) -> pd.DataFrame:
    actions = actions.copy()
    actions["_level_sort"] = actions["诊断层级"].apply(_level_sort)
    actions["_action_sort"] = actions["建议动作"].apply(_action_sort)
    actions = actions.sort_values(
        ["priority_score", "_level_sort", "_action_sort", "estimated_savings", "Spend"],
        ascending=[False, True, True, False, False],
    ).copy()
    p0_indices: list[int] = []
    p1_indices: list[int] = []
    type_counts: dict[str, int] = {}
    campaign_counts: dict[str, int] = {}
    ranks: dict[int, int] = {}

    for idx, row in actions.iterrows():
        action_type = str(row.get("建议动作", ""))
        campaign = str(row.get("Campaign Name", ""))
        eligible_p0 = (
            row["execution_tier"] == "P0"
            and not row.get("downgrade_reason")
            and type_counts.get(action_type, 0) < config.max_actions_per_type_p0
            and campaign_counts.get(campaign, 0) < config.max_actions_per_campaign_p0
            and len(p0_indices) < config.max_p0_actions
        )
        if eligible_p0:
            p0_indices.append(idx)
            type_counts[action_type] = type_counts.get(action_type, 0) + 1
            campaign_counts[campaign] = campaign_counts.get(campaign, 0) + 1
            ranks[idx] = len(p0_indices)
            continue
        if row["execution_tier"] == "P0":
            actions.at[idx, "execution_tier"] = "P1"
            reason = actions.at[idx, "downgrade_reason"]
            extra = _limit_downgrade_reason(row, config, type_counts, campaign_counts, len(p0_indices))
            actions.at[idx, "downgrade_reason"] = "；".join(part for part in [reason, extra] if part)
        if actions.at[idx, "execution_tier"] == "P1" and len(p1_indices) < config.max_p1_actions:
            p1_indices.append(idx)
            ranks[idx] = len(p1_indices)
        elif actions.at[idx, "execution_tier"] == "P1":
            actions.at[idx, "execution_tier"] = "P2"
            reason = actions.at[idx, "downgrade_reason"]
            actions.at[idx, "downgrade_reason"] = "；".join(part for part in [reason, "超出本周重点数量上限，降级为待观察"] if part)

    actions["action_rank"] = [ranks.get(idx, 9999) for idx in actions.index]
    return actions.drop(columns=["_level_sort", "_action_sort"], errors="ignore")


def _limit_downgrade_reason(row: pd.Series, config: ActionLimitConfig, type_counts: dict[str, int], campaign_counts: dict[str, int], p0_count: int) -> str:
    if p0_count >= config.max_p0_actions:
        return "超出今日处理数量上限，降级为本周重点"
    if type_counts.get(str(row.get("建议动作", "")), 0) >= config.max_actions_per_type_p0:
        return "同一动作类型已达到今日上限"
    if campaign_counts.get(str(row.get("Campaign Name", "")), 0) >= config.max_actions_per_campaign_p0:
        return "同一 Campaign 已有更高优先级动作"
    return "不满足 P0 收敛条件"


def _level_sort(level: object) -> int:
    value = str(level)
    if value in {"搜索词", "Search Term", "Targeting", "ASIN"}:
        return 0
    if value in {"广告组", "Ad Group"}:
        return 1
    if value in {"广告活动", "Campaign"}:
        return 2
    return 3


def _action_sort(action: object) -> int:
    value = str(action)
    order = {
        "否定精准": 0,
        "降低竞价": 1,
        "提取精准投放": 2,
        "提高竞价": 3,
        "暂停": 4,
        "增加预算": 5,
        "检查 Listing": 6,
        "继续观察": 7,
    }
    return order.get(value, 9)


def _priority_reason(row: pd.Series, config: DiagnosisConfig) -> str:
    return (
        f"影响层级 {row.get('materiality_level', '')}；"
        f"预估节省 ${float(row.get('estimated_savings', 0) or 0):.2f}；"
        f"预估机会 {float(row.get('estimated_opportunity', 0) or 0):.2f}；"
        f"花费占比 {float(row.get('spend_share', 0) or 0):.2%}；"
        f"数据充分性 {row.get('数据充分性', '')}；"
        f"置信度 {row.get('置信度', '')}。"
    )


def _campaign_high_share(actions: pd.DataFrame) -> dict[str, float]:
    if actions.empty or "Campaign Name" not in actions.columns:
        return {}
    if "execution_tier" in actions.columns and actions["execution_tier"].astype(str).isin(["P0", "P1", "P2", "P3"]).any():
        high = actions[actions["execution_tier"].astype(str).eq("P0")]
    else:
        high = actions[actions.get("优先级", pd.Series(dtype=str)).eq("高")]
    if high.empty:
        return {}
    counts = high["Campaign Name"].fillna("").astype(str).value_counts()
    total = counts.sum()
    return {campaign: float(count / total) for campaign, count in counts.items()}


def _low_impact_high_priority_count(actions: pd.DataFrame, config: DiagnosisConfig) -> int:
    if actions.empty:
        return 0
    if "execution_tier" in actions.columns and actions["execution_tier"].astype(str).isin(["P0", "P1", "P2", "P3"]).any():
        high = actions[actions["execution_tier"].astype(str).eq("P0")]
    else:
        high = actions[actions.get("优先级", pd.Series(dtype=str)).eq("高")]
    if high.empty:
        return 0
    low_impact = (
        (high.get("Spend", pd.Series(dtype=float)).fillna(0).astype(float) < config.min_waste_spend)
        | (high.get("Clicks", pd.Series(dtype=float)).fillna(0).astype(float) < 8)
        | high.get("数据充分性", pd.Series(dtype=str)).astype(str).eq("不足")
    )
    return int(low_impact.sum())


def _tier_value_counts(actions: pd.DataFrame, tier: str, column: str) -> dict[str, int]:
    if actions.empty or "execution_tier" not in actions.columns or column not in actions.columns:
        return {}
    target = actions[actions["execution_tier"].astype(str).eq(tier)]
    if target.empty:
        return {}
    return {str(key): int(value) for key, value in target[column].fillna("").astype(str).value_counts().items()}


def _duplicate_object_count(actions: pd.DataFrame) -> int:
    if actions.empty:
        return 0
    keys = ["诊断层级", "诊断对象", "Campaign Name", "Ad Group Name"]
    available = [column for column in keys if column in actions.columns]
    if not available:
        return 0
    return int(actions.duplicated(available).sum())
