from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from modules.aggregation import aggregate_by_dimension
from modules.field_mapping import CANONICAL_FIELDS
from modules.metrics import format_percent
from modules.rules_config import DEFAULT_DIAGNOSIS_STRICTNESS, DEFAULT_PROTECTED_TERMS, thresholds_for_strictness


ACTIONS = [
    "暂停",
    "否定精准",
    "否定词组",
    "降低竞价",
    "提高竞价",
    "增加预算",
    "提取精准投放",
    "检查 Listing",
    "继续观察",
]


PRIORITY_RANK = {"高": 0, "中": 1, "低": 2}
ACTION_RANK = {action: index for index, action in enumerate(ACTIONS)}
IRRELEVANT_TERM_HINTS = [
    "free",
    "used",
    "second hand",
    "pdf",
    "manual",
    "repair",
    "replacement parts",
    "parts only",
    "diy",
    "wholesale",
    "coupon",
    "免费",
    "二手",
    "维修",
    "说明书",
    "配件",
    "批发",
]


ACTION_COLUMNS = [
    "诊断规则",
    "问题类型",
    "建议动作",
    "合并动作",
    "优先级",
    "优先级评分",
    "置信度",
    "操作风险",
    "数据充分性",
    "诊断严格度",
    "诊断层级",
    "诊断对象",
    "原因",
    "命中规则",
    "命中条件",
    "关键证据",
    "为什么不是更强动作",
    "为什么不是更弱动作",
    "证据说明",
    "运营解释",
    "人工复核原因",
    "执行建议",
    "复核提醒",
    "目标 ACOS",
    "目标 CPA",
    "账户平均 CTR",
    "账户平均 CVR",
    "是否保护词",
    "是否存在规则冲突",
    "Campaign Name",
    "Ad Group Name",
    "Customer Search Term",
    "Targeting",
    "Match Type",
    "Ad Product",
    "Advertised ASIN",
    "Purchased ASIN",
    "Impressions",
    "Clicks",
    "Spend",
    "Sales",
    "Orders",
    "CTR",
    "CPC",
    "CVR",
    "ACOS",
    "ROAS",
    "Source Report",
]


@dataclass(frozen=True)
class DiagnosisConfig:
    target_acos: float = 0.30
    min_waste_clicks: int = 8
    hard_waste_clicks: int = 15
    min_waste_spend: float = 5.0
    high_acos_multiplier: float = 1.25
    low_acos_multiplier: float = 0.70
    min_quality_orders: int = 2
    high_ctr: float = 0.008
    low_ctr: float = 0.002
    low_cvr: float = 0.03
    high_impressions: int = 1000
    low_impressions: int = 300
    min_sales_low_exposure: float = 20.0
    budget_pressure_ratio: float = 0.80
    pause_spend_multiplier: float = 1.50
    exact_opportunity_orders: int = 2
    protected_terms: tuple[str, ...] = DEFAULT_PROTECTED_TERMS
    diagnosis_strictness: str = DEFAULT_DIAGNOSIS_STRICTNESS


@dataclass(frozen=True)
class AccountContext:
    target_acos: float
    account_ctr: float
    account_cvr: float
    account_cpc: float
    account_acos: float
    account_roas: float
    account_aov: float
    target_cpa: float
    diagnosis_strictness: str = DEFAULT_DIAGNOSIS_STRICTNESS
    protected_terms: tuple[str, ...] = DEFAULT_PROTECTED_TERMS


@dataclass(frozen=True)
class DataSufficiency:
    data_sufficiency: str
    confidence_base: str
    explanation: str


def run_diagnosis(
    df: pd.DataFrame,
    config: DiagnosisConfig | float | None = None,
    mode: str = "完整版",
) -> pd.DataFrame:
    if config is None:
        config = DiagnosisConfig()
    elif isinstance(config, (int, float)):
        config = DiagnosisConfig(target_acos=float(config))
    config = _apply_strictness_to_config(config)

    account_context = build_account_context(df, config)
    actions: list[dict[str, object]] = []

    search_term_df = aggregate_by_dimension(
        df,
        [
            CANONICAL_FIELDS["campaign_name"],
            CANONICAL_FIELDS["ad_group_name"],
            CANONICAL_FIELDS["customer_search_term"],
        ],
        "搜索词",
    )
    targeting_df = aggregate_by_dimension(
        df,
        [
            CANONICAL_FIELDS["campaign_name"],
            CANONICAL_FIELDS["ad_group_name"],
            CANONICAL_FIELDS["targeting"],
            CANONICAL_FIELDS["match_type"],
        ],
        "Targeting",
    )

    for _, row in search_term_df.iterrows():
        if _text(row, CANONICAL_FIELDS["customer_search_term"]):
            actions.extend(_diagnose_keyword_like_row(row, config, account_context, "搜索词"))

    for _, row in targeting_df.iterrows():
        if _text(row, CANONICAL_FIELDS["targeting"]):
            actions.extend(_diagnose_keyword_like_row(row, config, account_context, "Targeting"))

    if mode == "完整版":
        campaign_df = aggregate_by_dimension(df, [CANONICAL_FIELDS["campaign_name"]], "广告活动")
        ad_group_df = aggregate_by_dimension(
            df,
            [CANONICAL_FIELDS["campaign_name"], CANONICAL_FIELDS["ad_group_name"]],
            "广告组",
        )

        for _, row in campaign_df.iterrows():
            actions.extend(_diagnose_campaign_row(row, config, account_context, df))

        for _, row in ad_group_df.iterrows():
            actions.extend(_diagnose_ad_group_row(row, config, account_context, df))

    if not actions:
        return pd.DataFrame(columns=ACTION_COLUMNS)

    action_df = pd.DataFrame(actions)
    action_df = deduplicate_and_resolve_conflicts(action_df)
    if action_df.empty:
        return pd.DataFrame(columns=ACTION_COLUMNS)

    action_df["优先级评分"] = action_df.apply(
        lambda row: calculate_priority_score(
            row,
            _text(row, "问题类型") or _text(row, "诊断规则"),
            _text(row, "建议动作"),
            account_context,
            config,
        ),
        axis=1,
    )
    action_df["优先级"] = action_df["优先级评分"].apply(_priority_label_from_score)
    action_df["动作排序"] = action_df["建议动作"].map(ACTION_RANK).fillna(99)
    action_df["优先级排序"] = action_df["优先级"].map(PRIORITY_RANK).fillna(3)
    action_df = action_df.sort_values(
        by=["优先级排序", "优先级评分", "动作排序", CANONICAL_FIELDS["spend"], CANONICAL_FIELDS["clicks"]],
        ascending=[True, False, True, False, False],
    )
    return action_df[ACTION_COLUMNS].reset_index(drop=True)


def summarize_recommendations(actions: pd.DataFrame) -> dict[str, object]:
    if actions.empty:
        return {
            "总建议数": 0,
            "高优先级": 0,
            "否定建议": 0,
            "暂停建议": 0,
            "调价建议": 0,
            "增长建议": 0,
            "Listing问题": 0,
            "观察项": 0,
            "摘要文本": "暂未发现触发完整诊断规则的问题项，当前数据可以继续观察。",
        }

    action_counter = _count_actions(actions)
    priority_counter = Counter(actions["优先级"])
    summary = {
        "总建议数": int(len(actions)),
        "高优先级": int(priority_counter.get("高", 0)),
        "否定建议": int(action_counter.get("否定精准", 0) + action_counter.get("否定词组", 0)),
        "暂停建议": int(action_counter.get("暂停", 0)),
        "调价建议": int(action_counter.get("降低竞价", 0) + action_counter.get("提高竞价", 0)),
        "增长建议": int(action_counter.get("增加预算", 0) + action_counter.get("提取精准投放", 0)),
        "Listing问题": int(action_counter.get("检查 Listing", 0)),
        "观察项": int(action_counter.get("继续观察", 0)),
    }
    summary["摘要文本"] = (
        f"本次共生成 {summary['总建议数']} 条动作建议，其中高优先级 {summary['高优先级']} 条；"
        f"否定 {summary['否定建议']} 条，暂停 {summary['暂停建议']} 条，"
        f"调价 {summary['调价建议']} 条，增长放量 {summary['增长建议']} 条，"
        f"Listing 检查 {summary['Listing问题']} 条。"
    )
    return summary


def _apply_strictness_to_config(config: DiagnosisConfig) -> DiagnosisConfig:
    thresholds = thresholds_for_strictness(config.diagnosis_strictness)
    return DiagnosisConfig(
        target_acos=config.target_acos,
        min_waste_clicks=thresholds.min_waste_clicks,
        hard_waste_clicks=thresholds.hard_waste_clicks,
        min_waste_spend=max(config.min_waste_spend, thresholds.min_waste_spend),
        high_acos_multiplier=thresholds.high_acos_multiplier,
        low_acos_multiplier=thresholds.low_acos_multiplier,
        min_quality_orders=max(config.min_quality_orders, thresholds.min_orders_for_scale),
        high_ctr=config.high_ctr,
        low_ctr=config.low_ctr,
        low_cvr=config.low_cvr,
        high_impressions=config.high_impressions,
        low_impressions=config.low_impressions,
        min_sales_low_exposure=config.min_sales_low_exposure,
        budget_pressure_ratio=config.budget_pressure_ratio,
        pause_spend_multiplier=thresholds.pause_spend_multiplier,
        exact_opportunity_orders=max(config.exact_opportunity_orders, thresholds.min_orders_for_scale),
        protected_terms=config.protected_terms,
        diagnosis_strictness=config.diagnosis_strictness,
    )


def calculate_account_aov(df: pd.DataFrame) -> float:
    sales = float(df[CANONICAL_FIELDS["sales"]].sum()) if CANONICAL_FIELDS["sales"] in df.columns else 0.0
    orders = float(df[CANONICAL_FIELDS["orders"]].sum()) if CANONICAL_FIELDS["orders"] in df.columns else 0.0
    return sales / orders if orders > 0 else 0.0


def calculate_target_cpa(account_aov: float, target_acos: float) -> float:
    return account_aov * target_acos if account_aov > 0 and target_acos > 0 else 0.0


def build_account_context(df: pd.DataFrame, config: DiagnosisConfig) -> AccountContext:
    impressions = float(df[CANONICAL_FIELDS["impressions"]].sum()) if CANONICAL_FIELDS["impressions"] in df.columns else 0.0
    clicks = float(df[CANONICAL_FIELDS["clicks"]].sum()) if CANONICAL_FIELDS["clicks"] in df.columns else 0.0
    spend = float(df[CANONICAL_FIELDS["spend"]].sum()) if CANONICAL_FIELDS["spend"] in df.columns else 0.0
    sales = float(df[CANONICAL_FIELDS["sales"]].sum()) if CANONICAL_FIELDS["sales"] in df.columns else 0.0
    orders = float(df[CANONICAL_FIELDS["orders"]].sum()) if CANONICAL_FIELDS["orders"] in df.columns else 0.0
    account_aov = sales / orders if orders > 0 else 0.0
    return AccountContext(
        target_acos=config.target_acos,
        account_ctr=clicks / impressions if impressions > 0 else 0.0,
        account_cvr=orders / clicks if clicks > 0 else 0.0,
        account_cpc=spend / clicks if clicks > 0 else 0.0,
        account_acos=spend / sales if sales > 0 else 0.0,
        account_roas=sales / spend if spend > 0 else 0.0,
        account_aov=account_aov,
        target_cpa=calculate_target_cpa(account_aov, config.target_acos),
        diagnosis_strictness=config.diagnosis_strictness,
        protected_terms=config.protected_terms,
    )


def assess_data_sufficiency(row: pd.Series, thresholds: DiagnosisConfig | dict[str, object]) -> DataSufficiency:
    config = thresholds if isinstance(thresholds, DiagnosisConfig) else thresholds.get("config", DiagnosisConfig())
    account_context = thresholds.get("account_context") if isinstance(thresholds, dict) else None
    target_cpa = getattr(account_context, "target_cpa", 0.0) or 0.0
    clicks = _value(row, CANONICAL_FIELDS["clicks"])
    spend = _value(row, CANONICAL_FIELDS["spend"])
    impressions = _value(row, CANONICAL_FIELDS["impressions"])
    orders = _value(row, CANONICAL_FIELDS["orders"])

    if clicks < 5 or spend < config.min_waste_spend or (impressions < 300 and clicks < 3):
        return DataSufficiency(
            "不足",
            "低",
            f"点击 {clicks:.0f} 次、花费 ${spend:.2f}，样本偏少，暂不适合做否定、暂停或大幅降价。",
        )
    if 5 <= clicks <= 14 or orders == 1:
        return DataSufficiency(
            "一般",
            "中",
            f"点击 {clicks:.0f} 次、花费 ${spend:.2f}，已有方向性信号，但样本仍需复核。",
        )
    if clicks >= 15 or orders >= 2 or (target_cpa > 0 and spend >= target_cpa) or (
        orders == 0 and spend >= config.min_waste_spend * 2
    ):
        return DataSufficiency(
            "充分",
            "中",
            f"点击 {clicks:.0f} 次、花费 ${spend:.2f}，样本已具备判断价值。",
        )
    return DataSufficiency("一般", "中", f"点击 {clicks:.0f} 次、花费 ${spend:.2f}，建议结合相关性复核。")


def build_negative_keywords(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Campaign Name",
        "Ad Group Name",
        "Negative Keyword",
        "Negative Match Type",
        "Source Action",
        "Reason",
        "优先级评分",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)

    negative_actions = actions[_action_mask(actions, {"否定精准", "否定词组"})].copy()
    if not negative_actions.empty:
        negative_actions = negative_actions[negative_actions["Orders"].fillna(0).astype(float) <= 0]
    if not negative_actions.empty and "数据充分性" in negative_actions.columns:
        negative_actions = negative_actions[negative_actions["数据充分性"] != "不足"]
    if negative_actions.empty:
        return pd.DataFrame(columns=columns)

    negative_actions["Negative Keyword"] = negative_actions["Customer Search Term"].where(
        negative_actions["Customer Search Term"].astype(str).str.strip().ne(""),
        negative_actions["Targeting"],
    )
    negative_actions["Negative Match Type"] = negative_actions.apply(_negative_match_type, axis=1)
    negative_actions["Source Action"] = negative_actions.apply(
        lambda row: _source_action(row, {"否定精准", "否定词组"}),
        axis=1,
    )
    negative_actions["Reason"] = negative_actions["原因"]
    return negative_actions[columns].drop_duplicates().reset_index(drop=True)


def build_bid_adjustments(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Campaign Name",
        "Ad Group Name",
        "Targeting",
        "Customer Search Term",
        "建议调价方向",
        "Source Action",
        "Reason",
        "ACOS",
        "Orders",
        "优先级评分",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)

    bid_actions = actions[_action_mask(actions, {"降低竞价", "提高竞价"})].copy()
    if bid_actions.empty:
        return pd.DataFrame(columns=columns)

    bid_actions["建议调价方向"] = bid_actions.apply(_bid_direction, axis=1)
    bid_actions["Source Action"] = bid_actions.apply(
        lambda row: _source_action(row, {"降低竞价", "提高竞价"}),
        axis=1,
    )
    bid_actions["Reason"] = bid_actions["原因"]
    return bid_actions[columns].drop_duplicates().reset_index(drop=True)


def build_pause_list(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "诊断层级",
        "Campaign Name",
        "Ad Group Name",
        "诊断对象",
        "Reason",
        "Spend",
        "Sales",
        "Orders",
        "ACOS",
        "优先级评分",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)
    pause_actions = actions[_action_mask(actions, {"暂停"})].copy()
    if not pause_actions.empty:
        pause_actions = pause_actions[pause_actions["Orders"].fillna(0).astype(float) <= 0]
    if not pause_actions.empty and "数据充分性" in pause_actions.columns:
        pause_actions = pause_actions[pause_actions["数据充分性"] != "不足"]
    if pause_actions.empty:
        return pd.DataFrame(columns=columns)
    pause_actions["Reason"] = pause_actions["原因"]
    return pause_actions[columns].drop_duplicates().reset_index(drop=True)


def build_growth_list(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "建议动作",
        "诊断层级",
        "Campaign Name",
        "Ad Group Name",
        "Customer Search Term",
        "Targeting",
        "Reason",
        "Impressions",
        "Clicks",
        "Orders",
        "ACOS",
        "优先级评分",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)
    growth_actions = actions[_action_mask(actions, {"提高竞价", "增加预算", "提取精准投放"})].copy()
    if growth_actions.empty:
        return pd.DataFrame(columns=columns)
    growth_actions["Reason"] = growth_actions["原因"]
    return growth_actions[columns].drop_duplicates().reset_index(drop=True)


def build_exact_targeting_opportunities(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Campaign Name",
        "Ad Group Name",
        "Customer Search Term",
        "建议投放方式",
        "Reason",
        "Impressions",
        "Clicks",
        "Spend",
        "Sales",
        "Orders",
        "ACOS",
        "ROAS",
        "优先级评分",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)

    exact_actions = actions[_action_mask(actions, {"提取精准投放"})].copy()
    if exact_actions.empty:
        return pd.DataFrame(columns=columns)

    exact_actions["建议投放方式"] = "Exact"
    exact_actions["Reason"] = exact_actions["原因"]
    return exact_actions[columns].drop_duplicates().reset_index(drop=True)


def build_priority_list(actions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "优先级",
        "优先级评分",
        "建议动作",
        "诊断规则",
        "问题类型",
        "合并动作",
        "诊断层级",
        "诊断对象",
        "原因",
        "证据说明",
        "Spend",
        "Sales",
        "Orders",
        "ACOS",
    ]
    if actions.empty:
        return pd.DataFrame(columns=columns)

    priority = actions.copy()
    priority["优先级排序"] = priority["优先级"].map(PRIORITY_RANK).fillna(3)
    priority = priority.sort_values(["优先级排序", "优先级评分", "Spend"], ascending=[True, False, False])
    return priority[[column for column in columns if column in priority.columns]].reset_index(drop=True)


def run_diagnosis_self_check(actions: pd.DataFrame, config: DiagnosisConfig) -> dict[str, object]:
    columns = ["检查项", "严重程度", "诊断层级", "诊断对象", "建议动作", "原因", "修复建议"]
    if actions.empty:
        return {
            "通过项数量": 0,
            "警告项数量": 0,
            "高风险异常数量": 0,
            "异常明细": pd.DataFrame(columns=columns),
            "修复建议": "暂无动作建议，无法执行规则自检。",
        }

    issues: list[dict[str, object]] = []
    checks = [
        ("有订单对象是否被建议否定", _check_ordered_negative),
        ("低数据量对象是否被建议强动作", lambda df: _check_low_data_strong_action(df, config)),
        ("数据不足对象是否被错误标记为高优先级", _check_insufficient_high_priority),
        ("低 ACOS 对象是否被建议降价", lambda df: _check_low_acos_bid_down(df, config)),
        ("高 ACOS 有订单对象是否被正确处理", lambda df: _check_ordered_high_acos_action(df, config)),
        ("无订单高花费对象是否被识别", lambda df: _check_high_spend_no_order_identified(df, config)),
    ]
    passed_count = 0
    for name, checker in checks:
        found = checker(actions)
        if found.empty:
            passed_count += 1
            continue
        for _, row in found.iterrows():
            issues.append(
                {
                    "检查项": name,
                    "严重程度": row.get("_severity", "警告"),
                    "诊断层级": row.get("诊断层级", ""),
                    "诊断对象": row.get("诊断对象", ""),
                    "建议动作": row.get("合并动作", row.get("建议动作", "")),
                    "原因": row.get("_reason", row.get("原因", "")),
                    "修复建议": row.get("_fix", "复核规则阈值和动作冲突处理。"),
                }
            )

    detail = pd.DataFrame(issues, columns=columns)
    warning_count = int((detail["严重程度"] == "警告").sum()) if not detail.empty else 0
    high_count = int((detail["严重程度"] == "高风险").sum()) if not detail.empty else 0
    return {
        "通过项数量": passed_count,
        "警告项数量": warning_count,
        "高风险异常数量": high_count,
        "异常明细": detail,
        "修复建议": _self_check_fix_summary(detail),
    }


def _diagnose_keyword_like_row(
    row: pd.Series,
    config: DiagnosisConfig,
    account_context: AccountContext,
    level: str,
) -> list[dict[str, object]]:
    actions = []
    clicks = _value(row, CANONICAL_FIELDS["clicks"])
    orders = _value(row, CANONICAL_FIELDS["orders"])
    spend = _value(row, CANONICAL_FIELDS["spend"])
    impressions = _value(row, CANONICAL_FIELDS["impressions"])
    ctr = _value(row, "CTR")
    cvr = _value(row, "CVR")
    acos = _value(row, "ACOS")
    sufficiency = assess_data_sufficiency(row, {"config": config, "account_context": account_context})
    search_text = _text(row, CANONICAL_FIELDS["customer_search_term"]) or _text(row, CANONICAL_FIELDS["targeting"])
    protected = is_protected_term(search_text, config.protected_terms)

    if sufficiency.data_sufficiency == "不足":
        actions.append(
            _build_action(
                row,
                rule="数据不足观察",
                issue_type="数据不足",
                action="继续观察",
                level=level,
                reason=sufficiency.explanation,
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="低",
                risk="中",
                explanation="当前点击和花费样本较少，暂时不足以判断好坏。建议继续观察，避免过早否定或暂停导致错失潜在转化。",
                execution="继续积累点击和花费样本，下一轮结合订单、ACOS、CVR 和相关性再判断。",
                review="如果搜索词明显不相关，可人工提前处理；否则不建议直接否定。",
            )
        )
        return actions

    if orders == 0 and protected:
        actions.append(
            _build_action(
                row,
                rule="保护词无订单复核",
                issue_type="保护词复核",
                action="继续观察",
                level=level,
                reason=f"该对象包含保护词，点击 {clicks:.0f} 次、花费 ${spend:.2f} 且无订单，但不建议直接否定。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="低",
                risk="高",
                evidence=f"包含保护词；{sufficiency.explanation}",
                explanation="保护词可能代表品牌词、核心词或战略词，误否定会影响长期流量结构。",
                execution="人工复核相关性；如果相关但效率低，优先小幅降价观察。",
                review="包含保护词，必须人工确认后再操作。",
            )
        )
    elif (
        orders == 0
        and level == "搜索词"
        and _text(row, CANONICAL_FIELDS["customer_search_term"])
        and sufficiency.data_sufficiency == "充分"
        and clicks >= config.hard_waste_clicks
        and spend >= max(config.min_waste_spend, account_context.target_cpa * 0.8 if account_context.target_cpa else config.min_waste_spend)
        and _looks_irrelevant(row, ctr, cvr, impressions, clicks)
    ):
        actions.append(
            _build_action(
                row,
                rule="明显不相关无订单词",
                issue_type="高花费无转化",
                action="否定精准",
                level=level,
                reason=f"点击 {clicks:.0f} 次、花费 ${spend:.2f} 且无订单，搜索意图疑似不相关，建议否定精准。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="高",
                risk="低",
                explanation="该对象已经产生较多点击和花费，但尚未产生订单，且存在不相关信号。继续投放可能扩大无效花费。",
                execution="先复核相关性；确认不相关后添加为否定精准。",
                review="如果属于品牌词、核心词或新品测试词，请不要直接否定。",
            )
        )
    elif orders == 0 and clicks >= config.min_waste_clicks and spend >= config.min_waste_spend:
        actions.append(
            _build_action(
                row,
                rule="无订单消耗复核",
                issue_type="高花费无转化",
                action="降低竞价",
                level=level,
                reason=f"点击 {clicks:.0f} 次、花费 ${spend:.2f} 且无订单，但相关性仍需人工复核。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="中",
                risk="中",
                explanation="该对象已经产生一定消耗但没有订单，可能是相关性不足，也可能只是转化样本不足。",
                execution="先检查搜索词相关性；明显不相关再否定，相关但效率差则小幅降低竞价。",
                review="不要批量否定可能相关的核心流量词。",
            )
        )

    if orders >= 1 and acos > config.target_acos * config.high_acos_multiplier and sufficiency.data_sufficiency in {"一般", "充分"}:
        actions.append(
            _build_action(
                row,
                rule="高 ACOS 有订单",
                issue_type="高 ACOS 有订单",
                action="降低竞价",
                level=level,
                reason=f"已有订单但 ACOS {format_percent(acos)} 高于目标 {format_percent(config.target_acos)}。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence=sufficiency.confidence_base,
                risk="中",
                explanation="该对象已经产生订单，说明并非完全无效流量，但 ACOS 高于目标，当前成本效率偏低。",
                execution=_bid_down_execution(acos, config.target_acos),
                review="执行后观察 3-7 天，确认订单量和 ACOS 是否改善。",
            )
        )

    if (
        orders >= config.min_quality_orders
        and 0 < acos <= config.target_acos * config.low_acos_multiplier
        and sufficiency.data_sufficiency == "充分"
        and (account_context.account_cvr <= 0 or cvr >= account_context.account_cvr * 1.1)
    ):
        actions.append(
            _build_action(
                row,
                rule="低 ACOS 优质词",
                issue_type="优质低 ACOS 机会",
                action="提高竞价",
                level=level,
                reason=f"订单 {orders:.0f} 个且 ACOS {format_percent(acos)} 低于目标的 70%，具备放量空间。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="高",
                risk="高",
                explanation="该对象已产生订单且 ACOS 低于目标，具备放量潜力。",
                execution="可提高竞价 5%-15%，并密切观察提价后的 ACOS 变化。",
                review="提价和放量可能推高 ACOS，建议小步测试。",
            )
        )

    if level == "搜索词" and orders >= config.exact_opportunity_orders and 0 < acos <= config.target_acos * config.low_acos_multiplier:
        actions.append(
            _build_action(
                row,
                rule="精准投放机会词",
                issue_type="优质低 ACOS 机会",
                action="提取精准投放",
                level=level,
                reason="搜索词已有稳定转化且 ACOS 健康，建议单独提取为精准投放。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="高",
                risk="中",
                explanation="该搜索词已有稳定订单且 ACOS 健康，单独拉精准便于控价和放量。",
                execution="复制该搜索词，新建 Exact 投放或加入精准关键词，并单独设置竞价。",
                review="提取后避免原广告活动与新精准广告互相抢量，必要时调整原匹配出价。",
            )
        )

    if clicks >= 15 and account_context.account_ctr > 0 and account_context.account_cvr > 0 and ctr >= account_context.account_ctr * 1.3 and cvr <= account_context.account_cvr * 0.5:
        actions.append(
            _build_action(
                row,
                rule="高 CTR 低 CVR",
                issue_type="高 CTR 低 CVR",
                action="检查 Listing",
                level=level,
                reason=f"CTR {format_percent(ctr)} 高于账户平均，但 CVR {format_percent(cvr)} 明显偏低。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence=sufficiency.confidence_base,
                risk="中",
                explanation="流量有吸引力，但转化承接弱，可能是相关性、价格、主图、评论或详情页问题。",
                execution="检查 Listing、价格、优惠、评价数量和投放相关性；必要时小幅降价观察。",
                review="不要只靠否定解决转化承接问题。",
            )
        )

    if impressions >= config.high_impressions and (
        ctr < config.low_ctr or (account_context.account_ctr > 0 and ctr <= account_context.account_ctr * 0.5)
    ):
        actions.append(
            _build_action(
                row,
                rule="低 CTR 高曝光",
                issue_type="高曝光低 CTR",
                action="检查 Listing",
                level=level,
                reason=f"曝光 {impressions:.0f} 且 CTR {format_percent(ctr)} 偏低，建议检查主图、标题、价格和相关性。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence=sufficiency.confidence_base,
                risk="中",
                explanation="曝光已经足够，但点击弱，可能广告位、主图、价格、标题或关键词相关性不足。",
                execution="优先检查主图、价格、标题和投放词相关性，再决定是否调价或换词。",
                review="低 CTR 问题通常不适合直接暂停，需要先确认展示词和商品承接。",
            )
        )

    if orders >= 2 and account_context.account_cvr > 0 and cvr >= account_context.account_cvr * 1.3 and impressions < config.low_impressions:
        actions.append(
            _build_action(
                row,
                rule="高 CVR 低曝光",
                issue_type="优质低 ACOS 机会",
                action="提高竞价",
                level=level,
                reason=f"已有 {orders:.0f} 个订单且 CVR {format_percent(cvr)} 高于账户平均，但曝光仅 {impressions:.0f}。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="中",
                risk="高",
                explanation="该对象转化效率好但曝光不足，可能有进一步拿量空间。",
                execution="可小幅提高竞价或拓展匹配，优先小预算测试。",
                review="放量后可能带来更宽泛流量，需要持续观察 ACOS。",
            )
        )

    return actions


def _check_ordered_negative(actions: pd.DataFrame) -> pd.DataFrame:
    mask = (actions["Orders"].fillna(0).astype(float) > 0) & actions["合并动作"].astype(str).str.contains("否定", na=False)
    return _issue_rows(actions, mask, "高风险", "有订单对象被建议否定。", "禁止 Orders > 0 对象进入否定动作和否定词清单。")


def _check_low_data_strong_action(actions: pd.DataFrame, config: DiagnosisConfig) -> pd.DataFrame:
    low_data = (actions["Clicks"].fillna(0).astype(float) < 5) | (actions["Spend"].fillna(0).astype(float) < config.min_waste_spend)
    strong = actions["合并动作"].astype(str).str.contains("暂停|否定", na=False)
    return _issue_rows(actions, low_data & strong, "高风险", "低数据量对象被建议强动作。", "数据不足时只允许继续观察、检查或轻量复核。")


def _check_insufficient_high_priority(actions: pd.DataFrame) -> pd.DataFrame:
    mask = (actions["数据充分性"].astype(str) == "不足") & (actions["优先级"].astype(str) == "高")
    return _issue_rows(actions, mask, "警告", "数据不足对象被标记为高优先级。", "降低优先级或补充数据充分性扣分。")


def _check_low_acos_bid_down(actions: pd.DataFrame, config: DiagnosisConfig) -> pd.DataFrame:
    mask = (
        (actions["ACOS"].fillna(0).astype(float) > 0)
        & (actions["ACOS"].fillna(0).astype(float) <= config.target_acos * config.low_acos_multiplier)
        & actions["合并动作"].astype(str).str.contains("降低竞价", na=False)
    )
    return _issue_rows(actions, mask, "高风险", "低 ACOS 对象被建议降低竞价。", "低 ACOS 对象应进入放量、精准投放或观察，不应降价。")


def _check_ordered_high_acos_action(actions: pd.DataFrame, config: DiagnosisConfig) -> pd.DataFrame:
    target = (actions["Orders"].fillna(0).astype(float) > 0) & (actions["ACOS"].fillna(0).astype(float) > config.target_acos)
    bad_action = actions["合并动作"].astype(str).str.contains("否定|暂停", na=False)
    return _issue_rows(actions, target & bad_action, "高风险", "高 ACOS 有订单对象被建议强动作。", "有订单高 ACOS 应优先降低竞价或复核，不应否定或暂停。")


def _check_high_spend_no_order_identified(actions: pd.DataFrame, config: DiagnosisConfig) -> pd.DataFrame:
    high_spend = (
        (actions["Orders"].fillna(0).astype(float) <= 0)
        & (actions["Spend"].fillna(0).astype(float) >= actions.get("目标 CPA", pd.Series(0, index=actions.index)).fillna(0).astype(float).where(lambda value: value > 0, config.min_waste_spend))
        & ~actions["合并动作"].astype(str).str.contains("否定|降低竞价|暂停|继续观察", na=False)
    )
    return _issue_rows(actions, high_spend, "警告", "无订单高花费对象没有进入浪费或复核动作。", "检查高花费无转化规则是否漏判。")


def _issue_rows(actions: pd.DataFrame, mask: pd.Series, severity: str, reason: str, fix: str) -> pd.DataFrame:
    rows = actions[mask].copy()
    if rows.empty:
        return rows
    rows["_severity"] = severity
    rows["_reason"] = reason
    rows["_fix"] = fix
    return rows


def _self_check_fix_summary(detail: pd.DataFrame) -> str:
    if detail.empty:
        return "诊断自检通过，未发现明显规则冲突。"
    high_count = int((detail["严重程度"] == "高风险").sum())
    if high_count:
        return "存在高风险异常，建议先检查否定、暂停和数据充分性门禁。"
    return "存在轻量警告，建议复核优先级评分和规则解释。"


def _diagnose_campaign_row(
    row: pd.Series,
    config: DiagnosisConfig,
    account_context: AccountContext,
    raw_df: pd.DataFrame,
) -> list[dict[str, object]]:
    actions = []
    spend = _value(row, CANONICAL_FIELDS["spend"])
    sales = _value(row, CANONICAL_FIELDS["sales"])
    orders = _value(row, CANONICAL_FIELDS["orders"])
    acos = _value(row, "ACOS")
    impressions = _value(row, CANONICAL_FIELDS["impressions"])
    clicks = _value(row, CANONICAL_FIELDS["clicks"])
    budget = _campaign_budget(row, raw_df)
    sufficiency = assess_data_sufficiency(row, {"config": config, "account_context": account_context})
    weak_child_ratio, weak_child_count = _weak_child_stats(
        raw_df,
        [CANONICAL_FIELDS["campaign_name"]],
        row,
        config,
        account_context,
    )
    has_protected_child = _parent_has_protected_term(raw_df, [CANONICAL_FIELDS["campaign_name"]], row, config.protected_terms)

    if budget > 0 and spend >= budget * config.budget_pressure_ratio and orders >= 2 and acos <= config.target_acos:
        actions.append(
            _build_action(
                row,
                rule="预算可能不足的广告活动",
                issue_type="优质低 ACOS 机会",
                action="增加预算",
                level="广告活动",
                reason=f"花费 {spend:.2f} 已接近预算 {budget:.2f}，且 ACOS {format_percent(acos)} 不高于目标。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="中",
                risk="高",
                explanation="广告活动已有订单且成本效率在目标内，同时花费接近预算，可能存在预算限制。",
                execution="检查 Campaign 是否受预算限制；如确实受限，可小幅增加预算并观察 ACOS。",
                review="如果没有预算限制，不要仅凭花费接近预算就加预算。",
            )
        )

    if (
        orders == 0
        and sales == 0
        and clicks >= 30
        and sufficiency.data_sufficiency == "充分"
        and spend >= max(config.min_waste_spend * 3, account_context.target_cpa * config.pause_spend_multiplier)
        and weak_child_count >= 2
        and weak_child_ratio >= 0.6
        and not has_protected_child
    ):
        actions.append(
            _build_action(
                row,
                rule="暂停候选广告活动",
                issue_type="高花费无转化",
                action="暂停",
                level="广告活动",
                reason=f"广告活动花费 {spend:.2f}、点击 {clicks:.0f} 且无订单，且多数子对象表现较差。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="中",
                risk="中",
                evidence=f"{sufficiency.explanation}；低效子对象占比约 {weak_child_ratio:.0%}。",
                explanation="该广告活动整体消耗较高但没有订单，且低效对象占比较高，继续投放可能持续消耗预算。",
                execution="暂停前先复核活动结构、投放目标和库存利润；确认无战略测试价值后再暂停或重构。",
                review="不要因为单个搜索词差就暂停整个 Campaign。",
            )
        )
    elif orders == 0 and clicks >= config.hard_waste_clicks and spend >= config.min_waste_spend * 3:
        actions.append(
            _build_action(
                row,
                rule="广告活动无订单消耗复核",
                issue_type="高花费无转化",
                action="降低竞价",
                level="广告活动",
                reason=f"广告活动花费 {spend:.2f}、点击 {clicks:.0f} 且无订单，但暂未达到暂停条件。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence=sufficiency.confidence_base,
                risk="中",
                explanation="广告活动层级出现无订单消耗，但暂停需要更充分证据，建议先复核结构和低效对象。",
                execution="检查该 Campaign 下的搜索词和 Targeting，先处理低效对象或降低整体出价。",
                review="不要因为单个对象表现差就暂停整个 Campaign。",
            )
        )

    if orders >= config.min_quality_orders and 0 < acos <= config.target_acos and impressions < config.low_impressions:
        actions.append(
            _build_action(
                row,
                rule="有销量但曝光少",
                issue_type="优质低 ACOS 机会",
                action="增加预算",
                level="广告活动",
                reason=f"广告活动已有 {orders:.0f} 个订单但曝光 {impressions:.0f} 偏少，可检查预算或扩大流量。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="中",
                risk="高",
                explanation="广告活动已有订单且效率不差，曝光偏少时可以考虑扩量。",
                execution="优先检查预算是否受限，再考虑小幅加预算或拓展流量。",
                review="加预算属于放量动作，可能带来 ACOS 上升。",
            )
        )

    return actions


def _diagnose_ad_group_row(
    row: pd.Series,
    config: DiagnosisConfig,
    account_context: AccountContext,
    raw_df: pd.DataFrame,
) -> list[dict[str, object]]:
    actions = []
    spend = _value(row, CANONICAL_FIELDS["spend"])
    orders = _value(row, CANONICAL_FIELDS["orders"])
    clicks = _value(row, CANONICAL_FIELDS["clicks"])
    acos = _value(row, "ACOS")
    sufficiency = assess_data_sufficiency(row, {"config": config, "account_context": account_context})
    weak_child_ratio, weak_child_count = _weak_child_stats(
        raw_df,
        [CANONICAL_FIELDS["campaign_name"], CANONICAL_FIELDS["ad_group_name"]],
        row,
        config,
        account_context,
    )
    has_protected_child = _parent_has_protected_term(
        raw_df,
        [CANONICAL_FIELDS["campaign_name"], CANONICAL_FIELDS["ad_group_name"]],
        row,
        config.protected_terms,
    )

    if (
        orders == 0
        and clicks >= 30
        and sufficiency.data_sufficiency == "充分"
        and spend >= max(config.min_waste_spend * 3, account_context.target_cpa * config.pause_spend_multiplier)
        and weak_child_count >= 2
        and weak_child_ratio >= 0.6
        and not has_protected_child
    ):
        actions.append(
            _build_action(
                row,
                rule="暂停候选广告组",
                issue_type="高花费无转化",
                action="暂停",
                level="广告组",
                reason=f"广告组点击 {clicks:.0f}、花费 {spend:.2f} 且无订单，且多数子对象表现较差。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="中",
                risk="中",
                evidence=f"{sufficiency.explanation}；低效子对象占比约 {weak_child_ratio:.0%}。",
                explanation="该广告组整体消耗较高但没有订单，且低效对象占比较高，适合作为暂停或重构候选。",
                execution="暂停前先复核广告组内关键词、Targeting 和商品承接；确认结构低效后暂停或重建。",
                review="有订单广告组不要直接暂停，优先拆分或调价。",
            )
        )
    elif orders == 0 and clicks >= config.hard_waste_clicks and spend >= config.min_waste_spend * 3:
        actions.append(
            _build_action(
                row,
                rule="广告组无订单消耗复核",
                issue_type="高花费无转化",
                action="降低竞价",
                level="广告组",
                reason=f"广告组点击 {clicks:.0f}、花费 {spend:.2f} 且无订单，但暂未达到暂停条件。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence=sufficiency.confidence_base,
                risk="中",
                explanation="广告组层级出现无订单消耗，但暂停需要更充分证据，建议先复核关键词、Targeting 和商品承接。",
                execution="优先处理广告组内低效对象，或小幅降低广告组整体出价。",
                review="确认不是新品测试或战略流量后再采取更强动作。",
            )
        )

    if orders >= config.min_quality_orders and 0 < acos <= config.target_acos * config.low_acos_multiplier:
        actions.append(
            _build_action(
                row,
                rule="低 ACOS 优质广告组",
                issue_type="优质低 ACOS 机会",
                action="增加预算",
                level="广告组",
                reason=f"广告组订单 {orders:.0f} 且 ACOS {format_percent(acos)} 健康，可承接更多预算。",
                account_context=account_context,
                data_sufficiency=sufficiency,
                confidence="中",
                risk="高",
                explanation="广告组转化效率较好，可能具备扩量空间。",
                execution="检查是否受预算或出价限制，可小幅加预算或提高优质词出价。",
                review="扩量后需要持续观察 ACOS。",
            )
        )

    return actions


def _build_action(
    row: pd.Series,
    rule: str,
    issue_type: str,
    action: str,
    level: str,
    reason: str,
    account_context: AccountContext,
    data_sufficiency: DataSufficiency,
    confidence: str | None = None,
    risk: str = "中",
    evidence: str | None = None,
    explanation: str = "",
    execution: str = "",
    review: str = "",
) -> dict[str, object]:
    campaign = _text(row, CANONICAL_FIELDS["campaign_name"])
    ad_group = _text(row, CANONICAL_FIELDS["ad_group_name"])
    search_term = _text(row, CANONICAL_FIELDS["customer_search_term"])
    targeting = _text(row, CANONICAL_FIELDS["targeting"])
    diagnosis_object = _diagnosis_object(level, campaign, ad_group, search_term, targeting, row)
    confidence_value = confidence or data_sufficiency.confidence_base
    protected = is_protected_term(" ".join([search_term, targeting]), account_context.protected_terms) if hasattr(account_context, "protected_terms") else False
    metrics_evidence = _metrics_evidence(row, account_context)
    stronger_reason, weaker_reason = _action_strength_explanation(action, row, data_sufficiency, protected)
    review_reason = _manual_review_reason(confidence_value, risk, row, protected, data_sufficiency, action)

    return {
        "诊断规则": rule,
        "问题类型": issue_type,
        "建议动作": action,
        "合并动作": action,
        "优先级": "低",
        "优先级评分": 0,
        "置信度": confidence_value,
        "操作风险": risk,
        "数据充分性": data_sufficiency.data_sufficiency,
        "诊断严格度": account_context.diagnosis_strictness if hasattr(account_context, "diagnosis_strictness") else DEFAULT_DIAGNOSIS_STRICTNESS,
        "诊断层级": level,
        "诊断对象": diagnosis_object,
        "原因": reason,
        "命中规则": rule,
        "命中条件": _rule_condition_text(rule),
        "关键证据": metrics_evidence,
        "为什么不是更强动作": stronger_reason,
        "为什么不是更弱动作": weaker_reason,
        "证据说明": evidence or f"{data_sufficiency.explanation}；{metrics_evidence}",
        "运营解释": explanation or reason,
        "人工复核原因": review_reason,
        "执行建议": execution or action,
        "复核提醒": review or "执行前请结合库存、利润、活动目标和广告预算确认。",
        "目标 ACOS": account_context.target_acos,
        "目标 CPA": account_context.target_cpa,
        "账户平均 CTR": account_context.account_ctr,
        "账户平均 CVR": account_context.account_cvr,
        "是否保护词": "是" if protected else "否",
        "是否存在规则冲突": "否",
        "Campaign Name": campaign,
        "Ad Group Name": ad_group,
        "Customer Search Term": search_term,
        "Targeting": targeting,
        "Match Type": _text(row, CANONICAL_FIELDS["match_type"]),
        "Ad Product": _text(row, CANONICAL_FIELDS["ad_product"]),
        "Advertised ASIN": _text(row, CANONICAL_FIELDS["advertised_asin"]),
        "Purchased ASIN": _text(row, CANONICAL_FIELDS["purchased_asin"]),
        "Impressions": _value(row, CANONICAL_FIELDS["impressions"]),
        "Clicks": _value(row, CANONICAL_FIELDS["clicks"]),
        "Spend": _value(row, CANONICAL_FIELDS["spend"]),
        "Sales": _value(row, CANONICAL_FIELDS["sales"]),
        "Orders": _value(row, CANONICAL_FIELDS["orders"]),
        "CTR": _value(row, "CTR"),
        "CPC": _value(row, "CPC"),
        "CVR": _value(row, "CVR"),
        "ACOS": _value(row, "ACOS"),
        "ROAS": _value(row, "ROAS"),
        "Source Report": _text(row, CANONICAL_FIELDS["source_report"]),
    }


def deduplicate_and_resolve_conflicts(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    allowed = recommendations[recommendations.apply(_is_allowed_action, axis=1)].copy()
    if allowed.empty:
        return allowed

    deduped_rows = []
    identity_columns = ["诊断层级", "诊断对象", "Campaign Name", "Ad Group Name"]
    for _, group in allowed.groupby(identity_columns, dropna=False):
        ranked = group.copy()
        ranked["动作排序"] = ranked["建议动作"].map(ACTION_RANK).fillna(99)
        ranked = ranked.sort_values(["动作排序", "Spend", "Clicks"], ascending=[True, False, False])
        selected = ranked.iloc[0].copy()
        selected["诊断规则"] = _join_unique(group["诊断规则"])
        selected["问题类型"] = _join_unique(group["问题类型"])
        selected["合并动作"] = _join_unique(group["建议动作"])
        selected["原因"] = _join_unique(group["原因"])
        selected["证据说明"] = _join_unique(group["证据说明"])
        selected["运营解释"] = _join_unique(group["运营解释"])
        selected["人工复核原因"] = _join_unique(group["人工复核原因"])
        selected["执行建议"] = _join_unique(group["执行建议"])
        selected["复核提醒"] = _join_unique(group["复核提醒"])
        selected["命中规则"] = _join_unique(group["命中规则"])
        selected["命中条件"] = _join_unique(group["命中条件"])
        selected["关键证据"] = _join_unique(group["关键证据"])
        selected["为什么不是更强动作"] = _join_unique(group["为什么不是更强动作"])
        selected["为什么不是更弱动作"] = _join_unique(group["为什么不是更弱动作"])
        selected["置信度"] = _merge_confidence(group["置信度"])
        selected["操作风险"] = _merge_risk(group["操作风险"])
        selected["是否存在规则冲突"] = _detect_rule_conflict(selected)
        deduped_rows.append(selected)

    return pd.DataFrame(deduped_rows).reset_index(drop=True)


def calculate_priority_score(
    row: pd.Series,
    issue_type: str,
    action_type: str,
    account_context: AccountContext,
    config: DiagnosisConfig | None = None,
) -> int:
    config = config or DiagnosisConfig()
    action = action_type or _text(row, "建议动作")
    spend = _value(row, CANONICAL_FIELDS["spend"])
    clicks = _value(row, CANONICAL_FIELDS["clicks"])
    orders = _value(row, CANONICAL_FIELDS["orders"])
    acos = _value(row, "ACOS")
    sufficiency = _text(row, "数据充分性")
    confidence = _text(row, "置信度")

    if "高花费无转化" in issue_type:
        score = 70
    elif "高 ACOS" in issue_type:
        score = 60
    elif "优质低 ACOS" in issue_type:
        score = 65
    elif "高曝光低 CTR" in issue_type:
        score = 45
    elif "高 CTR 低 CVR" in issue_type:
        score = 55
    elif "数据不足" in issue_type:
        score = 20
    else:
        score = 40

    spend_base = account_context.target_cpa if account_context.target_cpa > 0 else config.min_waste_spend
    score += min(spend / max(spend_base, 1) * 5, 16)
    score += min(clicks / 15 * 4, 8)
    if acos > config.target_acos and config.target_acos > 0:
        score += min((acos / config.target_acos - 1) * 7, 12)
    if action in {"提高竞价", "增加预算", "提取精准投放"} and orders > 0 and 0 < acos <= config.target_acos:
        score += min(orders * 3, 10)

    if sufficiency == "不足":
        score -= 30
    elif sufficiency == "一般":
        score -= 10
    if orders > 0 and action in {"暂停", "否定精准", "否定词组"}:
        score -= 20
    if confidence == "低":
        score -= 20
    if action == "继续观察":
        score = min(score, 35)

    return int(max(0, min(round(score), 100)))


def _campaign_budget(row: pd.Series, raw_df: pd.DataFrame) -> float:
    campaign = _text(row, CANONICAL_FIELDS["campaign_name"])
    budget_column = CANONICAL_FIELDS["budget"]
    if not campaign or budget_column not in raw_df.columns:
        return 0.0

    campaign_rows = raw_df[raw_df[CANONICAL_FIELDS["campaign_name"]].astype(str) == campaign]
    if campaign_rows.empty:
        return 0.0
    return float(campaign_rows[budget_column].max())


def _diagnosis_object(
    level: str,
    campaign: str,
    ad_group: str,
    search_term: str,
    targeting: str,
    row: pd.Series,
) -> str:
    if level == "广告活动":
        return campaign
    if level == "广告组":
        return " / ".join(part for part in [campaign, ad_group] if part)
    if level == "搜索词":
        return search_term or targeting
    if level == "Targeting":
        return targeting or search_term
    asin = _text(row, "ASIN")
    return asin or search_term or targeting or campaign


def _action_mask(actions: pd.DataFrame, expected_actions: set[str]) -> pd.Series:
    if actions.empty:
        return pd.Series(dtype=bool)
    primary = actions["建议动作"].isin(expected_actions)
    if "合并动作" not in actions.columns:
        return primary
    merged = actions["合并动作"].fillna("").astype(str).apply(
        lambda value: any(action in value for action in expected_actions)
    )
    return primary | merged


def _count_actions(actions: pd.DataFrame) -> Counter:
    counter: Counter = Counter()
    if actions.empty:
        return counter
    for _, row in actions.iterrows():
        merged = _text(row, "合并动作") or _text(row, "建议动作")
        matched = [action for action in ACTIONS if action in merged]
        if not matched:
            matched = [_text(row, "建议动作")]
        counter.update(action for action in matched if action)
    return counter


def _source_action(row: pd.Series, expected_actions: set[str]) -> str:
    primary = _text(row, "建议动作")
    if primary in expected_actions:
        return primary
    merged = _text(row, "合并动作")
    for action in ACTIONS:
        if action in expected_actions and action in merged:
            return action
    return primary


def _bid_direction(row: pd.Series) -> str:
    source_action = _source_action(row, {"降低竞价", "提高竞价"})
    return {"降低竞价": "降低", "提高竞价": "提高"}.get(source_action, "")


def _negative_match_type(row: pd.Series) -> str:
    source_action = _source_action(row, {"否定精准", "否定词组"})
    return {"否定精准": "Negative Exact", "否定词组": "Negative Phrase"}.get(source_action, "")


def is_protected_term(search_term: str, protected_terms: tuple[str, ...] | list[str]) -> bool:
    normalized = str(search_term or "").lower().strip()
    if not normalized:
        return False
    return any(term.lower().strip() in normalized for term in protected_terms if term and term.strip())


def _weak_child_stats(
    raw_df: pd.DataFrame,
    parent_columns: list[str],
    parent_row: pd.Series,
    config: DiagnosisConfig,
    account_context: AccountContext,
) -> tuple[float, int]:
    if raw_df.empty:
        return 0.0, 0
    filtered = _filter_parent_rows(raw_df, parent_columns, parent_row)
    if filtered.empty:
        return 0.0, 0

    children = _child_dimension_rows(filtered, parent_columns)
    if children.empty:
        return 0.0, 0

    target_cpa = account_context.target_cpa if account_context.target_cpa > 0 else config.min_waste_spend * 2
    weak_mask = (
        (children[CANONICAL_FIELDS["orders"]].fillna(0).astype(float) <= 0)
        & (children[CANONICAL_FIELDS["clicks"]].fillna(0).astype(float) >= config.min_waste_clicks)
        & (children[CANONICAL_FIELDS["spend"]].fillna(0).astype(float) >= min(target_cpa, config.min_waste_spend * 2))
    )
    weak_count = int(weak_mask.sum())
    return float(weak_count / len(children)), weak_count


def _parent_has_protected_term(
    raw_df: pd.DataFrame,
    parent_columns: list[str],
    parent_row: pd.Series,
    protected_terms: tuple[str, ...] | list[str],
) -> bool:
    if not protected_terms:
        return False
    filtered = _filter_parent_rows(raw_df, parent_columns, parent_row)
    if filtered.empty:
        return False
    for column in [CANONICAL_FIELDS["customer_search_term"], CANONICAL_FIELDS["targeting"]]:
        if column not in filtered.columns:
            continue
        if filtered[column].fillna("").astype(str).apply(lambda value: is_protected_term(value, protected_terms)).any():
            return True
    return False


def _filter_parent_rows(raw_df: pd.DataFrame, parent_columns: list[str], parent_row: pd.Series) -> pd.DataFrame:
    filtered = raw_df.copy()
    for column in parent_columns:
        if column not in filtered.columns:
            return pd.DataFrame()
        parent_value = _text(parent_row, column)
        filtered = filtered[filtered[column].fillna("").astype(str).str.strip() == parent_value]
    return filtered


def _child_dimension_rows(filtered: pd.DataFrame, parent_columns: list[str]) -> pd.DataFrame:
    child_columns = [
        column
        for column in [CANONICAL_FIELDS["customer_search_term"], CANONICAL_FIELDS["targeting"]]
        if column in filtered.columns and filtered[column].fillna("").astype(str).str.strip().ne("").any()
    ]
    if not child_columns:
        return pd.DataFrame()

    child_frames = [aggregate_by_dimension(filtered, parent_columns + [column], "子对象") for column in child_columns]
    return pd.concat(child_frames, ignore_index=True).drop_duplicates()


def _looks_irrelevant(
    row: pd.Series,
    ctr: float,
    cvr: float,
    impressions: float,
    clicks: float,
) -> bool:
    search_text = " ".join(
        [
            _text(row, CANONICAL_FIELDS["customer_search_term"]),
            _text(row, CANONICAL_FIELDS["targeting"]),
        ]
    ).lower()
    if any(term in search_text for term in IRRELEVANT_TERM_HINTS):
        return True
    return clicks >= 20 and impressions >= 1000 and ctr < 0.002 and cvr == 0


def _bid_down_execution(acos: float, target_acos: float) -> str:
    ratio = acos / max(target_acos, 0.01)
    if ratio <= 1.5:
        return "建议小幅降低竞价 5%-10%，不要直接暂停或否定。"
    if ratio <= 2:
        return "建议降低竞价 10%-20%，并复核搜索词相关性。"
    return "建议降低竞价 20%-30% 或重新评估相关性，但不要直接否定已有订单对象。"


def _metrics_evidence(row: pd.Series, account_context: AccountContext) -> str:
    parts = [
        f"曝光 {_value(row, CANONICAL_FIELDS['impressions']):.0f}",
        f"点击 {_value(row, CANONICAL_FIELDS['clicks']):.0f}",
        f"花费 ${_value(row, CANONICAL_FIELDS['spend']):.2f}",
        f"销售额 ${_value(row, CANONICAL_FIELDS['sales']):.2f}",
        f"订单 {_value(row, CANONICAL_FIELDS['orders']):.0f}",
        f"ACOS {format_percent(_value(row, 'ACOS'))}",
        f"CVR {format_percent(_value(row, 'CVR'))}",
        f"目标 CPA ${account_context.target_cpa:.2f}",
    ]
    return "；".join(parts)


def _rule_condition_text(rule: str) -> str:
    if "高 ACOS" in rule:
        return "Orders >= 1 且 ACOS 高于目标 ACOS"
    if "无订单" in rule or "否定" in rule:
        return "Orders = 0，点击和花费达到判断阈值，并结合目标 CPA 与相关性判断"
    if "暂停" in rule:
        return "Campaign / Ad Group 无订单无销售，点击和花费充分，且多数子对象表现差"
    if "低 ACOS" in rule or "精准投放" in rule or "高 CVR" in rule:
        return "Orders 达到放量阈值，ACOS 低于目标，或 CVR 明显高于账户平均"
    if "低 CTR" in rule:
        return "Impressions 较高且 CTR 低于账户平均或低 CTR 阈值"
    if "高 CTR 低 CVR" in rule:
        return "CTR 高于账户平均，但 CVR 明显低于账户平均"
    if "数据不足" in rule:
        return "Clicks、Spend 或 Impressions 未达到最小判断样本"
    return "命中当前诊断规则阈值"


def _action_strength_explanation(
    action: str,
    row: pd.Series,
    data_sufficiency: DataSufficiency,
    protected: bool,
) -> tuple[str, str]:
    orders = _value(row, CANONICAL_FIELDS["orders"])
    if action in {"继续观察", "检查 Listing"}:
        stronger = "当前证据不足以执行否定、暂停或大幅调价。"
        weaker = "已有异常信号，完全忽略会延误复盘。"
    elif action == "降低竞价":
        stronger = "该对象已有订单或相关性不确定，不建议直接否定或暂停。"
        weaker = "ACOS 或无订单消耗已经出现压力，仅继续观察可能扩大花费。"
    elif action in {"否定精准", "否定词组"}:
        stronger = "搜索词层级处理即可，不应升级为暂停整个广告活动。"
        weaker = "点击和花费样本已充分，继续观察可能继续浪费预算。"
    elif action == "暂停":
        stronger = "暂停已经是强动作，执行前仍需人工复核结构和目标。"
        weaker = "父层级整体无订单且多数子对象低效，仅降价可能止损较慢。"
    else:
        stronger = "放量动作有风险，不建议一次性大幅加预算或大幅提价。"
        weaker = "已有订单和效率优势，仅观察可能错过放量机会。"
    if orders > 0 and action in {"降低竞价", "提高竞价", "增加预算", "提取精准投放"}:
        stronger += " 已有订单对象受安全规则保护，不直接否定或暂停。"
    if data_sufficiency.data_sufficiency == "不足":
        stronger = "数据不足，安全规则禁止升级为强动作。"
    if protected:
        stronger += " 包含保护词，安全规则禁止否定或暂停。"
    return stronger, weaker


def _manual_review_reason(
    confidence: str,
    risk: str,
    row: pd.Series,
    protected: bool,
    data_sufficiency: DataSufficiency,
    action: str,
) -> str:
    reasons = []
    orders = _value(row, CANONICAL_FIELDS["orders"])
    if confidence in {"低", "中"}:
        reasons.append("置信度不是高，需要人工复核证据。")
    if risk in {"中", "高"}:
        reasons.append("操作风险不低，执行前需结合利润、库存和活动目标。")
    if data_sufficiency.data_sufficiency == "不足":
        reasons.append("数据量不足，建议继续观察。")
    if orders > 0:
        reasons.append("该对象已有订单，调整前需确认利润和转化稳定性。")
    if action in {"否定精准", "否定词组"}:
        reasons.append("否定前需人工判断搜索词相关性。")
    if protected:
        reasons.append("该词可能是品牌词或核心词，建议谨慎处理。")
    return "；".join(reasons) or "规则证据较充分，按执行建议小步操作并复盘结果。"


def _detect_rule_conflict(row: pd.Series) -> str:
    action_text = f"{_text(row, '建议动作')} {_text(row, '合并动作')}"
    orders = _value(row, CANONICAL_FIELDS["orders"])
    acos = _value(row, "ACOS")
    target_acos = _value(row, "目标 ACOS")
    data_sufficiency = _text(row, "数据充分性")
    if orders > 0 and any(action in action_text for action in ["否定", "暂停"]):
        return "是"
    if data_sufficiency == "不足" and any(action in action_text for action in ["否定", "暂停"]):
        return "是"
    if target_acos and 0 < acos <= target_acos * 0.7 and "降低竞价" in action_text:
        return "是"
    return "否"


def _is_allowed_action(row: pd.Series) -> bool:
    action_text = f"{_text(row, '建议动作')} {_text(row, '合并动作')}"
    orders = _value(row, CANONICAL_FIELDS["orders"])
    data_sufficiency = _text(row, "数据充分性")
    if orders > 0 and any(action in action_text for action in ["否定精准", "否定词组", "暂停"]):
        return False
    if data_sufficiency == "不足" and any(action in action_text for action in ["暂停", "否定精准", "否定词组"]):
        return False
    return True


def _priority_label_from_score(score: float) -> str:
    if score >= 80:
        return "高"
    if score >= 50:
        return "中"
    return "低"


def _merge_confidence(values: pd.Series) -> str:
    rank = {"低": 0, "中": 1, "高": 2}
    reverse = {value: key for key, value in rank.items()}
    scores = [rank.get(str(value), 1) for value in values]
    return reverse.get(min(scores) if scores else 1, "中")


def _merge_risk(values: pd.Series) -> str:
    rank = {"低": 0, "中": 1, "高": 2}
    reverse = {value: key for key, value in rank.items()}
    scores = [rank.get(str(value), 1) for value in values]
    return reverse.get(max(scores) if scores else 1, "中")


def _join_unique(values: pd.Series) -> str:
    seen = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return "；".join(seen)


def _value(row: pd.Series, column: str) -> float:
    if column not in row.index or pd.isna(row[column]):
        return 0.0
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return 0.0


def _text(row: pd.Series, column: str) -> str:
    if column not in row.index or pd.isna(row[column]):
        return ""
    text = str(row[column]).strip()
    return "" if text == "(空)" else text
