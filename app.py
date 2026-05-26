from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from modules.aggregation import build_dimension_aggregations
from modules.ai_report import generate_ai_report, report_to_dataframe
from modules.basic_data_audit import (
    AccountSummarySource,
    account_summary_source_note,
    build_file_audit,
    duplicate_metric_guard_messages,
    report_type_display,
    run_basic_data_audit,
    select_account_summary_source,
)
from modules.action_overload import enrich_action_prioritization
from modules.data_safety import (
    DataTrustResult,
    DiagnosisSafetyGateResult,
    ReconciliationInput,
    ReconciliationResult,
    apply_diagnosis_safety_to_actions,
    calculate_data_trust_score,
    data_trust_dataframe,
    ensure_feedback_columns,
    operator_feedback_dataframe,
    reconcile_external_totals,
    reconciliation_dataframe,
    rules_version_dataframe,
    run_diagnosis_safety_gate,
    safety_gate_dataframe,
    write_diagnosis_audit_report,
)
from modules.data_loader import read_report
from modules.deepseek_client import DEEPSEEK_MODELS, generate_deepseek_report
from modules.diagnosis import (
    DiagnosisConfig,
    build_bid_adjustments,
    build_exact_targeting_opportunities,
    build_growth_list,
    build_negative_keywords,
    build_pause_list,
    build_priority_list,
    run_diagnosis_self_check,
    run_diagnosis,
    summarize_recommendations,
)
from modules.exporter import build_excel_report
from modules.field_mapping import CANONICAL_FIELDS, apply_field_mapping, mapping_results, mapping_results_dataframe
from modules.field_mapping import infer_report_type, missing_required_fields
from modules.metrics import add_metrics, calculate_account_overview, format_percent, overview_dataframe
from modules.pivot import ACTION_PIVOT_PRESETS, build_action_pivot, build_export_pivots
from modules.rules_config import DEFAULT_DIAGNOSIS_STRICTNESS, DIAGNOSIS_ENGINE_VERSION, RULE_CONFIG_VERSION, STRICTNESS_OPTIONS
from modules.settings import AppSettings


st.set_page_config(
    page_title="亚马逊广告诊断工具",
    page_icon="AD",
    layout="wide",
)


DISPLAY_NAME_MAP = {
    "Report": "报表",
    "Campaign Name": "广告活动名称",
    "Ad Group Name": "广告组名称",
    "Customer Search Term": "客户搜索词",
    "Targeting": "投放定向",
    "Match Type": "匹配类型",
    "Impressions": "曝光量",
    "Clicks": "点击量",
    "Spend": "花费",
    "Sales": "销售额",
    "Orders": "订单量",
    "Ad Product": "广告产品",
    "Advertised ASIN": "广告 ASIN",
    "Purchased ASIN": "成交 ASIN",
    "Budget": "预算",
    "Campaign Status": "广告活动状态",
    "Ad Group Status": "广告组状态",
    "Source Report": "来源报表",
    "CTR": "点击率",
    "CPC": "平均点击花费",
    "CVR": "转化率",
    "ACOS": "广告成本销售比",
    "ROAS": "广告回报率",
    "Reason": "原因",
    "Source Action": "来源动作",
    "Negative Keyword": "否定词",
    "Negative Match Type": "否定匹配类型",
    "ASIN Type": "ASIN 类型",
}

DISPLAY_REPORT_TYPE_MAP = {
    "SP_SEARCH_TERM_REPORT": "商品推广搜索词报表",
    "SP_TARGETING_REPORT": "商品推广投放报表",
    "SP_CAMPAIGN_REPORT": "商品推广广告活动报表",
    "SP_BULK_FILE": "Bulk 文件",
    "SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS": "热门搜索词 / Search Query 报告",
    "UNKNOWN": "未识别报表",
    "Search Term Report": "搜索词报表",
    "Targeting Report": "定向报表",
    "Unknown Report": "未识别报表",
}

PIVOT_VIEW_PRESETS = {
    "日常巡检": {
        "dimension": "广告活动",
        "priorities": ["高", "中"],
        "actions": [],
        "caption": "先看活动层面的风险规模和消耗集中度，适合每天打开后快速扫一遍。",
    },
    "先止损": {
        "dimension": "建议动作",
        "priorities": ["高"],
        "actions": ["暂停", "否定精准", "否定词组", "降低竞价"],
        "caption": "聚焦高优先级止损动作，优先处理高花费、无转化或 ACOS 明显偏高对象。",
    },
    "查承接": {
        "dimension": "建议动作",
        "priorities": ["高", "中"],
        "actions": ["检查 Listing", "继续观察"],
        "caption": "适合排查主图、标题、价格、评价和详情页承接问题，避免只调广告不看转化。",
    },
    "找增长": {
        "dimension": "建议动作",
        "priorities": ["中", "低"],
        "actions": ["提高竞价", "增加预算", "提取精准投放"],
        "caption": "把低 ACOS、有订单、可放量的对象集中出来，避免只做止损不做增长。",
    },
}


@dataclass
class AnalysisState:
    settings: AppSettings
    file_signature: tuple[tuple[str, int], ...]
    uploaded_count: int
    file_summaries: list[dict[str, object]]
    mapping_df: pd.DataFrame
    cleaned_data: pd.DataFrame
    enriched_data: pd.DataFrame
    report_frames: list[dict[str, object]]
    file_audit: pd.DataFrame
    account_summary_source: AccountSummarySource | None
    account_summary_note: pd.DataFrame
    basic_data_audit: pd.DataFrame
    duplicate_guard_messages: list[str]
    data_trust_result: DataTrustResult
    data_trust_df: pd.DataFrame
    reconciliation_result: ReconciliationResult
    reconciliation_df: pd.DataFrame
    safety_gate: DiagnosisSafetyGateResult
    safety_gate_df: pd.DataFrame
    rules_version_df: pd.DataFrame
    audit_report_path: str
    generated_at: datetime
    aggregations: dict[str, pd.DataFrame]
    data_quality_notes: list[str]
    action_pivots: dict[str, pd.DataFrame]
    overview: dict[str, float]
    overview_df: pd.DataFrame
    actions: pd.DataFrame
    summary: dict[str, object]
    negative_keywords: pd.DataFrame
    bid_adjustments: pd.DataFrame
    pause_list: pd.DataFrame
    growth_list: pd.DataFrame
    exact_opportunities: pd.DataFrame
    priority_list: pd.DataFrame
    self_check: dict[str, object]
    ai_report_sections: list[dict[str, str]]


def main() -> None:
    inject_styles()

    settings, uploaded_files, start_clicked = render_sidebar_controls()
    signature = file_signature(uploaded_files)
    if st.session_state.get("active_file_signature") != signature:
        st.session_state.pop("analysis_state", None)
        clear_cached_exports()
        st.session_state["active_file_signature"] = signature

    cached_state = st.session_state.get("analysis_state")
    should_load_reports = bool(uploaded_files) and (
        start_clicked
        or cached_state is None
        or settings.manual_mapping_enabled
    )
    if should_load_reports:
        loaded_reports, file_summaries = load_reports(uploaded_files)
    else:
        loaded_reports = []
        file_summaries = cached_state.file_summaries if cached_state is not None else []
    manual_mappings = render_manual_mapping_controls(loaded_reports, settings.manual_mapping_enabled)

    render_hero(settings, len(uploaded_files))

    if not uploaded_files:
        render_empty_state()
        return

    if not start_clicked and "analysis_state" not in st.session_state:
        render_upload_status(file_summaries)
        render_pre_diagnosis_check(loaded_reports, file_summaries, manual_mappings, settings)
        render_waiting_state()
        return

    if start_clicked or "analysis_state" not in st.session_state:
        if not loaded_reports:
            render_upload_status(file_summaries)
            st.error("没有可分析的有效文件，请检查上传文件。")
            return
        clear_cached_exports()
        st.session_state.pop("diagnosis_error", None)
        st.session_state.pop("diagnosis_error_detail", None)
        with st.spinner("正在识别字段、计算指标并生成诊断..."):
            reset_report_view_state()
            try:
                st.session_state["analysis_state"] = build_analysis_state(
                    settings=settings,
                    uploaded_files=uploaded_files,
                    file_signature_value=signature,
                    loaded_reports=loaded_reports,
                    file_summaries=file_summaries,
                    manual_mappings=manual_mappings,
                )
            except Exception as exc:
                st.session_state["diagnosis_error"] = f"{type(exc).__name__}: {exc}"
                st.session_state["diagnosis_error_detail"] = format_exception_detail(exc)
                st.rerun()
                return

    if st.session_state.get("diagnosis_error"):
        render_diagnosis_error()
        return

    render_dashboard_tabs(st.session_state["analysis_state"])


def render_sidebar_controls() -> tuple[AppSettings, list[Any], bool]:
    default = DiagnosisConfig()
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-kicker">LOCAL ANALYTICS</div>
                <div class="sidebar-brand-title">控制中心</div>
                <div class="sidebar-brand-copy">本地运行 · 表格诊断 · AI 复核</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("诊断设置", expanded=True):
            mode = st.radio("诊断模式", ["基础版", "完整版"], index=1, horizontal=True)
            target_acos_percent = st.number_input("目标 ACOS（%）", 1.0, 300.0, 30.0, 1.0)
            rule_preset = st.selectbox(
                "诊断口径",
                ["稳健止损（推荐）", "平衡优化", "积极优化", "自定义高级"],
                index=0,
                help="新手建议使用预设口径，只需要确认目标 ACOS；熟练运营可以切到自定义高级。",
            )
            config = diagnosis_config_for_preset(rule_preset, target_acos_percent / 100, default)
            st.caption(rule_preset_caption(rule_preset))
            if rule_preset != "自定义高级":
                with st.expander("查看口径细节", expanded=False):
                    render_rule_preset_preview(config)

        if rule_preset == "自定义高级":
            with st.expander("高级阈值", expanded=True):
                min_waste_clicks = st.number_input("最低点击阈值", 1, 200, config.min_waste_clicks, 1)
                min_waste_spend = st.number_input("最低花费阈值", 0.0, 10000.0, config.min_waste_spend, 1.0)
                hard_waste_clicks = st.number_input("高点击无转化阈值", 1, 300, config.hard_waste_clicks, 1)
                high_acos_multiplier = st.number_input("高 ACOS 倍数", 1.0, 10.0, config.high_acos_multiplier, 0.05)
                low_acos_multiplier = st.number_input("低 ACOS 倍数", 0.05, 1.0, config.low_acos_multiplier, 0.05)
                min_quality_orders = st.number_input("优质词最低订单", 1, 100, config.min_quality_orders, 1)
                high_ctr_percent = st.number_input("高 CTR 阈值（%）", 0.01, 100.0, config.high_ctr * 100, 0.1)
                low_ctr_percent = st.number_input("低 CTR 阈值（%）", 0.01, 100.0, config.low_ctr * 100, 0.05)
                low_cvr_percent = st.number_input("低 CVR 阈值（%）", 0.01, 100.0, config.low_cvr * 100, 0.5)
                high_impressions = st.number_input("高曝光阈值", 1, 1_000_000, config.high_impressions, 100)
                low_impressions = st.number_input("低曝光阈值", 1, 1_000_000, config.low_impressions, 50)
                min_sales_low_exposure = st.number_input("有销量低曝光销售额阈值", 0.0, 100000.0, config.min_sales_low_exposure, 5.0)
                budget_pressure_percent = st.number_input("预算压力阈值（%）", 1.0, 100.0, config.budget_pressure_ratio * 100, 1.0)
                pause_spend_multiplier = st.number_input("暂停花费倍数", 0.1, 20.0, config.pause_spend_multiplier, 0.1)
                exact_opportunity_orders = st.number_input("精准机会最低订单", 1, 100, config.exact_opportunity_orders, 1)
            config = DiagnosisConfig(
                target_acos=target_acos_percent / 100,
                min_waste_clicks=int(min_waste_clicks),
                hard_waste_clicks=int(hard_waste_clicks),
                min_waste_spend=float(min_waste_spend),
                high_acos_multiplier=float(high_acos_multiplier),
                low_acos_multiplier=float(low_acos_multiplier),
                min_quality_orders=int(min_quality_orders),
                high_ctr=high_ctr_percent / 100,
                low_ctr=low_ctr_percent / 100,
                low_cvr=low_cvr_percent / 100,
                high_impressions=int(high_impressions),
                low_impressions=int(low_impressions),
                min_sales_low_exposure=float(min_sales_low_exposure),
                budget_pressure_ratio=budget_pressure_percent / 100,
                pause_spend_multiplier=float(pause_spend_multiplier),
                exact_opportunity_orders=int(exact_opportunity_orders),
                protected_terms=config.protected_terms,
                diagnosis_strictness=config.diagnosis_strictness,
            )

        with st.expander("上传与运行", expanded=True):
            uploaded_files = st.file_uploader(
                "上传亚马逊广告报表",
                type=["csv", "xlsx", "xls"],
                accept_multiple_files=True,
                key="amazon_ads_reports",
                help="支持搜索词报表、定向报表、广告活动报表，以及 Bulk 表格。",
            )
            render_sidebar_upload_summary(uploaded_files or [])
            if uploaded_files:
                st.markdown(
                    '<div class="sidebar-next-step">下一步：点击下方按钮生成运营动作清单</div>',
                    unsafe_allow_html=True,
                )
            start_clicked = st.button("开始诊断 · 已就绪" if uploaded_files else "开始诊断", type="primary", width="stretch")

        with st.expander("高级选项", expanded=False):
            diagnosis_strictness = st.selectbox(
                "诊断严格度",
                list(STRICTNESS_OPTIONS),
                index=list(STRICTNESS_OPTIONS).index(DEFAULT_DIAGNOSIS_STRICTNESS),
                help="保守更少强动作，激进更快识别浪费项；安全底线始终保留。",
            )
            protected_terms_text = st.text_input(
                "保护词",
                value="",
                placeholder="品牌词、核心词，用逗号分隔",
                help="命中保护词的搜索词不会被直接否定或暂停。",
            )
            manual_mapping_enabled = st.checkbox("启用手动字段映射", value=False)
            ai_report_enabled = st.checkbox("启用本地 AI 报告模板", value=True)
            st.markdown("#### 外部对账")
            external_spend = st.number_input("外部系统总花费", min_value=0.0, value=0.0, step=100.0, help="例如领星 ERP 的广告总花费；不填写则保持 0。")
            external_sales = st.number_input("外部系统总销售额", min_value=0.0, value=0.0, step=100.0, help="例如领星 ERP 的广告销售额；不填写则保持 0。")
            external_orders = st.number_input("外部系统总订单（可选）", min_value=0.0, value=0.0, step=1.0)

        protected_terms = tuple(term.strip() for term in re.split(r"[,，\n]", protected_terms_text) if term.strip())
        config = DiagnosisConfig(
            **{
                **config.__dict__,
                "protected_terms": protected_terms,
                "diagnosis_strictness": diagnosis_strictness,
            }
        )

        if "analysis_state" in st.session_state:
            with st.expander("快捷导出", expanded=False):
                render_excel_download(st.session_state["analysis_state"], key="sidebar_export")

    return (
        AppSettings(
            mode,
            rule_preset,
            manual_mapping_enabled,
            ai_report_enabled,
            config,
            ReconciliationInput(
                external_spend=float(external_spend) if external_spend else None,
                external_sales=float(external_sales) if external_sales else None,
                external_orders=float(external_orders) if external_orders else None,
            ),
        ),
        uploaded_files or [],
        start_clicked,
    )


def diagnosis_config_for_preset(rule_preset: str, target_acos: float, default: DiagnosisConfig) -> DiagnosisConfig:
    values = {
        "target_acos": target_acos,
        "min_waste_clicks": default.min_waste_clicks,
        "hard_waste_clicks": default.hard_waste_clicks,
        "min_waste_spend": default.min_waste_spend,
        "high_acos_multiplier": default.high_acos_multiplier,
        "low_acos_multiplier": default.low_acos_multiplier,
        "min_quality_orders": default.min_quality_orders,
        "high_ctr": default.high_ctr,
        "low_ctr": default.low_ctr,
        "low_cvr": default.low_cvr,
        "high_impressions": default.high_impressions,
        "low_impressions": default.low_impressions,
        "min_sales_low_exposure": default.min_sales_low_exposure,
        "budget_pressure_ratio": default.budget_pressure_ratio,
        "pause_spend_multiplier": default.pause_spend_multiplier,
        "exact_opportunity_orders": default.exact_opportunity_orders,
        "protected_terms": default.protected_terms,
        "diagnosis_strictness": default.diagnosis_strictness,
    }
    if rule_preset == "稳健止损（推荐）":
        values.update(
            min_waste_clicks=8,
            hard_waste_clicks=20,
            min_waste_spend=max(default.min_waste_spend, 8.0),
            high_acos_multiplier=1.6,
            low_acos_multiplier=0.65,
            min_quality_orders=2,
            low_cvr=0.04,
            pause_spend_multiplier=3.0,
            exact_opportunity_orders=2,
        )
    elif rule_preset == "平衡优化":
        values.update(
            min_waste_clicks=5,
            hard_waste_clicks=15,
            min_waste_spend=max(default.min_waste_spend, 5.0),
            high_acos_multiplier=1.3,
            low_acos_multiplier=0.7,
            min_quality_orders=1,
            pause_spend_multiplier=2.0,
            exact_opportunity_orders=1,
        )
    elif rule_preset == "积极优化":
        values.update(
            min_waste_clicks=3,
            hard_waste_clicks=10,
            min_waste_spend=max(default.min_waste_spend, 3.0),
            high_acos_multiplier=1.15,
            low_acos_multiplier=0.8,
            min_quality_orders=1,
            pause_spend_multiplier=1.5,
            exact_opportunity_orders=1,
        )
    return DiagnosisConfig(**values)


def rule_preset_caption(rule_preset: str) -> str:
    captions = {
        "稳健止损（推荐）": "适合新手和日常巡检：减少误杀，优先找明确浪费。",
        "平衡优化": "适合常规运营：止损、调价和放量机会同步输出。",
        "积极优化": "适合成熟运营：更快暴露问题，但需要人工复核后执行。",
        "自定义高级": "适合熟悉业务阈值的运营，可逐项调整规则参数。",
    }
    return captions.get(rule_preset, "")


def render_rule_preset_preview(config: DiagnosisConfig) -> None:
    st.markdown(
        f"""
        <div class="sidebar-rule-card">
            <div><span>低效点击</span><b>≥ {config.min_waste_clicks}</b></div>
            <div><span>强止损点击</span><b>≥ {config.hard_waste_clicks}</b></div>
            <div><span>最低花费</span><b>${config.min_waste_spend:,.0f}</b></div>
            <div><span>高 ACOS</span><b>{config.high_acos_multiplier:.2f} × 目标</b></div>
            <div><span>优质订单</span><b>≥ {config.min_quality_orders}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_upload_summary(uploaded_files: list[Any]) -> None:
    count = len(uploaded_files or [])
    if not count:
        st.markdown(
            '<div class="sidebar-upload-summary">上传后会在本次会话中保留，刷新页面会清空文件。</div>',
            unsafe_allow_html=True,
        )
        return
    total_size = sum(int(getattr(file, "size", 0) or 0) for file in uploaded_files)
    size_mb = total_size / 1024 / 1024
    st.markdown(
        f"""
        <div class="sidebar-upload-summary sidebar-upload-ready">
            <b>已选择 {count} 个文件</b>
            <span>合计 {size_mb:.1f} MB · 可点击开始诊断</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_cached_exports() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("excel_export_bytes_") or str(key).startswith("excel_export_name_"):
            st.session_state.pop(key, None)


def reset_report_view_state() -> None:
    for key in ["ai_report_selected_section", "deepseek_report_section"]:
        st.session_state.pop(key, None)


def format_exception_detail(exc: Exception) -> str:
    import traceback

    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def render_diagnosis_error() -> None:
    st.error("诊断没有完成。请先检查上传文件是否为亚马逊广告报表，或切换到手动字段映射后重试。")
    st.caption(st.session_state.get("diagnosis_error", ""))
    with st.expander("查看技术细节", expanded=False):
        st.code(str(st.session_state.get("diagnosis_error_detail", "")), language="text")
    if st.button("清除错误并重新开始", type="primary"):
        st.session_state.pop("diagnosis_error", None)
        st.session_state.pop("diagnosis_error_detail", None)
        st.rerun()


def build_analysis_state(
    settings: AppSettings,
    uploaded_files: list[Any],
    file_signature_value: tuple[tuple[str, int], ...],
    loaded_reports: list[dict[str, object]],
    file_summaries: list[dict[str, object]],
    manual_mappings: dict[int, dict[str, str]],
) -> AnalysisState:
    generated_at = datetime.now()
    mapping_df, cleaned_data, report_frames = prepare_report_frames(loaded_reports, manual_mappings)
    enriched_data = add_metrics(cleaned_data)
    for report_frame in report_frames:
        report_frame["enriched_data"] = add_metrics(report_frame["cleaned_data"])
    account_summary_source = select_account_summary_source(report_frames)
    overview_source = account_summary_source.dataframe if account_summary_source else pd.DataFrame()
    overview = calculate_account_overview(overview_source)
    file_audit = build_file_audit(report_frames, account_summary_source)
    account_summary_note = account_summary_source_note(account_summary_source)
    basic_data_audit = run_basic_data_audit(report_frames, account_summary_source, overview)
    duplicate_guard_messages = duplicate_metric_guard_messages(report_frames, account_summary_source)
    data_trust_result = calculate_data_trust_score(report_frames, account_summary_source, file_audit, overview)
    reconciliation_result = reconcile_external_totals(overview, settings.reconciliation_input)
    safety_gate = run_diagnosis_safety_gate(
        report_frames,
        overview,
        data_trust_result,
        account_summary_source=account_summary_source,
        file_audit=file_audit,
        reconciliation_result=reconciliation_result,
    )
    data_trust_df = data_trust_dataframe(data_trust_result)
    reconciliation_df = reconciliation_dataframe(reconciliation_result)
    safety_gate_df = safety_gate_dataframe(safety_gate)
    rules_version_df = rules_version_dataframe(DIAGNOSIS_ENGINE_VERSION, RULE_CONFIG_VERSION, generated_at)
    file_summaries = enrich_file_summaries_with_audit(file_summaries, file_audit)
    aggregations = build_dimension_aggregations(enriched_data)
    data_quality_notes = build_data_quality_notes(mapping_df, enriched_data)
    data_quality_notes.extend(duplicate_guard_messages)
    data_quality_notes.extend(data_trust_result.data_quality_warnings)
    data_quality_notes.extend(safety_gate.warning_reasons)
    if safety_gate.can_diagnose:
        actions = run_diagnosis(enriched_data, settings.diagnosis_config, settings.mode)
        actions = enrich_action_prioritization(actions, settings.diagnosis_config)
        actions = apply_diagnosis_safety_to_actions(actions, safety_gate, data_trust_result, reconciliation_result)
    else:
        actions = ensure_feedback_columns(pd.DataFrame())
    summary = summarize_recommendations(actions)
    action_pivots = build_export_pivots(actions)
    negative_keywords = build_negative_keywords(actions)
    bid_adjustments = build_bid_adjustments(actions)
    pause_list = build_pause_list(actions)
    growth_list = build_growth_list(actions)
    exact_opportunities = build_exact_targeting_opportunities(actions)
    priority_list = build_priority_list(actions)
    self_check = run_diagnosis_self_check(actions, settings.diagnosis_config)
    ai_report_sections = (
        generate_ai_report(overview, actions, aggregations, settings.diagnosis_config.target_acos)
        if settings.ai_report_enabled
        else [{"章节": "AI 模板报告", "报告内容": "本次已关闭本地 AI 模板报告。"}]
    )
    audit_report_path = str(Path("outputs") / "diagnosis_audit_report.md")
    write_diagnosis_audit_report(
        audit_report_path,
        file_audit,
        overview,
        account_summary_source,
        data_trust_result,
        safety_gate,
        actions,
        "",
        DIAGNOSIS_ENGINE_VERSION,
        RULE_CONFIG_VERSION,
        generated_at,
    )

    return AnalysisState(
        settings=settings,
        file_signature=file_signature_value,
        uploaded_count=len(uploaded_files),
        file_summaries=file_summaries,
        mapping_df=mapping_df,
        cleaned_data=cleaned_data,
        enriched_data=enriched_data,
        report_frames=report_frames,
        file_audit=file_audit,
        account_summary_source=account_summary_source,
        account_summary_note=account_summary_note,
        basic_data_audit=basic_data_audit,
        duplicate_guard_messages=duplicate_guard_messages,
        data_trust_result=data_trust_result,
        data_trust_df=data_trust_df,
        reconciliation_result=reconciliation_result,
        reconciliation_df=reconciliation_df,
        safety_gate=safety_gate,
        safety_gate_df=safety_gate_df,
        rules_version_df=rules_version_df,
        audit_report_path=audit_report_path,
        generated_at=generated_at,
        aggregations=aggregations,
        data_quality_notes=data_quality_notes,
        action_pivots=action_pivots,
        overview=overview,
        overview_df=overview_dataframe(overview),
        actions=actions,
        summary=summary,
        negative_keywords=negative_keywords,
        bid_adjustments=bid_adjustments,
        pause_list=pause_list,
        growth_list=growth_list,
        exact_opportunities=exact_opportunities,
        priority_list=priority_list,
        self_check=self_check,
        ai_report_sections=ai_report_sections,
    )


def render_hero(settings: AppSettings, uploaded_count: int) -> None:
    state = st.session_state.get("analysis_state")
    if state is not None:
        overview = state.overview
        summary = state.summary
        st.html(
            f"""
            <section class="analysis-status-bar">
                <div>
                    <span>当前诊断</span>
                    <strong>{escape_html(settings.rule_preset)}</strong>
                </div>
                <div>
                    <span>ACOS</span>
                    <strong>{format_percent(safe_float(overview.get("ACOS")))}</strong>
                </div>
                <div>
                    <span>高优先级</span>
                    <strong>{int(safe_float(summary.get("高优先级", 0))):,}</strong>
                </div>
                <div>
                    <span>建议动作</span>
                    <strong>{int(safe_float(summary.get("总建议数", 0))):,}</strong>
                </div>
                <div>
                    <span>文件</span>
                    <strong>{uploaded_count}</strong>
                </div>
            </section>
            """
        )
        return

    st.markdown(
        f"""
        <div class="saas-hero">
            <div class="hero-main">
                <div class="hero-kicker">Amazon Ads Diagnostic SaaS</div>
                <div class="hero-title">亚马逊广告诊断工具</div>
                <p class="hero-subtitle">本地解析广告报表，聚合关键指标，输出可执行动作，并支持 AI 复核。</p>
                <div class="tag-row">
                    <span class="tag">搜索词</span>
                    <span class="tag">投放定向</span>
                    <span class="tag">广告活动</span>
                    <span class="tag">表格导出</span>
                </div>
            </div>
            <div class="hero-side">
                <div class="hero-side-row"><span class="side-label">运行</span><span class="side-value">本地无网</span></div>
                <div class="hero-side-row"><span class="side-label">模式</span><span class="side-value">{escape_html(settings.mode)}</span></div>
                <div class="hero-side-row"><span class="side-label">口径</span><span class="side-value">{escape_html(settings.rule_preset)}</span></div>
                <div class="hero-side-row"><span class="side-label">目标 ACOS</span><span class="side-value">{settings.diagnosis_config.target_acos:.0%}</span></div>
                <div class="hero-side-row"><span class="side-label">文件</span><span class="side-value">{uploaded_count}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_tabs(state: AnalysisState) -> None:
    tab_options = ["总览", "数据透视", "动作建议", "问题诊断", "机会分析", "AI 报告", "导出"]
    forced_tab = st.session_state.pop("force_dashboard_tab", None)
    if forced_tab in tab_options:
        st.session_state["dashboard_tab"] = forced_tab
    if st.session_state.get("dashboard_tab") not in tab_options:
        st.session_state["dashboard_tab"] = tab_options[0]

    if hasattr(st, "segmented_control"):
        selected_tab = st.segmented_control(
            "页面导航",
            tab_options,
            key="dashboard_tab",
            label_visibility="collapsed",
            width="stretch",
        )
    else:
        selected_tab = st.radio(
            "页面导航",
            tab_options,
            horizontal=True,
            key="dashboard_tab",
            label_visibility="collapsed",
        )
    selected_tab = selected_tab or st.session_state.get("dashboard_tab", tab_options[0])
    st.divider()

    if selected_tab == "总览":
        render_overview_tab(state)
    elif selected_tab == "数据透视":
        render_pivot_tab(state)
    elif selected_tab == "动作建议":
        render_actions_tab(state)
    elif selected_tab == "问题诊断":
        render_problems_tab(state)
    elif selected_tab == "机会分析":
        render_opportunities_tab(state)
    elif selected_tab == "AI 报告":
        render_ai_report_tab(state)
    elif selected_tab == "导出":
        render_export_tab(state)


def render_overview_tab(state: AnalysisState) -> None:
    render_operator_brief(state)
    render_diagnosis_accuracy_note()

    render_section_header("关键指标", "账户总览只基于一个权威数据源计算，避免不同维度报表重复累计。")
    overview = state.overview
    target_acos = state.settings.diagnosis_config.target_acos
    kpis = [
        ("总花费", f"${overview['总花费']:,.2f}", "需要关注" if overview["总花费"] else "暂无花费", "warning"),
        ("总销售额", f"${overview['总销售额']:,.2f}", "健康" if overview["总销售额"] else "暂无销售", "success" if overview["总销售额"] else "warning"),
        ("总订单", f"{overview['总订单']:,.0f}", "健康" if overview["总订单"] else "需要关注", "success" if overview["总订单"] else "danger"),
        ("ACOS", format_percent(overview["ACOS"]), "高于目标" if overview["ACOS"] > target_acos else "健康", "danger" if overview["ACOS"] > target_acos else "success"),
        ("ROAS", f"{overview['ROAS']:,.2f}", "健康" if overview["ROAS"] >= 1 else "需要关注", "success" if overview["ROAS"] >= 1 else "warning"),
        ("CTR", format_percent(overview["CTR"]), "流量信号", "neutral"),
        ("CPC", f"${overview['CPC']:,.2f}", "单次点击成本", "neutral"),
        ("CVR", format_percent(overview["CVR"]), "转化信号", "neutral"),
    ]
    for row in chunked(kpis, 4):
        columns = st.columns(4)
        for column, item in zip(columns, row):
            with column:
                render_kpi_card(*item)

    render_account_summary_source_note(state)
    render_duplicate_metric_guard(state)

    render_section_header("诊断摘要", "展示当前分析覆盖度、风险规模和机会数量。")
    summary_items = [
        ("广告活动", len(state.aggregations.get("广告活动", pd.DataFrame()))),
        ("广告组", len(state.aggregations.get("广告组", pd.DataFrame()))),
        ("搜索词", len(state.aggregations.get("搜索词", pd.DataFrame()))),
        ("ACOS 状态", "高于目标" if overview["ACOS"] > target_acos else "健康"),
        ("高优先级", state.summary["高优先级"]),
        ("机会项", len(state.exact_opportunities) + len(state.growth_list)),
        ("建议动作", state.summary["总建议数"]),
        ("文件数", state.uploaded_count),
    ]
    for row in chunked(summary_items, 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, row):
            with column:
                render_stat_card(label, value, "neutral")

    render_upload_status(state.file_summaries)
    render_diagnosis_audit_summary(state)
    render_file_audit_table(state)
    render_basic_data_audit_panel(state)


def render_account_summary_source_note(state: AnalysisState) -> None:
    source = state.account_summary_source
    source_text = (
        f"{display_report_type(source.report_type)} | {source.filename}"
        if source
        else "未选择"
    )
    currency = source.currency if source else "CAD"
    st.markdown(
        f"""
        <div class="table-note">
            账户总览基于：{escape_html(source_text)}。其他报表仅用于维度诊断，未参与总花费 / 总销售额重复汇总。
            当前统计货币：{escape_html(currency)}。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_safety_summary(state: AnalysisState) -> None:
    trust = state.data_trust_result
    safety = state.safety_gate
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_stat_card("数据可信度", f"{trust.data_trust_score}/100 · {trust.data_trust_level}", "danger" if trust.data_trust_level == "低" else "warning" if trust.data_trust_level == "中" else "success")
    with c2:
        render_stat_card("诊断安全阀", safety.safety_level, "danger" if not safety.can_diagnose else "warning" if not safety.can_generate_p0 else "success")
    with c3:
        render_stat_card("外部对账", state.reconciliation_result.reconciliation_status, "danger" if state.reconciliation_result.reconciliation_status == "阻止诊断" else "warning" if state.reconciliation_result.reconciliation_status in {"警告", "严重警告"} else "success")
    with c4:
        render_stat_card("规则版本", DIAGNOSIS_ENGINE_VERSION, "neutral")
    if trust.data_trust_level == "低":
        st.error("当前数据可信度较低，系统已关闭 P0 今日必做动作，建议先修复数据源或确认报表口径。")
    if state.reconciliation_result.reconciliation_status in {"严重警告", "阻止诊断"}:
        st.warning("工具计算结果与外部系统差异较大，建议先复核报表口径，不建议直接执行广告动作。")
    if not safety.can_diagnose:
        st.error("诊断安全阀已阻断动作生成。当前仅建议查看数据审计并修复上传报表。")
    render_data_trust_detail_panel(
        trust,
        state.safety_gate,
        state.account_summary_source,
        state.file_audit,
    )


def render_data_trust_detail_panel(
    trust: DataTrustResult,
    safety: DiagnosisSafetyGateResult,
    source: AccountSummarySource | None,
    file_audit: pd.DataFrame,
) -> None:
    source_text = f"{display_report_type(source.report_type)} | {source.filename}" if source else "未选择"
    duplicate_risk = "否" if not file_audit.empty and file_audit["是否参与账户总览"].eq("是").sum() <= 1 else "是"
    warnings_text = "；".join(trust.data_quality_warnings)
    rows = [
        ("数据可信度等级", trust.data_trust_level, "通过" if trust.data_trust_level == "高" else "警告" if trust.data_trust_level == "中" else "阻止"),
        ("当前账户总览数据源", source_text, "通过" if source else "阻止"),
        ("是否存在重复计算风险", duplicate_risk, "通过" if duplicate_risk == "否" else "阻止"),
        ("核心字段是否完整", "否" if "缺少核心字段" in warnings_text else "是", "警告" if "缺少核心字段" in warnings_text else "通过"),
        ("金额字段是否解析正常", "否" if "无法解析" in warnings_text or "空值" in warnings_text else "是", "警告" if "无法解析" in warnings_text or "空值" in warnings_text else "通过"),
        ("是否有异常指标", "是" if any(keyword in warnings_text for keyword in ["Clicks > Impressions", "Orders > Clicks", "CVR > 100", "ACOS 极端"]) else "否", "警告" if any(keyword in warnings_text for keyword in ["Clicks > Impressions", "Orders > Clicks", "CVR > 100", "ACOS 极端"]) else "通过"),
        ("是否可以生成 P0 今日必做", "是" if safety.can_generate_p0 else "否", "通过" if safety.can_generate_p0 else "阻止"),
    ]
    with st.expander("数据可信度", expanded=trust.data_trust_level != "高"):
        if trust.data_trust_level == "低":
            st.error("当前数据可信度较低，系统已关闭强动作建议。请先检查报表类型、字段识别和数据口径。")
        st.dataframe(pd.DataFrame(rows, columns=["检查项", "结果", "状态"]), width="stretch", hide_index=True)
        if trust.data_quality_warnings:
            st.caption("数据提醒：" + "；".join(trust.data_quality_warnings[:4]))
        if trust.blocking_errors:
            st.caption("阻止项：" + "；".join(trust.blocking_errors))


def render_duplicate_metric_guard(state: AnalysisState) -> None:
    for message in state.duplicate_guard_messages:
        st.warning(message)


def render_file_audit_table(state: AnalysisState) -> None:
    if state.file_audit.empty:
        return
    render_section_header("上传文件审计", "确认每个文件的报表类型、用途和是否参与账户总览。")
    st.dataframe(state.file_audit, width="stretch", hide_index=True)


def render_basic_data_audit_panel(state: AnalysisState) -> None:
    if state.basic_data_audit.empty:
        return
    high_risk = state.basic_data_audit["风险级别"].isin(["严重错误", "高风险"]).any()
    with st.expander("基础数据自检", expanded=bool(high_risk)):
        if high_risk:
            st.error("基础数据自检发现高风险项，请先修正数据口径再解读诊断建议。")
        else:
            st.success("基础数据自检通过：账户总览未跨多个报表重复累计。")
        st.dataframe(state.basic_data_audit, width="stretch", hide_index=True)


def render_pre_diagnosis_check(
    loaded_reports: list[dict[str, object]],
    file_summaries: list[dict[str, object]],
    manual_mappings: dict[int, dict[str, str]],
    settings: AppSettings,
) -> None:
    if not loaded_reports:
        return
    render_section_header("一键自检", "上传报表后先检查数据是否可信，再开始诊断。")
    if not st.button("一键自检", type="secondary", width="stretch", key="run_pre_diagnosis_self_check"):
        st.caption("点击后会检查报表类型、字段识别、总览口径、重复计算风险、异常指标、P0 门禁和 Excel 导出可用性。")
        return

    try:
        _mapping_df, _cleaned_data, report_frames = prepare_report_frames(loaded_reports, manual_mappings)
        for report_frame in report_frames:
            report_frame["enriched_data"] = add_metrics(report_frame["cleaned_data"])
        source = select_account_summary_source(report_frames)
        overview = calculate_account_overview(source.dataframe if source else pd.DataFrame())
        file_audit = build_file_audit(report_frames, source)
        trust = calculate_data_trust_score(report_frames, source, file_audit, overview)
        reconciliation = reconcile_external_totals(overview, settings.reconciliation_input)
        safety = run_diagnosis_safety_gate(
            report_frames,
            overview,
            trust,
            account_summary_source=source,
            file_audit=file_audit,
            reconciliation_result=reconciliation,
        )
    except Exception as exc:
        st.error(f"诊断前检查失败：{type(exc).__name__}: {exc}")
        return

    rows = build_pre_diagnosis_check_rows(loaded_reports, file_summaries, file_audit, source, trust, safety)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    render_data_trust_detail_panel(trust, safety, source, file_audit)
    if any(row["状态"] == "阻止" for row in rows):
        st.error("存在阻止项，本次不会生成强动作建议。请先修复字段、报表类型或总览数据源。")
    elif any(row["状态"] == "警告" for row in rows):
        st.warning("存在警告项，可以继续诊断，但建议先复核数据口径。")
    else:
        st.success("诊断前检查通过，可以开始诊断。")


def build_pre_diagnosis_check_rows(
    loaded_reports: list[dict[str, object]],
    file_summaries: list[dict[str, object]],
    file_audit: pd.DataFrame,
    source: AccountSummarySource | None,
    trust: DataTrustResult,
    safety: DiagnosisSafetyGateResult,
) -> list[dict[str, str]]:
    report_types = [str(report.get("report_type", "")) for report in loaded_reports]
    missing_fields = "；".join(str(item.get("缺失必需字段", "")) for item in file_summaries if item.get("缺失必需字段"))
    included_count = int(file_audit["是否参与账户总览"].eq("是").sum()) if not file_audit.empty else 0
    bulk_wrong = bool(not file_audit.empty and ((file_audit["report_type"].eq("SP_BULK_FILE")) & file_audit["是否参与账户总览"].eq("是")).any())
    top_wrong = bool(not file_audit.empty and ((file_audit["report_type"].eq("SEARCH_QUERY_PERFORMANCE_OR_TOP_SEARCH_TERMS")) & file_audit["是否参与账户总览"].eq("是")).any())
    source_label = f"{display_report_type(source.report_type)} | {source.filename}" if source else "未选择"
    summary_metrics_from_source = source is not None and included_count == 1
    parse_warning_text = "；".join(trust.data_quality_warnings)
    parse_ok = not any(keyword in parse_warning_text for keyword in ["无法解析", "空值"])
    abnormal_metric = any(keyword in parse_warning_text for keyword in ["Clicks > Impressions", "Orders > Clicks", "CVR > 100", "ACOS 极端"])
    can_export_excel = bool(loaded_reports) and source is not None
    return [
        _check_row("是否已上传文件", bool(loaded_reports), "通过" if loaded_reports else "阻止"),
        _check_row("是否识别到报表类型", bool(report_types) and all(report_type not in {"读取失败", "UNKNOWN"} for report_type in report_types), "通过" if bool(report_types) and all(report_type not in {"读取失败", "UNKNOWN"} for report_type in report_types) else "警告"),
        _check_row("是否识别到 Spend / Sales / Orders / Clicks / Impressions", not missing_fields, "通过" if not missing_fields else "警告", missing_fields or "核心指标字段已识别"),
        _check_row("是否明确 account_summary_source", source is not None, "通过" if source else "阻止", source_label),
        _check_row("总花费、总销售额是否来自权威报表", summary_metrics_from_source, "通过" if summary_metrics_from_source else "阻止", source_label),
        _check_row("是否存在多个报表重复汇总风险", included_count <= 1, "通过" if included_count <= 1 else "阻止", f"参与总览文件数：{included_count}"),
        _check_row("Bulk 文件是否被错误纳入总览", not bulk_wrong, "通过" if not bulk_wrong else "阻止"),
        _check_row("热门搜索词报告是否被错误纳入广告表现统计", not top_wrong, "通过" if not top_wrong else "阻止"),
        _check_row("Spend / Sales / Orders / Clicks / Impressions 是否解析正常", parse_ok, "通过" if parse_ok else "警告", "解析正常" if parse_ok else parse_warning_text),
        _check_row("是否存在 Clicks > Impressions、Orders > Clicks 等异常", not abnormal_metric, "通过" if not abnormal_metric else "警告", "未发现账户级异常" if not abnormal_metric else parse_warning_text),
        _check_row("数据可信度评分", trust.data_trust_score >= 70, "通过" if trust.data_trust_score >= 85 else "警告" if trust.data_trust_score >= 70 else "阻止", f"{trust.data_trust_score}/100 · {trust.data_trust_level}"),
        _check_row("是否可以开始诊断", safety.can_diagnose, "通过" if safety.can_diagnose else "阻止", "；".join(safety.blocking_reasons) or "可以开始"),
        _check_row("是否可以生成 P0 今日必做", safety.can_generate_p0, "通过" if safety.can_generate_p0 else "阻止", f"data_trust_score={trust.data_trust_score}"),
        _check_row("是否可以导出 Excel", can_export_excel, "通过" if can_export_excel else "阻止", "可导出完整诊断包" if can_export_excel else "缺少有效报表或总览来源"),
    ]


def _check_row(item: str, passed: bool, status: str, detail: str = "") -> dict[str, str]:
    return {"检查项": item, "结果": "是" if passed else "否", "状态": status, "说明": detail}


def render_diagnosis_audit_summary(state: AnalysisState) -> None:
    with st.expander("诊断审计摘要", expanded=False):
        tier_counts = state.actions.get("execution_tier", pd.Series(dtype=str)).value_counts().to_dict() if not state.actions.empty else {}
        rows = [
            ("审计报告", state.audit_report_path),
            ("诊断引擎版本", DIAGNOSIS_ENGINE_VERSION),
            ("规则配置版本", RULE_CONFIG_VERSION),
            ("数据可信度", f"{state.data_trust_result.data_trust_score}/100 · {state.data_trust_result.data_trust_level}"),
            ("安全阀", state.safety_gate.safety_level),
            ("诊断信号数", len(state.actions)),
            ("P0 / P1 / P2 / P3", f"{tier_counts.get('P0', 0)} / {tier_counts.get('P1', 0)} / {tier_counts.get('P2', 0)} / {tier_counts.get('P3', 0)}"),
            ("高风险需复核动作", int(state.actions.get("需要人工复核", pd.Series(dtype=str)).astype(str).eq("是").sum()) if not state.actions.empty else 0),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["项目", "内容"]), width="stretch", hide_index=True)


def render_diagnosis_accuracy_note() -> None:
    st.markdown(
        """
        <div class="table-note">
            <strong>诊断准确性说明</strong>
            本工具基于广告报表中的曝光、点击、花费、订单、销售额等数据进行规则诊断。
            系统会优先保护已有订单对象和数据不足对象，避免过早否定或暂停。
            所有建议仍建议结合利润率、库存、活动目标和搜索词相关性复核后执行。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_operator_brief(state: AnalysisState) -> None:
    overview = state.overview
    summary = state.summary
    target_acos = state.settings.diagnosis_config.target_acos
    acos = safe_float(overview.get("ACOS"))
    orders = safe_float(overview.get("总订单"))
    spend = safe_float(overview.get("总花费"))
    high_priority = int(safe_float(summary.get("高优先级", 0)))

    if spend > 0 and orders == 0:
        status = "严重风险"
        status_tone = "danger"
        conclusion = "账户已有广告消耗但暂无订单，先确认转化窗口和报表口径，再处理高花费无转化对象。"
    elif acos > target_acos:
        status = "需要优化"
        status_tone = "warning"
        conclusion = f"当前 ACOS {format_percent(acos)} 高于目标 {format_percent(target_acos)}，先压降无效花费，再复查转化承接。"
    elif high_priority:
        status = "局部风险"
        status_tone = "warning"
        conclusion = f"整体效率可控，但仍有 {high_priority} 条高优先级动作需要按花费排序处理。"
    else:
        status = "表现可控"
        status_tone = "success"
        conclusion = "当前没有明显高风险信号，建议保留健康活动预算，并继续寻找低 ACOS 放量机会。"

    top_actions = build_operator_next_steps(state)
    action_html = "".join(
        f'<div class="ops-brief-step"><span>{index}</span><p>{escape_html(action)}</p></div>'
        for index, action in enumerate(top_actions, start=1)
    )
    st.html(
        f"""
        <section class="ops-brief ops-brief-{escape_html(status_tone)}">
            <div class="ops-brief-main">
                <div class="ops-brief-kicker">今日运营 Brief</div>
                <h2>{escape_html(status)}</h2>
                <p>{escape_html(conclusion)}</p>
            </div>
            <div class="ops-brief-actions">
                <div class="ops-brief-actions-title">建议处理顺序</div>
                {action_html}
            </div>
        </section>
        """
    )


def build_operator_next_steps(state: AnalysisState) -> list[str]:
    summary = state.summary
    notes = getattr(state, "data_quality_notes", [])
    actions: list[str] = []
    if notes:
        actions.append("先复核数据口径，尤其是有花费无销售额、点击有订单为 0 的记录。")
    if int(safe_float(summary.get("暂停建议", 0))):
        actions.append(f"立即查看暂停建议，优先处理 {summary['暂停建议']} 个明显浪费的活动或广告组。")
    if int(safe_float(summary.get("否定建议", 0))):
        actions.append(f"批量检查 {summary['否定建议']} 条否定词建议，确认无误后再执行。")
    if int(safe_float(summary.get("调价建议", 0))):
        actions.append(f"处理 {summary['调价建议']} 条调价建议，先降高 ACOS，再提低 ACOS 优质对象。")
    if int(safe_float(summary.get("Listing问题", 0))):
        actions.append(f"复查 {summary['Listing问题']} 条 Listing 承接问题，重点看主图、标题、价格和评价。")
    if int(safe_float(summary.get("增长建议", 0))):
        actions.append(f"保留并放量 {summary['增长建议']} 个增长机会，避免只止损不增效。")
    if not actions:
        actions.append("保持当前投放节奏，继续观察花费、订单和 ACOS 的 7 天趋势。")
    return actions[:4]


def render_pivot_tab(state: AnalysisState) -> None:
    render_section_header("数据透视", "按广告活动、广告组、搜索词、Targeting、动作或优先级汇总，便于运营快速定位问题。")
    if state.actions.empty:
        st.info("暂无动作建议，无法生成透视表。")
        return

    view = st.selectbox("常用运营视角", list(PIVOT_VIEW_PRESETS.keys()), index=0, key="pivot_operator_view")
    view_config = PIVOT_VIEW_PRESETS[view]
    st.markdown(
        f'<div class="ops-view-note"><strong>{escape_html(view)}</strong><span>{escape_html(view_config["caption"])}</span></div>',
        unsafe_allow_html=True,
    )

    all_priorities = sorted(state.actions["优先级"].dropna().unique())
    all_actions = sorted(state.actions["建议动作"].dropna().unique())
    default_priorities = [item for item in view_config["priorities"] if item in all_priorities] or all_priorities
    default_actions = [item for item in view_config["actions"] if item in all_actions]
    dimensions = list(ACTION_PIVOT_PRESETS.keys())
    dimension_index = dimensions.index(view_config["dimension"]) if view_config["dimension"] in dimensions else 0

    c1, c2, c3 = st.columns([1.2, 1.25, 1])
    preset = c1.selectbox("透视维度", dimensions, index=dimension_index, key=f"pivot_dimension_{view}")
    priorities = c2.multiselect(
        "优先级",
        all_priorities,
        default=default_priorities,
        key=f"pivot_priority_filter_{view}",
    )
    min_clicks = c3.number_input("最低点击", min_value=0, max_value=100000, value=0, step=1, key=f"pivot_min_clicks_filter_{view}")

    if default_actions:
        action_types = st.multiselect(
            "建议动作",
            all_actions,
            default=default_actions,
            key=f"pivot_action_filter_{view}",
            help="当前运营视角会预设一组常用动作，也可以手动增减。",
        )
    else:
        action_types = all_actions
        st.caption("当前视角默认包含全部建议动作。")

    filtered = state.actions[
        state.actions["优先级"].isin(priorities)
        & state.actions["建议动作"].isin(action_types)
    ].copy()
    pivot = build_action_pivot_cached(filtered, tuple(ACTION_PIVOT_PRESETS[preset]))
    if min_clicks and "Clicks" in pivot.columns:
        pivot = pivot[pivot["Clicks"] >= min_clicks]

    sort_options = [column for column in ["高优先级数", "建议数", "Spend", "ACOS", "Orders", "Sales", "Clicks"] if column in pivot.columns]
    if sort_options:
        sort_by = st.selectbox("排序指标", sort_options, index=0, key=f"pivot_sort_by_{view}_{preset}")
        pivot = pivot.sort_values(sort_by, ascending=False)

    render_pivot_snapshot(pivot, preset)
    st.markdown('<div class="table-note">透视表优先展示“有多少问题、问题有多严重、消耗了多少钱”，比纯文字报告更适合日常复盘和派单。</div>', unsafe_allow_html=True)
    st.dataframe(format_pivot_dataframe(pivot.head(200)), width="stretch", hide_index=True, height=520)

    if st.toggle("显示导出透视表预览", value=False):
        export_pivots = getattr(state, "action_pivots", build_export_pivots(state.actions))
        for name, dataframe in export_pivots.items():
            with st.expander(name, expanded=False):
                st.dataframe(format_pivot_dataframe(dataframe.head(80)), width="stretch", hide_index=True)


@st.cache_data(show_spinner=False)
def build_action_pivot_cached(actions: pd.DataFrame, group_columns: tuple[str, ...]) -> pd.DataFrame:
    return build_action_pivot(actions, list(group_columns))


def render_pivot_snapshot(pivot: pd.DataFrame, preset: str) -> None:
    if pivot.empty:
        st.info("当前筛选条件下暂无数据。")
        return
    total_items = len(pivot)
    high_count = int(safe_float(pivot.get("高优先级数", pd.Series(dtype=float)).sum()))
    spend = safe_float(pivot.get("Spend", pd.Series(dtype=float)).sum())
    top_name = ""
    group_columns = ACTION_PIVOT_PRESETS.get(preset, [])
    for column in group_columns:
        if column in pivot.columns and not pivot.empty:
            top_name = str(pivot.iloc[0].get(column, "") or "")
            break
    top_text = top_name if top_name else "当前维度首项"
    st.html(
        f"""
        <section class="pivot-snapshot">
            <div><span>透视对象</span><strong>{total_items:,.0f}</strong></div>
            <div><span>高优先级</span><strong>{high_count:,.0f}</strong></div>
            <div><span>涉及花费</span><strong>${spend:,.2f}</strong></div>
            <div><span>首要关注</span><strong>{escape_html(limit_report_text(top_text, 24))}</strong></div>
        </section>
        """
    )


def render_actions_tab(state: AnalysisState) -> None:
    render_section_header("动作建议", "优先级按浪费金额、点击样本、ACOS 偏离、转化缺口和动作紧急度综合计算。")
    st.markdown(
        '<div class="table-note">操作前请结合库存、利润率、活动目标和搜索词相关性复核。已有订单的对象不建议直接否定或暂停，优先考虑调价。高风险动作已标记为“需人工复核”。</div>',
        unsafe_allow_html=True,
    )
    if not state.safety_gate.can_generate_p0:
        st.warning("诊断安全阀已关闭 P0 今日必做动作；当前建议只作为复核线索。")
    render_copy_zones(state)
    render_action_queue(state)
    render_action_overload_summary_note(state)
    columns = st.columns(4)
    stats = [
        ("今日必做 P0", int(state.actions.get("execution_tier", pd.Series(dtype=str)).eq("P0").sum()), "danger"),
        ("本周重点 P1", int(state.actions.get("execution_tier", pd.Series(dtype=str)).eq("P1").sum()), "warning"),
        ("待观察 P2", int(state.actions.get("execution_tier", pd.Series(dtype=str)).eq("P2").sum()), "neutral"),
        ("否定词", state.summary["否定建议"], "danger"),
    ]
    for column, stat in zip(columns, stats):
            with column:
                render_stat_card(*stat)

    st.markdown('<div class="table-note">默认只展示 P0 今日必做和 P1 本周重点；P2 待观察折叠查看，P3 仅进入 Excel 完整明细。</div>', unsafe_allow_html=True)
    must_do_only = st.toggle(
        "只看今日必做",
        value=True,
        key="actions_today_must_do",
        help="聚焦 P0 今日必做动作，默认最多 10 条。",
    )
    if must_do_only and "execution_tier" in state.actions.columns:
        action_source = state.actions[state.actions["execution_tier"].eq("P0")]
    else:
        action_source = state.actions[state.actions.get("execution_tier", pd.Series("", index=state.actions.index)).isin(["P0", "P1"])] if "execution_tier" in state.actions.columns else state.actions
    if must_do_only:
        st.caption(f"已筛出 {len(action_source):,} 条今日必做动作；关闭开关可查看全部建议。")
    filtered = render_action_filters(action_source)
    if "优先级评分" in filtered.columns:
        filtered = filtered.sort_values("优先级评分", ascending=False)
    st.dataframe(style_action_table(filtered), width="stretch", hide_index=True, height=520)
    if "execution_tier" in state.actions.columns:
        with st.expander("查看 P2 待观察动作", expanded=False):
            p2 = state.actions[state.actions["execution_tier"].eq("P2")]
            st.dataframe(style_action_table(p2.head(200)), width="stretch", hide_index=True, height=420)
    render_operator_feedback_panel(state)


def render_copy_zones(state: AnalysisState) -> None:
    negative_text = build_negative_copy_text(state.actions)
    exact_text = build_exact_copy_text(state.actions)
    with st.expander("复制到亚马逊广告后台", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**否定词复制区**")
            st.caption("仅包含 P0 / P1 的否定建议，已排除 Orders > 0 的对象。")
            st.text_area("否定词复制区", value=negative_text, height=180, label_visibility="collapsed", key="negative_copy_area")
        with c2:
            st.markdown("**精准投放词复制区**")
            st.caption("仅包含 P0 / P1 中建议提取为 Exact 的搜索词。")
            st.text_area("精准投放词复制区", value=exact_text, height=180, label_visibility="collapsed", key="exact_copy_area")


def build_negative_copy_text(actions: pd.DataFrame) -> str:
    if actions.empty or "execution_tier" not in actions.columns:
        return ""
    source = actions[
        actions["execution_tier"].isin(["P0", "P1"])
        & actions["合并动作"].astype(str).str.contains("否定", na=False)
        & (actions["Orders"].fillna(0).astype(float) <= 0)
    ].copy()
    terms = source["Customer Search Term"].fillna("").astype(str).str.strip()
    terms = terms[terms.ne("")]
    return "\n".join(dict.fromkeys(terms.tolist()))


def build_exact_copy_text(actions: pd.DataFrame) -> str:
    if actions.empty or "execution_tier" not in actions.columns:
        return ""
    source = actions[
        actions["execution_tier"].isin(["P0", "P1"])
        & actions["合并动作"].astype(str).str.contains("提取精准投放", na=False)
    ].copy()
    terms = source["Customer Search Term"].fillna("").astype(str).str.strip()
    terms = terms[terms.ne("")]
    return "\n".join(dict.fromkeys(terms.tolist()))


def render_operator_feedback_panel(state: AnalysisState) -> None:
    if state.actions.empty:
        return
    with st.expander("运营反馈记录", expanded=False):
        st.caption("反馈会保存在当前会话，并进入 Excel 的“运营反馈记录” Sheet。")
        feedback_columns = [
            "诊断对象",
            "建议动作",
            "需要人工复核",
            "operator_feedback",
            "feedback_reason",
            "reviewed_by",
            "reviewed_at",
        ]
        editor_source = state.actions[[column for column in feedback_columns if column in state.actions.columns]].copy()
        edited = st.data_editor(
            editor_source,
            width="stretch",
            hide_index=True,
            key="operator_feedback_editor",
            column_config={
                "operator_feedback": st.column_config.SelectboxColumn("operator_feedback", options=["", "准确", "不准确", "太激进", "太保守", "已执行", "暂不执行", "需要复核"]),
                "feedback_reason": st.column_config.TextColumn("feedback_reason"),
                "reviewed_by": st.column_config.TextColumn("reviewed_by"),
                "reviewed_at": st.column_config.TextColumn("reviewed_at"),
            },
            disabled=[column for column in editor_source.columns if column not in {"operator_feedback", "feedback_reason", "reviewed_by", "reviewed_at"}],
        )
        for column in ["operator_feedback", "feedback_reason", "reviewed_by", "reviewed_at"]:
            if column in edited.columns:
                state.actions.loc[edited.index, column] = edited[column]


def render_action_overload_summary_note(state: AnalysisState) -> None:
    total = len(state.actions)
    p0 = int(state.actions.get("execution_tier", pd.Series(dtype=str)).eq("P0").sum()) if not state.actions.empty else 0
    st.markdown(
        f"""
        <div class="table-note">
            <strong>动作收敛说明</strong>
            系统识别到 {total:,} 条诊断信号，已根据影响金额、数据充分性、置信度和操作风险，
            筛选出 {p0:,} 条今日必做动作，避免动作过载。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_queue(state: AnalysisState) -> None:
    if state.actions.empty:
        return
    queue = state.actions.copy()
    queue["_priority_sort"] = queue["优先级"].map({"高": 0, "中": 1, "低": 2}).fillna(3)
    if "优先级评分" in queue.columns:
        queue = queue.sort_values(["_priority_sort", "优先级评分", "Spend"], ascending=[True, False, False])
    queue = queue.head(6)
    cards = "".join(render_action_queue_card(row, index) for index, (_, row) in enumerate(queue.iterrows(), start=1))
    st.html(
        f"""
        <section class="action-queue">
            <div class="action-queue-heading">
                <div>
                    <span>运营执行队列</span>
                    <p>按风险和影响排序，先处理这些对象，再进入下方表格做批量筛选。</p>
                </div>
                <strong>{len(state.actions):,.0f} 条建议</strong>
            </div>
            <div class="action-queue-grid">{cards}</div>
        </section>
        """
    )


def filter_today_must_do_actions(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return actions
    urgent_actions = {"暂停", "否定精准", "否定词组", "降低竞价", "检查 Listing"}
    dataframe = actions.copy()
    action_text = dataframe["合并动作"].fillna(dataframe["建议动作"]).astype(str)
    mask = dataframe["优先级"].eq("高") | action_text.apply(lambda value: any(action in value for action in urgent_actions))
    filtered = dataframe[mask].copy()
    if "优先级评分" in filtered.columns:
        filtered = filtered.sort_values(["优先级评分", "Spend"], ascending=[False, False])
    return filtered


def render_action_queue_card(row: pd.Series, index: int) -> str:
    priority = str(row.get("优先级", "") or "低")
    tone = "danger" if priority == "高" else "warning" if priority == "中" else "success"
    action = str(row.get("合并动作") or row.get("建议动作") or "继续观察")
    obj = best_action_object(row)
    reason = limit_report_text(row.get("原因", ""), 92)
    spend = safe_float(row.get("Spend", 0))
    acos = format_percent(safe_float(row.get("ACOS", 0)))
    orders = safe_float(row.get("Orders", 0))
    return (
        f'<article class="action-task action-task-{tone}">'
        f'<div class="action-task-top"><span>#{index}</span><b>{escape_html(priority)}优先级</b></div>'
        f'<h3>{escape_html(action)}</h3>'
        f'<p>{escape_html(obj)}</p>'
        f'<div class="action-task-metrics">'
        f'<em>花费 ${spend:,.2f}</em><em>订单 {orders:,.0f}</em><em>ACOS {escape_html(acos)}</em>'
        f'</div>'
        f'<small>{escape_html(reason)}</small>'
        f'</article>'
    )


def best_action_object(row: pd.Series) -> str:
    for column in ["诊断对象", "Customer Search Term", "Targeting", "Campaign Name", "Ad Group Name"]:
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    return "未命名对象"


def render_problems_tab(state: AnalysisState) -> None:
    render_section_header("问题诊断", "每个分区仅展示前 10 条重点问题。")
    render_priority_standard(state)
    sections = [
        ("高花费无转化", filter_actions(state.actions, rules=["明显不相关无订单词", "相关但高点击无转化", "相关但中等消耗无转化"], action_names=["否定精准", "降低竞价", "继续观察"]), "danger"),
        ("高 ACOS 对象", filter_actions(state.actions, rules=["高 ACOS 低效词"], action_names=["降低竞价"]), "warning"),
        ("高曝光低 CTR", filter_actions(state.actions, rules=["低 CTR 高曝光"], action_names=["检查 Listing"]), "warning"),
        ("高点击低 CVR", filter_actions(state.actions, rules=["高 CTR 低 CVR"], action_names=["检查 Listing"]), "warning"),
        ("需要暂停或重构的广告活动", filter_actions(state.actions, rules=["需要暂停的广告活动 / 广告组"], action_names=["暂停"]), "danger"),
    ]
    for title, dataframe, tone in sections:
        render_problem_section(title, dataframe, tone)

    with st.expander("查看完整问题明细", expanded=False):
        problem_actions = state.actions[state.actions["建议动作"].isin(["暂停", "否定精准", "否定词组", "降低竞价", "检查 Listing"])] if not state.actions.empty else state.actions
        st.dataframe(style_action_table(problem_actions), width="stretch", hide_index=True)


def render_priority_standard(state: AnalysisState) -> None:
    config = state.settings.diagnosis_config
    st.html(
        f"""
        <section class="priority-standard">
            <div>
                <span>高优先级</span>
                <strong>立即处理</strong>
                <p>达到强止损、明显浪费或 ACOS 显著高于目标，优先看花费和点击样本。</p>
            </div>
            <div>
                <span>中优先级</span>
                <strong>本周优化</strong>
                <p>有消耗或转化信号但效率偏低，适合降价、复查 Listing 或继续观察。</p>
            </div>
            <div>
                <span>低优先级</span>
                <strong>保留观察</strong>
                <p>样本不足或风险较轻，先看趋势，不建议一次性大批量调整。</p>
            </div>
            <div>
                <span>当前口径</span>
                <strong>{escape_html(state.settings.rule_preset)}</strong>
                <p>低效点击 ≥ {config.min_waste_clicks}，强止损点击 ≥ {config.hard_waste_clicks}，最低花费 ${config.min_waste_spend:,.0f}。</p>
            </div>
        </section>
        """
    )


def render_opportunities_tab(state: AnalysisState) -> None:
    render_section_header("机会分析", "增长机会使用绿色和金色高亮。")
    quality_terms = filter_actions(state.actions, rules=["低 ACOS 优质词"], action_names=["提高竞价", "增加预算"]).head(10)
    exact_terms = state.exact_opportunities.head(10)
    budget_campaigns = filter_actions(state.actions, rules=["预算可能不足的广告活动"], action_names=["增加预算"]).head(10)
    high_conversion = build_high_conversion_table(state)
    long_tail = build_long_tail_table(state)

    render_opportunity_section("低 ACOS 高转化搜索词", quality_terms, "gold")
    render_opportunity_section("可提取精准投放词", exact_terms, "success")
    render_opportunity_section("可加预算广告活动", budget_campaigns, "success")
    render_opportunity_section("高转化 ASIN / 定向", high_conversion, "gold")
    render_opportunity_section("潜在长尾词", long_tail, "success")


def render_ai_report_tab(state: AnalysisState) -> None:
    st.html('<div class="ai-page-anchor"></div>')
    st.html(
        """
        <div class="ai-page-heading">
            <div class="ai-page-kicker">AI Review Report</div>
            <h2>AI 报告</h2>
            <p>展示 DeepSeek 复核生成的广告诊断报告，底部保留重新生成和原文核对入口。</p>
        </div>
        """
    )
    render_deepseek_panel(state)


def render_ai_status_summary(state: AnalysisState) -> None:
    report_ready = bool(st.session_state.get("deepseek_report"))
    reminders = len(getattr(state, "data_quality_notes", []))
    finish_reason = str(st.session_state.get("deepseek_finish_reason", "") or "")
    model = str(st.session_state.get("ds_selected_model", DEEPSEEK_MODELS[0]))
    status_items = [
        ("DeepSeek 复核", "已生成" if report_ready else "未生成", "success" if report_ready else "neutral"),
        ("数据提醒", str(reminders), "warning" if reminders else "neutral"),
        ("诊断口径", state.settings.rule_preset, "neutral"),
        ("当前模型", model, "neutral"),
    ]
    if finish_reason:
        status_items[0] = ("DeepSeek 复核", f"已生成 · {finish_reason}", "success")
    cards = "".join(
        f'<div class="ai-status-item ai-status-{escape_html(tone)}">'
        f"<span>{escape_html(label)}</span>"
        f"<strong>{escape_html(value)}</strong>"
        "</div>"
        for label, value, tone in status_items
    )
    st.html(f'<div class="ai-status-strip">{cards}</div>')


def render_ai_report_reading_flow(state: AnalysisState, content: str) -> None:
    sections = split_markdown_sections(clean_deepseek_report(content)) if content else {}
    render_ai_core_conclusion(state, sections, bool(content))
    render_ai_metric_section(state)
    render_ai_problem_section(state, sections)
    render_ai_action_section(state, sections)
    render_ai_quality_section(state)


def render_ai_core_conclusion(state: AnalysisState, sections: dict[str, str], has_report: bool) -> None:
    conclusion = build_core_conclusion(state, sections, has_report)
    target_acos = state.settings.diagnosis_config.target_acos
    acos = safe_float(state.overview.get("ACOS", 0))
    tone_label = "需要优先优化" if acos > target_acos else "表现可控"
    st.html(
        f"""
        <section class="ai-report-module ai-core-module">
            <div class="ai-core-label">核心结论 · 账户整体判断</div>
            <h3>{escape_html(tone_label)}</h3>
            <p>{escape_html(conclusion)}</p>
        </section>
        """
    )


def render_ai_metric_section(state: AnalysisState) -> None:
    overview = state.overview
    target_acos = state.settings.diagnosis_config.target_acos
    metrics = [
        ("ROAS", f"{safe_float(overview.get('ROAS')):,.2f}", "广告回报率"),
        ("ACOS", format_percent(safe_float(overview.get("ACOS"))), f"目标 {format_percent(target_acos)}"),
        ("点击量", f"{safe_float(overview.get('总点击')):,.0f}", "账户总点击"),
        ("订单数", f"{safe_float(overview.get('总订单')):,.0f}", "广告归因订单"),
        ("CTR", format_percent(safe_float(overview.get("CTR"))), "点击效率"),
        ("CVR", format_percent(safe_float(overview.get("CVR"))), "转化效率"),
        ("总花费", f"${safe_float(overview.get('总花费')):,.2f}", "广告消耗"),
        ("销售额", f"${safe_float(overview.get('总销售额')):,.2f}", "广告销售额"),
    ]
    metric_cards = "".join(
        '<div class="ai-metric-card">'
        f"<span>{escape_html(label)}</span>"
        f"<strong>{escape_html(value)}</strong>"
        f"<small>{escape_html(note)}</small>"
        "</div>"
        for label, value, note in metrics
    )
    st.html(
        f"""
        <section class="ai-report-module">
            <div class="ai-module-heading">
                <span>关键数据</span>
                <em>用于判断账户效率和转化质量的核心指标</em>
            </div>
            <div class="ai-metric-grid">{metric_cards}</div>
        </section>
        """
    )


def render_ai_problem_section(state: AnalysisState, sections: dict[str, str]) -> None:
    cards = build_problem_cards(state, sections)
    card_html = "".join(
        f'<div class="ai-risk-card ai-risk-{escape_html(card["tone"])}">'
        "<div>"
        f"<span>{escape_html(card['badge'])}</span>"
        f"<h4>{escape_html(card['title'])}</h4>"
        "</div>"
        f"<p>{escape_html(card['body'])}</p>"
        "</div>"
        for card in cards
    )
    st.html(
        f"""
        <section class="ai-report-module">
            <div class="ai-module-heading">
                <span>主要问题</span>
                <em>先看风险，再决定是否否定、降价、暂停或继续观察</em>
            </div>
            <div class="ai-risk-grid">{card_html}</div>
        </section>
        """
    )


def render_ai_action_section(state: AnalysisState, sections: dict[str, str]) -> None:
    actions = build_action_checklist(state, sections)
    action_html = "".join(
        '<div class="ai-action-item">'
        "<span></span>"
        f"<p>{escape_html(action)}</p>"
        "</div>"
        for action in actions
    )
    st.html(
        f"""
        <section class="ai-report-module">
            <div class="ai-module-heading">
                <span>优先动作建议</span>
                <em>按复核口径从数据确认到投放调整依次执行</em>
            </div>
            <div class="ai-action-list">{action_html}</div>
        </section>
        """
    )


def render_ai_quality_section(state: AnalysisState) -> None:
    notes = getattr(state, "data_quality_notes", [])
    if not notes:
        return
    note_html = "".join(f"<li>{escape_html(note)}</li>" for note in notes[:4])
    st.html(
        f"""
        <section class="ai-quality-note">
            <div class="ai-quality-title">数据质量提醒</div>
            <ul>{note_html}</ul>
        </section>
        """
    )


def build_core_conclusion(state: AnalysisState, sections: dict[str, str], has_report: bool) -> str:
    if has_report:
        body = find_section_body(sections, ["账户整体", "整体判断", "核心结论", "最大问题", "复核摘要"])
        candidate = first_report_sentence(body)
        if candidate:
            return candidate

    overview = state.overview
    target_acos = state.settings.diagnosis_config.target_acos
    acos = safe_float(overview.get("ACOS"))
    if not has_report:
        return "DeepSeek 复核尚未生成。生成后，这里会优先呈现账户结论、关键数据、主要风险和下一步动作。"
    if acos > target_acos:
        return (
            f"当前广告存在较明显的投放效率问题，ACOS {format_percent(acos)} 高于目标 "
            f"{format_percent(target_acos)}，需要优先排查数据口径与核心活动消耗结构。"
        )
    return (
        f"当前账户整体 ACOS 为 {format_percent(acos)}，未高于目标 {format_percent(target_acos)}，"
        "建议在保持健康活动预算的同时继续排查局部浪费。"
    )


def build_problem_cards(state: AnalysisState, sections: dict[str, str]) -> list[dict[str, str]]:
    overview = state.overview
    target_acos = state.settings.diagnosis_config.target_acos
    acos = safe_float(overview.get("ACOS"))
    cards: list[dict[str, str]] = []

    if acos > target_acos:
        cards.append(
            {
                "tone": "high",
                "badge": "高风险",
                "title": "ACOS 高于目标",
                "body": (
                    f"当前 ACOS {format_percent(acos)}，目标 {format_percent(target_acos)}。"
                    "建议先处理高花费、高 ACOS 或无转化的消耗来源。"
                ),
            }
        )

    notes = getattr(state, "data_quality_notes", [])
    if notes:
        cards.append(
            {
                "tone": "amber",
                "badge": "需复核",
                "title": "数据存在口径异常",
                "body": limit_report_text(notes[0], 118),
            }
        )

    high_priority = int(safe_float(state.summary.get("高优先级", 0)))
    if high_priority:
        cards.append(
            {
                "tone": "medium",
                "badge": "优先处理",
                "title": "高优先级动作较多",
                "body": f"当前共有 {high_priority} 条高优先级动作，适合先按花费和风险排序逐项处理。",
            }
        )

    listing_count = int(safe_float(state.summary.get("Listing问题", 0)))
    if listing_count:
        cards.append(
            {
                "tone": "medium",
                "badge": "承接问题",
                "title": "Listing 承接需要复查",
                "body": f"诊断中有 {listing_count} 条建议指向 Listing 检查，需复核主图、标题、价格、评价和详情页承接。",
            }
        )

    for item in collect_report_items(sections, ["最大问题", "问题诊断", "广告结构"], 2):
        if len(cards) >= 4:
            break
        cards.append({"tone": "neutral", "badge": "DeepSeek", "title": "复核补充", "body": item})

    if not cards:
        cards.append(
            {
                "tone": "neutral",
                "badge": "观察",
                "title": "暂未发现突出风险",
                "body": "当前规则没有触发明显高风险项，建议继续观察消耗变化并保留表现稳定的活动。",
            }
        )
    return cards[:4]


def build_action_checklist(state: AnalysisState, sections: dict[str, str]) -> list[str]:
    report_actions = collect_report_items(sections, ["未来", "行动", "优先", "建议"], 4)
    fallback_actions = [
        "先排查转化延迟和归因窗口，确认销售额与订单是否存在统计滞后。",
        "核查有花费无销售额、点击有订单为 0 的数据，避免把口径问题误判为后台故障。",
        "优先处理高花费、高 ACOS 或无转化的广告活动、搜索词和投放定向。",
        "保留表现优异的品牌词、高 ROAS 活动和低 ACOS 转化来源。",
        "完成数据复核后，再决定是否执行否定、降价、暂停或预算调整。",
    ]
    if int(safe_float(state.summary.get("增长建议", 0))):
        fallback_actions.append("对低 ACOS 且有订单的对象补充预算或提取精准投放，避免只做止损不做增长。")

    actions: list[str] = []
    for action in report_actions + fallback_actions:
        if action and action not in actions:
            actions.append(action)
        if len(actions) >= 6:
            break
    return actions


def find_section_body(sections: dict[str, str], keywords: list[str]) -> str:
    for title, body in sections.items():
        if any(keyword in title for keyword in keywords):
            return body
    return next(iter(sections.values()), "") if sections else ""


def collect_report_items(sections: dict[str, str], keywords: list[str], limit: int) -> list[str]:
    body = find_section_body(sections, keywords)
    paragraphs, items = split_report_body(body)
    candidates = items or paragraphs
    cleaned: list[str] = []
    for item in candidates:
        text = limit_report_text(item, 132)
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def first_report_sentence(body: str) -> str:
    paragraphs, items = split_report_body(body)
    for item in paragraphs + items:
        text = limit_report_text(item, 176)
        if text:
            return text
    return ""


def limit_report_text(text: object, max_chars: int) -> str:
    cleaned = strip_inline_markdown(str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ；;")
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip("，,。；; ") + "。"


def render_export_tab(state: AnalysisState) -> None:
    render_section_header("导出中心", "按运营场景导出，执行、复盘和深度分析使用不同文件。")
    st.html(
        """
        <section class="export-choice-grid">
            <article>
                <span>执行用</span>
                <h3>运营动作清单</h3>
                <p>给一线运营逐条处理，重点包含动作、对象、原因、优先级和核心数据。</p>
            </article>
            <article>
                <span>复盘用</span>
                <h3>管理摘要</h3>
                <p>给主管快速看账户状态、关键指标、建议数量和诊断口径。</p>
            </article>
            <article>
                <span>分析用</span>
                <h3>完整诊断包</h3>
                <p>包含透视表、动作建议、清洗明细、AI 报告和字段识别结果。</p>
            </article>
        </section>
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "下载运营动作清单 CSV",
            data=dataframe_to_csv_bytes(style_export_actions(state.actions)),
            file_name=f"amazon_ads_action_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width="stretch",
        )
    with c2:
        st.download_button(
            "下载管理摘要 CSV",
            data=dataframe_to_csv_bytes(build_management_summary_dataframe(state)),
            file_name=f"amazon_ads_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width="stretch",
        )
    with c3:
        render_excel_download(state, key="main_export")

    with st.container(border=True):
        st.subheader("完整 Excel 报告包说明")
        st.caption("包含账户总览、AI 报告、动作建议、透视表、否定词、暂停清单、调价清单、精准机会和清洗后明细。")
    render_diagnosis_self_check_panel(state)
    with st.expander("字段识别结果", expanded=False):
        st.dataframe(state.mapping_df, width="stretch", hide_index=True)
    with st.expander("清洗后数据明细", expanded=False):
        preview_limit = 500
        st.caption(f"前端仅预览前 {preview_limit:,} 行，完整清洗明细会进入 Excel 导出，避免大表预览导致页面报错。")
        st.dataframe(
            format_pivot_dataframe(state.enriched_data.head(preview_limit)),
            width="stretch",
            hide_index=True,
        )


def render_diagnosis_self_check_panel(state: AnalysisState) -> None:
    result = state.self_check
    detail = result.get("异常明细", pd.DataFrame())
    passed = int(result.get("通过项数量", 0))
    warning = int(result.get("警告项数量", 0))
    high = int(result.get("高风险异常数量", 0))
    with st.expander("诊断自检", expanded=bool(high)):
        c1, c2, c3 = st.columns(3)
        with c1:
            render_stat_card("通过项", passed, "success")
        with c2:
            render_stat_card("警告项", warning, "warning")
        with c3:
            render_stat_card("高风险异常", high, "danger" if high else "success")
        st.caption(str(result.get("修复建议", "")))
        if isinstance(detail, pd.DataFrame) and not detail.empty:
            st.dataframe(detail, width="stretch", hide_index=True)
        else:
            st.success("诊断自检通过，未发现明显规则冲突。")


def render_deepseek_panel(state: AnalysisState) -> None:
    # ━━━ Phase 1: Loading ━━━
    if st.session_state.get("ds_loading"):
        st.info("正在调用 DeepSeek API 生成复核报告，预计需要 30-90 秒，请耐心等待...")

        try:
            result = generate_deepseek_report(
                st.session_state.get("ds_api_key", ""),
                st.session_state.get("ds_selected_model", DEEPSEEK_MODELS[0]),
                state.overview,
                state.actions,
                state.aggregations,
                state.settings.diagnosis_config.target_acos,
                getattr(state, "data_quality_notes", []),
                timeout=120,
            )
        except Exception as exc:
            st.session_state["deepseek_error"] = f"{type(exc).__name__}: {exc}"
            st.session_state["deepseek_error_detail"] = format_exception_detail(exc)
            st.session_state.pop("ds_loading", None)
            st.rerun()
            return

        st.session_state.pop("ds_loading", None)
        if result.ok:
            st.session_state["deepseek_report"] = result.content
            st.session_state["deepseek_finish_reason"] = result.finish_reason
        else:
            st.session_state["deepseek_error"] = result.error
        st.session_state["force_dashboard_tab"] = "AI 报告"
        st.rerun()
        return

    # ━━━ Phase 2: Cached result / error ━━━
    if st.session_state.get("deepseek_error"):
        st.error("DeepSeek 复核没有生成成功。请检查密钥、模型名称和网络连接后重试。")
        st.caption(st.session_state["deepseek_error"])
        if st.session_state.get("deepseek_error_detail"):
            with st.expander("查看技术细节", expanded=False):
                st.code(str(st.session_state.get("deepseek_error_detail", "")), language="text")
        if st.button("清除错误", key="clear_ds_error"):
            st.session_state.pop("deepseek_error", None)
            st.session_state.pop("deepseek_error_detail", None)
            st.rerun()

    content = str(st.session_state.get("deepseek_report", "") or "")
    if content:
        render_ai_trust_note(state)
        render_deepseek_report_content(content)
    else:
        render_deepseek_empty_state()
    render_deepseek_form(state)
    if content:
        render_deepseek_raw_panel(content)


def render_ai_trust_note(state: AnalysisState) -> None:
    notes = getattr(state, "data_quality_notes", [])
    if not notes:
        return
    st.html(
        f"""
        <section class="ai-trust-note">
            <strong>AI 可信度提示</strong>
            <p>{escape_html(limit_report_text(notes[0], 132))}</p>
        </section>
        """
    )


def render_deepseek_empty_state() -> None:
    st.html(
        """
        <section class="deepseek-empty-state">
            <div>
                <span>等待生成</span>
                <h3>尚未生成 DeepSeek 复核报告</h3>
                <p>点击下方按钮后，这里会直接展示 AI 生成的完整诊断报告。</p>
            </div>
        </section>
        """
    )


def render_deepseek_report_content(content: str) -> None:
    sections = split_markdown_sections(clean_deepseek_report(content))
    if not sections:
        st.info("DeepSeek 已返回内容，但暂时无法解析为报告章节。可在下方查看完整原文。")
        return

    section_html = "".join(
        render_deepseek_section_html(title, body)
        for title, body in sections.items()
    )
    st.html(
        f"""
        <section class="deepseek-report-shell">
            <div class="deepseek-report-heading">
                <span>DeepSeek 复核报告</span>
                <p>以下内容来自 DeepSeek 返回结果，已做基础清洗和排版。</p>
            </div>
            <div class="deepseek-report-flow">{section_html}</div>
        </section>
        """
    )


def render_deepseek_section_html(title: str, body: str) -> str:
    lines = [line.strip() for line in str(body or "").splitlines()]
    pieces: list[str] = []
    list_open = False

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            pieces.append("</ul>")
            list_open = False

    for raw_line in lines:
        if not raw_line:
            close_list()
            continue
        if extract_markdown_heading(raw_line):
            continue

        bullet_match = re.match(r"^(\d+[.、]|[-*•])\s*(.+)$", raw_line)
        if bullet_match:
            if not list_open:
                pieces.append("<ul>")
                list_open = True
            pieces.append(f"<li>{escape_html(strip_inline_markdown(bullet_match.group(2)))}</li>")
            continue

        close_list()
        pieces.append(f"<p>{escape_html(strip_inline_markdown(raw_line))}</p>")

    close_list()
    body_html = "".join(pieces) or "<p>暂无内容。</p>"
    return (
        '<article class="deepseek-report-section">'
        f"<h3>{escape_html(clean_section_title(title))}</h3>"
        f"{body_html}"
        "</article>"
    )


def render_deepseek_form(state: AnalysisState) -> None:
    has_report = bool(st.session_state.get("deepseek_report"))
    stored_key = str(st.session_state.get("ds_api_key", "") or "")
    current_model = str(st.session_state.get("ds_selected_model", DEEPSEEK_MODELS[0]))
    if current_model not in DEEPSEEK_MODELS:
        current_model = DEEPSEEK_MODELS[0]

    st.html(
        """
        <section class="ai-settings-intro">
            <div>
                <strong>DeepSeek 复核</strong>
                <span>仅发送账户总览、重点动作、聚合数据和数据质量提醒，不发送全量明细。</span>
            </div>
        </section>
        """
    )

    with st.expander("AI 复核设置", expanded=not has_report):
        show_advanced = st.toggle(
            "显示高级设置",
            value=not bool(stored_key),
            key="deepseek_show_advanced",
            help="高级设置用于修改 DeepSeek 密钥或切换模型；平时只需要重新生成报告。",
        )

        with st.form("deepseek_review_form"):
            api_key = stored_key
            model = current_model
            if show_advanced:
                model_index = DEEPSEEK_MODELS.index(current_model) if current_model in DEEPSEEK_MODELS else 0
                api_key = st.text_input(
                    "DeepSeek 密钥",
                    value=stored_key,
                    type="password",
                    placeholder="sk-...",
                    key="deepseek_api_key_input",
                )
                model = st.selectbox("模型", DEEPSEEK_MODELS, index=model_index, key="deepseek_model")
            button_label = "重新生成 AI 报告" if has_report else "生成 AI 报告"
            submitted = st.form_submit_button(button_label, type="primary", width="stretch")

    if submitted:
        st.session_state.pop("deepseek_error", None)
        st.session_state.pop("deepseek_error_detail", None)
        st.session_state.pop("deepseek_report", None)
        st.session_state.pop("deepseek_finish_reason", None)
        if not api_key.strip():
            st.session_state["deepseek_error"] = "请先输入 DeepSeek 密钥。"
            st.rerun()

        st.session_state["ds_api_key"] = api_key.strip()
        st.session_state["ds_selected_model"] = model
        st.session_state["ds_loading"] = True
        st.session_state["force_dashboard_tab"] = "AI 报告"
        st.rerun()


def render_deepseek_raw_panel(content: str) -> None:
    with st.expander("查看完整 DeepSeek 原文", expanded=False):
        finish_reason = st.session_state.get("deepseek_finish_reason", "")
        meta = f"原文字符数：{len(content):,}"
        if finish_reason:
            meta += f" · finish_reason：{finish_reason}"
        st.caption(meta)
        st.text_area(
            "DeepSeek 原始返回内容",
            value=content,
            height=420,
            label_visibility="collapsed",
            disabled=True,
            key="deepseek_raw_text",
        )
        st.download_button(
            "下载 DeepSeek 原文",
            data=content.encode("utf-8"),
            file_name="deepseek_raw_report.txt",
            mime="text/plain",
            key="download_deepseek_raw",
            width="stretch",
        )


def render_upload_status(file_summaries: list[dict[str, object]]) -> None:
    if not file_summaries:
        return
    render_section_header("上传状态", "每个文件都会独立读取并识别用途。")
    for row in chunked(file_summaries, 3):
        columns = st.columns(3)
        for column, item in zip(columns, row):
            with column:
                status = str(item.get("读取状态", ""))
                tone = "success" if status == "成功" else "warning" if "字段不完整" in status else "danger"
                with st.container(border=True):
                    st.markdown(f"**{item.get('文件名', '')}**")
                    st.markdown(render_status_badge(status or "未知状态", tone), unsafe_allow_html=True)
                    st.caption(f"报表类型：{display_report_type(item.get('识别到的报表类型', ''))}")
                    st.caption(f"数据行数：{item.get('行数', 0)} · 列数：{item.get('列数', 0)}")
                    if item.get("是否参与账户总览"):
                        st.caption(f"参与账户总览：{item.get('是否参与账户总览')}")
                        st.caption(f"仅用于诊断辅助：{item.get('是否只用于诊断辅助')}")
                        st.caption(f"不参与花费 / 销售额汇总：{item.get('是否不应参与广告花费 / 销售额汇总')}")
                    if item.get("用途说明"):
                        st.caption(f"用途：{item.get('用途说明')}")
                    if item.get("排除原因"):
                        st.caption(f"排除原因：{item.get('排除原因')}")
                    missing = item.get("缺失必需字段", "") or "无"
                    st.caption(f"缺失字段：{missing}")


def render_empty_state() -> None:
    st.html(
        """
        <section class="workflow-empty">
            <div>
                <span>开始诊断</span>
                <h3>上传亚马逊广告报表后开始分析</h3>
                <p>建议同时上传搜索词、定向、广告活动或 Bulk 报表，系统会先做字段识别、数据清洗，再输出运营动作。</p>
            </div>
            <ol>
                <li>左侧上传 CSV / Excel 文件</li>
                <li>确认诊断口径和目标 ACOS</li>
                <li>点击“开始诊断”生成执行清单</li>
            </ol>
        </section>
        """
    )


def render_waiting_state() -> None:
    st.html(
        """
        <section class="workflow-empty workflow-ready">
            <div>
                <span>文件已就绪</span>
                <h3>现在可以开始诊断</h3>
                <p>下方会先展示文件识别结果。如果字段不完整，可以打开左侧高级选项启用手动字段映射。</p>
            </div>
            <ol>
                <li>检查文件读取状态</li>
                <li>确认缺失字段是否影响分析</li>
                <li>点击左侧“开始诊断”</li>
            </ol>
        </section>
        """
    )


def load_reports(uploaded_files: list[Any]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    loaded_reports: list[dict[str, object]] = []
    file_summaries: list[dict[str, object]] = []
    for uploaded_file in uploaded_files or []:
        summary = {
            "文件名": uploaded_file.name,
            "识别到的报表类型": "读取失败",
            "行数": 0,
            "列数": 0,
            "缺失必需字段": "",
            "读取状态": "失败",
            "错误信息": "",
        }
        try:
            df = read_report(uploaded_file, uploaded_file.name)
            report_type = infer_report_type(df.columns, uploaded_file.name)
            missing_fields = missing_required_fields(df.columns, report_type)
            summary.update(
                {
                    "识别到的报表类型": report_type,
                    "行数": len(df),
                    "列数": len(df.columns),
                    "缺失必需字段": "、".join(display_field_name(field) for field in missing_fields),
                    "读取状态": "成功" if not missing_fields else "成功，字段不完整",
                }
            )
            loaded_reports.append({"filename": uploaded_file.name, "report_type": report_type, "dataframe": df, "missing_fields": missing_fields})
        except Exception as exc:
            summary["错误信息"] = str(exc)
        file_summaries.append(summary)
    return loaded_reports, file_summaries


def render_manual_mapping_controls(loaded_reports: list[dict[str, object]], enabled: bool) -> dict[int, dict[str, str]]:
    if not enabled:
        return {}
    manual_mappings: dict[int, dict[str, str]] = {}
    with st.sidebar:
        st.divider()
        st.markdown("### 字段映射")
        for index, report in enumerate(loaded_reports):
            dataframe = report["dataframe"]
            with st.expander(str(report["filename"]), expanded=index == 0):
                options = ["自动识别"] + [str(column).strip() for column in dataframe.columns]
                selected_mapping: dict[str, str] = {}
                for field_key, label in CANONICAL_FIELDS.items():
                    if field_key == "source_report":
                        continue
                    selected = st.selectbox(display_field_name(label), options=options, index=0, key=f"manual_mapping_{index}_{field_key}")
                    if selected != "自动识别":
                        selected_mapping[field_key] = selected
                manual_mappings[index] = selected_mapping
    return manual_mappings


def prepare_report_frames(
    loaded_reports: list[dict[str, object]],
    manual_mappings: dict[int, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    mapping_rows = []
    cleaned_frames = []
    report_frames: list[dict[str, object]] = []
    for index, report in enumerate(loaded_reports):
        report_name = f"{report['report_type']} | {report['filename']}"
        dataframe = report["dataframe"]
        manual_mapping = (manual_mappings or {}).get(index, {})
        mapping_rows.extend(mapping_results(report_name, dataframe.columns, manual_mapping))
        cleaned = apply_field_mapping(dataframe, report_name, manual_mapping)
        cleaned_frames.append(cleaned)
        report_frames.append(
            {
                "filename": report["filename"],
                "report_type": report["report_type"],
                "source_report": report_name,
                "raw_data": dataframe,
                "cleaned_data": cleaned,
            }
        )
    mapping_df = mapping_results_dataframe(mapping_rows)
    if not mapping_df.empty:
        mapping_df["报表"] = mapping_df["Report"].astype(str).apply(_display_report_cell)
        mapping_df["标准字段"] = mapping_df["标准字段"].astype(str).apply(display_field_name)
        mapping_df = mapping_df.drop(columns=["Report"])
    cleaned_data = pd.concat(cleaned_frames, ignore_index=True) if cleaned_frames else pd.DataFrame()
    return mapping_df, cleaned_data, report_frames


def prepare_data(loaded_reports: list[dict[str, object]], manual_mappings: dict[int, dict[str, str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_df, cleaned_data, _report_frames = prepare_report_frames(loaded_reports, manual_mappings)
    return mapping_df, cleaned_data


def enrich_file_summaries_with_audit(file_summaries: list[dict[str, object]], file_audit: pd.DataFrame) -> list[dict[str, object]]:
    if file_audit.empty:
        return file_summaries
    audit_by_name = {str(row["文件名"]): row for _, row in file_audit.iterrows()}
    enriched: list[dict[str, object]] = []
    for summary in file_summaries:
        item = dict(summary)
        audit = audit_by_name.get(str(item.get("文件名", "")))
        if audit is not None:
            item.update(
                {
                    "识别到的报表类型": audit.get("report_type", item.get("识别到的报表类型", "")),
                    "是否参与账户总览": audit.get("是否参与账户总览", ""),
                    "是否只用于诊断辅助": audit.get("是否只用于诊断辅助", ""),
                    "是否不应参与广告花费 / 销售额汇总": audit.get("是否不应参与广告花费 / 销售额汇总", ""),
                    "用途说明": audit.get("用途说明", ""),
                    "排除原因": audit.get("排除原因", ""),
                    "Spend 合计": audit.get("Spend 合计", 0),
                    "Sales 合计": audit.get("Sales 合计", 0),
                    "Orders 合计": audit.get("Orders 合计", 0),
                    "Clicks 合计": audit.get("Clicks 合计", 0),
                    "Impressions 合计": audit.get("Impressions 合计", 0),
                }
            )
        enriched.append(item)
    return enriched


def build_data_quality_notes(mapping_df: pd.DataFrame, enriched_data: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if not mapping_df.empty and {"标准字段", "状态"}.issubset(mapping_df.columns):
        required_display_fields = {"曝光量", "点击量", "花费", "销售额", "订单量"}
        mapped_fields = set(
            mapping_df.loc[mapping_df["状态"].astype(str).str.contains("成功", na=False), "标准字段"].astype(str)
        )
        missing_fields = sorted(required_display_fields - mapped_fields)
        if missing_fields:
            notes.append("关键字段未完整识别：" + "、".join(missing_fields))

    if enriched_data.empty:
        notes.append("清洗后没有可分析的数据行")
        return notes

    spend_no_sales = int(((enriched_data.get("Spend", 0) > 0) & (enriched_data.get("Sales", 0) == 0)).sum())
    clicks_no_orders = int(((enriched_data.get("Clicks", 0) > 0) & (enriched_data.get("Orders", 0) == 0)).sum())
    clicks_gt_impressions = int((enriched_data.get("Clicks", 0) > enriched_data.get("Impressions", 0)).sum())
    revenue_without_spend = int(((enriched_data.get("Spend", 0) == 0) & ((enriched_data.get("Sales", 0) > 0) | (enriched_data.get("Orders", 0) > 0))).sum())

    if spend_no_sales:
        notes.append(f"{spend_no_sales} 行存在花费但销售额为 0，ACOS 会显示为 ∞，AI 不应推断后台故障")
    if clicks_no_orders:
        notes.append(f"{clicks_no_orders} 行存在点击但订单为 0，需要结合转化窗口和报表口径复核")
    if clicks_gt_impressions:
        notes.append(f"{clicks_gt_impressions} 行点击量大于曝光量，可能存在字段映射或源报表口径问题")
    if revenue_without_spend:
        notes.append(f"{revenue_without_spend} 行存在无花费销售/订单，建议确认是否混入非广告归因数据")
    return notes


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def style_export_actions(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame(columns=["优先级", "建议动作", "诊断对象", "原因"])
    columns = [
        column
        for column in [
            "优先级",
            "优先级评分",
            "诊断严格度",
            "命中规则",
            "数据充分性",
            "置信度",
            "操作风险",
            "合并动作",
            "建议动作",
            "诊断对象",
            "原因",
            "证据说明",
            "人工复核原因",
            "执行建议",
            "复核提醒",
            "目标 CPA",
            "账户平均 CTR",
            "账户平均 CVR",
            "是否保护词",
            "是否存在规则冲突",
            "Campaign Name",
            "Ad Group Name",
            "Customer Search Term",
            "Targeting",
            "Spend",
            "Sales",
            "Orders",
            "ACOS",
            "ROAS",
        ]
        if column in actions.columns
    ]
    export = actions[columns].copy()
    return rename_display_columns(export)


def build_management_summary_dataframe(state: AnalysisState) -> pd.DataFrame:
    overview = state.overview
    summary = state.summary
    source = state.account_summary_source
    rows = [
        ("诊断时间", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("诊断引擎版本", DIAGNOSIS_ENGINE_VERSION),
        ("规则配置版本", RULE_CONFIG_VERSION),
        ("诊断口径", state.settings.rule_preset),
        ("账户总览数据源", f"{display_report_type(source.report_type)} | {source.filename}" if source else "未选择"),
        ("重复计算防护", "是"),
        ("数据可信度", f"{state.data_trust_result.data_trust_score}/100 · {state.data_trust_result.data_trust_level}"),
        ("诊断安全阀", state.safety_gate.safety_level),
        ("目标 ACOS", format_percent(state.settings.diagnosis_config.target_acos)),
        ("ACOS", format_percent(safe_float(overview.get("ACOS")))),
        ("ROAS", f"{safe_float(overview.get('ROAS')):,.2f}"),
        ("总花费", f"${safe_float(overview.get('总花费')):,.2f}"),
        ("销售额", f"${safe_float(overview.get('总销售额')):,.2f}"),
        ("订单数", f"{safe_float(overview.get('总订单')):,.0f}"),
        ("高优先级动作", int(safe_float(summary.get("高优先级", 0)))),
        ("全部建议动作", int(safe_float(summary.get("总建议数", 0)))),
        ("数据质量提醒", "；".join(getattr(state, "data_quality_notes", [])) or "无"),
    ]
    return pd.DataFrame(rows, columns=["项目", "内容"])


def render_excel_download(state: AnalysisState, key: str) -> None:
    bytes_key = f"excel_export_bytes_{key}"
    name_key = f"excel_export_name_{key}"
    if st.button("生成完整诊断 Excel", type="primary", width="stretch", key=f"{key}_build"):
        with st.spinner("正在生成 Excel 工作簿..."):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state[name_key] = f"amazon_ads_diagnostic_pro_{timestamp}.xlsx"
            st.session_state[bytes_key] = build_report_bytes(state, st.session_state[name_key])

    if st.session_state.get(bytes_key):
        st.download_button(
            "下载完整诊断 Excel",
            data=st.session_state[bytes_key],
            file_name=st.session_state.get(name_key, "amazon_ads_diagnostic_pro.xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary",
            width="stretch",
            key=key,
        )


def build_report_bytes(state: AnalysisState, excel_filename: str = "") -> bytes:
    ai_sections = list(state.ai_report_sections)
    if st.session_state.get("deepseek_report") and not any(item["章节"] == "DeepSeek 复核报告" for item in ai_sections):
        ai_sections.append({"章节": "DeepSeek 复核报告", "报告内容": str(st.session_state["deepseek_report"])})
    export_tables = {
        **getattr(state, "action_pivots", build_export_pivots(state.actions)),
        **getattr(state, "aggregations", {}),
    }
    if excel_filename:
        write_diagnosis_audit_report(
            state.audit_report_path,
            state.file_audit,
            state.overview,
            state.account_summary_source,
            state.data_trust_result,
            state.safety_gate,
            state.actions,
            excel_filename,
            DIAGNOSIS_ENGINE_VERSION,
            RULE_CONFIG_VERSION,
            datetime.now(),
        )
    return build_excel_report(
        state.overview_df,
        report_to_dataframe(ai_sections),
        state.actions,
        state.negative_keywords,
        state.pause_list,
        state.bid_adjustments,
        state.exact_opportunities,
        state.enriched_data,
        state.priority_list,
        export_tables,
        file_audit=state.file_audit,
        account_summary_note=state.account_summary_note,
        basic_data_audit=state.basic_data_audit,
        data_trust=state.data_trust_df,
        reconciliation=state.reconciliation_df,
        safety_gate=state.safety_gate_df,
        operator_feedback=operator_feedback_dataframe(state.actions),
        rules_version=state.rules_version_df,
    )


def render_section_header(title: str, caption: str = "") -> None:
    st.markdown(f"## {title}")
    if caption:
        st.caption(caption)


def render_kpi_card(label: str, value: str, status: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card kpi-{escape_html(tone)}">
            <div class="kpi-card-top">
                <div class="kpi-icon">{escape_html(kpi_icon(label))}</div>
                <div class="kpi-label">{escape_html(label)}</div>
            </div>
            <div class="kpi-value">{escape_html(value)}</div>
            <div class="kpi-footer">{render_status_badge(status, tone)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: object, tone: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card stat-{escape_html(tone)}">
            <div class="stat-label">{escape_html(label)}</div>
            <div class="stat-value">{escape_html(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_icon(label: str) -> str:
    icons = {
        "总花费": "$",
        "总销售额": "¥",
        "总订单": "#",
        "ACOS": "%",
        "ROAS": "R",
        "CTR": "C",
        "CPC": "$",
        "CVR": "%",
    }
    return icons.get(label, "•")


def render_problem_section(title: str, dataframe: pd.DataFrame, tone: str) -> None:
    with st.container(border=True):
        st.markdown(render_status_badge(title, tone), unsafe_allow_html=True)
        st.caption("按优先级评分与花费排序，仅展示前 10 条。")
        if dataframe.empty:
            st.info("没有触发该类问题。")
        else:
            st.dataframe(style_action_table(dataframe.head(10)), width="stretch", hide_index=True, height=320)


def render_opportunity_section(title: str, dataframe: pd.DataFrame, tone: str) -> None:
    with st.container(border=True):
        st.markdown(render_status_badge(title, tone), unsafe_allow_html=True)
        st.caption("仅展示前 10 条可控增长机会。")
        if dataframe.empty:
            st.info("暂未发现该类机会。")
        else:
            st.dataframe(format_metric_dataframe(dataframe.head(10)), width="stretch", hide_index=True, height=320)


def render_report_card(title: str, body: str) -> None:
    with st.container(border=False):
        st.markdown(f"### {escape_html(title)}")
        paragraphs, items = split_report_body(body)
        if not paragraphs and not items:
            st.info("暂无内容")
            return
        for index, paragraph in enumerate(paragraphs):
            class_name = "ai-report-paragraph ai-report-lead" if index == 0 else "ai-report-paragraph"
            st.markdown(f'<p class="{class_name}">{escape_html(paragraph)}</p>', unsafe_allow_html=True)
        if items:
            st.markdown('<div class="ai-report-list-title">关键要点</div>', unsafe_allow_html=True)
            for item in items:
                st.markdown(
                    f'<div class="ai-report-item ai-report-item-{report_item_tone(item)}">{escape_html(item)}</div>',
                    unsafe_allow_html=True,
                )


def report_item_tone(item: str) -> str:
    text = str(item)
    if any(keyword in text for keyword in ["风险", "浪费", "偏高", "无转化", "为 0", "不足", "失真"]):
        return "risk"
    if any(keyword in text for keyword in ["建议", "执行", "优化", "降低", "否定", "暂停", "提取"]):
        return "action"
    if any(keyword in text for keyword in ["数据", "花费", "点击", "订单", "ACOS", "ROAS", "CTR", "CVR"]):
        return "evidence"
    return "neutral"


def split_report_body(body: str) -> tuple[list[str], list[str]]:
    text = normalize_report_text(body)
    if not text:
        return [], []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    items = []
    paragraphs = []
    for line in lines:
        if extract_markdown_heading(line):
            continue
        line = strip_inline_markdown(line)
        if re.match(r"^(\d+[.、]|[-*•])\s*", line):
            items.append(re.sub(r"^(\d+[.、]|[-*•])\s*", "", line).strip())
        elif "；" in line and len(line) > 120:
            paragraphs.extend([part.strip(" ；。") + "。" for part in line.split("；") if part.strip()])
        else:
            paragraphs.append(line)

    if not items and len(paragraphs) == 1 and len(paragraphs[0]) > 140:
        pieces = re.split(r"(?<=[。！？])", paragraphs[0])
        paragraphs = [piece.strip() for piece in pieces if piece.strip()]
    return paragraphs[:4], items[:5]


def normalize_report_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"</?div[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("\u3000", " ")
    text = strip_inline_markdown(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([。！？；])\s*", r"\1\n", text)
    return text.strip()


def clean_deepseek_report(content: str) -> str:
    text = str(content or "").strip()
    text = re.sub(r"^#+\s*DeepSeek\s*复核报告\s*", "", text, flags=re.I)
    text = re.sub(r"^#+\s*亚马逊广告.*?报告\s*", "", text, flags=re.I)
    text = re.sub(r"^(好的|收到|您好|你好)[，,。\s]*", "", text)
    text = re.sub(r"</?div[^>]*>", "", text)
    return text.strip()


def split_markdown_sections(content: str) -> dict[str, str]:
    text = clean_deepseek_report(content)
    if not text:
        return {}

    sections: dict[str, str] = {}
    current_title = "复核摘要"
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections[current_title] = body

    for raw_line in text.splitlines():
        line = raw_line.strip()
        title = extract_markdown_heading(line)
        if title:
            flush()
            current_title = title
            current_lines = []
            continue
        current_lines.append(raw_line)
    flush()

    if sections:
        return sections
    return {"复核摘要": text}


def extract_markdown_heading(line: str) -> str | None:
    text = str(line or "").strip()
    if not text:
        return None
    if re.match(r"^[-*•]\s+", text):
        return None

    has_heading_marker = bool(
        re.match(r"^#{1,6}\s+", text)
        or re.match(r"^(?:\*\*|__)?\s*\d{1,2}\s*[.、]\s*", text)
        or re.match(r"^(?:\*\*|__).{2,60}(?:\*\*|__)\s*[:：]?$", text)
    )
    if not has_heading_marker:
        return None

    title = clean_section_title(text)
    if title in {"报告", "结论"} or len(title) < 2 or len(title) > 40:
        return None
    return title


def clean_section_title(text: str) -> str:
    title = str(text or "").strip()
    title = re.sub(r"^#{1,6}\s*", "", title)
    title = title.strip()
    title = re.sub(r"^(?:\*\*|__)\s*", "", title)
    title = re.sub(r"\s*(?:\*\*|__)\s*$", "", title)
    title = title.strip("*_` \t")
    title = re.sub(r"^\d{1,2}\s*[.、]\s*", "", title)
    title = title.rstrip(":：").strip()
    return strip_inline_markdown(title)


def strip_inline_markdown(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    return cleaned


def render_status_badge(label: object, tone: str = "neutral") -> str:
    tone_class = {
        "success": "badge-success",
        "warning": "badge-warning",
        "danger": "badge-danger",
        "gold": "badge-gold",
        "neutral": "badge-neutral",
    }.get(tone, "badge-neutral")
    return f'<span class="badge {tone_class}">{escape_html(label)}</span>'


def render_action_filters(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        st.info("没有动作建议。")
        return actions
    c1, c2, c3, c4 = st.columns(4)
    priorities = c1.multiselect(
        "优先级",
        sorted(actions["优先级"].dropna().unique()),
        default=sorted(actions["优先级"].dropna().unique()),
        key="actions_priority_filter",
    )
    action_types = c2.multiselect(
        "动作类型",
        sorted(actions["建议动作"].dropna().unique()),
        default=sorted(actions["建议动作"].dropna().unique()),
        key="actions_type_filter",
    )
    object_types = c3.multiselect(
        "对象类型",
        sorted(actions["诊断层级"].dropna().unique()),
        default=sorted(actions["诊断层级"].dropna().unique()),
        key="actions_object_filter",
    )
    campaigns = c4.multiselect(
        "广告活动",
        sorted(actions["Campaign Name"].dropna().astype(str).unique()),
        format_func=lambda value: abbreviate_display_text(value, 28),
        key="actions_campaign_filter",
    )
    filtered = actions[
        actions["优先级"].isin(priorities)
        & actions["建议动作"].isin(action_types)
        & actions["诊断层级"].isin(object_types)
    ].copy()
    if campaigns:
        filtered = filtered[filtered["Campaign Name"].isin(campaigns)]
    return filtered


def filter_actions(action_df: pd.DataFrame, rules: list[str] | None = None, action_names: list[str] | None = None) -> pd.DataFrame:
    if action_df.empty:
        return action_df
    dataframe = action_df.copy()
    if rules:
        dataframe = dataframe[dataframe["诊断规则"].astype(str).apply(lambda value: any(rule in value for rule in rules))]
    if action_names:
        dataframe = dataframe[dataframe["合并动作"].fillna(dataframe["建议动作"]).astype(str).apply(lambda value: any(action in value for action in action_names))]
    if "优先级评分" in dataframe.columns:
        dataframe = dataframe.sort_values(["优先级评分", "Spend"], ascending=[False, False])
    return dataframe


def build_high_conversion_table(state: AnalysisState) -> pd.DataFrame:
    targeting = state.aggregations.get("Targeting", pd.DataFrame())
    asin = state.aggregations.get("ASIN", pd.DataFrame())
    if targeting.empty and asin.empty:
        return pd.DataFrame()
    combined = pd.concat([targeting, asin], ignore_index=True, sort=False)
    if combined.empty or "Orders" not in combined.columns:
        return pd.DataFrame()
    return combined[
        (combined["Orders"] >= 1)
        & (combined["ACOS"] <= state.settings.diagnosis_config.target_acos)
    ].sort_values("Orders", ascending=False).head(10)


def build_long_tail_table(state: AnalysisState) -> pd.DataFrame:
    search_terms = state.aggregations.get("搜索词", pd.DataFrame())
    if search_terms.empty:
        return pd.DataFrame()
    return search_terms[
        (search_terms["Orders"] >= 1)
        & (search_terms["Clicks"] <= 12)
    ].sort_values(["Orders", "ACOS"], ascending=[False, True]).head(10)


def style_action_table(dataframe: pd.DataFrame):
    display_columns = [
        column
        for column in [
            "优先级",
            "优先级评分",
            "execution_tier",
            "action_rank",
            "estimated_savings",
            "spend_share",
            "建议动作",
            "合并动作",
            "priority_reason",
            "downgrade_reason",
            "诊断层级",
            "诊断对象",
            "Campaign Name",
            "Ad Group Name",
            "Customer Search Term",
            "Targeting",
            "Spend",
            "Sales",
            "Orders",
            "ACOS",
            "CTR",
            "CVR",
            "原因",
        ]
        if column in dataframe.columns
    ]
    if dataframe.empty:
        return dataframe
    display_df = dataframe[display_columns].rename(columns=DISPLAY_NAME_MAP)
    display_df = truncate_display_dataframe(display_df)
    return display_df.style.format(metric_formatters(display_df)).apply(priority_row_style, axis=1)


def priority_row_style(row: pd.Series) -> list[str]:
    priority = row.get("优先级", "")
    action_text = f"{row.get('建议动作', '')} {row.get('合并动作', '')}"
    if priority == "高":
        color = "background-color: #FEF3F2; color: #7A271A;"
    elif priority == "中":
        color = "background-color: #FFFAEB; color: #7A2E0E;"
    elif any(word in action_text for word in ["提高竞价", "增加预算", "提取精准投放"]):
        color = "background-color: #ECFDF3; color: #054F31;"
    else:
        color = ""
    return [color] * len(row)


LONG_TEXT_COLUMNS = {
    "广告活动名称",
    "广告组名称",
    "客户搜索词",
    "投放定向",
    "诊断对象",
    "原因",
    "Negative Keyword",
    "否定词",
}


def truncate_display_dataframe(dataframe: pd.DataFrame, max_chars: int = 34) -> pd.DataFrame:
    display_df = dataframe.copy()
    for column in display_df.columns:
        if column in LONG_TEXT_COLUMNS or display_df[column].dtype == object:
            display_df[column] = display_df[column].apply(lambda value: abbreviate_display_text(value, max_chars))
    return display_df


def abbreviate_display_text(value: object, max_chars: int = 34) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip("_ -/，,") + "…"


def format_metric_dataframe(dataframe: pd.DataFrame):
    if dataframe.empty:
        return dataframe
    display_df = dataframe.rename(columns=DISPLAY_NAME_MAP)
    display_df = truncate_display_dataframe(display_df)
    return display_df.style.format(metric_formatters(display_df))


def format_pivot_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    display_df = dataframe.rename(columns=DISPLAY_NAME_MAP).copy()
    display_df = truncate_display_dataframe(display_df)
    for column in display_df.columns:
        if column in {"CTR", "CVR", "ACOS", "点击率", "转化率", "广告成本销售比"}:
            display_df[column] = display_df[column].apply(lambda value: format_percent(safe_float(value)))
        elif column in {"CPC", "ROAS", "Spend", "Sales", "Budget", "平均点击花费", "广告回报率", "花费", "销售额", "预算"}:
            display_df[column] = display_df[column].apply(lambda value: f"{safe_float(value):,.2f}")
        elif column in {"Impressions", "Clicks", "Orders", "曝光量", "点击量", "订单量", "优先级评分", "最高优先级评分"} or str(column).endswith("数"):
            display_df[column] = display_df[column].apply(lambda value: f"{safe_float(value):,.0f}")
    return display_df


def metric_formatters(dataframe: pd.DataFrame) -> dict[str, object]:
    percent_columns = {
        column: lambda value: format_percent(safe_float(value))
        for column in ["CTR", "CVR", "ACOS", "点击率", "转化率", "广告成本销售比"]
        if column in dataframe.columns
    }
    decimal_columns = {
        column: "{:.2f}"
        for column in ["CPC", "ROAS", "Spend", "Sales", "Budget", "平均点击花费", "广告回报率", "花费", "销售额", "预算"]
        if column in dataframe.columns
    }
    integer_columns = {
        column: "{:.0f}"
        for column in ["Impressions", "Clicks", "Orders", "曝光量", "点击量", "订单量", "优先级评分"]
        if column in dataframe.columns
    }
    return {**percent_columns, **decimal_columns, **integer_columns}


def safe_float(value: object) -> float:
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def count_bid_down(bid_adjustments: pd.DataFrame) -> int:
    if bid_adjustments.empty or "建议调价方向" not in bid_adjustments.columns:
        return 0
    return int((bid_adjustments["建议调价方向"] == "降低").sum())


def file_signature(uploaded_files: list[Any]) -> tuple[tuple[str, int], ...]:
    return tuple((file.name, int(getattr(file, "size", 0) or 0)) for file in uploaded_files or [])


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def escape_html(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def display_field_name(name: object) -> str:
    return DISPLAY_NAME_MAP.get(str(name), str(name))


def display_report_type(name: object) -> str:
    return DISPLAY_REPORT_TYPE_MAP.get(str(name), str(name))


def rename_display_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.rename(columns={column: display_field_name(column) for column in dataframe.columns})


def _display_report_cell(value: object) -> str:
    text = str(value)
    for source, target in DISPLAY_REPORT_TYPE_MAP.items():
        text = text.replace(source, target)
    return text


def inject_styles() -> None:
    css_path = Path(__file__).with_name("styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stMarkdownContainer"] h1 a,
    [data-testid="stMarkdownContainer"] h2 a,
    [data-testid="stMarkdownContainer"] h3 a,
    [data-testid="stMarkdownContainer"] h4 a,
    a.anchor-link {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        font-size: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "选择文件" !important;
        font-size: 14px !important;
    }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
