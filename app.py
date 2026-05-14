from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from modules.aggregation import build_dimension_aggregations
from modules.ai_report import generate_ai_report, report_to_dataframe
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
    run_diagnosis,
    summarize_recommendations,
)
from modules.exporter import build_excel_report
from modules.field_mapping import CANONICAL_FIELDS, apply_field_mapping, mapping_results, mapping_results_dataframe
from modules.field_mapping import infer_report_type, missing_required_fields
from modules.metrics import add_metrics, calculate_account_overview, overview_dataframe
from modules.settings import AppSettings


st.set_page_config(
    page_title="亚马逊广告诊断专家版",
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
    "Search Term Report": "搜索词报表",
    "Targeting Report": "定向报表",
    "Unknown Report": "未识别报表",
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
    aggregations: dict[str, pd.DataFrame]
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
    ai_report_sections: list[dict[str, str]]


def main() -> None:
    inject_styles()

    settings, uploaded_files, start_clicked = render_sidebar_controls()
    signature = file_signature(uploaded_files)
    if st.session_state.get("active_file_signature") != signature:
        st.session_state.pop("analysis_state", None)
        st.session_state["active_file_signature"] = signature

    loaded_reports, file_summaries = load_reports(uploaded_files)
    manual_mappings = render_manual_mapping_controls(loaded_reports, settings.manual_mapping_enabled)

    render_hero(settings, len(uploaded_files))

    if not uploaded_files:
        render_empty_state()
        return

    if not start_clicked and "analysis_state" not in st.session_state:
        render_upload_status(file_summaries)
        render_waiting_state()
        return

    if start_clicked or "analysis_state" not in st.session_state:
        if not loaded_reports:
            render_upload_status(file_summaries)
            st.error("没有可分析的有效文件，请检查上传文件。")
            return
        with st.spinner("正在识别字段、计算指标并生成诊断..."):
            st.session_state["analysis_state"] = build_analysis_state(
                settings=settings,
                uploaded_files=uploaded_files,
                file_signature_value=signature,
                loaded_reports=loaded_reports,
                file_summaries=file_summaries,
                manual_mappings=manual_mappings,
            )

    render_dashboard_tabs(st.session_state["analysis_state"])


def render_sidebar_controls() -> tuple[AppSettings, list[Any], bool]:
    default = DiagnosisConfig()
    with st.sidebar:
        st.markdown("## 控制中心")
        st.caption("支持本地无网运行")

        mode = "完整版"
        target_acos_percent = st.number_input("目标 ACOS（%）", 1.0, 300.0, 30.0, 1.0)
        min_waste_clicks = st.number_input("最低点击阈值", 1, 200, default.min_waste_clicks, 1)
        min_waste_spend = st.number_input("最低花费阈值", 0.0, 10000.0, default.min_waste_spend, 1.0)

        with st.expander("高级阈值", expanded=False):
            hard_waste_clicks = st.number_input("高点击无转化阈值", 1, 300, default.hard_waste_clicks, 1)
            high_acos_multiplier = st.number_input("高 ACOS 倍数", 1.0, 10.0, default.high_acos_multiplier, 0.05)
            low_acos_multiplier = st.number_input("低 ACOS 倍数", 0.05, 1.0, default.low_acos_multiplier, 0.05)
            min_quality_orders = st.number_input("优质词最低订单", 1, 100, default.min_quality_orders, 1)
            high_ctr_percent = st.number_input("高 CTR 阈值（%）", 0.01, 100.0, default.high_ctr * 100, 0.1)
            low_ctr_percent = st.number_input("低 CTR 阈值（%）", 0.01, 100.0, default.low_ctr * 100, 0.05)
            low_cvr_percent = st.number_input("低 CVR 阈值（%）", 0.01, 100.0, default.low_cvr * 100, 0.5)
            high_impressions = st.number_input("高曝光阈值", 1, 1_000_000, default.high_impressions, 100)
            low_impressions = st.number_input("低曝光阈值", 1, 1_000_000, default.low_impressions, 50)
            min_sales_low_exposure = st.number_input("有销量低曝光销售额阈值", 0.0, 100000.0, default.min_sales_low_exposure, 5.0)
            budget_pressure_percent = st.number_input("预算压力阈值（%）", 1.0, 100.0, default.budget_pressure_ratio * 100, 1.0)
            pause_spend_multiplier = st.number_input("暂停花费倍数", 0.1, 20.0, default.pause_spend_multiplier, 0.1)
            exact_opportunity_orders = st.number_input("精准机会最低订单", 1, 100, default.exact_opportunity_orders, 1)

        manual_mapping_enabled = st.checkbox("启用手动字段映射", value=False)
        ai_report_enabled = st.checkbox("启用本地 AI 报告模板", value=True)

        st.divider()
        uploaded_files = st.file_uploader(
            "上传亚马逊广告报表",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="amazon_ads_reports",
            help="支持搜索词报表、定向报表、广告活动报表，以及 Bulk 表格。",
        )
        start_clicked = st.button("开始诊断", type="primary", width="stretch")

        if "analysis_state" in st.session_state:
            st.divider()
            st.markdown("### 导出")
            render_excel_download(st.session_state["analysis_state"], key="sidebar_export")

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
    )
    return (
        AppSettings(mode, manual_mapping_enabled, ai_report_enabled, config),
        uploaded_files or [],
        start_clicked,
    )


def build_analysis_state(
    settings: AppSettings,
    uploaded_files: list[Any],
    file_signature_value: tuple[tuple[str, int], ...],
    loaded_reports: list[dict[str, object]],
    file_summaries: list[dict[str, object]],
    manual_mappings: dict[int, dict[str, str]],
) -> AnalysisState:
    mapping_df, cleaned_data = prepare_data(loaded_reports, manual_mappings)
    enriched_data = add_metrics(cleaned_data)
    aggregations = build_dimension_aggregations(enriched_data)
    overview = calculate_account_overview(enriched_data)
    actions = run_diagnosis(enriched_data, settings.diagnosis_config, settings.mode)
    summary = summarize_recommendations(actions)
    negative_keywords = build_negative_keywords(actions)
    bid_adjustments = build_bid_adjustments(actions)
    pause_list = build_pause_list(actions)
    growth_list = build_growth_list(actions)
    exact_opportunities = build_exact_targeting_opportunities(actions)
    priority_list = build_priority_list(actions)
    ai_report_sections = (
        generate_ai_report(overview, actions, aggregations, settings.diagnosis_config.target_acos)
        if settings.ai_report_enabled
        else [{"章节": "AI 模板报告", "报告内容": "本次已关闭本地 AI 模板报告。"}]
    )

    return AnalysisState(
        settings=settings,
        file_signature=file_signature_value,
        uploaded_count=len(uploaded_files),
        file_summaries=file_summaries,
        mapping_df=mapping_df,
        cleaned_data=cleaned_data,
        enriched_data=enriched_data,
        aggregations=aggregations,
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
        ai_report_sections=ai_report_sections,
    )


def render_hero(settings: AppSettings, uploaded_count: int) -> None:
    st.markdown(
        f"""
        <div class="saas-hero">
            <div>
                <div class="hero-title">亚马逊广告诊断专家版</div>
                <p class="hero-subtitle">上传亚马逊广告报表，立即获得可执行的优化动作建议。</p>
                <div class="tag-row">
                    <span class="tag">搜索词</span>
                    <span class="tag">投放定向</span>
                    <span class="tag">广告活动</span>
                    <span class="tag">表格导出</span>
                </div>
            </div>
            <div class="hero-side">
                <div class="hero-side-row"><span class="side-label">运行环境</span><span class="side-value">支持本地无网运行</span></div>
                <div class="hero-side-row"><span class="side-label">目标 ACOS</span><span class="side-value">{settings.diagnosis_config.target_acos:.0%}</span></div>
                <div class="hero-side-row"><span class="side-label">已上传文件</span><span class="side-value">{uploaded_count}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_tabs(state: AnalysisState) -> None:
    tabs = st.tabs(["总览", "动作建议", "问题诊断", "机会分析", "AI 报告", "导出"])
    with tabs[0]:
        render_overview_tab(state)
    with tabs[1]:
        render_actions_tab(state)
    with tabs[2]:
        render_problems_tab(state)
    with tabs[3]:
        render_opportunities_tab(state)
    with tabs[4]:
        render_ai_report_tab(state)
    with tabs[5]:
        render_export_tab(state)


def render_overview_tab(state: AnalysisState) -> None:
    render_section_header("表现总览", "所有核心指标都基于聚合后的总量重新计算。")
    overview = state.overview
    target_acos = state.settings.diagnosis_config.target_acos
    kpis = [
        ("总花费", f"${overview['总花费']:,.2f}", "需要关注" if overview["总花费"] else "暂无花费", "warning"),
        ("总销售额", f"${overview['总销售额']:,.2f}", "健康" if overview["总销售额"] else "暂无销售", "success" if overview["总销售额"] else "warning"),
        ("总订单", f"{overview['总订单']:,.0f}", "健康" if overview["总订单"] else "需要关注", "success" if overview["总订单"] else "danger"),
        ("ACOS", f"{overview['ACOS']:.2%}", "高于目标" if overview["ACOS"] > target_acos else "健康", "danger" if overview["ACOS"] > target_acos else "success"),
        ("ROAS", f"{overview['ROAS']:,.2f}", "健康" if overview["ROAS"] >= 1 else "需要关注", "success" if overview["ROAS"] >= 1 else "warning"),
        ("CTR", f"{overview['CTR']:.2%}", "流量信号", "neutral"),
        ("CPC", f"${overview['CPC']:,.2f}", "单次点击成本", "neutral"),
        ("CVR", f"{overview['CVR']:.2%}", "转化信号", "neutral"),
    ]
    for row in chunked(kpis, 4):
        columns = st.columns(4)
        for column, item in zip(columns, row):
            with column:
                render_kpi_card(*item)

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


def render_actions_tab(state: AnalysisState) -> None:
    render_section_header("动作建议", "优先处理红色高优先级，再处理调价和增长机会。")
    columns = st.columns(4)
    stats = [
        ("高优先级", state.summary["高优先级"], "danger"),
        ("否定词", state.summary["否定建议"], "danger"),
        ("降低竞价", count_bid_down(state.bid_adjustments), "warning"),
        ("精准机会", len(state.exact_opportunities), "success"),
    ]
    for column, stat in zip(columns, stats):
        with column:
            render_stat_card(*stat)

    st.markdown('<div class="table-note">建议先处理高优先级浪费项：暂停、否定、明显高 ACOS；再执行调价和精准提取。</div>', unsafe_allow_html=True)
    filtered = render_action_filters(state.actions)
    if "优先级评分" in filtered.columns:
        filtered = filtered.sort_values("优先级评分", ascending=False)
    st.dataframe(style_action_table(filtered), width="stretch", hide_index=True, height=520)


def render_problems_tab(state: AnalysisState) -> None:
    render_section_header("问题诊断", "每个分区仅展示前 10 条重点问题。")
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
    render_section_header("AI 报告", "展示本地模板报告，并可选择 DeepSeek 复核。")
    report_map = {item.get("章节", ""): item.get("报告内容", "") for item in state.ai_report_sections}
    cards = [
        ("账户整体表现", "账户整体表现总结", "总"),
        ("最大问题", "当前最大问题", "重"),
        ("浪费花费分析", "浪费花费分析", "费"),
        ("转化效率分析", "转化效率分析", "转"),
        ("流量质量分析", "流量质量分析", "流"),
        ("机会分析", "关键词机会分析", "机"),
        ("广告结构问题", "广告活动结构问题", "构"),
        ("优先级行动", "优先级行动计划", "先"),
        ("未来 7 天计划", "未来 7 天优化建议", "周"),
        ("预期效果", "预期改善效果", "效"),
    ]

    summary_cols = st.columns(4)
    summary_items = [
        ("本地模板", "已生成", "success"),
        ("DeepSeek 复核", "已生成" if st.session_state.get("deepseek_report") else "未生成", "success" if st.session_state.get("deepseek_report") else "neutral"),
        ("报告章节", len(cards), "neutral"),
        ("当前模式", state.settings.mode, "neutral"),
    ]
    for column, (label, value, tone) in zip(summary_cols, summary_items):
        with column:
            render_stat_card(label, value, tone)

    labels = [title for title, _, _ in cards]
    default_title = st.session_state.get("ai_report_selected_section", labels[0])
    if default_title not in labels:
        default_title = labels[0]
    selected_title = st.radio(
        "AI 报告章节",
        options=labels,
        index=labels.index(default_title),
        horizontal=True,
        key="ai_report_selected_section",
        label_visibility="collapsed",
    )
    selected_card = next(item for item in cards if item[0] == selected_title)
    render_report_card(
        selected_card[0],
        report_map.get(selected_card[1], ""),
        selected_card[2],
    )

    render_deepseek_panel(state)

    if st.session_state.get("deepseek_report"):
        render_deepseek_report_panel(str(st.session_state["deepseek_report"]))


def render_deepseek_report_panel(content: str) -> None:
    with st.container(border=True):
        st.subheader("DeepSeek 复核报告")
        st.markdown(content)


def render_export_tab(state: AnalysisState) -> None:
    render_section_header("导出中心", "下载带格式的完整 Excel 工作簿。")
    with st.container(border=True):
        st.subheader("Excel 报告包")
        st.caption("包含账户总览、AI 报告、动作建议、否定词、暂停清单、调价清单、精准机会和清洗后明细。")
        render_excel_download(state, key="main_export")
    with st.expander("字段识别结果", expanded=False):
        st.dataframe(state.mapping_df, width="stretch", hide_index=True)
    with st.expander("清洗后数据明细", expanded=False):
        st.dataframe(format_metric_dataframe(state.enriched_data), width="stretch", hide_index=True)


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
                timeout=120,
            )
        except Exception as exc:
            import traceback
            st.session_state["deepseek_error"] = (
                f"异常：{type(exc).__name__}: {exc}\n\n```\n{traceback.format_exc()}\n```"
            )
            st.session_state.pop("ds_loading", None)
            st.rerun()
            return

        st.session_state.pop("ds_loading", None)
        if result.ok:
            st.session_state["deepseek_report"] = result.content
        else:
            st.session_state["deepseek_error"] = result.error
        st.rerun()
        return

    # ━━━ Phase 2: Cached result / error ━━━
    if st.session_state.get("deepseek_error"):
        st.error(st.session_state["deepseek_error"])
        if st.button("清除错误", key="clear_ds_error"):
            st.session_state.pop("deepseek_error", None)
            st.rerun()

    # ━━━ Phase 3: Form (always visible, no expander) ━━━
    with st.container(border=True):
        st.subheader("DeepSeek 复核")
        st.caption("发送账户总览、重点动作和重点聚合数据，不发送全量明细。")

        with st.form("deepseek_review_form"):
            c1, c2 = st.columns([3, 1])
            api_key = c1.text_input(
                "DeepSeek 密钥",
                value=st.session_state.get("ds_api_key", ""),
                type="password",
                placeholder="sk-...",
                key="deepseek_api_key_input",
            )
            model = c2.selectbox("模型", DEEPSEEK_MODELS, index=0, key="deepseek_model")
            submitted = st.form_submit_button("AI 复核并生成报告", type="primary")

        if submitted:
            st.session_state.pop("deepseek_error", None)
            st.session_state.pop("deepseek_report", None)
            if not api_key.strip():
                st.session_state["deepseek_error"] = "请先输入 DeepSeek 密钥。"
                st.rerun()

            st.session_state["ds_api_key"] = api_key.strip()
            st.session_state["ds_selected_model"] = model
            st.session_state["ds_loading"] = True
            st.rerun()

    # ━━━ Show report ━━━
    if st.session_state.get("deepseek_report"):
        render_deepseek_report_panel(st.session_state["deepseek_report"])


def render_upload_status(file_summaries: list[dict[str, object]]) -> None:
    if not file_summaries:
        return
    render_section_header("上传状态", "每个文件都会独立读取。")
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
                    missing = item.get("缺失必需字段", "") or "无"
                    st.caption(f"缺失字段：{missing}")


def render_empty_state() -> None:
    with st.container(border=True):
        st.markdown("### 上传亚马逊广告报表后开始分析")
        st.caption("支持搜索词报表、定向报表、广告活动报表，以及亚马逊 Bulk 表格。")
        st.info("请在左侧上传 CSV / Excel 文件，然后点击“开始诊断”。")


def render_waiting_state() -> None:
    with st.container(border=True):
        st.markdown("### 已准备好开始诊断")
        st.caption("文件已上传。请先查看上传状态，然后在左侧点击“开始诊断”。")


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


def prepare_data(loaded_reports: list[dict[str, object]], manual_mappings: dict[int, dict[str, str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_rows = []
    cleaned_frames = []
    for index, report in enumerate(loaded_reports):
        report_name = f"{report['report_type']} | {report['filename']}"
        dataframe = report["dataframe"]
        manual_mapping = (manual_mappings or {}).get(index, {})
        mapping_rows.extend(mapping_results(report_name, dataframe.columns, manual_mapping))
        cleaned_frames.append(apply_field_mapping(dataframe, report_name, manual_mapping))
    mapping_df = mapping_results_dataframe(mapping_rows)
    if not mapping_df.empty:
        mapping_df["报表"] = mapping_df["Report"].astype(str).apply(_display_report_cell)
        mapping_df["标准字段"] = mapping_df["标准字段"].astype(str).apply(display_field_name)
        mapping_df = mapping_df.drop(columns=["Report"])
    cleaned_data = pd.concat(cleaned_frames, ignore_index=True) if cleaned_frames else pd.DataFrame()
    return mapping_df, cleaned_data


def render_excel_download(state: AnalysisState, key: str) -> None:
    excel_bytes = build_report_bytes(state)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "下载完整诊断 Excel",
        data=excel_bytes,
        file_name=f"amazon_ads_diagnostic_pro_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        width="stretch",
        key=key,
    )


def build_report_bytes(state: AnalysisState) -> bytes:
    ai_sections = list(state.ai_report_sections)
    if st.session_state.get("deepseek_report") and not any(item["章节"] == "DeepSeek 复核报告" for item in ai_sections):
        ai_sections.append({"章节": "DeepSeek 复核报告", "报告内容": str(st.session_state["deepseek_report"])})
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
        state.aggregations,
    )


def render_section_header(title: str, caption: str = "") -> None:
    st.markdown(f"## {title}")
    if caption:
        st.caption(caption)


def render_kpi_card(label: str, value: str, status: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{escape_html(label)}</div>
            <div class="kpi-value">{escape_html(value)}</div>
            {render_status_badge(status, tone)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: object, tone: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{escape_html(label)}</div>
            <div class="stat-value">{escape_html(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def render_report_card(title: str, body: str, icon: str) -> None:
    with st.container(border=True):
        st.markdown(f"### {icon} · {title}")
        st.markdown(body or "暂无内容")


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
    priorities = c1.multiselect("优先级", sorted(actions["优先级"].dropna().unique()), default=sorted(actions["优先级"].dropna().unique()))
    action_types = c2.multiselect("动作类型", sorted(actions["建议动作"].dropna().unique()), default=sorted(actions["建议动作"].dropna().unique()))
    object_types = c3.multiselect("对象类型", sorted(actions["诊断层级"].dropna().unique()), default=sorted(actions["诊断层级"].dropna().unique()))
    campaigns = c4.multiselect("广告活动", sorted(actions["Campaign Name"].dropna().astype(str).unique()))
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
            "建议动作",
            "合并动作",
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


def format_metric_dataframe(dataframe: pd.DataFrame):
    if dataframe.empty:
        return dataframe
    display_df = dataframe.rename(columns=DISPLAY_NAME_MAP)
    return display_df.style.format(metric_formatters(display_df))


def metric_formatters(dataframe: pd.DataFrame) -> dict[str, str]:
    percent_columns = {
        column: "{:.2%}"
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
