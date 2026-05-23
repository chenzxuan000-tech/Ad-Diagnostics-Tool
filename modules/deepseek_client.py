from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import requests

from modules.metrics import format_percent


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_PRO_MODEL = "deepseek-v4-pro"
DEEPSEEK_FLASH_MODEL = "deepseek-v4-flash"
DEEPSEEK_MODELS = [DEEPSEEK_PRO_MODEL, DEEPSEEK_FLASH_MODEL]


@dataclass(frozen=True)
class DeepSeekResult:
    ok: bool
    content: str
    error: str = ""
    finish_reason: str = ""


def generate_deepseek_report(
    api_key: str,
    model: str,
    overview: dict[str, float],
    actions: pd.DataFrame,
    aggregations: dict[str, pd.DataFrame],
    target_acos: float,
    data_quality_notes: list[str] | None = None,
    timeout: int = 120,
) -> DeepSeekResult:
    api_key = api_key.strip()
    model = model.strip() or DEEPSEEK_MODELS[0]
    if not api_key:
        return DeepSeekResult(False, "", "请先输入 DeepSeek 密钥。")
    if model not in DEEPSEEK_MODELS:
        return DeepSeekResult(False, "", f"不支持的模型：{model}")

    user_prompt = _build_prompt(overview, actions, aggregations, target_acos, data_quality_notes)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是资深亚马逊广告顾问。请基于用户提供的结构化广告诊断数据，"
                    "输出中文、专业、可执行的广告优化报告。必须严格遵守事实边界，"
                    "不要编造未提供的数据、日期、报表周期、产品信息或后台故障。"
                    "不要寒暄，不要重复标题，直接输出结构化结论。"
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 3600,
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=(15, timeout),
        )
    except requests.exceptions.ConnectionError as exc:
        return DeepSeekResult(False, "", f"无法连接 DeepSeek API：{_s(str(exc))}")
    except requests.exceptions.Timeout:
        return DeepSeekResult(False, "", f"DeepSeek API 请求超时（{timeout}s），请稍后重试。")
    except requests.exceptions.RequestException as exc:
        return DeepSeekResult(False, "", f"DeepSeek API 请求异常：{_s(str(exc))}")

    # Force UTF-8 before reading body
    resp.encoding = "utf-8"

    if resp.status_code == 401:
        return DeepSeekResult(False, "", "密钥无效（401 Unauthorized），请检查 DeepSeek 密钥是否正确。")
    if resp.status_code == 402:
        return DeepSeekResult(False, "", "账户余额不足（402 Payment Required），请充值。")
    if resp.status_code == 429:
        return DeepSeekResult(False, "", "请求频率超限（429），请稍后重试。")
    if not resp.ok:
        return DeepSeekResult(False, "", f"API 返回错误 HTTP {resp.status_code}：{resp.text[:300]}")

    try:
        body = resp.json()
    except ValueError:
        return DeepSeekResult(False, "", f"API 返回非 JSON 数据（前300字符）：{resp.text[:300]}")

    choice = body.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "").strip()
    finish_reason = str(choice.get("finish_reason", ""))
    if not content:
        # Show raw response for debugging
        finish_reason = choice.get("finish_reason", "unknown")
        return DeepSeekResult(False, "", f"DeepSeek 返回空内容。finish_reason={finish_reason}，raw={json.dumps(body, ensure_ascii=False)[:400]}")
    return DeepSeekResult(True, content, finish_reason=finish_reason)


def deepseek_report_to_dataframe(content: str, model: str) -> pd.DataFrame:
    return pd.DataFrame([{"章节": f"DeepSeek 复核报告（{model}）", "报告内容": content}])


def _s(text: str) -> str:
    return text[:300]


# ── prompt builder ──

def _build_prompt(
    overview: dict[str, float],
    actions: pd.DataFrame,
    aggregations: dict[str, pd.DataFrame],
    target_acos: float,
    data_quality_notes: list[str] | None = None,
) -> str:
    context = {
        "目标 ACOS": format_percent(target_acos),
        "报表周期": "用户未提供，禁止编造具体日期或日期范围",
        "数据质量提醒": data_quality_notes or [],
        "账户总览": _overview_payload(overview),
        "高优先级动作 Top 25": _records(actions, 25),
        "广告活动 Top 15": _records(aggregations.get("广告活动", pd.DataFrame()), 15),
        "搜索词 Top 30": _records(aggregations.get("搜索词", pd.DataFrame()), 30),
        "ASIN Top 15": _records(aggregations.get("ASIN", pd.DataFrame()), 15),
    }
    return (
        "请基于以下 Amazon Ads 诊断数据，生成一份专业广告顾问报告。\n"
        "硬性约束：\n"
        "1. 所有数字只能来自下方 JSON，不得自行计算不存在的字段或编造后台数据。\n"
        "2. 报表周期未知，禁止写具体报告日期、近几天、上周、本月等时间判断。\n"
        "3. 如果销售额或订单为 0，只能表述为“上传数据口径显示为 0”；不得直接断言追踪代码故障、Listing 致命问题或广告系统故障。\n"
        "4. 对 Listing、投放匹配、追踪异常等原因只能作为假设，并标注高/中/低可能性及依据。\n"
        "5. 缺少字段或数据质量提醒中的问题必须先提示，再给行动建议。\n"
        "6. 如果某项结论缺少数据支持，请写“当前数据不足以判断”，不要补故事。\n\n"
        "输出格式：不要寒暄；不要写“好的/收到”；不要重复报告标题。"
        "请输出完整复核报告，总长度控制在 2200-3200 个中文字符。"
        "每个章节 3-5 条要点，优先用短句和项目符号；关键结论必须附带数据依据。\n\n"
        "报告必须包含：1. 账户整体判断；2. 最大问题；3. 浪费花费分析；"
        "4. 转化效率分析；5. 流量质量分析；6. 关键词/Targeting 机会；"
        "7. 广告活动结构问题；8. 未来 7 天行动计划；9. 预期改善效果。\n"
        "请给出具体可执行的动作建议，避免泛泛而谈。\n\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def _overview_payload(overview: dict[str, float]) -> dict[str, object]:
    return {
        "总曝光": round(float(overview.get("总曝光", 0)), 2),
        "总点击": round(float(overview.get("总点击", 0)), 2),
        "总花费": round(float(overview.get("总花费", 0)), 2),
        "总销售额": round(float(overview.get("总销售额", 0)), 2),
        "总订单": round(float(overview.get("总订单", 0)), 2),
        "CTR": format_percent(float(overview.get("CTR", 0))),
        "CPC": round(float(overview.get("CPC", 0)), 2),
        "CVR": format_percent(float(overview.get("CVR", 0))),
        "ACOS": format_percent(float(overview.get("ACOS", 0))),
        "ROAS": round(float(overview.get("ROAS", 0)), 2),
    }


def _records(dataframe: pd.DataFrame, limit: int) -> list[dict[str, object]]:
    if dataframe.empty:
        return []
    columns = [
        column
        for column in [
            "优先级", "优先级评分", "建议动作", "合并动作", "诊断规则",
            "诊断层级", "诊断对象", "Campaign Name", "Ad Group Name",
            "Customer Search Term", "Targeting", "ASIN", "ASIN Type",
            "Impressions", "Clicks", "Spend", "Sales", "Orders",
            "CTR", "CPC", "CVR", "ACOS", "ROAS", "原因",
        ]
        if column in dataframe.columns
    ]
    prepared = dataframe[columns].head(limit).copy()
    for col in ["CTR", "CVR", "ACOS"]:
        if col in prepared.columns:
            prepared[col] = prepared[col].apply(lambda v: format_percent(float(v or 0)))
    for col in ["Spend", "Sales", "CPC", "ROAS"]:
        if col in prepared.columns:
            prepared[col] = prepared[col].apply(lambda v: round(float(v or 0), 2))
    return prepared.fillna("").to_dict(orient="records")
