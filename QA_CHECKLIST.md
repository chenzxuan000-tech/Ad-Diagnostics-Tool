# QA Checklist

## 数据口径
- 账户总览必须只有一个 `account_summary_source`。
- Campaign / Targeting / Search Term 同时上传时，不允许跨报表累加 Spend / Sales / Orders / Clicks / Impressions。
- Bulk 文件和热门搜索词 / Search Query 报告不得参与广告表现汇总。
- 总览指标必须基于汇总总量重新计算，不能平均行级 CTR / CPC / CVR / ACOS / ROAS。

## 诊断安全
- `data_trust_level = 低` 时不能生成 P0。
- 外部对账差异超过 15% 时不能生成强动作。
- Orders > 0 的对象不得进入否定词清单。
- 数据不足对象不得进入暂停建议。
- 高风险动作必须标记为“需要人工复核”。

## 每次修改后验证
```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai_ads_pycache python3 -m py_compile app.py
python3 scripts/run_basic_data_audit.py
python3 scripts/run_golden_case_tests.py
python3 scripts/run_diagnosis_self_check.py
python3 -m unittest
```
