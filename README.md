# Amazon Ads 完整诊断版

一个本地运行的亚马逊广告诊断工具。当前版本使用 Streamlit + Pandas + OpenPyXL，不接 Amazon Ads API，不接 OpenAI API；默认使用本地模板报告，也可手动填入 DeepSeek API Key 调用 DeepSeek 生成复核报告。

## 功能

- 上传 Amazon Ads Search Term Report 和 Targeting Report，支持一次上传多个文件
- 支持 CSV、XLSX、XLS
- 自动识别常见字段
- 支持手动字段映射：自动识别不准时，可以为每个上传文件单独指定标准字段
- 支持基础版 / 完整版模式切换：
  - 基础版：聚焦 Search Term / Targeting 的否定、调价、Listing 和精准投放机会
  - 完整版：额外启用 Campaign / Ad Group 结构、预算和暂停诊断
- 支持 Sponsored Products、Sponsored Brands、Sponsored Display，Bulk 报表中的 `产品 / Ad Product` 会被保留到清洗数据和动作建议中
- 自动计算 CTR、CPC、CVR、ACOS、ROAS
- 支持输入目标 ACOS，默认 30%，并提供“稳健止损 / 平衡优化 / 积极优化 / 自定义高级”诊断口径
- 支持多维度聚合：
  - 广告活动维度
  - 广告组维度
  - 搜索词维度
  - Targeting 维度
  - ASIN 维度，如果存在 `Advertised ASIN` 或 `Purchased ASIN`
- 展示账户总览 Dashboard、诊断摘要、AI 详情报告、维度聚合表、动作建议表
- 新增数据透视视图，可按广告活动、广告组、搜索词、Targeting、动作类型和优先级汇总建议数、高中低优先级数、动作数量、花费、订单、ACOS 等指标
- 动作建议支持优先级评分和高 / 中 / 低分层：高优先级聚焦证据充分且影响预算的止损项，中优先级覆盖调价、Listing 复查和明确机会，低优先级用于样本不足或温和放量
- 自动输出数据质量提醒；当销售额或订单为 0、关键字段未识别、点击量异常等情况出现时，会提示先复查字段映射和报表口径
- 导出 Excel，包含：
  - 账户总览
  - AI 详情报告
  - 动作建议清单
  - 否定词清单
  - 暂停清单
  - 调价清单
  - 精准投放机会
  - 账户总数据明细
  - 优先级清单
  - 数据透视表：广告活动、广告组、搜索词、Targeting、建议动作、优先级
  - 各维度聚合表
- Excel 自动格式化：表头加粗、冻结首行、自动筛选、自动列宽、百分比格式、美元金额格式、优先级标色、评分色阶、长文本换行、Sheet 标签上色
- 导出文件名自动包含日期时间
- 支持 DeepSeek API 复核报告：用户填入 API Key 并点击按钮后才调用；只发送账户总览、Top 动作建议和各维度 Top 聚合数据，不发送全量明细

## 🚀 快速启动（双击运行，零命令行）

### macOS 用户

1. 安装 Python3：https://www.python.org/downloads/（安装时勾选 "Add to PATH"）
2. 解压项目文件夹
3. **双击 `启动.command`**（首次可能需右键 → 打开）
4. 浏览器自动打开 http://localhost:8501

### Windows 用户

1. 安装 Python3：https://www.python.org/downloads/（安装时勾选 "Add to PATH"）
2. 解压项目文件夹
3. **双击 `启动.bat`**
4. 浏览器自动打开 http://localhost:8501

> 首次运行会自动创建虚拟环境并安装依赖，约需 1-2 分钟。之后启动秒开。

---

## 命令行启动（备用）

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 使用样例数据测试

项目内置两份样例数据：

- `sample_data/sample_search_term_report.csv`
- `sample_data/sample_targeting_report.csv`

测试步骤：

1. 启动应用。
2. 在上传区域一次性选择这两个 CSV 文件。
3. 查看“上传文件列表”，确认两个文件都显示为读取成功。
4. 查看“字段匹配预览”，确认 Spend/Cost、7 Day/14 Day Sales、7 Day/14 Day Orders、Click/Clicks 等别名能被识别。
5. 可在侧边栏开启“启用手动字段映射”，为某个样例文件手动指定字段，确认“字段匹配预览”状态变成“手动指定”。
6. 在侧边栏切换“基础版 / 完整版”，观察完整版会多出广告活动和广告组结构类建议。
7. 查看“账户总览 Dashboard”和“动作建议表”。
8. 点击“下载完整诊断 Excel”，打开导出的 Excel 检查 Sheet、筛选器、冻结首行、金额/百分比格式和优先级颜色。

样例数据覆盖：

- 有曝光无点击
- 有点击无订单
- 有订单但 ACOS 高
- ACOS 低的优质词
- 高曝光低 CTR
- `Sales = 0`
- `Clicks = 0`

## 如何判断计算结果是否正确

所有指标都使用聚合后的总量重新计算，不使用行级指标简单平均：

- `CTR = total_clicks / total_impressions`
- `CPC = total_spend / total_clicks`
- `CVR = total_orders / total_clicks`
- `ACOS = total_spend / total_sales`
- `ROAS = total_sales / total_spend`

分母为 0 时结果显示为 0，不应出现 `inf`、`Infinity`、`NaN`。

## 示例报表需要哪些字段

至少建议包含以下字段。工具会自动识别常见英文和部分中文表头：

| 标准字段 | 常见表头示例 |
| --- | --- |
| Campaign Name | Campaign Name, Campaign, 广告活动名称 |
| Ad Group Name | Ad Group Name, Ad group, 广告组名称 |
| Customer Search Term | Customer Search Term, Search Term, 搜索词 |
| Targeting | Targeting, Keyword, Product Targeting, Target, 关键词 |
| Match Type | Match Type, Match, 匹配类型 |
| Ad Product | Ad Product, Advertising Product, Product, 产品, 广告产品类型 |
| Impressions | Impressions, 展示量, 曝光量 |
| Clicks | Clicks, 点击量 |
| Spend / Cost | Spend, Cost, 花费 |
| Sales | Sales, Total Sales, 7 Day Total Sales, 14 Day Total Sales, 销售额 |
| Orders | Orders, 7 Day Total Orders (#), 14 Day Total Orders, Purchases, 订单 |
| Advertised ASIN | Advertised ASIN, Advertised Product ASIN, 广告商品 ASIN |
| Purchased ASIN | Purchased ASIN, Purchased Product ASIN, 成交 ASIN |
| Budget | Budget, Campaign Budget, Daily Budget, 预算 |

Search Term Report 通常需要 `Customer Search Term`，Targeting Report 通常需要 `Targeting`。如果部分维度字段缺失，工具会填空；如果数字字段缺失，会按 0 处理。ASIN 和 Budget 是可选字段，存在时会启用对应聚合或预算诊断。

如果你的 Bulk 报表来自 Amazon Ads 控制台，常见中文列如 `产品`、`广告活动名称（仅供参考）`、`广告组名称（仅供参考）`、`顾客搜索词`、`投放表达式`、`展示量`、`点击量`、`花费`、`销量`、`订单数量` 已内置识别。若仍识别不到，开启手动字段映射即可覆盖。

## 常见错误排查

- 文件读取失败：确认文件是 `.csv`、`.xlsx` 或 `.xls`，CSV 建议使用 UTF-8 编码。
- 字段未识别：查看“字段匹配预览”和“缺失字段提醒”，把表头改成 README 中列出的常见字段名。
- 字段识别错列：开启侧边栏“启用手动字段映射”，在对应文件展开面板里指定正确列。
- 指标全为 0：通常是数字字段未识别，重点检查 `Impressions`、`Clicks`、`Spend/Cost`、`Sales`、`Orders`。
- Search Term Report 缺少搜索词：确认表头是 `Customer Search Term` 或 `Search Term`。
- Targeting Report 缺少投放字段：确认表头是 `Targeting`、`Keyword` 或 `Product Targeting`。
- Excel 无法打开：重新导出一次，并确认本地没有用 Excel 打开同名旧文件。

## 诊断动作

- 否定精准
- 否定词组
- 暂停
- 降低竞价
- 提高竞价
- 增加预算
- 提取精准投放
- 检查 Listing
- 继续观察

## 诊断规则

规则集中在 `modules/diagnosis.py`，后续调阈值或新增规则主要改这个文件。

当前内置规则：

1. 高 ACOS 低效词
2. 低 ACOS 优质词
3. 高 CTR 低 CVR
4. 低 CTR 高曝光
5. 有销量但曝光少
6. 预算可能不足的广告活动
7. 需要暂停的广告活动 / 广告组
8. 精准投放机会词

## AI 详情报告

AI 报告默认由本地模板生成。报告会结合账户总览、动作建议、浪费花费、转化效率、流量质量、关键词机会和广告活动结构输出顾问式分析。报告结构固定为：

1. 账户整体表现总结
2. 当前最大问题
3. 浪费花费分析
4. 转化效率分析
5. 流量质量分析
6. 关键词机会分析
7. 广告活动结构问题
8. 优先级行动计划
9. 未来 7 天优化建议
10. 预期改善效果

## DeepSeek API 复核

在”AI 报告”标签页找到 DeepSeek 复核面板：

1. 输入 `DeepSeek API Key`（从 platform.deepseek.com 获取）。
2. 选择模型：`deepseek-v4-flash`（推荐，速度快）或 `deepseek-v4-pro`（深度推理）。
3. 点击”AI 复核并生成报告”。
4. 成功后报告会显示在页面，并追加到 Excel 的 `AI 详情报告` Sheet。

DeepSeek 调用只会发送：

- 账户总览指标
- 高优先级动作 Top 15
- Campaign / Search Term / ASIN 聚合 Top 数据
- 数据质量提醒

不会发送全量明细数据。没有填写 API Key 或调用失败时，本地模板报告仍可正常使用。
Prompt 中已限定事实边界：禁止编造报表日期、日期范围、后台销售数据、产品信息或追踪故障；销售额/订单为 0 时只能表述为“上传数据口径显示为 0”。

## 项目结构

```text
.
├── app.py                  # 主入口（Streamlit 应用）
├── requirements.txt        # Python 依赖
├── README.md               # 本文档
├── start.sh                # 一键启动脚本（macOS/Linux）
├── styles.css              # 自定义样式
├── .streamlit/
│   └── config.toml         # Streamlit 主题配置
├── sample_data/            # 样例数据，可用于测试
│   ├── sample_search_term_report.csv
│   └── sample_targeting_report.csv
└── modules/
    ├── __init__.py
    ├── aggregation.py       # 多维度聚合
    ├── ai_report.py         # 本地 AI 模板报告
    ├── data_loader.py       # 报表文件读取
    ├── deepseek_client.py   # DeepSeek API 客户端
    ├── diagnosis.py         # 诊断规则引擎
    ├── diagnostics.py       # 诊断配置
    ├── exporter.py          # Excel 导出
    ├── field_mapping.py     # 字段映射与识别
    ├── metrics.py           # 指标计算
    └── settings.py          # 应用设置
```

## 分享给别人

### 方式一：打包 zip（推荐）

```bash
# 在项目目录下执行
zip -r amazon-ads-diagnosis.zip . \
  -x ".venv/*" "__pycache__/*" "*.pyc" ".DS_Store"

# 把 zip 发给对方，对方解压后：
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### 方式二：Streamlit Community Cloud（免费托管）

1. 把项目推送到 GitHub 仓库
2. 访问 share.streamlit.io，用 GitHub 账号登录
3. 选择仓库和分支，点击 Deploy
4. 获得一个公开 URL，任何人都能访问

### 方式三：Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
