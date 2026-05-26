# AI Advertising Diagnosis Assistant

## 项目定位
- 本项目是本地运行的亚马逊广告诊断工具，主入口为 `app.py`。
- 技术栈以 Streamlit、Pandas、OpenPyXL 为主，优先保持简单、可读、可维护。
- 默认不接 Amazon Ads API；DeepSeek 调用为用户手动输入 API Key 后的可选功能。

## 启动方式
- macOS 双击 `启动.command`，或命令行运行 `./start.sh`。
- 备用命令：
  ```bash
  .venv/bin/python -m streamlit run app.py --server.port 8503
  ```

## 目录约定
- `app.py`：Streamlit 主界面和流程编排。
- `modules/`：核心功能模块。
- `sample_data/`：本地测试样例数据。
- `.streamlit/`：Streamlit 配置。
- 新增导出文件默认放入 `exports/`，不要混入根目录。
- 新增文档默认放入 `docs/`。
- 新增测试默认放入 `tests/`。

## 修改原则
- 先定位根因，再改代码；不要凭猜测堆补丁。
- 优先沿用现有函数和模块边界，不为了小改动引入复杂框架。
- 涉及字段识别、指标计算、导出格式时，要用样例数据或最小复现验证。
- 不要随意修改广告指标计算逻辑，包括 CTR、CPC、CVR、ACOS、ROAS。
- 不要随意修改诊断规则、阈值含义和动作生成逻辑；如确需修改，必须说明原因并补充验证。
- 不允许多个广告表现报表同时参与账户总览求和。
- Bulk 文件默认不参与账户总览。
- 热门搜索词 / Search Query 类报告不参与广告表现统计。
- Orders > 0 的对象不能进入否定词清单。
- 数据不足对象不能进入暂停建议。
- data_trust_level = 低 时不能生成 P0 今日必做。
- 外部对账差异过大时不能生成强动作。
- 页面默认不能展示海量动作。
- 运营同事易用性优先于视觉效果。
- 专业诊断宁愿少而准，不要多而乱。
- UI 优化必须保持数据可读性，运营同事易用性优先于纯视觉效果。
- 每次修改后至少检查 `app.py` 语法是否正常。
- 涉及导出、字段、指标、诊断展示时，要确认 Excel 导出仍可用。
- 每次涉及数据口径、诊断、安全阀、导出或报表识别的修改后必须运行：
  ```bash
  python3 scripts/run_basic_data_audit.py
  python3 scripts/run_golden_case_tests.py
  python3 scripts/run_diagnosis_self_check.py
  ```
- 如果测试失败，不要宣称完成。
- 每次修改后必须确认 `app.py` 可以启动。
- 不覆盖 `sample_data/` 中的原始样例文件。
- 不提交 `.venv/`、临时缓存、系统文件或本地导出结果。

## 代码风格
- Python 代码保持类型标注和清晰函数名。
- 面向用户的界面文案使用中文。
- 文件名、变量名、函数名使用英文。
- 注释只写必要的业务规则或非显然逻辑。
