"""独立调试脚本 — 直接测试你的报表文件是否能正确读取销售额"""
import sys
import io
import traceback

# Ensure we're importing from this project
sys.path.insert(0, '/Users/czx/Downloads/amazon-ads-diagnosis')

print("=" * 60)
print("  亚马逊广告报表读取调试")
print("=" * 60)

# 1. Check module versions
print("\n[1] 模块加载路径:")
import modules.metrics as m
import modules.field_mapping as fm
import modules.data_loader as dl
print(f"  metrics:       {m.__file__}")
print(f"  field_mapping: {fm.__file__}")
print(f"  data_loader:   {dl.__file__}")

# 2. Read the file
print("\n[2] 读取报表文件...")
target = '/Users/czx/Downloads/US商品推广_搜索词_报告.xlsx'

with open(target, 'rb') as fh:
    bio = io.BytesIO(fh.read())
bio.name = 'US商品推广_搜索词_报告.xlsx'

try:
    df = dl.read_report(bio, bio.name)
    print(f"  ✓ 读取成功: {df.shape[0]} 行 x {df.shape[1]} 列")
except Exception as e:
    print(f"  ✗ 读取失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 3. Check raw column data
print("\n[3] 原始数据校验:")
for col_name, expected_sum in [
    ('展示量', 39425),
    ('点击量', 1088),
    ('花费', 917.65),
    ('7天总销售额', 2322.08),
    ('7天总订单数(#)', 69),
]:
    if col_name in df.columns:
        actual = df[col_name].sum()
        match = "✓" if abs(actual - expected_sum) < 0.02 else "✗ MISMATCH"
        print(f"  {match} {col_name}: sum={actual}, expected={expected_sum}")
    else:
        print(f"  ✗ {col_name}: 列不存在!")

# 4. Field mapping
print("\n[4] 字段映射识别:")
report_type = fm.infer_report_type(df.columns, bio.name)
print(f"  报表类型: {report_type}")

candidates = fm.detect_field_candidates(df.columns)
for kf in ['impressions', 'clicks', 'spend', 'sales', 'orders']:
    matched = candidates.get(kf, [])
    status = '✓' if matched else '✗ 未识别!'
    print(f"  {status} {fm.CANONICAL_FIELDS[kf]}: {matched}")

# 5. Apply field mapping
print("\n[5] 应用字段映射...")
cleaned = fm.apply_field_mapping(df, f'{report_type} | {bio.name}')
sales_series = cleaned['Sales']
print(f"  Sales dtype: {sales_series.dtype}")
print(f"  Sales 前10个值: {list(sales_series.head(10))}")
non_zero = 0
for v in sales_series:
    try:
        if float(v) > 0:
            non_zero += 1
    except (ValueError, TypeError):
        pass
print(f"  Sales 非零值数量: {non_zero}")

# 6. Metrics
print("\n[6] 计算指标...")
import pandas as pd
enriched = m.add_metrics(cleaned)
overview = m.calculate_account_overview(enriched)

# 7. Final results
print("\n" + "=" * 60)
print("  最终结果")
print("=" * 60)
fields = [
    ('总花费', '$917.65'),
    ('总销售额', '$2,322.08'),
    ('总订单', '69'),
]
all_ok = True
for label, expected in fields:
    actual = overview[label]
    expected_val = float(expected.replace('$','').replace(',',''))
    ok = abs(actual - expected_val) < 0.02
    if not ok:
        all_ok = False
    status = '✓' if ok else '✗ 错误!'
    print(f"  {status} {label}: {actual} (期望: {expected})")

print(f"\n  ACOS: {overview['ACOS']:.2%}")
print(f"  ROAS: {overview['ROAS']:.2f}")
print()

if all_ok:
    print("✓ 所有数据正确！如果 Streamlit app 仍显示 $0，")
    print("  请 Ctrl+C 完全停止 Streamlit 后重新运行：")
    print("  cd /Users/czx/Downloads/amazon-ads-diagnosis")
    print("  source .venv/bin/activate")
    print("  streamlit run app.py")
else:
    print("✗ 数据不正确，需要进一步排查。")
