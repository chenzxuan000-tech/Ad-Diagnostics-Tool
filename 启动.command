#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "================================================"
echo "  亚马逊广告诊断工具 - 安装 & 启动"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python3，请先安装 Python："
    echo "   https://www.python.org/downloads/"
    echo ""
    read -p "按回车键退出..."
    exit 1
fi

echo "✅ Python3 已就绪"

# Create venv if needed
if [ ! -d ".venv" ]; then
    echo ""
    echo ">>> 正在创建虚拟环境（首次运行需要 1-2 分钟）..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies if needed
if [ ! -f ".venv/.installed" ]; then
    echo ""
    echo ">>> 正在安装依赖包..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    touch .venv/.installed
    echo "✅ 依赖安装完成"
fi

echo ""
echo "================================================"
echo "  🚀 启动中，浏览器将自动打开..."
echo "  如未打开，访问: http://localhost:8501"
echo "  按 Ctrl+C 停止"
echo "================================================"
echo ""

sleep 1
open http://localhost:8501 2>/dev/null || true
streamlit run app.py --server.headless true --browser.gatherUsageStats false
