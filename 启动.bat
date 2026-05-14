@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ================================================
echo   亚马逊广告诊断工具 - 安装 ^& 启动
echo ================================================
echo.

where python3 >nul 2>nul
if %errorlevel% neq 0 (
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        echo ❌ 未找到 Python，请先安装 Python：
        echo    https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
    set PYTHON=python
) else (
    set PYTHON=python3
)

echo ✅ Python 已就绪

if not exist ".venv" (
    echo.
    echo ^>^>^> 正在创建虚拟环境（首次运行需要 1-2 分钟）...
    %PYTHON% -m venv .venv
)

call .venv\Scripts\activate.bat

if not exist ".venv\.installed" (
    echo.
    echo ^>^>^> 正在安装依赖包...
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    type nul > .venv\.installed
    echo ✅ 依赖安装完成
)

echo.
echo ================================================
echo   🚀 启动中，浏览器将自动打开...
echo   如未打开，访问: http://localhost:8501
echo   按 Ctrl+C 停止
echo ================================================
echo.

start http://localhost:8501
streamlit run app.py --server.headless true --browser.gatherUsageStats false
pause
