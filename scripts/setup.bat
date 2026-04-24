@echo off
REM ============================================================
REM scripts\setup.bat — Windows 环境初始化
REM 用法: 双击运行，或在 cmd/PowerShell 中执行 scripts\setup.bat
REM ============================================================

echo === 沈阳街区能耗预测 · 环境初始化 ===

REM 1. 创建虚拟环境（若不存在）
IF NOT EXIST ".venv\" (
    echo [INFO] 创建虚拟环境 .venv ...
    python -m venv .venv
)

REM 2. 激活虚拟环境
call .venv\Scripts\activate.bat
echo [INFO] 虚拟环境已激活

REM 3. 升级 pip
python -m pip install --upgrade pip

REM 4. 安装依赖
echo [INFO] 安装 requirements.txt ...
pip install -r requirements.txt

REM 5. 检查 .env
IF NOT EXIST ".env" (
    copy .env.example .env > nul
    echo [WARN] 已自动复制 .env.example -^> .env
    echo [WARN] 请用文本编辑器打开 .env，填入 DATA_ROOT 路径
)

echo.
echo === 安装完成 ===
echo 下一步: 编辑 .env 填入 DATA_ROOT（如 DATA_ROOT=G:/Knowcl）
echo 然后运行: python scripts\check_data.py
pause
