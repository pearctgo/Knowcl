#!/usr/bin/env bash
# ============================================================
# scripts/setup.sh — Linux / Mac 环境初始化
# 用法: bash scripts/setup.sh
# ============================================================
set -e

echo "=== 沈阳街区能耗预测 · 环境初始化 ==="

# 1. 检查 Python 版本
python3 --version | grep -E "Python 3\.(9|10|11|12)" > /dev/null 2>&1 || {
    echo "[ERROR] 需要 Python 3.9 ~ 3.12，当前版本不符合要求"
    exit 1
}

# 2. 创建虚拟环境（若不存在）
if [ ! -d ".venv" ]; then
    echo "[INFO] 创建虚拟环境 .venv ..."
    python3 -m venv .venv
fi

# 3. 激活虚拟环境
source .venv/bin/activate
echo "[INFO] 虚拟环境已激活: $(which python)"

# 4. 升级 pip
pip install --upgrade pip

# 5. 安装依赖
echo "[INFO] 安装 requirements.txt ..."
pip install -r requirements.txt

# 6. 检查 .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[WARN] 已自动复制 .env.example → .env"
    echo "[WARN] ⚠️  请编辑 .env，填入 DATA_ROOT 路径，然后重新运行 check_data.py"
fi

echo ""
echo "=== 安装完成 ==="
echo "下一步: 编辑 .env 填入 DATA_ROOT，然后运行:"
echo "  source .venv/bin/activate"
echo "  python scripts/check_data.py"
