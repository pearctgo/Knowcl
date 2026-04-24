# 沈阳街区能耗预测 · Shenyang Block-level Energy Prediction

基于 [UrbanKG-KnowCL](https://github.com/tsinghua-fib-lab/UrbanKG-KnowCL)（WWW'23）的迁移改造，
将纽约社经指标预测框架迁移至**沈阳市街区能耗回归**，核心贡献为将基础 KG 扩展为建筑物城市知识图谱（bldg-UKG）。

## 精度目标

\[
\text{Acc}(\text{SV}) < \text{Acc}(\text{SI}) < \text{Acc}(\text{base-KG}) < \text{Acc}(\text{bldg-UKG}) < \text{Acc}(\text{bldg-UKG+SV}) < \text{Acc}(\text{bldg-UKG+SI})
\]

---

## 快速开始（3 步）

### 步骤 1 · 克隆并安装依赖

```bash
git clone https://github.com/<your-username>/shenyang-energy-kg.git
cd shenyang-energy-kg

# Windows
scripts\setup.bat

# Linux / Mac
bash scripts/setup.sh
```

### 步骤 2 · 配置数据路径

```bash
# 复制模板
cp .env.example .env

# 用任意文本编辑器打开 .env，填入你的数据根目录
# Windows 示例:  DATA_ROOT=G:/Knowcl
# Linux 示例:    DATA_ROOT=/data/knowcl
```

### 步骤 3 · 数据诊断（Phase 1）

```bash
python scripts/check_data.py
```

脚本会在 `<DATA_ROOT>/999-输出成果文件/00-数据检查报告/` 下生成 `data_check_report.md`。

---

## 项目结构

```
shenyang-energy-kg/
├── .env.example          # 路径配置模板（复制为 .env 后填写）
├── .gitignore
├── README.md
├── requirements.txt
├── config/
│   ├── paths.yaml        # 所有子目录路径，从 DATA_ROOT 派生
│   └── experiments/      # 每个实验一个 yaml（Phase 4 以后）
├── docs/
│   ├── CLAUDE.md         # 协作协议
│   ├── task_plan.md      # 路线图
│   ├── findings.md       # 已积累知识
│   └── progress.md       # 会话流水
├── scripts/
│   ├── check_data.py     # 数据诊断
│   ├── setup.sh          # Linux/Mac 环境初始化
│   └── setup.bat         # Windows 环境初始化
├── src/                  # Phase 2 以后填充
├── results/
│   └── experiments.csv   # 实验汇总表（入 Git）
└── tests/                # Phase 2 以后填充
```

---

## 可移植性保证

- 所有路径从 `config/paths.yaml` 派生，由 `.env` 中的 `DATA_ROOT` 驱动
- 代码使用 `pathlib.Path`，不含任何 `G:\` 或 `/data/` 硬编码
- GPU 运行时检测（`torch.cuda.is_available()`），自动回退 CPU
- 所有文件 IO 使用 `encoding='utf-8'`

---

## 引用

```bibtex
@inproceedings{knowcl2023,
  title     = {Urban Knowledge Graph Enhanced Visual Representation Learning},
  author    = {...},
  booktitle = {WWW},
  year      = {2023}
}
```
