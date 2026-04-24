# task_plan.md · 项目路线图

> 本文件记录 **路线图 / 阶段状态 / 决策 / 错误**.
> 维护协议见 `CLAUDE.md § 4`. append-only, 不擦除历史.

---

## Goal (项目总目标)

参考 `tsinghua-fib-lab/UrbanKG-KnowCL` (WWW'23, 纽约社经指标预测),
在沈阳市做**街区能耗回归预测**, 验证如下精度等级:

\[
\mathrm{Acc}(\text{SV}) < \mathrm{Acc}(\text{SI}) < \mathrm{Acc}(\text{base-KG}) < \mathrm{Acc}(\text{bldg-UKG}) < \mathrm{Acc}(\text{bldg-UKG+SV}) < \mathrm{Acc}(\text{bldg-UKG+SI})
\]

指标: R² / RMSE / MAE / MAPE, log1p 空间和原空间都报.

副目标:
- **可移植**: 任何机器 `clone → pip install → edit .env → run`, 不出硬编码路径.
- **可续接**: 任何新对话用 `CLAUDE.md § 6` 模板开场即可续.
- **可复现**: 每次实验一行写进 `results/experiments.csv`, 含 git commit hash.

---

## 实验列表 (6 个, 均在 208-block 主实验集上比较)

| 实验ID | 模态 | KG | 说明 |
|---|---|---|---|
| E1 | SV (街景) | 无 | 纯视觉 baseline |
| E2 | SI (遥感) | 无 | 纯遥感 baseline |
| E3 | 无图像 | base-KG | KG-only baseline |
| E4 | 无图像 | bldg-UKG | 扩展 KG baseline |
| E5 | SV | bldg-UKG | KnowCL Stage1+2 (街景版) |
| E6 | SI | bldg-UKG | KnowCL Stage1+2 (遥感版) |

每个实验跑 7 个 visual backbone (ResNet-50 / ConvNeXt / DenseNet121 / ViT / MobileNetV3 / AttentionCNN / EfficientNet).

---

## 目录结构约定

```
shenyang-energy-kg/              ← 代码仓库, 入 Git
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── config/
│   ├── paths.yaml
│   └── experiments/
│       ├── e1_sv_resnet50.yaml
│       └── ...
├── docs/
│   ├── CLAUDE.md
│   ├── task_plan.md
│   ├── findings.md
│   └── progress.md
├── scripts/
│   ├── check_data.py            ← Phase 1, 13 项检查 C01-C13
│   ├── make_block_whitelist.py  ← Phase 1 收尾, 生成 whitelist
│   ├── setup.sh
│   └── setup.bat
├── src/
│   ├── utils/
│   │   ├── label_transform.py   ← log1p/expm1
│   │   └── metrics.py           ← RMSE/MAE/MAPE/R², MetricsLogger
│   ├── datasets/                ← Phase 2 填充
│   ├── models/                  ← Phase 4-5 填充
│   ├── losses/                  ← Phase 6 填充
│   └── engine/                  ← Phase 4-6 填充
├── results/
│   └── experiments.csv
└── tests/

<DATA_ROOT>/
├── 1-能源数据/ ... 15-遥感影像/
└── 999-输出成果文件/
    ├── 00-数据检查报告/
    │   ├── data_check_report.md
    │   └── data_check_summary.json
    ├── 01-预处理中间件/
    │   ├── block_whitelist.csv          ← 208-block 主实验集 + 划分
    │   └── sv_spatial_join.csv          ← 街景坐标空间 join 结果
    ├── 02-Stage1预训练权重/
    ├── 03-Stage2下游结果/
    ├── 04-可视化/
    └── 05-最终对比表/
```

---

## Phases

### ✅ Phase 0 · 可移植项目骨架
- [x] 新建本地代码仓库目录 (与数据目录分离)
- [x] 创建 `.env.example`
- [x] 创建 `config/paths.yaml`
- [x] 写 `.gitignore`
- [x] 写 `requirements.txt`
- [x] 写 `README.md`
- [x] 写 `scripts/setup.sh` 和 `scripts/setup.bat`
- [x] 写 `scripts/check_data.py` (C01-C13, 从零编写)
- [ ] `git init` + 首次 commit (用户执行)
- [ ] 在 GitHub 建 Private 仓库并 push (用户执行)
- [ ] 打 tag: `v0.0-scaffold` (用户执行)
- **Status:** complete (代码产出完成, git 操作待用户执行)
- **产出**: 9 个骨架文件 + check_data.py (C01-C13)

### ▶ Phase 1 · 数据诊断
- [x] 运行 `python scripts/check_data.py` → `data_check_report.md` + `data_check_summary.json`
- [x] 填写 `findings.md § 4` 所有数据事实
- [x] 确认标签格式 (JSON dict-of-dict) + 能耗列名 (energy)
- [x] 确认划分文件无泄漏 (train ∩ val ∩ test = 0)
- [x] 确认 KG Region 实体与 label block_id 完全对齐 (757/757)
- [x] 确认 SI 已有 757 张 per-block PNG (15-遥感影像)
- [x] 确认 SV 空间 join 覆盖 208 街区 (三模态交集 = 208, < 500 阈值)
- [x] 确认 KG 已含 buildingFunction + belongsToLand, Phase 3 只需补 buildingIn + buildingHeight
- [ ] **[blocking]** 运行 `make_block_whitelist.py` 生成 208-block 主实验集 + 6:2:2 划分
- [ ] 确认 208-block 划分 (train≈125/val≈41/test≈42) 无泄漏
- [ ] 将 208-block 划分写入标准文件 (`01-预处理中间件/block_whitelist.csv`)
- **Status:** in_progress
- **预计会话数:** 1 (收尾)
- **产出**: `block_whitelist.csv`, `findings.md § 4` 全部填完

### ▶ Phase 2 · 流水线基石
- [ ] `src/datasets/block_index.py`: block_id → {sv_paths, si_path, label, split}
  - SV: 基于空间 join 结果 (`sv_spatial_join.csv`) 查找图片路径
  - SI: `15-遥感影像/<block_id>.png` (直接对应)
  - label: `shenyang_region2allinfo.json` 读取 `raw[block_id]["energy"]`
- [ ] `src/datasets/splits.py`: 读 `block_whitelist.csv`, 返回 train/val/test block_id 列表
- [x] `src/utils/label_transform.py`: log1p/expm1 封装 (已完成)
- [x] `src/utils/metrics.py`: RMSE/MAE/MAPE/R², MetricsLogger (已完成)
- [ ] 单元测试: 随机抽 20 个 block, 验证 block_index 返回文件均存在
- **Status:** pending
- **预计会话数:** 2-3
- **依赖**: Phase 1 block_whitelist.csv 生成

### ▶ Phase 3 · KG 扩展 (base-KG → bldg-UKG)
- [ ] 分析现有 KG 关系: `buildingFunction` (Building→Type) + `belongsToLand` (Building→Land) 已有
- [ ] `src/kg/build_bldg_ukg.py`:
  - 从 `processed_shenyang20230318.shp` 读 Height 字段 → 生成 `buildingHeight` (Building→HeightBin)
  - 空间 join buildings ↔ 沈阳L4.shp → 生成 `buildingIn` (Building→Region)
  - 合并到现有 KG → `complete_knowledge_graph_bldg.txt`
- [ ] 验证: 新 KG 关系数 ≤ 20 (当前 15 + 扩展 2-3 = 17-18, 安全)
- [ ] 可视化: 随机抽 1 个 region, 画其 1-hop 子图
- **Status:** pending
- **预计会话数:** 2
- **依赖**: Phase 1, 2

### ▶ Phase 4 · 单模态 Baselines (E1, E2)
- [ ] `src/models/single_backbone.py`: 7 backbone 统一封装
- [ ] `src/engine/train_supervised.py`: backbone → MLP → label
- [ ] E1 (SV): ResNet50 先跑通, 获得第一个 R²
- [ ] E1 补全 7 backbone
- [ ] E2 (SI): 替换为 15-遥感影像, 跑 7 backbone
- **Status:** pending
- **预计会话数:** 5-7
- **依赖**: Phase 2

### ▶ Phase 5 · KG-only Baselines (E3, E4)
- [ ] `src/models/compgcn.py`: CompGCN (dgl 实现)
- [ ] TuckER 预训练嵌入初始化 (读 14-预训练文件/*.npz)
- [ ] E3: base-KG → region emb → MLP → energy
- [ ] E4: bldg-UKG → 同流程
- **Status:** pending
- **预计会话数:** 3-4
- **依赖**: Phase 3

### ▶ Phase 6 · KnowCL Stage-1 对比预训练 (E5, E6 上半段)
- [ ] `src/losses/info_nce.py`: 对称点积 InfoNCE, tau=0.07
- [ ] `src/models/pair_clip.py`: KnowCL 主模型
- [ ] `src/engine/train_pretrain.py`: Stage 1 脚本
- [ ] E5 (SV+bldg-UKG): CompGCN + ResNet50 跑通
- [ ] 监控: InfoNCE < 1.0? alignment/uniformity?
- [ ] E6 (SI+bldg-UKG): 同流程
- **Status:** pending
- **预计会话数:** 4-6
- **依赖**: Phase 3, 4, 5

### ▶ Phase 7 · KnowCL Stage-2 下游回归 (E5, E6 下半段)
- [ ] `src/engine/train_downstream.py`
- [ ] E5 7 backbone 全跑
- [ ] E6 7 backbone 全跑
- [ ] 验证 6 级精度不等式是否全部成立
- **Status:** pending
- **预计会话数:** 5-7
- **依赖**: Phase 6

### ▶ Phase 8 · 消融、可视化、写作
- [ ] 关系消融: 去掉 buildingIn / buildingHeight 分别跑
- [ ] backbone 消融对比图
- [ ] UMAP embedding 可视化 (按能耗着色)
- [ ] 失败案例分析 (|y - ŷ| > 3σ)
- [ ] 最终汇总图表
- [ ] 论文/毕业设计写作
- **Status:** pending
- **预计会话数:** 5-8
- **依赖**: Phase 7

---

## Decisions Made (append-only)

| 日期 | 决策 | 原因 | Phase |
|---|---|---|---|
| 2026-04-23 | 采用 planning-with-files 4 文件协议 | 解决长对话上下文丢失 | - |
| 2026-04-23 | 代码仓库与数据目录物理分离 | 代码入 Git, 数据不入 | Phase 0 |
| 2026-04-23 | 路径全部走 `config/paths.yaml` + `.env.DATA_ROOT` | 可移植性核心约束 | Phase 0 |
| 2026-04-23 | 能耗标签 log1p 变换, 指标原/log 双空间报 | KnowCL 论文范式 + 长尾数据 | Phase 2 |
| 2026-04-23 | 语义编码器用 CompGCN + TuckER 初始化 | 继承 KnowCL 论文最优配置 | Phase 5, 6 |
| 2026-04-24 | 主建筑文件选 `processed_shenyang20230318.shp` (含 Height/Function/Age/Quality) | 字段最完整 | Phase 3 |
| 2026-04-24 | SV 坐标用空间 join, 不信任 CSV `街区ID` 列 | 原始 CSV ID 有 1225 个不对应 757 块 | Phase 2 |
| 2026-04-24 | **主实验采用 208-block 子集**, 在其内重新 6:2:2 划分 | SV 空间 join 只覆盖 208 块, 需统一比较基准 | Phase 1 收尾 |
| 2026-04-24 | Phase 3 KG 扩展只补 `buildingIn` + `buildingHeight` 两种关系 | 现有 KG 已含 buildingFunction/belongsToLand, 只缺区级归属和高度 | Phase 3 |
| 2026-04-24 | SI 数据使用 `15-遥感影像/*.png` (已预裁切), 不重新裁 11-卫星数据大 TIF | 757 张 PNG 已全覆盖, 无需重复处理 | Phase 2 |

---

## Errors Encountered (append-only)

| 日期 | 现象 | 根因 | 解法 | Phase |
|---|---|---|---|---|
| 2026-04-24 | check_data C03 未识别 JSON 标签文件 | 脚本只查找 CSV/Excel | 重写 C03 支持 JSON dict-of-dict 三种结构 | Phase 0 |
| 2026-04-24 | check_data C08 未识别 shenyang_zl15_*.csv | 脚本只认 train.txt 固定名 | 重写 C08 兼容任意前缀命名, 自动检测 ID 列 | Phase 0 |
| 2026-04-24 | SV 三模态交集仅 208 (< 500 阈值) | SV CSV 原始 block_id 与主街区 ID 体系不一致, 空间 join 才是正确匹配方式 | ① 用空间 join 结果定义 SV 覆盖范围; ② 主实验改用 208-block 子集; ③ 不直接训练, 先完成 Phase 1 收尾 | Phase 1 |

---

## 当前活动 Phase

**Phase 1 in_progress** — 最后一步: 运行 `make_block_whitelist.py` 生成 208-block 主实验集.

下一个 Phase → **Phase 2 · 流水线基石** (block_index.py 是核心).
