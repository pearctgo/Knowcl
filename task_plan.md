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

副目标 (均需保障):
- **可移植**: 任何机器 `clone → pip install → edit .env → run`, 不出硬编码路径.
- **可续接**: 任何新对话用 `CLAUDE.md § 6` 模板开场即可续.
- **可复现**: 每次实验一行写进 `results/experiments.csv`, 含 git commit hash.

---

## 目录结构约定 (Phase 0 冻结后填入)

```
shenyang-energy-kg/              ← 代码仓库, 入 Git
├── .env.example                 ← DATA_ROOT= 占位
├── .gitignore
├── README.md                    ← 3 步安装说明
├── requirements.txt             ← pin 版本
├── config/
│   ├── paths.yaml               ← 派生所有路径 (依赖 .env 的 DATA_ROOT)
│   └── experiments/             ← 每个实验一个 yaml
│       ├── e1_sv_resnet50.yaml
│       └── ...
├── docs/
│   ├── CLAUDE.md                ← 规则
│   ├── task_plan.md             ← 本文件
│   ├── findings.md              ← 知识
│   ├── progress.md              ← 流水
│   └── CONV_NOTES/              ← 每次对话摘要存档 (可选)
├── src/                         ← 等 Phase 2 之后再细分
├── scripts/
│   ├── check_data.py            ← Phase 1 已产出
│   ├── setup.sh
│   └── setup.bat
├── results/
│   └── experiments.csv          ← 入 Git, 主汇总表
└── tests/                       ← 可选

<DATA_ROOT>/                     ← 用户自己的数据根, 不入 Git
├── 1-能源数据/ ... 15-遥感影像/
└── 999-输出成果文件/
    ├── 00-数据检查报告/
    ├── 01-预处理中间件/
    ├── 02-Stage1预训练权重/
    ├── 03-Stage2下游结果/
    ├── 04-可视化/
    └── 05-最终对比表/
```

> `src/` 内部目录 **刻意不定**. 看完数据、Phase 2 实现 `block_index` 后再决定细分方式.

---

## Phases

### ▶ Phase 0 · 可移植项目骨架
**目的**: 使项目从 day 1 起就是"别的电脑也能直接跑".
- [ ] 新建本地代码仓库目录 (**与数据目录分离**, 如 `D:\code\shenyang-energy-kg`)
- [ ] 创建 `.env.example`, 内容为 `DATA_ROOT=G:/Knowcl` 这种占位行
- [ ] 创建 `config/paths.yaml`, 所有子目录路径从 `${DATA_ROOT}` 派生
- [ ] 写 `.gitignore` (模板在 `findings.md § .gitignore`)
- [ ] 写 `requirements.txt`, 版本全部 pin
- [ ] 写 `README.md`: 项目一句话 + 3 步安装
- [ ] 写 `scripts/setup.sh` 和 `scripts/setup.bat` (创建 venv + pip install)
- [ ] 把上一次产出的 `check_data.py` 挪进来, 改成读 `config/paths.yaml` 而不是硬编码 `G:\Knowcl`
- [ ] `git init` + 首次 commit
- [ ] 在 GitHub 建 Private 仓库 `shenyang-energy-kg`, push
- [ ] 打 tag: `v0.0-scaffold`
- **Status:** pending
- **预计会话数:** 1-2

### ▶ Phase 1 · 数据诊断
**目的**: 摸清真实数据长什么样, 填 `findings.md § 数据事实`.
- [ ] 在目标机器 `python scripts/check_data.py` (借 `.env` 指向 `G:\Knowcl`)
- [ ] 读 `G:\Knowcl\999-输出成果文件\00-数据检查报告\data_check_report.md`
- [ ] 把报告数字填到 `findings.md § 数据事实` 每一格
- [ ] 解决所有 blocking 问题:
  - [ ] 划分泄漏 (交集 > 0 必须清零)
  - [ ] 缺目录
  - [ ] 标签偏度 (> 3 上 log1p) / 零值比 > 30% 核查
  - [ ] CRS 不一致的图层统一方案
- [ ] 精算三模态交集 (label ∩ SV ∩ SI ∩ building), 落到 `01-预处理中间件/block_whitelist.csv`
- [ ] 若交集 < 500: **暂停建模, 扩数据**
- **Status:** pending
- **预计会话数:** 2-3
- **产出**: `data_check_report.md` + `findings.md` 填空 + `block_whitelist.csv`

### ▶ Phase 2 · 流水线基石
**目的**: 所有模型共用的"地基", 保证跨模态对齐 + 防泄漏.
- [ ] `src/datasets/block_index.py`: 给定 `block_id`, 返回该街区的 SV 路径列表 / SI 路径 / KG 子图 / label
- [ ] `src/datasets/splits.py`: 基于 `block_whitelist.csv` 冻结 6:2:2 split, 落盘
- [ ] `src/utils/label_transform.py`: `y_train = log1p(y)`, `y_hat = expm1(pred)`
- [ ] `src/utils/metrics.py`: RMSE/MAE/MAPE/R² 统一实现, **log 和 raw 空间都报**
- [ ] 单元测试 (tests/): 随机抽 20 个 block, 验证 block_index 返回的文件都存在
- [ ] 所有路径经 `config/paths.yaml`
- **Status:** pending
- **预计会话数:** 3-4
- **依赖**: Phase 1 完成

### ▶ Phase 3 · KG 扩展 (base → bldg-UKG)
**目的**: 在 `7-知识图谱` 的基础上, 追加 building / plot 实体和边.
- [ ] 读 `findings.md § KG 关系清单`, 确认新加的 5-10 种关系
- [ ] `src/kg/build_bldg_ukg.py`:
  - 从 `9-建筑物数据` 抽 building 实体 + 属性 (高度, 类型, 面积)
  - 从 `8-街区数据` 抽 plot 实体 (若建筑数据含 plot_id)
  - 生成新三元组: `(building, locatedIn, region)`, `(plot, adjacent, plot)` 等
  - 合并原 KG + 新三元组, 输出 `triples_bldg_ukg.tsv` + `entity2id.txt` + `relation2id.txt`
- [ ] 验证: 实体/关系/三元组统计, 度数分布, 与原 KG 的增量对比
- [ ] 可视化: 随机抽 1 个 region, 画其 1-hop 子图
- **Status:** pending
- **预计会话数:** 2-3
- **依赖**: Phase 1, 2
- **产出**: `01-预处理中间件/bldg_ukg/` 下完整 KG 文件

### ▶ Phase 4 · 单模态 Baselines (E1, E2)
**目的**: 把最简单的两条路先跑通, 拿到 baseline 数字.
- [ ] `src/models/single_backbone.py`: 统一封装 7 个 backbone 的 forward (输入 image, 输出 embedding)
- [ ] `src/engine/train_supervised.py`: 直接 backbone → MLP → label
- [ ] `config/experiments/e1_sv_resnet50.yaml`: 先跑通这一个
- [ ] E1-resnet50 出第一个能读的 RMSE/R²
- [ ] 补齐 E1 其余 6 backbone
- [ ] E2 (遥感): 替换 backbone 为 SI 版配置, 跑全 7 个
- [ ] 每跑完一个 append 一行到 `results/experiments.csv` + 一段到 `progress.md`
- **Status:** pending
- **预计会话数:** 5-7 (每 backbone ≈ 一次对话)
- **依赖**: Phase 2

### ▶ Phase 5 · KG-only Baselines (E3, E4)
- [ ] `src/models/compgcn.py`: 基于 dgl 实现 CompGCN (论文原版)
- [ ] 用 TuckER 预训练嵌入初始化 (从 `findings.md § 关键数字` 查)
- [ ] E3: 用 base-KG 训练, 得 region emb → MLP → label
- [ ] E4: 用 bldg-UKG 训练, 同流程
- **Status:** pending
- **预计会话数:** 3-4
- **依赖**: Phase 3

### ▶ Phase 6 · KnowCL Stage-1 对比预训练 (E5, E6 的上半段)
- [ ] `src/losses/info_nce.py`: 对称 InfoNCE (Image→KG + KG→Image), `tau=0.07`
- [ ] `src/models/pair_clip.py`: KnowCL 主模型, 两个 encoder + 两个 projection head
- [ ] `src/engine/train_pretrain.py`: Stage 1 训练脚本
- [ ] 跑通 E5 的 CompGCN + ResNet50 组合, 保存权重到 `02-Stage1预训练权重/`
- [ ] 监控: InfoNCE loss 末端 < 1.0? alignment/uniformity 曲线?
- [ ] 扩展到 E6 (SI)
- **Status:** pending
- **预计会话数:** 4-6
- **依赖**: Phase 3, 4, 5

### ▶ Phase 7 · KnowCL Stage-2 下游回归 (E5, E6 的下半段)
- [ ] `src/engine/train_downstream.py`: 加载 Stage-1 权重, 冻结 encoder, 训 MLP
- [ ] E5 的 7 个 backbone 全跑
- [ ] E6 的 7 个 backbone 全跑
- [ ] 验证精度等级 (Goal 的 6 级不等式是否全部成立)
- **Status:** pending
- **预计会话数:** 5-7
- **依赖**: Phase 6

### ▶ Phase 8 · 消融、可视化、写作
- [ ] 关系消融: bldg-UKG 去掉一类关系 (building / plot / flow) 再跑, 看 R² 变化
- [ ] backbone 消融: 每种模态下 7 backbone 的对比图
- [ ] UMAP embedding 可视化, 按能耗高低着色
- [ ] 失败案例分析: |y - ŷ| > 3σ 的 block 画出来看
- [ ] 最终 `results/experiments.csv` 汇总图表
- [ ] 论文/毕业设计写作
- **Status:** pending
- **预计会话数:** 5-8
- **依赖**: Phase 7

---

## Decisions Made (append-only 决策日志)

> 每个重大选型**只决定一次**, 之后不再反复争论. 模板: `YYYY-MM-DD | 决策 | 原因 | Phase`.

| 日期 | 决策 | 原因 | Phase |
|---|---|---|---|
| 2026-04-23 | 采用 planning-with-files 4 文件协议 | Manus 风格, 解决长对话上下文丢失 | - |
| 2026-04-23 | 代码仓库与数据目录**物理分离** (`D:\code\` vs `G:\Knowcl\`) | 代码入 Git, 数据不入 | Phase 0 |
| 2026-04-23 | 路径全部走 `config/paths.yaml` 派生自 `.env.DATA_ROOT` | 可移植性核心约束 | Phase 0 |
| 2026-04-23 | 能耗标签 log1p 变换, 指标原/log 空间双报 | KnowCL 论文范式 + 长尾数据 | Phase 2 |
| 2026-04-23 | 语义编码器用 CompGCN (非普通 GCN) + TuckER 初始化 | 继承 KnowCL 论文最优配置 | Phase 5, 6 |
| 2026-04-23 | 划分按 region 级 6:2:2, 固化到 `block_whitelist.csv` | 所有实验可比性 + 防泄漏 | Phase 1, 2 |
| | | | |

---

## Errors Encountered (append-only, 踩过不再踩)

> 模板: `YYYY-MM-DD | 现象 | 根因 | 解法 | Phase`.

| 日期 | 现象 | 根因 | 解法 | Phase |
|---|---|---|---|---|
| (示例) | log 训练 loss NaN | `log(0)` | 改 `log1p` | - |
| (示例) | val R² 为负 | train/val block_id 泄漏 | 按 region 重划 | - |
| | | | | |

---

## 当前活动 Phase

**现在**: 无 phase 在 `in_progress`. 下一步 → 新开对话做 **Phase 0**.

---

## 实验汇总指针

所有实验的详细结果在 `results/experiments.csv` (入 Git, 每次一行).
最近 3 条结果在 `progress.md` 最新 3 条 session 里有引用.
本文件 **不记录** 每次实验的具体数字 — 只记 phase 状态.
