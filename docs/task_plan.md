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

> ⚠ 若 Phase 1.5 街景重采成功并将 SV 覆盖率提到 ≥ 500 块, 主实验基础将切换到全量 757-block.

---

## 目录结构约定

```
shenyang-energy-kg/              ← 代码仓库, 入 Git
├── .env.example
├── .env                         ← 实际 AK / DATA_ROOT, 不入 Git
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
│   ├── progress.md
│   └── baidu_ak_setup_guide.md  ← Plan B (官方 API 路径), 当前不需要
├── scripts/
│   ├── check_data.py            ← Phase 1, 13 项检查 C01-C13
│   ├── make_block_whitelist.py  ← Phase 1 收尾, 生成 whitelist
│   ├── collect_streetview_baidu_full.py  ← Phase 1.5, 走 mapsv0 内部端点
│   ├── setup.sh
│   └── setup.bat
├── phase_b_scripts/             ← Phase B 快速原型脚本 (独立轨道, 当前 G:\ 路径)
│   ├── 1_build_labels.py        ← SHP 能耗 → log+Z-score → 分层划分
│   ├── 2_predict_streetview.py  ← 7 backbone × 街景 MLP 回归
│   ├── 3_predict_remote_sensing.py  ← 7 backbone × 遥感 MLP 回归
│   ├── 4_build_base_kg.py       ← block-POI-类目-landuse KG
│   ├── 5_build_building_kg.py   ← + 建筑物 + sub_type KG
│   ├── 6_kg_models.py           ← 15 个 KG embedding 模型库
│   └── 7_train_kg.py            ← KG 训练驱动 + link prediction 评估
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
    ├── 001-街景重采_baidu/                ← 新增 Phase 1.5 输出
    │   ├── images/Block_<id>/*.jpg
    │   └── tables/
    │       ├── candidate_points_all_l4.csv
    │       ├── request_plan.csv
    │       ├── collection_log.csv
    │       ├── streetview_index.csv
    │       └── collection_summary.json
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

### ▶ Phase 1.5 · 街景全量重采 (Optional, 与 Phase 2 可并行) ⭐ 新增 2026-04-26
- [x] 用户提供线索: 项目原 `test_shenhe.py` 用 `mapsv0.bdimg.com` 内部端点采集, 不走开放平台 API
- [x] 调研: mapsv0.bdimg.com `qt=qsdata` (查 panoid) + `qt=pr3d` (下全景图) 不需要 AK, 仅需 `Referer: https://map.baidu.com/`
- [x] 写 `scripts/collect_streetview_baidu_full.py` (~960 行):
  * 全量 757 块, 不再"补 549 缺失"分支
  * WGS84 → BD09MC 全本地实现 (无 geoconv API 调用), 含 Baidu 分段多项式
  * panoid 不存在则跳过 (天然 probe, 不浪费下载)
  * 整点 4 张图全已存在则跳 panoid 查询 (二级断点续采)
  * 反爬: 真实浏览器 UA 池 + Referer + 点间默认 2s 暂停
  * 输出 streetview_index.csv 给 Phase 2 直接消费
- [x] (废弃) 申请百度 AK + 写 baidu_ak_setup_guide.md — **此方法不再需要**, 但文档保留作 Plan B
- [ ] **[smoke test]** `python scripts/collect_streetview_baidu_full.py --max-points 50` → 看 `panoid_hit_rate`
  - 决策点: panoid_hit_rate < 30% → 检查坐标转换公式或换 IP
  - 决策点: panoid_hit_rate ≥ 50% → 全跑
- [ ] **[full run]** 不带 `--max-points` 跑约 3k 候选点 × 4 方向 ≈ 12k 张
  - 估时: 3k 点 × 2s sleep + 4 张请求 × 0.3s ≈ 2 小时
- [ ] 检查 `tables/streetview_index.csv` 覆盖街区数 (目标 ≥ 500)
- [ ] 若覆盖率达标, 把 sv_index.csv 输入 make_block_whitelist.py 重新生成主实验集
- **Status:** in_progress (代码完成, 等用户跑 smoke test)
- **预计会话数:** 1-2 (smoke test 反馈 + 调参)
- **产出**: `999-输出成果文件/001-街景重采_baidu/` 整个目录, 含 streetview_index.csv
- **风险**:
  - mapsv0.bdimg.com 是百度内部端点, 非官方文档化, 后端调整可能随时让脚本失效
  - 严格说违反百度服务条款 § 2.2, 属灰色区域 (但与项目原 test_shenhe.py 同方法)
  - 部分坐标 panoid 缺失是正常 (小区/园区内部), 不能消除
  - 若 IP 被风控 (大量 http_403/418), 增大 `--sleep-between-points` 或换网络

### ▶ Phase 2 · 流水线基石
- [ ] `src/datasets/block_index.py`: block_id → {sv_paths, si_path, label, split}
  - SV: **优先**消费 Phase 1.5 输出的 `streetview_index.csv` (新一致来源); 退化时再用 12-街景文件 + 空间 join
  - SI: `15-遥感影像/<block_id>.png` (直接对应)
  - label: `shenyang_region2allinfo.json` 读取 `raw[block_id]["energy"]`
- [ ] `src/datasets/splits.py`: 读 `block_whitelist.csv`, 返回 train/val/test block_id 列表
- [x] `src/utils/label_transform.py`: log1p/expm1 封装 (已完成)
- [x] `src/utils/metrics.py`: RMSE/MAE/MAPE/R², MetricsLogger (已完成)
- [ ] 单元测试: 随机抽 20 个 block, 验证 block_index 返回文件均存在
- **Status:** pending
- **预计会话数:** 2-3
- **依赖**: Phase 1 block_whitelist.csv 生成; Phase 1.5 streetview_index.csv (可后补)

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

## Phase B · 快速原型脚本轨道 (与 Phase 1–8 并行, 独立运行) ⭐ 新增 2026-04-30

> **目的**: 在 KnowCL 主流程（Phase 2–7）完成之前, 通过独立脚本快速获得 7 backbone × 2 模态 + 15 KG 模型 × 2 KG 的对比基线数字, 为论文提供第一批实验证据.
>
> **架构特征**: 独立 `.py` 文件, 当前路径**硬编码 `G:\Knowcl`** (用户显式指定), 不依赖 `src/` 模块, 不走 `config/paths.yaml`. 正式集成进主流程前需重构.
>
> **与主流程关系**: Phase B 是 E1~E4 的"快速复现版". Phase 4/5/6/7 才是完整 KnowCL 管线. 两个轨道的结果应独立汇报, 不能互相替代.

### ⚠️ 待对账风险 (Phase B 开跑前必须确认)

| 风险 | 原始架构用法 | Phase B 脚本用法 | 状态 |
|---|---|---|---|
| 能耗标签来源 | `10-街区能耗标签/shenyang_region2allinfo.json`, 字段 `energy`, ID = `Region_N` | `8-街区数据/沈阳L4能耗.shp`, 字段 `E_Final_W5`, ID = `BlockID` | ⚠️ **待用户确认是否同一数据** |
| 街区 ID 格式 | `Region_N` (如 Region_1) | `BlockID` int | ⚠️ 需对应关系 |
| L4 BlockID 与 L5 LandID | L4 = 能耗主表 (757 块) | L5 = KG 地块 (LandID) | ⚠️ 两套编号是否互通待确认 |
| KG 来源 | 现有 `complete_knowledge_graph.txt` (852k 三元组, CompGCN 输入) | 从 SHP 零起构建 (block-POI-建筑-landuse) | ❗ 两套 KG 独立, 不能混用 embedding |
| SI 影像 | `15-遥感影像/<block_id>.png` (757 张已预裁切) | ESRI 瓦片下载 + fallback 本地大 TIF | ⚠️ Phase B SI 与主流程 SI 图源不同 |
| 遥感 fallback TIF | 不用 | `11-卫星数据/影像下载_2503152313.tif` | 确认文件存在即可 |

### Phase B 脚本清单与状态

| 脚本 | 作用 | 状态 | 输出位置 |
|---|---|---|---|
| `1_build_labels.py` | `沈阳L4能耗.shp` → log+Z-score → 5 分位分层 7:1.5:1.5 划分 | ✅ 代码完成 | `999-输出成果文件/002-能耗预测/` |
| `2_predict_streetview.py` | 7 backbone 街景特征 MLP 回归, per-backbone 缓存 | ✅ 代码完成 | 同上 |
| `3_predict_remote_sensing.py` | ESRI 优先 + TIF fallback, 7 backbone MLP 回归 | ✅ 代码完成 | 同上 |
| `4_build_base_kg.py` | block-POI-主类-子类-landuse KG | ✅ 代码完成 | `999-输出成果文件/003-知识图谱/base/` |
| `5_build_building_kg.py` | + 建筑物/高度/年代/质量/功能 + POI sub_type | ✅ 代码完成 | `999-输出成果文件/003-知识图谱/building/` |
| `6_kg_models.py` | 15 个 KG embedding 模型类 (库文件) | ✅ 代码完成, AST 通过 | — |
| `7_train_kg.py` | 自对抗训练 + filtered link prediction + embedding 导出 | ✅ 代码完成 | `003-知识图谱/<base\|building>/` |

### Phase B 运行顺序

```bash
# 先确认 ⚠️ 待对账风险全部 OK 再跑

# 能耗预测轨道
python 1_build_labels.py
python 2_predict_streetview.py --backbone all    # 首跑约 1.5-2h (GPU)
python 3_predict_remote_sensing.py --backbone all

# KG 轨道
python 4_build_base_kg.py
python 5_build_building_kg.py
python 6_kg_models.py                           # 自测 15 个 forward
python 7_train_kg.py --kg base     --model all
python 7_train_kg.py --kg building --model all
```

### Phase B 关键输出

```
999-输出成果文件/
├── 002-能耗预测/
│   ├── energy_labels.csv, label_stats.json
│   ├── sv_features_block_<bb>.npz × 7        # 特征缓存
│   ├── sv_predictions_<bb>.csv × 7
│   ├── sv_mlp_<bb>.pt × 7
│   ├── sv_all_models.csv                       # 7 backbone 预测并排
│   ├── sv_metrics_summary.csv
│   ├── rs_images/Block_*.jpg                   # 遥感影像 (一次性)
│   ├── rs_predictions_<bb>.csv × 7
│   ├── rs_metrics_summary.csv
│   ├── final_comparison.csv                    # sv × rs × 7 backbone = 14 路对比
│   └── metrics_summary_all.csv
└── 003-知识图谱/
    ├── base/
    │   ├── train/valid/test.tsv
    │   ├── entities.json, relations.json
    │   ├── block_to_entity.json               # 下游融合桥
    │   ├── embeddings_<model>.npz × 15        # 含 block_emb 子集
    │   ├── metrics_<model>.json × 15
    │   └── metrics_summary.csv
    └── building/ (结构同 base)
```

### Phase B 验收标准

- `metrics_summary_all.csv` 有 14 行 (7 sv + 7 rs), R²(log) ≥ 0.3 视为流程通畅
- `base/metrics_summary.csv` 有 15 行, MRR > 0.05 视为 KG 有信号
- 跑出数字后把两张 csv 贴回, 决定哪个 backbone / KG 模型进入 Phase 4/5 主流程

- **Status:** in_progress (代码完成, 等用户跑通并确认 ⚠️ 风险点)

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
| 2026-04-26 | **新增 Phase 1.5 街景重采**, 用百度地图前端内部端点 (`mapsv0.bdimg.com`) 全量采集 757 块 | 现有 SV 仅覆盖 208 块, 限制主实验样本量; 重采可能将基础提到 757 全量 | Phase 1.5 |
| 2026-04-26 | 街景采集方法 = 项目原 `test_shenhe.py` 同套路: qsdata 查 panoid + pr3d 下全景图 | 与原 12-街景文件 同来源, 保证图片风格一致, 训练时模态分布不漂移 | Phase 1.5 |
| 2026-04-26 | 坐标转换 WGS84 → BD09MC 全本地实现 (Baidu 分段多项式) | 不依赖任何外部 API, 也不依赖 geoconv 那个共享 AK; 坐标转换误差 < 500m, 同坐标系下完全可接受 | Phase 1.5 |
| 2026-04-26 | ~~官方 panorama/v2 API 路径~~ 降级为 Plan B, 仅当内部端点失效再启用 | 用户已确认走内部端点是项目既定方法 | Phase 1.5 |
| 2026-04-26 | API key 处理规范 § 7.1: 官方 API 三段式回退 + 灰色区域使用须知 | CLAUDE.md 协议级补全 | 协议级 |
| 2026-04-30 | 新增 Phase B 快速原型脚本轨道 (7 个独立 .py) | 在 KnowCL 主流程完成前快速获得 7 backbone × 2 模态 + 15 KG 模型基线数字 | Phase B |
| 2026-04-30 | Phase B 图像 backbone 7 选: ResNet50 / DenseNet121 / ConvNeXt-T / ViT-B16 / MobileNetV3-L / EfficientNet-B0 / AttentionCNN | 覆盖经典 CNN / 轻量 / Transformer / 注意力 4 类; AttentionCNN = ResNet50+CBAM (非 KG 侧的 AttH) | Phase B |
| 2026-04-30 | Phase B KG embedding 选 15 模型而非 CompGCN | Phase B KG 是从 SHP 零起构建的独立图, 不同于主流程 complete_knowledge_graph.txt; 用 TransE/RotatE 等标准链接预测模型做 link prediction 评估 | Phase B |
| 2026-04-30 | Phase B KG 分两层: base (block-POI-landuse) + building (+ 建筑物 + sub_type) | 对应主流程 base-KG vs bldg-UKG 的对比逻辑 | Phase B |
| 2026-04-30 | Phase B 脚本当前豁免 G:\\ 路径约束 (用户显式指定), 正式集成前重构 | 用户在本次会话中明确给定所有路径, 属于探索性脚本 | Phase B |
| 2026-04-30 | AttentionCNN 用 CBAM 实现 (ResNet50 + Channel Attention + Spatial Attention) | CBAM 是 attention CNN 文献最规范形式, 额外参数仅约 0.5M; 注意与 KG 侧 AttH (双曲注意力) 不是同一个东西 | Phase B |
| 2026-04-30 | Phase B 特征提取后缓存 .npz, MLP 重训只需秒级 | 样本量小 (~750), 端到端 finetune 过拟合风险大; 特征缓存便于反复调参 | Phase B |

---

## Errors Encountered (append-only)

| 日期 | 现象 | 根因 | 解法 | Phase |
|---|---|---|---|---|
| 2026-04-24 | check_data C03 未识别 JSON 标签文件 | 脚本只查找 CSV/Excel | 重写 C03 支持 JSON dict-of-dict 三种结构 | Phase 0 |
| 2026-04-24 | check_data C08 未识别 shenyang_zl15_*.csv | 脚本只认 train.txt 固定名 | 重写 C08 兼容任意前缀命名, 自动检测 ID 列 | Phase 0 |
| 2026-04-24 | SV 三模态交集仅 208 (< 500 阈值) | SV CSV 原始 block_id 与主街区 ID 体系不一致, 空间 join 才是正确匹配方式 | ① 用空间 join 结果定义 SV 覆盖范围; ② 主实验改用 208-block 子集; ③ 不直接训练, 先完成 Phase 1 收尾 | Phase 1 |
| 2026-04-26 | 旧 `街景采集.py` 跑到 [7/7] 抛 `ValueError: 未设置 BAIDU_MAP_AK` | 用错了路径: 旧脚本走百度开放平台官方 API, 必须 AK; 但项目原 `test_shenhe.py` 走 mapsv0 内部端点不需要 AK | 新版 `collect_streetview_baidu_full.py` 改回 mapsv0 内部端点路径, 与项目原 `test_shenhe.py` 同方法 | Phase 1.5 |
| 2026-04-26 | 旧脚本硬编码 `Path(r"G:\Knowcl")` | 违反 CLAUDE.md § 7 可移植性 | 新版从 `config/paths.yaml` 读 DATA_ROOT, 所有相对路径基于此解析 | Phase 1.5 |
| 2026-04-26 | 第一次 Claude 写的版本走百度开放平台官方 API | Claude 没看用户已上传的 `test_shenhe.py`, 误以为要从零设计采集方案 | 用户提示后, 切换到 mapsv0 内部端点同方法; 教训: 看完所有 project files 再设计 | Phase 1.5 |
| 2026-04-30 | Phase B 脚本标签来源 (`沈阳L4能耗.shp` / `E_Final_W5` / `BlockID`) 与主流程 (`shenyang_region2allinfo.json` / `energy` / `Region_N`) 不一致 | 用户在本次会话中提供的路径与原始数据文件不同 | **待用户确认**: 两份是否为同一数据不同格式? 若不同需建 BlockID↔Region_N 映射表; Phase B 跑前必须先验证 | Phase B |
| 2026-04-30 | Phase B KG 与主流程 KG 是两套独立图 (SHP 构建 vs complete_knowledge_graph.txt) | 两次构建逻辑完全不同, embedding 不可互换 | Phase B embedding 和主流程 CompGCN embedding 分开存储; 论文中分开报告, 标注来源 | Phase B |

---

## 当前活动 Phase

**Phase 1 in_progress** — 最后一步: 运行 `make_block_whitelist.py` 生成 208-block 主实验集.
**Phase 1.5 in_progress (并行)** — 代码完成, 等用户拿 AK 跑 smoke test.
**Phase B in_progress (并行)** — 7 脚本代码全部完成, 等用户: ① 确认 ⚠️ 待对账风险 ② 跑通后回填指标表.

下一个 Phase → **Phase 2 · 流水线基石** (block_index.py 是核心).
Phase B 下一步 → 用户确认 `E_Final_W5`/`BlockID` 与 `energy`/`Region_N` 对应关系, 然后跑 `1_build_labels.py` 验证.
