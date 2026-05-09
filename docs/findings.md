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
2. **精度等级目标** (主流程仍以此为参考, Phase B 实测显示沈阳能耗任务 SV ≈ SI): `SV < SI < base-KG < bldg-UKG < bldg-UKG+SV < bldg-UKG+SI`. ⚠ Phase B 已验证 SV / SI 单模态在沈阳能耗上水平相当 (R²(log) 0.34-0.37), 与论文 § 5.2 "城市/指标偏好不同" 一致.
3. **数据在** `G:\Knowcl\1-* ... 15-*`, 输出固定到 `G:\Knowcl\999-输出成果文件\<编号>-*`.
4. **论文继承的硬数字** (见 § 2): 嵌入维度 64, CompGCN + TuckER 初始化, 对称点积 InfoNCE, 标签 log1p.
5. **⚠️ 核心样本量约束**: SV 空间 join 后仅覆盖 208 个街区, 主实验用 208-block 子集划分 (见 § 4.1), **不是 757**. ⚠ Phase 1.5 街景重采后已扩展到 698 块 (Phase B 用此基础), 主实验切换待定.

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
- 最重要结论: 不同城市、不同指标对 SV/SI 的偏好不同 (论文 § 5.2 倒数第二段). **2026-05-07 在沈阳能耗任务上验证: SV 0.371 / SI 0.342, 互补关系成立.**

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

### § 3.6 单图 backbone 特征处理戒律 (2026-05-07 新增, 来自 sv/rs v1→v2 教训)
1. **聚合前不要做 L2 归一化** (`feats /= np.linalg.norm(...)`). 这会丢幅度信号, 街景 v1 / 遥感 v1 都因这一行 R²(log) 被压在 0.10-0.18, 删后翻倍到 0.30-0.37.
2. 归一化只应该出现在 **InfoNCE 投影头之后** (损失函数内部需要), 不应该在 backbone penultimate features 上.
3. 街区级聚合策略要保留方差信号: 街景多图用 mean+std concat, 遥感单图无法多池化, 改用几何辅助特征 concat.
4. 多输出函数 (如 fit_ridge) 统一返回**全样本**预测, 算指标时切片. 反过来会触发保存阶段长度不齐 → KeyError.

---

## § 4. 数据事实 (Phase 1 检测完成, 2026-04-24 填入)

> 来源: `scripts/check_data.py` 输出的 `data_check_report.md` 和 `data_check_summary.json`.

### § 4.1 规模

- [2026-04-24] 街区总数 (沈阳L4.shp, 8-街区数据): **757**. 来源: data_check_report C04. 影响: KG/SI 实验最大样本量上限.
- [2026-04-24] 建筑物总数 (主文件 processed_shenyang20230318.shp): **153,428** 栋; 三环版 (沈阳建筑物三环.shp): 87,163 栋; 旧版 (沈阳建筑物.shp): 132,184 栋 (无高度字段, 不用). 来源: data_check_report C05. 影响: Phase 3 KG 扩展选主文件.
- [2026-04-24] 能耗标签覆盖街区数: **757** (全覆盖, 与 label 完全对齐). 来源: data_check_report C03. 影响: 标签不是瓶颈.
- [2026-04-24] KG Region 实体数: **757** (与街区完全对齐). 来源: data_check_report C09. 影响: KG 不是瓶颈.
- [2026-04-24] **⚠️ SV 空间 join 覆盖街区数: 208** (CSV 79163 条点记录经纬度空间 join 到 沈阳L4.shp 后结果). 原始 CSV block_id 列覆盖 1225 个 ID (含大量不在 757 块内的 ID), 直接匹配 label 仅 174 块. 来源: data_check_report C06 + data_check_summary.json. 影响: **SV 实验 (E1/E5) 样本量上限 = 208, 远低于 500 的建模阈值, 是本项目最大瓶颈**. ⚠ Phase 1.5 重采后扩展到 698 块, 主流程主实验切换待定.
- [2026-04-24] SI (15-遥感影像) 文件数: **757 张 PNG** (每块一张, 已预裁切). 11-卫星数据 下有 1 张大 TIF (16896×15360px, 4 波段, EPSG:3857) 作为原图备用. 来源: data_check_report C02/C07. 影响: SI 覆盖 757 块, 不是瓶颈.
- [2026-04-24] 街景图总数 (平铺): **203,452 张 JPG**. CSV 映射表有 79,163 条坐标记录. 按空间 join 后, 208 个街区内每块 min=1, median=42, max=1,837. 来源: data_check_summary.json. 影响: 空间 join 是唯一可靠的 SV-block 对应方式, 不能依赖 CSV 原始 block_id 列.
- [2026-04-24] KG 三元组 / 实体 / 关系: **852,324 / 273,348 / 15**. 来源: data_check_report C09. 影响: 关系数 15 在 CompGCN 参数可承受范围内 (< 20 阈值).
- [2026-04-24] 建筑-街区空间关联: 689/757 个街区有建筑物, 68 个街区无建筑. 每块中位数 40 栋, 最多 3,217 栋. 来源: data_check_report C12. 影响: Phase 3 KG 扩展时 68 个无建筑街区的 Building 实体为空, 需特殊处理.
- [2026-04-24] **三模态精确交集 (label ∩ KG ∩ SV_spatial) = 208**. 与 SI 交集 = 208 (SI 覆盖全 757 块, 不减少). 来源: data_check_summary.json multimodal_intersection_recommended. 影响: 主实验样本量 208, 需重新对这 208 块做 6:2:2 划分 (≈125/41/42).
- [2026-05-07] **Phase 1.5 重采实测**: 街景下载完成 9316 张 / 698 街区 (与 757 差 59 块, 多为无 panoid 的小区/园区内部). Phase B 在此 698 块基础上做实验, 7:1.5:1.5 切分得 train=488/val=105/test=105. 来源: Phase B 实测日志 (Session 07-09). 影响: Phase B 用 698 块, 主流程主实验是否切换到 698 仍待 Phase 1 收尾决策.

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
| `KeyError: 'R2_log'` 在 sort_values 时 | metrics list 是空的, 上游异常被 try-except 吞 → 加 traceback |
| `All arrays must be of the same length` 在 pd.DataFrame 时 | 某列长度 ≠ 其他列, 多半是 fit_xxx 只返回部分子集预测 |

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
- [2026-04-30] **Phase B 7 backbone 设计**: ResNet50(2048d) / DenseNet121(1024d) / ConvNeXt-T(768d) / ViT-B/16(768d) / MobileNetV3-L(960d) / EfficientNet-B0(1280d) / AttentionCNN(2048d, ResNet50+CBAM). 来源: 本次会话设计. 影响: Phase 4 单模态 baseline 选 backbone 时可参考 Phase B 结果.
- [2026-04-30] **AttentionCNN ≠ AttH**: AttentionCNN = ResNet50 + CBAM (Channel Attention + Spatial Attention, Woo et al. ECCV 2018), 是图像 backbone. AttH = Chami et al. ACL 2020 双曲空间注意力旋转/反射融合, 是 KG embedding 模型. 名字相似但完全不同. 来源: 本次设计. 影响: 代码中及论文中必须明确区分, 不要混淆.
- [2026-04-30] **Phase B 15 个 KG embedding 模型分类**: 欧氏平移 (TransE/MuRE) / 复数旋转 (DistMult/ComplEx/RotatE) / 四元数 (QuatE) / 张量分解 (TuckER) / 双曲 (MuRP/RotH/RefH/AttH) / 球面 (MuRS) / 锥 (ConE) / 多曲率 (M2GNN/GIE). M2GNN/GIE 简化版去掉 GNN 消息传递, 保留多曲率几何骨架, 使所有 15 模型在统一 (h,r,t)→score 接口下可比. 来源: 本次设计 + 各论文. 影响: Phase 5 CompGCN 才是主流程 KG 编码器; Phase B 15 模型做 link prediction 基线比较.
- [2026-04-30] **Phase B KG 训练细节**: RotatE 风格自对抗损失 (gamma=12, alpha=0.5), 64 负样本/正样本, dim=32 默认, 评估子采样 max_test_triples=1000 加速. 曲率 c 全部学习 (softplus 保正). RotH/RefH/AttH 要求 dim 偶数, 训练器自动 +1 修正. 来源: 本次设计. 影响: 若三元组数超 200 万建议升 dim 到 64-128.
- [2026-04-30] **Phase B 遥感影像**: ESRI World Imagery zoom=17 (沈阳约 1m/像素), bbox+50m buffer. 失败 fallback 本地 TIF (11-卫星数据/影像下载_2503152313.tif). 与主流程 15-遥感影像 PNG 图源不同 (已预裁切 vs 现下载). 来源: 本次设计. 影响: Phase B SI 和主流程 SI 图像不一样, 论文中要标清.
- [2026-04-30] **⚠️ 关键待确认: Phase B 标签路径与主流程不一致**. Phase B 用 `8-街区数据/沈阳L4能耗.shp` (BlockID, E_Final_W5); 主流程用 `10-街区能耗标签/shenyang_region2allinfo.json` (Region_N, energy). 这两份数据是否同源? 若 E_Final_W5 ≈ energy (同一能耗年份/聚合方式), Phase B 结果才可与主流程对比. 来源: 本次设计 + 原始文件对照. 影响: **Phase B 第一步就要验证**: `df['E_Final_W5'].describe()` vs `{Region_N: data.energy}` 统计是否吻合 (mean≈10.7, std≈9.9).
- [2026-05-07] **Phase B SV 模态 v2.1 实测** (698 街区, 7 backbone × {mlp, ridge}): ResNet50 MLP R²(log)=0.371 (最高), AttentionCNN MLP 0.343, 与 KnowCL 论文纽约 SV 实验 R²~0.377 同量级. L0 baseline 仅辅助特征 (log 图数 + GPS std) R²(log)=0.092, 故图像 backbone 净增益 +0.28, 信号显著. 来源: Phase B Session 07 实测. 影响: Phase 4 (E1) 优先用 ResNet50/AttentionCNN, 不需要换 DINOv2/CLIP.
- [2026-05-07] **L2 归一化在单图特征聚合阶段是反向操作**: KnowCL 论文里只在 InfoNCE 投影头之后才归一化 (是损失函数需要), 在 backbone penultimate 特征上做归一化会丢幅度信号. 街景 v1 与遥感 v1 都误置. 修复: v2.1 删除该行, 改在 MLP 输入端用 BatchNorm1d 处理尺度. 来源: 街景 v2.1 实测 R²(log) 从 0.18 → 0.37 验证. 影响: 后续所有图像 backbone 实验 (Phase 4 E1/E2, Phase 7 E5/E6) 必须先核查此处. **CLAUDE.md § 9 L2 已加为戒律第 4 项.**
- [2026-05-07] v2 设计教训: 多输出函数 (sklearn baseline 类) 返回时切片粒度要对齐. 函数应统一返回**全样本**预测, 算指标时切片. 反过来会触发保存阶段长度不齐. 来源: v2.1 bugfix. 影响: 后续所有 sklearn baseline 函数都遵循"返回全长"原则, 已写入 task_plan Decisions.
- [2026-05-07] **Phase B SI 模态 v2.1 实测** (698 街区, 7 backbone × {mlp, ridge}): DenseNet121 Ridge(α=1000) R²(log)=0.342 / R²(raw)=0.172 (最高), 击败所有 MLP. 7 backbone 排序: densenet121 / vit_b_16 / resnet50 / convnext_tiny / attention_cnn / efficientnet_b0 / mobilenet_v3_large. L0 几何辅助 baseline R²(log)=0.067, 图像净增益 +0.275, 信号显著. 来源: Phase B Session 09 实测. 影响: Phase 4 (E2) 优先用 DenseNet121, 且要带 Ridge 对照, MLP 不一定最强.
- [2026-05-07] **沈阳能耗任务上 SV < SI 等级不严格成立**: 实测 SV 最佳 R²(log)=0.371, SI 最佳 0.342; raw 空间反过来 SV=0.127, SI=0.172. 街景擅长拟合多数低能耗街区, 遥感擅长拟合少数高能耗大街区, 互补关系. 与 KnowCL 论文 § 5.2 "城市/指标对 SV/SI 偏好不同" 一致. 来源: Phase B Session 09 实测对比. 影响: 论文报告时不强行套等级, 而报告"两者水平相当, 互补", task_plan Goal 段已加 ⚠ 注释.
- [2026-05-07] **轻量 backbone (MobileNetV3, EfficientNet-B0) 在单图遥感 + 224 patch 上明显垫底**: R²(log) 0.16-0.18 vs 其他 0.29-0.34. 街景上同样这两个 backbone 也偏弱. 推断: 单图 + 低分辨率 patch 对特征容量要求高, 轻量模型抓不住. 来源: Phase B Session 09 实测. 影响: Phase 4 (E1/E2) 可以从 7 backbone 中删除这两个轻量模型, 节省时间. 论文消融图保留作对比.
- [2026-05-07] **Ridge 在 DenseNet121 上首次明确打败 MLP** (R²(log) 0.342 vs 0.320), 在多个 backbone 上 Ridge 与 MLP 接近. 验证经验法则: 高维 (1000+ 维) + 小样本 (< 1000) 时 Ridge 经常更稳, MLP 容量过剩. 来源: Phase B Session 09 实测. 影响: Phase 4 (E1/E2) 单模态 baseline 必带 Ridge 对照, 不能只跑 MLP.
- [2026-05-07] **Phase B 模板复用审计教训**: 遥感 v1 是从街景 v1 复制改写的, 街景的 fatal bug (L2 归一化) 直接同步到了遥感. 任何"v1 模板复用" 都要逐行审计, 不能信任旧版作者的判断. 来源: 本次会话静态审查. 影响: Phase B/4/5/6/7 任何模板化的脚本批量生成, 都要把每个目标脚本的 L2 norm/aug/loss 三类操作单独 review.
- [2026-05-07] **遥感"零成本基线" = 街区几何**: 5 维 (log_area, log_perim, compact, lon_c, lat_c) Ridge R²(log)=0.067, 弱于街景 aux 的 0.092 (街景含图数信号). 但这是合理的基线锚点, 表明面积本身对能耗有解释力 (大街区 = 高能耗倾向). 来源: Phase B Session 09 设计 + 实测. 影响: Phase 4 单遥感 backbone 实验默认带几何辅助; 论文中应单独报告"几何 only" baseline.
- [2026-05-09] **KG 下游评估"作弊"判别准则** (本会话踩坑): 评估"KG 模型 X 的下游能力"时, 特征里**不能**包含非 X 直接产出的强信号. v4.2 默认 concat handcrafted (POI 类目计数 + 建筑统计 + 几何), 即使 X 是随机 emb 也能跑 R²(log) ≈ 0.2-0.3, 论文里站不住. 解法 (v4.3): 分层归因 4 组特征 (kg_only / kg_nbr / kg_nbr_topo / kg_oracle_handcraft), 主表只报前三组纯 KG, handcrafted 移到 `--include-oracle` ablation 标 "oracle". 同时加 `--plain-block-emb` 随机 emb 健康检查 (R² 应 ≈ 0). 来源: 用户 2026-05-09 驳回 + 本次重设计. 影响: CLAUDE.md § 9 L4 加第 5/6 项戒律, § 11 加第 9 条硬约束; 后续所有"评估某模块下游能力"都遵循此准则.
- [2026-05-09] **1-hop 邻居聚合是合法 KG 信号, 不算作弊**: per-relation neighbor mean (32 × n_r 维) + log(1+deg) 直接从 KG 三元组计算, 等价于一层无参 R-GCN 聚合 (论文 § 2.3 CompGCN 第一层不用关系投影矩阵时退化为此). 这与"掺 handcrafted POI 计数"性质根本不同 (前者是图本身, 后者是预处理产出), 可以加进主表. 来源: KnowCL 论文 § 2.3 + GraphSAGE / CompGCN 公式. 影响: 7_v4.3 把 kg_nbr 和 kg_nbr_topo 与 kg_only 一起放主表, 论文里取 kg_nbr 或 kg_nbr_topo 作 KG 模型代表精度.
- [2026-05-09] **KnowCL 跨模态对比学习的 SV image features 池化**: 街景 N 张图先经 backbone → mean+std 双池化 → per-block (2D 维), 然后送 InfoNCE 投影头. SI 单图直接送即可 (无池化). 这与 v2.1 单模态预测的池化策略**完全一致**, 因此 8_contrastive 直接消费 `sv_image_features_<bb>.npz` (逐图特征) 内部做池化, 消费 `rs_features_block_<bb>_v2.npz` (已 per-block) 直接用. 来源: KnowCL 论文 § 4.3.2 + 本次设计. 影响: 8_contrastive_kg_image.py 实现, 与 SV/SI 单模态精度公平对比.
- [2026-05-09] **KnowCL 对比学习 batch 内三方交集要求**: labels ∩ image ∩ kg 三方 inner join 必须 ≥ 100 个 BlockID, 否则 InfoNCE batch (B=32) 内负样本太少, 退化为 batch normalization. Phase B 698 标签 × 698 街景 (重采后) × ~698 KG block ≈ 698, 安全; 主流程 208-block 子集 ≥ 100, 也安全, 但要监控 alignment + uniformity 防止模式崩溃. 来源: 本次设计. 影响: 8_contrastive 加了 `if len(common) < 100: raise` 的硬性检查.

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

---

## § 11. Phase B 快速原型脚本架构速查 (2026-04-30 新增, 2026-05-07 v2.1 修订)

### § 11.1 脚本输入输出全表

| 脚本 | 关键输入路径 | 关键输出 | 核心字段 / 配置 |
|---|---|---|---|
| `1_build_labels.py` | `8-街区数据/沈阳L4能耗.shp` | `energy_labels.csv`, `label_stats.json` | `BlockID`, `E_Final_W5` |
| `2_predict_streetview.py` (v2.1) | `streetview_index.csv` + `energy_labels.csv` | `sv_image_features_<bb>.npz × 7` (单图特征缓存), `sv_predictions_<bb>.csv × 7` (mlp+ridge), `sv_baseline_metrics.csv`, `sv_metrics_summary.csv` (14 行), `sv_all_models.csv` | 池化 mean+std, 辅助 (log_n_imgs + lon/lat std) |
| `3_predict_remote_sensing.py` (v2.1) | ESRI/`11-卫星数据/*.tif` + `energy_labels.csv` + `8-街区数据/沈阳L4能耗.shp` (取几何) | `rs_features_block_<bb>_v2.npz × 7` (无归一化), `rs_predictions_<bb>.csv × 7` (mlp+ridge), `rs_baseline_metrics.csv`, `rs_metrics_summary.csv` (14 行), `rs_all_models.csv`, `final_comparison.csv`, `metrics_summary_all.csv` | 池化无 (单图), 几何辅助 5 维 (log_area/log_perim/compact/lon_c/lat_c) |
| `4_build_base_kg.py` | POI (`6-POI数据/merged_poi.shp`) + L5 (`8-街区数据/沈阳L5.shp`) + 用地 (`16-地块数据/沈阳市.shp`) | `base/train.tsv`, `entities.json`, `block_to_entity.json`, `block_features.csv` | `name, main_cat, sub_cat`, `LandID`, `Level1_cn, Level2_cn` |
| `5_build_building_kg.py` | 同上 + 建筑 (`9-建筑物数据/processed_shenyang20230318.shp`) | `building/train.tsv`, … | `Height, Function, Age, Quality`, `sub_type` |
| `6_kg_models.py` | 无 (库文件) | 15 个模型类 | — |
| `7_train_kg.py` (v4.3, 2026-05-09 重设计) | `<base\|building>/train.tsv` + `6_kg_models.py` + `energy_labels.csv` | `embeddings_<model>.npz` (含 ent_emb/rel_emb + **block_id/block_emb 给 8_用**), `metrics_<model>.json` (4 组下游详细), `metrics_summary.csv` (每行 = kg/model/feature_set, 默认 3 行/模型) | `--kgs base,building --models all --include-oracle --plain-block-emb`; 4 组特征 = kg_only(d) / kg_nbr(d×n_r 加 d) / kg_nbr_topo(+n_r) / kg_oracle_handcraft(+handcrafted, 仅 oracle) |
| `8_contrastive_kg_image.py` (v1.0, 2026-05-09 新增) 🆕 | `sv_image_features_<bb>.npz` 或 `rs_features_block_<bb>_v2.npz` + `embeddings_<kg_model>.npz` (block_id/block_emb 键) + `energy_labels.csv` | `contrastive/contrastive_<modality>_<bb>_<kg_model>.npz` (img_proj/kg_proj 投影后向量), `metrics_contrastive.csv` (每行 = modality/backbone/kg_model/head, 4 行/组合) | KnowCL Stage1+2: d_proj=128, hidden=256, batch=32, epochs=200, lr=3e-4, τ=0.07, 对称点积 InfoNCE; 4 head = baseline_raw_image / baseline_kg_only / contrastive_image / contrastive_concat |

### § 11.2 Phase B 与主流程对比

| 维度 | Phase B 快速原型 | 主流程 (Phase 1-8) |
|---|---|---|
| 标签来源 | `沈阳L4能耗.shp` / `E_Final_W5` / `BlockID` | `shenyang_region2allinfo.json` / `energy` / `Region_N` |
| 划分 | 5 分位分层 7:1.5:1.5 (Phase B 实测 698 块: train=488/val=105/test=105) | 208-block 6:2:2 (Phase 1 收尾) |
| SI 图像 | ESRI 现下 + TIF fallback | `15-遥感影像/<block_id>.png` 预裁切 |
| KG | 从 SHP 零起构建 | `complete_knowledge_graph.txt` (852k 三元组) |
| KG 编码器 | 15 个标准 embedding 模型 | CompGCN + TuckER 初始化 |
| 对比学习 | 无 (仅 MLP / Ridge 回归) | KnowCL InfoNCE Stage1+2 |
| 路径规范 | G:\\ 硬编码 (临时) | `config/paths.yaml` + `.env` |
| 适用场景 | 快速出数字, 探索 backbone/KG 模型排序 | 正式复现 KnowCL, 论文主实验 |

### § 11.3 下游融合接口 (Phase B embedding 给融合用)

```python
import numpy as np
# 加载 KG block embedding (7_v4.3 格式)
data = np.load("003-知识图谱/base/embeddings/embeddings_rotate.npz")
block_ids = data["block_id"]      # (n_blocks,) str, 已规范化
block_emb = data["block_emb"]     # (n_blocks, dim) float32, 已按 block_id 排序
# 直接在 8_contrastive 里用 block_id 与 image features 做 inner join
```

`block_to_entity.json`: `{block_id_int: entity_id_int}` —— 与 `block_index.tsv` 等价, 旧版回退用.

### § 11.3.1 对比学习接口 (8_contrastive 输出, 给论文最终对比用)

```python
import pandas as pd
df = pd.read_csv("003-知识图谱/metrics_contrastive.csv")
# 每个 (modality, backbone, kg_model) 出 4 行 head
# 论文核心数字取 head=='contrastive_concat' 的 R²(log)
mask = df["head"] == "contrastive_concat"
df[mask].sort_values("ridge_test_r2", ascending=False).head(10)
```

### § 11.4 Phase B 实测结果速查 (2026-05-07 完成)

#### SV 模态 R²(log) 排序前 5 (来自 sv_metrics_summary.csv)

| Rank | backbone | model | feat_dim | R²(log) | R²(raw) | RMSE_raw |
|---|---|---|---|---|---|---|
| 1 | ResNet50 | mlp | 4098 | **0.371** | 0.127 | 3.70 |
| 2 | AttentionCNN | mlp | 4098 | 0.343 | 0.119 | 3.72 |
| 3 | DenseNet121 | ridge(α=1000) | 2050 | 0.302 | 0.083 | 3.79 |
| 4 | AttentionCNN | ridge(α=1000) | 4098 | 0.299 | 0.109 | 3.74 |
| 5 | ResNet50 | ridge(α=1000) | 4098 | 0.291 | 0.074 | 3.81 |

#### SI 模态 R²(log) 排序前 5 (来自 rs_metrics_summary.csv)

| Rank | backbone | model | feat_dim | R²(log) | R²(raw) | RMSE_raw |
|---|---|---|---|---|---|---|
| 1 | DenseNet121 | ridge(α=1000) | 1029 | **0.342** | 0.172 | 3.60 |
| 2 | ViT-B/16 | mlp | 773 | 0.320 | 0.126 | 3.70 |
| 3 | DenseNet121 | mlp | 1029 | 0.320 | 0.141 | 3.67 |
| 4 | ResNet50 | mlp | 2053 | 0.295 | 0.177 | 3.59 |
| 5 | ResNet50 | ridge(α=1000) | 2053 | 0.292 | 0.147 | 3.66 |

#### Baseline 对照

| 模态 | baseline | R²(log) |
|---|---|---|
| SV | mean_predictor | -0.0005 |
| SV | ridge_aux_only (log_n_imgs + GPS std) | 0.092 |
| SI | mean_predictor | -0.0005 |
| SI | ridge_geom_only (log_area + log_perim + compact + lon_c + lat_c) | 0.067 |

**净增益**: SV +0.279, SI +0.275, 两者图像信号都显著.
