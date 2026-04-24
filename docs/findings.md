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
4. **论文继承的硬数字** (见 § 3): 嵌入维度 64, CompGCN + TuckER 初始化, 对称 InfoNCE, 标签 log1p.
5. **可移植性是硬约束**: 代码不准出现 `G:\` 绝对路径, 一律从 `config/paths.yaml` 派生.

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
  - SV 多图: 每 region 平均 `1/n Σ ResNet(img_i)`.
- **两个投影头** (各为 2-layer MLP + ReLU): 把 KG emb 和 image emb 投到同一 128 维空间.
- **对称 InfoNCE loss**:
  \[
  \mathcal{L}_a = -\log \frac{\exp(\mathrm{sim}(\tilde{I}_a, \tilde{e}_a))}{\sum_i \exp(\mathrm{sim}(\tilde{I}_a, \tilde{e}_i))}
          -\log \frac{\exp(\mathrm{sim}(\tilde{e}_a, \tilde{I}_a))}{\sum_i \exp(\mathrm{sim}(\tilde{e}_a, \tilde{I}_i))}
  \]
  其中 `sim(·)` 是点积 (注意**不是** cosine).
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

### § 2.6 在纽约数据上的基准数字 (我们要超越这个)
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

### § 3.5 划分戒律
- 按 `block_id` 划 6:2:2, **不按图像级**.
- 划分文件 (`splits/train.txt` 等) 冻结后**不得动**. 要调实验, 只改超参.
- 所有实验共用**同一份**划分, 保证互相可比.

---

## § 4. 数据事实 (Phase 1 检测后填入, 现在全部待填)

> 跑完 `scripts/check_data.py` 后, 从 `data_check_report.md` 抄到这里.

### § 4.1 规模
- 街区总数 (8-街区数据): `____`
- 建筑物总数 (9-建筑物数据): `____`
- 能耗标签覆盖的街区数 (10-街区能耗标签): `____`
- 三模态精确交集 (label ∩ SV ∩ SI ∩ building): `____` ← **决定样本量**
- 街景图总数 (12): `____`
- 街景 per-block min/median/max: `___/___/___` ← 决定是否能坚持原文 ≥ 40 阈值
- 卫星图总数 (11): `____`, 单张分辨率: `____`
- KG 三元组 / 实体 / 关系: `___ / ___ / ___`

### § 4.2 标签
- 能耗列名 (主 label): `____`
- 单位: `____` (kWh? MJ? GJ/m²? 年? 月?)
- 统计: mean=`___`, median=`___`, skew=`___`, kurt=`___`, zero_ratio=`___`
- 决策: skew > 3 强制 log1p; zero_ratio > 30% 需与业务核对是否是"真零"还是"缺失"

### § 4.3 空间
- 街区 shp CRS: `____`
- 建筑 shp CRS: `____`
- 卫星瓦片 CRS: `____`
- 是否存在 CRS 不一致: `____`

### § 4.4 划分
- `13-训练测试验证集` 里文件清单: `____`
- train/val/test 样本数: `___ / ___ / ___`
- 泄漏检查结果 (三两两交集): `____`

---

## § 5. KG 扩展设计 (Phase 3, 先列候选, 实装前再筛)

### § 5.1 新增实体
| 类型 | 来源 | 键 | 属性候选 |
|---|---|---|---|
| `Building` | 9-建筑物数据 | building_id | height, type, footprint_area, floors |
| `Plot` | 8-街区数据 / 9- | plot_id (或从 shp 生成) | land_use, area, FAR |

### § 5.2 新增关系 (候选, 最终选 5-8 种)
| 关系 | 头-尾 | 语义 | 备注 |
|---|---|---|---|
| `buildingIn` | Building → Region | 建筑落在街区 | 必加 |
| `plotIn` | Plot → Region | 地块属于街区 | 若 plot ≠ region |
| `buildingOn` | Building → Plot | 建筑坐落地块 | |
| `adjacentBuilding` | Building → Building | 距离 ≤ 10m 的邻接建筑 | |
| `sameType` | Building → Building | 同功能类型 | |
| `heightRange` | Building → HeightBin | 高度分档 | 新增 Category 子类 `HeightBin`? |
| `majorUse` | Plot → LandUseCategory | 地块主要用途 | |

### § 5.3 反向边
原 KnowCL 对每种关系加反向 (`~locateAt` 之类). 我们保持同一约定.

### § 5.4 关系数量监控
- 原纽约版 \|R\|=6, 北京/上海 =10.
- 本项目扩展后预期 \|R\| = 10-15. **超过 20 就要警惕** — CompGCN 参数 O(\|R\|·d²) 会爆.

来源: KnowCL 论文 Table 1 + 附录 A.1. 影响: Phase 3.

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

**决定硬件**: `____` (Phase 0 会话时用户告诉 Claude)

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

### § 8.1 `.gitignore` 模板 (Phase 0 用)
```gitignore
# 数据 (在 G:\ 或任何 DATA_ROOT, 绝不入库)
data/
999-输出成果文件/
**/01-预处理中间件/
**/02-Stage1预训练权重/
**/03-Stage2下游结果/
**/04-可视化/

# 模型权重
*.pt
*.pth
*.ckpt
*.bin
*.safetensors
*.npz
*.npy
*.pkl

# 环境
.env
.venv/
venv/
__pycache__/
*.py[cod]
*.egg-info/

# IDE
.idea/
.vscode/
*.swp

# 系统
.DS_Store
Thumbs.db

# 日志
*.log
logs/
wandb/
mlruns/
tensorboard/

# 白名单例外
!results/experiments.csv
!configs/**/*.yaml
!.env.example
```

### § 8.2 `.env.example` 模板
```dotenv
# 复制为 .env 并填入本机实际路径
# Windows: DATA_ROOT=G:/Knowcl  (正斜杠, 避免转义)
# Linux/Mac: DATA_ROOT=/data/knowcl
DATA_ROOT=
```

### § 8.3 `config/paths.yaml` 骨架
```yaml
# 所有路径都从 DATA_ROOT 派生, 不要再硬编码下面任何一行的前缀
data_root: ${DATA_ROOT}

raw:
  energy:       ${data_root}/1-能源数据
  bldg_height:  ${data_root}/2-建筑高度数据
  nightlight:   ${data_root}/3-夜光数据
  population:   ${data_root}/4-人口数据
  edgar:        ${data_root}/5-EDGAR
  poi:          ${data_root}/6-POI数据
  kg:           ${data_root}/7-知识图谱
  blocks:       ${data_root}/8-街区数据
  buildings:    ${data_root}/9-建筑物数据
  labels:       ${data_root}/10-街区能耗标签
  satellite:    ${data_root}/11-卫星数据
  streetview:   ${data_root}/12-街景文件
  splits:       ${data_root}/13-训练测试验证集
  pretrained:   ${data_root}/14-预训练文件
  rs_raw:       ${data_root}/15-遥感影像

outputs:
  root:         ${data_root}/999-输出成果文件
  check:        ${outputs.root}/00-数据检查报告
  preproc:      ${outputs.root}/01-预处理中间件
  stage1:       ${outputs.root}/02-Stage1预训练权重
  stage2:       ${outputs.root}/03-Stage2下游结果
  vis:          ${outputs.root}/04-可视化
  summary:      ${outputs.root}/05-最终对比表
```

> 使用时: 读 `.env` 取 `DATA_ROOT`, 用 `os.path.expandvars` 或 python-dotenv + jinja 展开.

---

## § 9. 研究过程中的新发现 (2-Action Rule 的落点, 历次会话 append)

> 每条新条目**必须**有日期、来源、影响三个字段.
> 写在这一节, 不要散落到其他地方.

- [2026-04-23] planning-with-files skill 由 OthmanAdi 维护, 核心是 3 份 md + 4 个 hook + 2-Action Rule + 5-Question Reboot Test. 来源: github.com/OthmanAdi/planning-with-files. 影响: 本项目协作协议整体采用.
- [2026-04-23] KnowCL 论文确认 Semantic Encoder 是 CompGCN 而不是普通 GCN, 且用 TuckER 做 embedding 初始化. 来源: arXiv 2302.13094 § 4.3.1. 影响: Phase 5, 6 实现时必须用 `dgl.nn.CompGraphConv` 或等价实现.
- [2026-04-23] KnowCL 的 InfoNCE 相似度是**点积**不是 cosine (论文式 (5) 下方). 来源: arXiv 2302.13094 § 4.4. 影响: 写 `info_nce.py` 时别想当然换 cosine.
- [2026-04-23] KnowCL batch size: SI=128, SV=16 (不对称, 因 SV 每 region 多图, 有效样本本来就小). 来源: 论文 README 命令行. 影响: Phase 6 初值.
- [2026-04-23] 原 KnowCL 每 region 需 ≥ 40 张街景才纳入. 沈阳数据可能达不到, 需 Phase 1 数据检查后决定是否放宽. 来源: 论文 § 5.1.1. 影响: Phase 1 的 blocking issue.
-
