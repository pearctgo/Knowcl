# CLAUDE.md · 新对话必读

> **⚠️ 新对话的 Claude, 这是你的第一份读物.**
> 读完这份 **和** `task_plan.md` / `findings.md` / `progress.md` 最新一条,
> 通过 § 2 的 5 问自检, 再动手. 任何时候不确定: 先问, 不要编.

---

## § 0. 这是什么项目

**沈阳市街区能耗预测** · 参考清华 FIB-Lab 的 [UrbanKG-KnowCL](https://github.com/tsinghua-fib-lab/UrbanKG-KnowCL) 迁移改造.

- 原论文: 纽约社经指标预测 (人口/教育/犯罪)
- 本项目: **沈阳街区能耗回归**, 基础 KG 扩展为"建筑物城市 KG" (base + building)
- 目标精度等级 (主流程参考):
  SV < SI < base-KG < bldg-UKG < bldg-UKG+SV < bldg-UKG+SI
  ⚠ 2026-05-07 Phase B 实测: 在沈阳能耗任务上 SV ≈ SI (0.371 vs 0.342, 互补关系), 与 KnowCL 论文 § 5.2 "城市/指标偏好不同" 一致, 不强行套等级.
- 7 个视觉 backbone 对比池: ResNet-50 / ConvNeXt / DenseNet121 / ViT / MobileNetV3 / AttentionCNN / EfficientNet
- ⚠️ **关键约束**: SV 仅覆盖 208 个街区 (三模态交集), 主实验在 208-block 子集上跑, 不是全部 757 块
- ⚠️ **新增分支 (2026-04-26)**: Phase 1.5 用百度地图前端内部端点 (`mapsv0.bdimg.com`) 重采全 757 街区街景, **无需申请 AK** (与项目原 `test_shenhe.py` 同方法). 若成功, 主实验可升级为 757-block 全量. **2026-05-07 已完成: 实际产出 9316 张 / 698 街区**.
- ⚠️ **新增并行轨道 (2026-04-30)**: Phase B 快速原型脚本 (`1_build_labels.py` ~ `7_train_kg.py`) —— 用于在 KnowCL 主流程完成前快速出指标对比 (7 视觉 backbone × 街景/遥感 + 15 KG embedding 模型 × base/building KG). **与 src/ 主流程独立, 当前硬编码 G:\\ 路径 (待迁移到 paths.yaml)**. 见 `task_plan.md § Phase B`.
  **2026-05-07 实测**: SV+SI 步骤完成, R²(log) SV=0.371, SI=0.342, 净增益 +0.28; 进入 KG 轨道.
- ⚠️ **Phase B 第 8 步追加 (2026-05-09)**: 新增 `8_contrastive_kg_image.py` (KnowCL 论文 § 4 Stage 1+2 实现) —— 这是项目核心目的: image (SV/SI) ↔ KG block embedding 跨模态对比学习, 对称点积 InfoNCE, τ=0.07. 输出 `metrics_contrastive.csv` 含 4 行/(modality,backbone,kg_model): `baseline_raw_image / baseline_kg_only / contrastive_image / contrastive_concat`. 完整精度等级 SV < SI < base-KG < bldg-UKG < bldg-UKG+SV < bldg-UKG+SI 在三张表里就齐了 (SV/SI 在 `sv|rs_metrics_summary.csv`, KG 在 `metrics_summary.csv` 取 feature_set=B/C, +SV/+SI 在 `metrics_contrastive.csv` 取 head=contrastive_concat).

完整背景、关键论文数字、目录清单、约束 — 见 `findings.md`.
路线图、阶段、已做决策 — 见 `task_plan.md`.
每次会话干了啥 — 见 `progress.md`.

---

## § 1. 本项目的 4 文件架构 (planning-with-files 约定)

| 文件 | 存什么 | 谁写 | 何时写 |
|---|---|---|---|
| `CLAUDE.md` (本文) | 协议 / 规则 / 模板 | 用户 + Claude (稳定, 极少改) | 踩到协议层面的新坑才改 |
| `task_plan.md` | **路线图** (目标 / 阶段 / 状态 / 决策 / 错误) | Claude 起草, 用户认可后落地 | 每次 phase 状态变化 |
| `findings.md` | **已掌握的知识** (数据事实 / 技术选型 / 约束 / 论文摘要) | Claude 研究过程 append | **2-Action Rule: 每 2 次查询强制更新** |
| `progress.md` | **会话流水**, 一次对话一条 | Claude 在会话尾写, 用户 append | 每次会话收尾 |

**类比**: 对话窗口 = RAM (会丢), 这 4 份文件 = 磁盘 (持久化).

---

## § 2. 🔴 5-Question Reboot Test (开场自检, 必做)

新对话开场, 你 (Claude) **必须**先在回复里公开回答以下 5 题 **再动手**.

| # | 问 | 答在哪找 |
|---|---|---|
| 1 | 我现在在哪个 Phase? | `task_plan.md § Phases` 里 status=in_progress 那条 |
| 2 | 下一个 Phase 是什么? | 同上, 向下看 |
| 3 | 项目总目标? | `task_plan.md § Goal` |
| 4 | 最关键的 5 条 findings 是什么? | `findings.md § 1` |
| 5 | 上一次 session 做了什么? 留了什么坑? | `progress.md` 最后一条 |

---

## § 3. 🔴 2-Action Rule (研究过程中强制)

每完成 **2 次** 以下任一动作, 必须立刻把新知识 append 进 `findings.md`, 再继续:
- `web_search` / `web_fetch` / `view`(读陌生文件) / `bash`(ls/cat 未知数据) / 用户贴的新材料

格式 (append 到 `findings.md § 9`):
```markdown
- [YYYY-MM-DD] <发现>. 来源: <url / 文件名 / 用户>. 影响: <对 task_plan 哪个 phase 的决策>.
```

---

## § 4. 🔴 Phase 状态机

```
pending ──开始做──▶ in_progress ──完成──▶ complete
                        │
                        └──卡住──▶ blocked (写到 Errors 表)
```

错误处理三步:
1. append 到 `task_plan.md § Errors`
2. append 到 `progress.md` 当前 session 的 Errors 段
3. **绝不重复同样的失败动作**

---

## § 5. 🔴 Compact 会话收尾协议 (每次必做)

**会话结束前**, 输出如下 4 块:

```
### BLOCK 1 · progress.md 新增条目
## YYYY-MM-DD · Session NN · <一句话主题>
- **Phase**: ...
- **Actions**: ...
- **Files produced/modified**: ...
- **Open issues**: ...
- **Next session**: ...
- **Commit message**: ...

### BLOCK 2 · task_plan.md 更新建议
- 要勾选的 checkbox: ...
- 要切换的 status: ...
- 要新增到 Decisions 表的行: ...
- 要新增到 Errors 表的行 (若有): ...

### BLOCK 3 · findings.md 更新建议
- 要新增到 § <章节> 的内容: ...

### BLOCK 4 · git 一行
<type>(<scope>): <简述>
```

---

## § 6. 🔴 新对话开场模板

```
你好. 沈阳街区能耗预测项目, 继续.

本次任务: <一句话>

请:
1. 先按 CLAUDE.md § 2 公开回答 5-Question Reboot Test
2. 复述你对本次任务的理解, 我确认后再动手
3. 本次最多改 2 个文件

[附 A] CLAUDE.md 全文: <粘贴>
[附 B] task_plan.md 全文: <粘贴>
[附 C] findings.md 全文 (或只粘相关章节): <粘贴>
[附 D] progress.md 最后 2 条: <粘贴>
[附 E] 本次涉及的代码 (若有): <粘贴>
```

---

## § 7. 可移植性约束

| 规则 | 正确 | 错误 |
|---|---|---|
| 数据路径 | 从 `config/paths.yaml` 读 | `Path(r"G:\Knowcl")` 硬编码 |
| 依赖 | `requirements.txt` pin 版本 | 不 pin |
| OS | `pathlib.Path`, `os.path.join` | 反斜杠拼接 |
| GPU | `torch.cuda.is_available()` 检测 | 写死 `.cuda()` |
| 中文路径 | `encoding='utf-8'` | 默认 gbk |
| **API 密钥** ⭐ | **从 `.env` 读 (env var 优先)** | 写进代码 commit 到 git |

> **§ 7.1 第三方 API 密钥处理 (新增 2026-04-26, 修订 2026-04-26)**
> 当代码需要第三方 API key (官方途径 — 百度地图开放平台 AK / 高德 / Mapbox / 微软 Bing 等) 时:
> 1. 三段式回退读取: 环境变量 → `.env` 文件 → 默认值 (空字符串)
> 2. `.gitignore` 必须屏蔽 `.env`
> 3. 仓库提供 `.env.example` 占位
> 4. 代码若检测不到 key, 报清晰错误并指向申请文档, 不要默默静默失败
> 5. ⚠ Claude 不能替用户申请 key — 国内 API 都强制实名认证.
>
> **若使用前端内部端点路径** (例如本项目 Phase 1.5 走 `mapsv0.bdimg.com`):
> - 不需要 AK, 但需要带正确的 `Referer` 请求头伪装成浏览器
> - 严格说违反服务条款 § 2.2 "不得直接存取 ... 内部数据", 属灰色区域
> - 仅限学位论文/学术研究等非营利场景使用
> - 不要将抓下的内容转分发或商用
> - 风险: 后端调整可能随时让脚本失效, 需有备用方案

---

## § 8. 协作颗粒度

**一次会话 = 一次 git commit = 一件原子可回滚的事.**

| 对的 | 错的 |
|---|---|
| "实现 `src/datasets/block_index.py`" | "帮我写整个 Stage 1" |
| "把 lr 从 1e-4 调 3e-4 重跑 E1" | "跑所有实验看哪个好" |

标尺: 单文件 ≤ 200 行, 单函数 ≤ 50 行, 一次对话 ≤ 2 文件.

> ⚠ 例外 1: 数据采集类一次性脚本 (如 `collect_streetview_baidu_full.py`) 因为耦合需求和容错代码, 单文件可放宽到 ≤ 1000 行, 但仍要按职责分节注释.
>
> ⚠ 例外 2: docs 维护协议本身支持 4 doc 联动改 (progress + task_plan + findings + CLAUDE 同步更新), 不算在 ≤ 2 文件限制内.

---

## § 9. Debug 5 层排查

### L0 · Sanity (必做, 10 分钟)
- [ ] 均值预测器 RMSE/R²? 打不过 = 模型有问题
- [ ] `sklearn.LinearRegression(特征=[建面, POI数, 夜光均, 建高均])` R²? 打不过 = **去修数据, 别调模型**
- [ ] `sklearn.Ridge(alpha=tuned)` 和 MLP 同表对比 — 高维 + 小样本时 Ridge 经常更稳, MLP 容量过剩 (Phase B Session 09 验证)
- [ ] train/val/test block_id 交集 = 0?
- [ ] label 打乱后模型仍高精度 = 数据泄漏

### L1 · 拟合状态
| 现象 | 诊断 | 对策 |
|---|---|---|
| train↓ val↑ | 过拟合 | Dropout↑, aug↑, wd↑, 早停 |
| 都平 | 欠拟合 | lr×3/÷3, 加特征, 换 loss |
| 都↓持平 | 天花板 | 加数据/模态/KG |
| 震荡 | lr 太大 | lr÷3, grad clip |

### L2 · 特征质量
1. Pearson/Spearman < 0.05 = 信号不存在
2. 方差≈0 的特征删掉
3. KS 检验 train vs test 分布漂移
4. ⭐ **(2026-05-07 新增)** 单图 backbone 特征**聚合前不要做 L2 归一化** (`feats /= np.linalg.norm(...)`). 这会丢幅度信号. 街景 v1 / 遥感 v1 都因这一行 R²(log) 被压在 0.10-0.18, 删后翻倍到 0.30-0.37. 归一化只应该出现在 InfoNCE 投影头之后 (损失函数内部需要), 不应该在 backbone penultimate features 上. 改在 MLP 输入端用 `BatchNorm1d` 处理尺度.
5. ⭐ **(2026-05-07 新增)** 街区级聚合策略要保留方差信号: 街景多图用 `mean+std concat`, 遥感单图无法多池化, 改用几何辅助特征 (面积/周长/紧凑度/坐标) concat. 单 mean pooling 是丢方差的反向操作.
6. ⭐ **(2026-05-07 新增)** 任何 sklearn baseline / 多输出回归函数应统一返回**全样本预测**, 算指标时切片. 反过来 (只返 test) 会触发保存阶段长度不齐 → KeyError.

### L3 · 对比学习专属
- InfoNCE 初值 ≈ \(\log N\), 末端 < 1.0
- alignment + uniformity 同步下降
- UMAP: 全挤 = 模式崩溃; 全散 = 对齐失败
- batch 内负样本不含同 region 的另一张图

### L4 · KG 质量
- 平均实体度 < 3 = KG 太稀疏
- KG emb concat 进线性回归: R² 无提升 = KG 挂错实体
- ⭐ **(2026-05-09 新增) KG 下游评估"作弊"判别准则**: 评估"KG 模型 X 的下游能力"时, 特征**不能**包含非 X 直接产出的信号 (handcrafted POI 类目计数 / 建筑统计 / 几何这些是数据预处理产出, 不是 KG 模型产出). 若把 X 替换为**随机 emb** 后下游 R²(log) 仍 ≥ 0.2, 说明手工特征 / 标签泄漏 dominant, 不能算 KG 模型精度. **解法**: 7_v4.3 改成分层归因, 主表只报 `kg_only / kg_nbr / kg_nbr_topo` 三组纯 KG 特征 R², handcrafted 仅在 `--include-oracle` 下作 ablation 上界. 必须用 `--plain-block-emb` 跑随机健康检查, R² ≈ 0 才算下游链路对齐 OK.
- ⭐ **(2026-05-09 新增) 1-hop 邻居聚合是合法 KG 信号**: per-relation neighbor mean + log(1+deg) 是从 KG 三元组直接计算的, 等价于一层无参 R-GCN 聚合, 仍属"KG 模型在该图上的输出". 这与"掺 handcrafted 特征"在性质上根本不同, 可以加进主表.

### L5 · 代码 bug
- `model.eval()` 切了?
- log1p 训练, 指标时 `expm1` 回原空间?
- DataLoader `num_workers > 0` 种子固定?
- 预训练权重 `missing_keys` 查了?
- `try-except` 包裹的代码块要带 `traceback.print_exc()`, 否则异常会被静默吞 (Phase B Session 07 教训: ridge 长度不齐异常被吞导致 metrics 始终空)

### L6 · 数据采集类专属 (新增 2026-04-26)
- API 返回 JSON 不是图片? → 看 `Content-Type`, 多半是 key 错或配额超
- 大量 `no_panorama`? → 候选点都挂在小区内部, 改用道路 buffer 重采
- 坐标偏移? → 检查 coordtype (wgs84ll vs bd09ll vs gcj02ll), 国内 API 默认 bd09ll
- 断点续采没省时间? → 检查 `quick_image_is_valid` 阈值, 太严会反复重下

### L7 · 模板复用类专属 (新增 2026-05-07)
- v1 模板复制到第二个 backbone/模态/任务 时, **逐行审计**, 不能信任旧版作者. Phase B Session 08 教训: 街景 v1 的 L2 归一化 fatal bug 直接同步到了遥感 v1.
- 重点审计三类操作: ① 特征归一化 / 标准化 ② 数据增强 ③ 损失函数 — 这三类最易在跨任务时引入隐藏 bug.

---

## § 10. Git 速查

### 一次性初始化
```bash
cd D:\code\shenyang-energy-kg
git init && git branch -M main
git add .gitignore README.md docs/ scripts/ configs/ src/ requirements.txt .env.example results/experiments.csv
git commit -m "chore(init): scaffold phase 0"
git remote add origin https://github.com/<user>/shenyang-energy-kg.git
git push -u origin main
git tag -a v0.0-scaffold -m "phase 0 complete"
git push --tags
```

### 日常
```bash
git status
git add <明确的文件>    # 不要 git add .
git commit -m "<BLOCK 4>"
git push
```

### Commit Message type 速查
| type | 场合 |
|---|---|
| `chore` | 脚手架/配置/依赖 |
| `feat` | 新功能模块 |
| `fix` | 修 bug |
| `refactor` | 重构 |
| `data` | 数据处理脚本 |
| `exp` | 实验结果 (`exp(e1-v01): resnet50 R²=0.31`) |
| `docs` | 改 md |

---

## § 11. 提示词硬约束

```
硬约束:
1. 不确定就问, 不要编 API/列名/路径.
2. 本次改 ≤ 2 个文件, 不新建目录 (除非我同意).
3. 产出必须含: (a) 代码/diff (b) 4 个 BLOCK 更新建议 (c) 一行 commit.
4. 任何代码不得出现 G:\ 绝对路径.
5. 收到任务先复述, 我确认才动手.
6. ⚠ API key 永远不出现在代码里, 永远从 .env 读.
7. ⚠ Phase B 快速原型脚本当前豁免 G:\ 限制 (用户显式指定路径), 但正式集成进 src/ 前必须迁移到 paths.yaml.
8. ⚠ (2026-05-07) v1 → v2 模板复用时必须逐行审计. 先看完旧版再改, 不要"假定旧版没问题". 街景/遥感 v1 共用的 L2 归一化 bug 就是这么传过来的.
9. ⚠ (2026-05-09) **评估"模型 X 的下游能力"时, 特征里不能掺非 X 直接产出的信号**. 评估 KG 模型时不混 handcrafted, 评估 image backbone 时不混 KG, 评估对比学习模块时基线必须显式分离. 任何"混合特征"都必须放进 ablation 表显式标 oracle, 不能进主表打分. 否则视作"作弊", 论文里站不住.
```

---

## § 12. 何时开新对话

出现任一信号就立即关掉开新 (走 § 5 收尾):
- Claude 重复给已驳回的建议
- 同一 bug 反复 3 轮没修
- 回复明显变短、敷衍
- 滚屏到顶要 10 秒以上

---

## § 13. 本文件维护规则

- append-only: 改了用删除线保留旧内容, 旁边写新内容
- 只在协议层面的新坑才改本文件
- `git commit -m "docs(claude): <改动摘要>"`

---

*"Context is volatile. Files are forever." — planning-with-files 核心哲学.*
