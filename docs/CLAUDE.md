# CLAUDE.md · 新对话必读

> **⚠️ 新对话的 Claude, 这是你的第一份读物.**
> 读完这份 **和** `task_plan.md` / `findings.md` / `progress.md` 最新一条,
> 通过 § 2 的 5 问自检, 再动手. 任何时候不确定: 先问, 不要编.

---

## § 0. 这是什么项目

**沈阳市街区能耗预测** · 参考清华 FIB-Lab 的 [UrbanKG-KnowCL](https://github.com/tsinghua-fib-lab/UrbanKG-KnowCL) 迁移改造.

- 原论文: 纽约社经指标预测 (人口/教育/犯罪)
- 本项目: **沈阳街区能耗回归**, 基础 KG 扩展为"建筑物城市 KG" (base + building + plot)
- 目标精度等级 (必须做出):
  SV < SI < base-KG < bldg-UKG < bldg-UKG+SV < bldg-UKG+SI
- 7 个视觉 backbone 对比池: ResNet-50 / ConvNeXt / DenseNet121 / ViT / MobileNetV3 / AttentionCNN / EfficientNet

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
长对话 / 换新对话的焦虑, 本质是"上下文没落到磁盘". 跑完本协议就解决.

---

## § 2. 🔴 5-Question Reboot Test (开场自检, 必做)

新对话开场, 你 (Claude) **必须**先在回复里公开回答以下 5 题 **再动手**.
答不全就向用户要缺失文件, 不要凭空假设.

| # | 问 | 答在哪找 |
|---|---|---|
| 1 | 我现在在哪个 Phase? | `task_plan.md § Phases` 里 status=in_progress 那条 |
| 2 | 下一个 Phase 是什么? | 同上, 向下看 |
| 3 | 项目总目标? | `task_plan.md § Goal` |
| 4 | 最关键的 5 条 findings 是什么? | `findings.md` 顶部 |
| 5 | 上一次 session 做了什么? 留了什么坑? | `progress.md` 最后一条 |

用户会看着你这段回答. 答得含糊 = 用户补材料. 答得清晰 = 用户给你任务.

---

## § 3. 🔴 2-Action Rule (研究过程中强制)

每完成 **2 次** 以下任一动作, 必须立刻把新知识 append 进 `findings.md`, 再继续下一个动作:
- `web_search` / `web_fetch` / `view`(读陌生文件) / `bash`(ls/cat 未知数据) / 用户贴的新材料

**不允许**: 先研究 10 步再一次性写 findings.
**原因**: 对话一长就丢信息. 边做边写, 未来新对话才能复现.

格式 (append 到 `findings.md` 对应章节):
```markdown
- [YYYY-MM-DD] <发现>. 来源: <url / 文件名 / 用户>. 影响: <对 task_plan 哪个 phase 的决策>.
```

---

## § 4. 🔴 Phase 状态机 (task_plan.md 的维护规则)

```
pending ──开始做──▶ in_progress ──完成──▶ complete
                        │
                        └──卡住──▶ blocked (写到 Errors 表)
```

- **开始**: 把 phase status 从 `pending` 改成 `in_progress`.
- **完成**: 勾选 `[x]`, status 改 `complete`.
- **错误**:
  1. append 到 `task_plan.md § Errors Encountered` 表 (日期 / 现象 / 根因 / 解法 / Phase).
  2. append 到 `progress.md` 当前 session 的 Errors 段.
  3. **绝不重复同样的失败动作** — 这是 Manus 协议的核心纪律.

---

## § 5. 🔴 Compact 会话收尾协议 (每次必做)

**会话结束前**, 你必须输出如下 4 块标记清晰的内容, 用户会 append 进对应文件.
不允许省略、不允许合并.

```
### BLOCK 1 · progress.md 新增条目 (append 到文件末尾)
## YYYY-MM-DD · Session NN · <一句话主题>
- **Phase**: <phase 编号 / 名字>
- **Actions**: <3-6 行项目符号>
- **Files produced/modified**: <逐个列>
- **Open issues**: <未解决的>
- **Next session**: <下次重点>
- **Commit message**: <type>(<scope>): <≤70 字符>

### BLOCK 2 · task_plan.md 更新建议
- 要勾选的 checkbox: <哪个 phase 的哪几项>
- 要切换的 status: <phase N: in_progress → complete>
- 要新增到 Decisions 表的行: <日期 / 决策 / 原因 / Phase>
- 要新增到 Errors 表的行 (若有): <日期 / 现象 / 根因 / 解法 / Phase>

### BLOCK 3 · findings.md 更新建议
- 要新增到 § <章节> 的内容: <具体文本>
- 每条遵守 [日期] 发现. 来源. 影响. 的格式

### BLOCK 4 · git 一行
格式: <type>(<scope>): <简述>
```

用户操作 (标准流程):
1. 把 BLOCK 1 粘到 `progress.md` 末尾.
2. 按 BLOCK 2 逐条改 `task_plan.md`.
3. 按 BLOCK 3 逐条改 `findings.md`.
4. 执行:
   ```
   git add docs/ src/ scripts/ configs/ <改了的代码>
   git commit -m "<粘 BLOCK 4>"
   git push
   ```

---

## § 6. 🔴 新对话开场模板 (用户用这个开头)

```
你好. 沈阳街区能耗预测项目, 继续.

本次任务: <一句话, 如 "Phase 2: 实现 block_index.py">

请:
1. 先按 CLAUDE.md § 2 公开回答 5-Question Reboot Test
2. 复述你对本次任务的理解, 我确认后再动手
3. 本次最多改 2 个文件; 长代码走文件而不是聊天

[附 A] CLAUDE.md 全文:
<粘贴>

[附 B] task_plan.md 全文:
<粘贴>

[附 C] findings.md 全文 (或只粘相关章节):
<粘贴>

[附 D] progress.md 最后 2 条:
<粘贴>

[附 E] 本次涉及的代码 (若有, 只贴 1-2 个文件):
<粘贴>
```

---

## § 7. 可移植性约束 (写任何代码时遵守)

本项目**必须**能拷到任何 Windows/Linux 机器直接跑. 规则:

| 规则 | 正确 | 错误 |
|---|---|---|
| 数据路径 | 从 `config/paths.yaml` 或 `.env` 读取 | `Path(r"G:\Knowcl")` 硬编码 |
| 依赖 | `requirements.txt` 全部 pin 版本 (`torch==2.1.0`) | 不 pin / 用 conda 特定 channel 的包 |
| OS 假设 | `pathlib.Path`, `os.path.join` | 反斜杠字符串拼接 |
| GPU | 运行时检测 `torch.cuda.is_available()`, 回退 CPU | 写死 `.cuda()` |
| 中文路径 | utf-8 读写, 文件 IO 加 `encoding='utf-8'` | 默认 gbk |
| 绝对路径 | 任何代码里都不应有 | 注释 `# 我自己电脑的路径` 也不行 |

**新机器跑起来的 3 步** (README.md 里要写):
1. `git clone` + `pip install -r requirements.txt`
2. `cp .env.example .env` 然后编辑 `.env` 里的 `DATA_ROOT`
3. `python scripts/check_data.py`

---

## § 8. 协作颗粒度

**一次会话 = 一次 git commit = 一件原子可回滚的事.**

| 对的 | 错的 |
|---|---|
| "实现 `src/datasets/sv_dataset.py`" | "帮我写整个 Stage 1" |
| "把 InfoNCE temperature 从 0.07 调 0.1 重跑" | "跑所有实验看哪个好" |
| "诊断为什么 E1-resnet50 val R² 为负" | "模型效果不好" |

标尺: 单文件 ≤ 200 行, 单函数 ≤ 50 行, 一次对话 ≤ 2 文件.

---

## § 9. Debug 5 层排查 (实验不好时按序走, 不跳)

### L0 · Sanity (10 分钟能做完, 必做)
- [ ] 均值预测器 (train 均值当全部预测) 的 RMSE/MAE/R² 是多少? 你模型打得过吗?
- [ ] `sklearn.LinearRegression(特征 = 建面 + POI 数 + 夜光均 + 建高均)` 的 R² 多少? 打不过这个 = **别调模型, 去修数据**.
- [ ] train/val/test 的 block_id 交集 **=0**? `check_data.py` 会报.
- [ ] label 打乱测试: 真 label 随机置换后模型仍高精度 = 数据泄漏.

### L1 · 拟合状态
| 现象 | 诊断 | 对策 |
|---|---|---|
| train 降 val 升 | 过拟合 | Dropout↑, aug↑, wd↑, 缩模型, 早停 |
| train+val 都平 | 欠拟合 | lr × 3 / ÷ 3, 加特征, 换 loss |
| 都降且持平 | 触到天花板 | 加数据 / 加模态 / 加 KG |
| 剧烈震荡 | lr 太大 / batch 太小 | lr ÷ 3, grad clip |

### L2 · 特征质量
1. 每列特征 vs label 的 Pearson/Spearman, 绝对值普遍 < 0.05 = 信号不存在
2. 方差 ≈ 0 的特征全删
3. train vs test 分布漂移 (KS 检验 p < 0.01)

### L3 · 对比学习专属 (E5/E6)
- InfoNCE 初值 ≈ \(\log N\), 末端必须 **< 1.0**. 不降 → lr/temperature 错.
- alignment / uniformity 应同步下降
- UMAP 看 embedding: 全挤 = 模式崩溃 (temperature 太低); 全散 = 对齐失败
- batch 内负样本不能含同 region 的另一张图 (原 KnowCL 最易错处)

### L4 · KG 质量 (E3/E4)
- 平均实体度 < 3 = KG 太稀疏
- 把 KG emb concat 进线性回归: R² 若无提升 = KG 挂错实体

### L5 · 代码 bug
- `model.eval()` 切了吗?
- log1p 训练, 指标算时记得 `expm1` 回原空间?
- DataLoader `num_workers > 0` 每 worker 种子固定?
- 预训练权重 `missing_keys` / `unexpected_keys` 查了?

---

## § 10. Git 速查

### 一次性初始化
```bash
cd <你的代码目录>                    # 不是 G:\Knowcl!
# 先放好 .gitignore (见 findings.md § .gitignore 模板)
git init && git branch -M main
git add .gitignore README.md docs/ scripts/ configs/ src/ requirements.txt .env.example
git commit -m "init: project scaffold"
git remote add origin https://github.com/<user>/shenyang-energy-kg.git
git push -u origin main
```

### 日常 (每次会话收尾)
```bash
git status                          # 看改了啥
git add <明确的文件>                # 不要 git add .
git commit -m "<BLOCK 4 的消息>"
git push
```

### Commit Message 规范 (Conventional Commits)
| type | 场合 |
|---|---|
| `feat` | 新功能/新模型 |
| `fix` | 修 bug |
| `refactor` | 重构 |
| `docs` | 改 md |
| `exp` | 一次实验结果 (本项目特化) |
| `data` | 数据处理 |
| `chore` | 依赖/配置 |

示例: `exp(e5-v03): knowcl_sv resnet50 lr=1e-4, R²=0.48`

### 分支
- `main`: 只留跑得通的
- `exp/<短名>`: 每次大实验一条; 跑完值得留就 squash-merge, 不留就 `git branch -D`
- 里程碑打 tag: `git tag -a v0.1-data-checked -m "..."`, `git push --tags`

---

## § 11. 提示词硬约束 (用户粘给 Claude 的, 每次)

```
硬约束 (必须遵守):
1. 不确定就问, 不要编 API/列名/路径.
2. 本次改 ≤ 2 个文件, 不新建目录 (除非我同意).
3. 产出必须含: (a) 代码 diff 或整文件 (b) CLAUDE/task_plan/findings/progress 的更新建议 (c) 一行 commit.
4. 任何代码**不得**出现 G:\ 绝对路径, 路径都从 config/paths.yaml 读.
5. 收到任务先复述, 我确认才动手.
```

---

## § 12. 何时开新对话

出现任一信号就**立即**关掉开新的 (走 § 5 Compact 收尾):
- Claude 重复给已驳回的建议
- 同一个 bug 反复 3 轮还没修
- 回复明显变短、敷衍
- 你滚屏到顶要 10 秒以上
- 话题漂了 (从模型聊到了 Git, 又聊到了 IDE 快捷键)

---

## § 13. 本文件的维护规则

- **append-only 原则**: 协议改了, 用删除线 `~~旧~~` 保留, 旁边写新. 历史不擦.
- 只在遇到**协议层面**的新坑才改本文件. 项目层面的坑进 `task_plan.md § Errors`.
- 每次改动一律 `git commit -m "docs(claude): <改动摘要>"`.

---

*"Context is volatile. Files are forever." — planning-with-files skill 核心哲学.*
