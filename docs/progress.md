# progress.md · 会话流水

> 一次会话 = 一条 entry. Append-only, 不改历史.
> 维护协议: `CLAUDE.md § 5` (Compact 收尾).
>
> **新对话的 Claude 必读**: 最近 1-2 条 entry 帮你回忆"上次做到哪".

---

## 格式模板 (每条 entry 都用这个)

```markdown
## YYYY-MM-DD · Session NN · <一句话主题>
- **Phase**: <phase 编号及名称>
- **Goal of this session**: <1 句>
- **Actions**:
  - <3-8 个动作项, 每项 1 行>
- **Files produced/modified**:
  - `<路径>` · <创建 / 修改 / 删除>
- **Key findings** (已 append 到 findings.md 的关键条目):
  - <摘要 + findings.md 的章节号>
- **Experiments run** (若有):
  - `<exp_id>`: <结果行>
- **Errors encountered**:
  - <现象 + 根因 + 解法, 已在 task_plan.md § Errors 记了>
- **Open issues**:
  - <给下次会话看的>
- **Next session**:
  - <建议的下一个任务, 一句>
- **Git**: `<commit hash>` · `<type>(<scope>): <message>`
```

---

## 2026-04-23 · Session 01 · 初始化设计 (planning-with-files 协议落地)

- **Phase**: 规划前置 (Phase 0 尚未开始)
- **Goal of this session**:
  在第一次对话里, 把整个长期项目的协作框架定下, 产出 4 份持久化文件,
  使得后续任意一个新对话都能在 1 分钟内对齐上下文.

- **Actions**:
  - 读完 `tsinghua-fib-lab/UrbanKG-KnowCL` README (含 4 个 subfolder / 要求的命令行 / 依赖)
  - 读完 KnowCL 论文原文 (arxiv 2302.13094), 提取关键架构与超参
  - 读完 `OthmanAdi/planning-with-files` skill 的 workflow 和 quickstart, 掌握其协议
  - 按 skill 规范搭建 4 文件架构 (CLAUDE / task_plan / findings / progress)

- **Files produced/modified**:
  - `docs/CLAUDE.md` · 创建 (协议 + 规则 + 模板)
  - `docs/task_plan.md` · 创建 (8 个 phase 的路线图 + 决策表 + 错误表)
  - `docs/findings.md` · 创建 (论文摘要 + 硬约束 + 数据事实空格 + KG 扩展设计)
  - `docs/progress.md` · 创建 (本文件, 含第一条 entry)

- **Key findings** (已 append 到 findings.md):
  - KnowCL 语义编码器是 CompGCN (非普通 GCN), + TuckER 初始化 · findings.md § 2.3 / § 9
  - InfoNCE 用点积相似度非 cosine · § 2.3 / § 9
  - SI/SV batch size 不对称 (128 vs 16) · § 2.5 / § 9
  - 原文要求每 region ≥ 40 张街景, 沈阳需核实 · § 2.5 / § 9
  - 精度目标等级的 6 级不等式已定稿, 6 组实验 E1-E6 清晰 · task_plan.md § Goal

- **Experiments run**: 无.

- **Errors encountered**: 无.

- **Open issues**:
  1. `findings.md § 4` 所有【数据事实】空格都是 `____`, 需 Phase 1 填.
  2. `findings.md § 6` 技术栈中的 PyTorch / DGL 版本、硬件 GPU 未决, 需 Phase 0 用户告知.
  3. 代码目录细分 (`src/` 内部) 刻意留白, 等 Phase 2 看完真实数据再定.
  4. 上一轮已产出的 `check_data.py` 位于 `/mnt/user-data/outputs/`, Phase 0 第一件事就是把它挪进代码仓库并改成读 `config/paths.yaml`.

- **Next session (建议)**:
  **开启 Phase 0 · 可移植项目骨架**. 具体子任务:
  - 用户创建 `D:\code\shenyang-energy-kg\` (与数据目录 `G:\Knowcl\` 分离)
  - 粘 4 份 md 进 `docs/`, 粘上一轮的 `check_data.py` 进 `scripts/`
  - 然后开新对话, 把 4 份 md 按 `CLAUDE.md § 6` 模板贴给新 Claude, 任务描述写:
    "执行 task_plan.md Phase 0, 先把 check_data.py 改成从 config/paths.yaml 读路径, 再写 .env.example / paths.yaml / requirements.txt / README.md / .gitignore, 最后给我 git init 命令."

- **Git**: (用户首次 commit 后填 hash)
  `chore(init): scaffold with planning-with-files protocol (CLAUDE+task_plan+findings+progress)`

---

<!--
后续 session 都从这行下面 append. 新对话 Claude 生成 BLOCK 1 后,
用户直接粘到这里. 格式照抄上面模板.

-->
## 2026-04-24 · Session 04 · 深化 check_data + Phase 2 基础工具

- **Phase**: Phase 1 尾声 → Phase 2 起步
- **Actions**:
  - check_data.py：重写 C06（读 SV CSV 映射）、C07（地理范围+覆盖率）、C09（实体类型分布+关系明细）、C10（自动精确交集）、新增 C12（建筑-街区空间join）、C13（标签-KG对齐）
  - 新建 scripts/make_block_whitelist.py（Phase 1 收尾，生成 block_whitelist.csv）
  - 新建 src/utils/label_transform.py（log1p/expm1 封装）
  - 新建 src/utils/metrics.py（RMSE/MAE/MAPE/R²，log+raw 双空间，MetricsLogger）
- **Files produced/modified**:
  - `scripts/check_data.py` · 修改（C06/C07/C09/C10 重写，C12/C13 新增）
  - `scripts/make_block_whitelist.py` · 创建
  - `src/utils/label_transform.py` · 创建
  - `src/utils/metrics.py` · 创建
- **Key findings**（来自 16:17 报告）:
  - 已填入 findings.md § 4
- **Open issues**:
  1. 街景 CSV 映射文件内容未知，C06 是否能自动读取待跑脚本验证
  2. KG Region 实体格式（Region_xxx vs 纯数字）待 C13 确认
  3. 卫星大 TIF 尚未裁切为 per-block，C07 覆盖率检查待运行
- **Next session**: 运行新版 check_data.py，确认 C06/C10/C12/C13 结果，再开始 block_index.py（Phase 2 核心）
- **Commit message**: feat(phase1+2): deep check C06/C09/C12/C13 + make_whitelist + label_transform + metrics
