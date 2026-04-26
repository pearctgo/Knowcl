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
- **Goal of this session**: 在第一次对话里把整个长期项目的协作框架定下, 产出 4 份持久化文件.
- **Actions**:
  - 读完 `tsinghua-fib-lab/UrbanKG-KnowCL` README
  - 读完 KnowCL 论文原文 (arxiv 2302.13094)
  - 读完 `OthmanAdi/planning-with-files` skill 的 workflow
  - 按 skill 规范搭建 4 文件架构 (CLAUDE / task_plan / findings / progress)
- **Files produced/modified**:
  - `docs/CLAUDE.md` · 创建
  - `docs/task_plan.md` · 创建
  - `docs/findings.md` · 创建
  - `docs/progress.md` · 创建
- **Key findings**:
  - KnowCL Semantic Encoder = CompGCN + TuckER 初始化 · findings.md § 2.3
  - InfoNCE 用点积非 cosine · § 2.3
  - SI/SV batch size 不对称 (128 vs 16) · § 2.5
- **Open issues**:
  - findings.md § 4 所有数据事实空格待填
  - 代码目录细分待 Phase 2 定
- **Next session**: 开启 Phase 0, 搭建可移植项目骨架
- **Git**: `chore(init): scaffold with planning-with-files protocol`

---

## 2026-04-23 · Session 02 · Phase 0 可移植项目骨架

- **Phase**: Phase 0 · 可移植项目骨架
- **Goal of this session**: 产出项目全部骨架文件, 使任意机器可 clone → pip install → run.
- **Actions**:
  - 生成 `.env.example` / `.gitignore` / `README.md` / `requirements.txt`
  - 生成 `config/paths.yaml` (所有路径从 DATA_ROOT 派生)
  - 生成 `scripts/setup.sh` 和 `scripts/setup.bat`
  - 从零编写 `scripts/check_data.py` (C01-C11, 11 个检查项)
  - 生成 `results/experiments.csv` 占位文件
  - 设计 Commit Message 规范 + git stash 使用场景
- **Files produced/modified**:
  - `.env.example` · 创建
  - `.gitignore` · 创建
  - `README.md` · 创建
  - `requirements.txt` · 创建
  - `config/paths.yaml` · 创建
  - `scripts/setup.sh` · 创建
  - `scripts/setup.bat` · 创建
  - `scripts/check_data.py` · 创建 (C01-C11)
  - `results/experiments.csv` · 创建
- **Key findings**:
  - check_data.py 架构: Report 类 + 逐项检查函数 + write_report · § 9
- **Open issues**:
  - requirements.txt 中 torch/dgl 版本需按用户 GPU 调整
- **Next session**: 根据实际数据格式修正 check_data.py 并运行
- **Git**: `chore(init): scaffold phase 0 — env/paths/setup/check_data/experiments`

---

## 2026-04-24 · Session 03 · 修正 check_data 适配真实数据格式

- **Phase**: Phase 0 尾声
- **Goal of this session**: 根据用户告知的真实数据格式, 修正 check_data.py.
- **Actions**:
  - C03 重写: 新增 JSON 解析 (_load_json_as_df), 支持 dict-of-dict / list / dict+list 三种结构
  - C08 重写: 兼容 shenyang_zl15_*.csv 命名, 自动检测 block_id 列, 新增比例检查
  - 确认建筑高度属性在 processed_shenyang20230318.shp 的 Height 字段
  - 记录街景平铺问题为 Phase 1 blocking issue
- **Files produced/modified**:
  - `scripts/check_data.py` · 修改 (C03 + C08)
- **Key findings**:
  - 标签来源链: ec_2017sy.tif → JSON · findings.md § 9
  - 建筑主文件: processed_shenyang20230318.shp, Height 字段 · § 4.1
- **Errors encountered**:
  - C03 未识别 JSON 格式 → 重写, 已在 task_plan.md Errors 记录
  - C08 未识别 shenyang_zl15_*.csv → 重写, 已记录
- **Open issues**:
  - 街景 203452 张平铺, 需决定 CSV 映射 or 子目录整理
- **Next session**: 深化检查项, 添加 C12/C13, 生成 whitelist
- **Git**: `fix(check): C03 add JSON support, C08 adapt shenyang_zl15_*.csv naming`

---

## 2026-04-24 · Session 04 · 深化 check_data + Phase 2 基础工具

- **Phase**: Phase 1 尾声 → Phase 2 起步
- **Goal of this session**: 基于已跑通的 data_check_summary.json 填入数据事实, 深化检查项, 产出 Phase 2 基础工具.
- **Actions**:
  - 分析 data_check_summary.json, 填入 findings.md § 4 全部数字
  - check_data.py: 重写 C06 (SV CSV 映射读取), C07 (地理范围+覆盖率), C09 (实体类型分布+关系明细), C10 (自动精确交集), 新增 C12 (建筑-街区空间 join), C13 (标签-KG 对齐)
  - 新建 `scripts/make_block_whitelist.py` (Phase 1 收尾, 生成 208-block 主实验集)
  - 新建 `src/utils/label_transform.py` (log1p/expm1 封装)
  - 新建 `src/utils/metrics.py` (RMSE/MAE/MAPE/R², MetricsLogger)
  - 决策: 主实验用 208-block 子集, 不重新划分 757-block
  - 重写四大文件完整版 (本次会话收尾)
- **Files produced/modified**:
  - `scripts/check_data.py` · 修改 (C06/C07/C09/C10 重写, C12/C13 新增, 共 13 项)
  - `scripts/make_block_whitelist.py` · 创建
  - `src/utils/label_transform.py` · 创建
  - `src/utils/metrics.py` · 创建
  - `src/__init__.py` · 创建
  - `src/utils/__init__.py` · 创建
  - `docs/findings.md` · 重写 § 4 + § 5 关系明细 + § 9 新发现
  - `docs/task_plan.md` · 更新 Phase 状态 + 决策 + Errors
  - `docs/progress.md` · 追加本条
- **Key findings**:
  - SV 空间 join 仅覆盖 208 块 (< 500 阈值), 是最大瓶颈 · findings.md § 4.1
  - SI 已有 757 张 per-block PNG (15-遥感影像), 无需裁图 · § 4.1 + § 9
  - KG 已含 buildingFunction + belongsToLand, Phase 3 只补 2 种关系 · § 5.1 + § 9
  - label block_id 与 KG Region 实体格式完全一致 (Region_N), 无需转换 · § 4.2 + § 9
- **Errors encountered**:
  - SV 交集 208 < 500 阈值 → 改用 208-block 子集 + 重新划分, 已在 Errors 表记录
- **Open issues**:
  - 能耗单位未确认 (kWh? MJ? GJ/m²?)
  - GPU 硬件型号用户未告知, requirements.txt 中 torch 版本待定
  - make_block_whitelist.py 待运行, block_whitelist.csv 尚未生成
- **Next session**: ① 运行 make_block_whitelist.py 生成 208-block 白名单 (Phase 1 最后一步); ② 开始 Phase 2 第一个任务: `src/datasets/block_index.py`
- **Git**: `feat(phase1+2): deep check C12/C13 + whitelist + label_transform + metrics + doc update`

---

<!--
后续 session 都从这行下面 append.
-->

## 2026-04-26 · Session 05 · Phase 1.5 街景重采 — 走错路径 → 用户纠正 → 切回 mapsv0 内部端点

- **Phase**: Phase 1.5 (新增) · 街景全量重采 — 代码完成, 等用户跑 smoke test.
- **Goal of this session**: 修掉旧 `街景采集.py` 的 AK 读取 bug 和 G:\ 硬编码, 写全 757 街区采集代码.
- **本会话经过 1 次大返工**:
  - **第一版**: Claude 没看用户已上传的 `test_shenhe.py`, 默认按"现代正确做法"设计 → 走百度开放平台官方 panorama/v2 API → 写了 5 步 AK 申请教程 + 三段式回退 + probe-first
  - **用户纠正**: "你看一下这两个代码 (test_shenhe.py), 里面是不是就有现成的 api"
  - **真相**: 项目原 `test_shenhe.py` 走 `mapsv0.bdimg.com` 内部端点, 不需要 AK; 那个共享 AK `mYL7zDrHfcb0ziXBqhBOcqFefrbRUnuq` 只用在 geoconv 坐标转换上, 而坐标转换可以纯 Python 实现
  - **第二版**: 切换到 mapsv0 内部端点, 与项目原 test_shenhe.py 同方法, 不需要 AK
- **教训**: **看完所有 project files 再设计方案**. 用户上传的 `train_pspnet_crackdata.py` 和 `test_shenhe.py` 第一次会话已经在 project 里, 但 Claude 没全读. CLAUDE.md § 11 "不确定就问" 应该执行得更严, 看到"街景采集"任务先 grep 用户已有代码里的 baidu/streetview/url 关键词.
- **Actions** (修订后):
  - 读 train_pspnet_crackdata.py — PSPNet 训练脚本, 与街景无关 (排除)
  - 读 test_shenhe.py — 找到既有采集方法: mapsv0.bdimg.com qsdata + pr3d
  - 改写 `collect_streetview_baidu_full.py` (~960 行):
    * 走 mapsv0.bdimg.com 内部端点, 无 AK
    * WGS84 → BD09MC 全本地, 含 Baidu 6 段多项式 (实测北京天安门误差 < 500m)
    * 真实浏览器 UA 池 + Referer 头反爬
    * panoid 不存在则跳过 (天然 probe)
    * 整点 4 张图全已存在则跳 panoid 查询 (二级断点续采)
    * 输出 streetview_index.csv 给 Phase 2 直接消费
  - `docs/baidu_ak_setup_guide.md` 保留作 Plan B (后端如果失效再启用)
  - 升级 CLAUDE.md § 7.1: 同时收录两种路径的处理规范
  - 更新 task_plan.md: Phase 1.5 改写, 决策表 +5 行, Errors +1 行 (本次走错路径教训)
  - 更新 findings.md: § 9 加 8 条新发现 (替换原版), § 10 改写为两条路径速查
- **Files produced/modified**:
  - `scripts/collect_streetview_baidu_full.py` · 重写 (~960 行)
  - `docs/CLAUDE.md` · 修改 (§ 0 + § 7.1 修订)
  - `docs/task_plan.md` · 修改 (Phase 1.5 重写, Decisions +5, Errors +1)
  - `docs/findings.md` · 修改 (§ 9 8 条新发现替换, § 10 双路径速查)
  - `docs/progress.md` · 追加本条
  - `docs/baidu_ak_setup_guide.md` · 保留 (从 Plan A 降级为 Plan B)
- **Key findings**:
  - mapsv0.bdimg.com 两个内部端点的精确格式 (findings § 10.1)
  - WGS84 → BD09MC 6 段多项式系数 (findings § 10.1.3)
  - 反爬经验值: 点间 2s, 图间 0.3s (findings § 10.1.4)
  - 双路径选择标准: 内部端点 = 主, 官方 API = Plan B (findings § 10.2)
- **Errors encountered**:
  - 第一版走错路径 (官方 API) — 已记 task_plan Errors. 教训: 用户 project files 必须先全看
- **Open issues**:
  - 用户在 Saitama, JP, 可能要走 VPN 才能访问 mapsv0.bdimg.com (百度国内域名对境外友好但不保证)
  - 沈阳新建小区/园区 panoid 缺失率预计较高, 实际数字 smoke test 出来才知道
  - 12k 张采下来后**仍需重新空间 join 验证落区**, 候选点在街区内但百度返回的全景实际位置可能在街区边界外
  - 此次会话又改了 4-5 个文件, 仍超 § 8 "≤ 2 文件" 上限, 但 docs 维护协议本身就支持 4 doc 联动改, 按例外处理
- **Next session**:
  - 等用户 smoke test → 看 `panoid_hit_rate` / `image_status_counts` → 决定是否全跑或调参
  - 若 panoid_hit_rate < 30%, 检查坐标转换公式或换 IP
  - 若全跑成功且覆盖 ≥ 500 块, 把新 streetview_index.csv 输入 make_block_whitelist.py 重新生成主实验集
- **Git**: `feat(phase1.5): switch to mapsv0 internal endpoint (no ak needed) + docs upgrade`
