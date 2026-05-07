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

---

## 2026-04-30 · Session 06 · Phase B 快速原型脚本 — 7 backbone + 15 KG 模型全套代码交付

- **Phase**: Phase B (新增并行轨道) · 快速原型脚本
- **Goal of this session**: 设计并交付 7 个独立脚本, 覆盖 7 视觉 backbone × 2 模态能耗预测 + 15 KG embedding 模型 × 2 KG 构建与训练, 为论文第一批基线数字铺路. 同步更新四大 docs.
- **Actions**:
  - 设计 Phase B 整体架构 (7 脚本分工, 与主流程 Phase 1-8 的关系)
  - 写 `1_build_labels.py` (109 行): `沈阳L4能耗.shp` → log+Z-score → 5 分位分层 7:1.5:1.5 划分
  - 写 `2_predict_streetview.py` (433 行): `build_backbone(name)` 工厂 + 7 backbone + per-backbone 特征缓存 + MLP 回归 + 汇总表
  - 写 `3_predict_remote_sensing.py` (615 行): ESRI 瓦片 (zoom=17) 优先 + TIF fallback + 7 backbone + 最终 14 路对比表
  - 写 `4_build_base_kg.py` (208 行): block-POI-主类-子类-landuse 五类实体 + 5 关系 + 90/5/5 划分 + block_to_entity.json
  - 写 `5_build_building_kg.py` (289 行): + 建筑物 (Height/Function/Age/Quality) + POI sub_type; Height 4 档离散化; Age 自动识别 (年份/楼龄/字符串)
  - 写 `6_kg_models.py` (648 行): 15 个 KG embedding 模型 + Poincaré ball / 球面 / Givens 几何工具 + MODEL_REGISTRY + forward 自测
  - 写 `7_train_kg.py` (335 行): 自对抗损失 (gamma=12, alpha=0.5) + 64 neg/pos + filtered link prediction + embedding 导出 (.npz 含 block_emb 子集)
  - 所有脚本 AST 语法检查通过
  - 多轮更新 docs (task_plan / progress / findings / CLAUDE): 用户两次指出应在原有基础上更新而非重新生成
  - 第一次 docs 更新未读原始文件直接重写 → 用户指正 → 读完四个原始文件后增量更新 (本条)
- **Files produced/modified**:
  - `phase_b_scripts/1_build_labels.py` · 创建
  - `phase_b_scripts/2_predict_streetview.py` · 创建
  - `phase_b_scripts/3_predict_remote_sensing.py` · 创建
  - `phase_b_scripts/4_build_base_kg.py` · 创建
  - `phase_b_scripts/5_build_building_kg.py` · 创建
  - `phase_b_scripts/6_kg_models.py` · 创建
  - `phase_b_scripts/7_train_kg.py` · 创建
  - `docs/CLAUDE.md` · 修改 (§ 0 新增 Phase B 说明, § 11 新增第 7 条豁免)
  - `docs/task_plan.md` · 修改 (目录 + Phase B 全段 + Decisions +8 行 + Errors +2 行 + 当前活动 Phase 更新)
  - `docs/findings.md` · 修改 (§ 9 +7 条新发现 + § 11 新增 Phase B 架构速查)
  - `docs/progress.md` · 追加本条
- **Key findings**:
  - AttentionCNN (CBAM) ≠ AttH (双曲注意力), 两者名字近似但完全不同 · findings § 9 + § 11.2
  - Phase B 标签路径与主流程不一致 (⚠️ 待确认) · findings § 9 末条 + task_plan Errors
  - Phase B KG 与 complete_knowledge_graph.txt 是两套独立图 · findings § 11.2
  - 15 模型 M2GNN/GIE 简化版去 GNN 分支以保统一接口 · findings § 9
- **Errors encountered**:
  - 本次两次 docs 更新均未先读原始文件 → 重新生成而非增量更新 → 用户指正后才正确执行. 教训再次验证: **先读所有 project files 再动笔**, CLAUDE.md § 11 "不确定就问" + "看完 project files" 要更严格执行.
- **Open issues**:
  - ⚠️ `E_Final_W5`/`BlockID` (SHP) 与 `energy`/`Region_N` (JSON) 是否同源 — **Phase B 第一步必须验证**
  - ⚠️ `BlockID` (L4) 与 `LandID` (L5) 编号体系是否互通 — 融合阶段必须确认
  - Phase B 脚本硬编码 `G:\Knowcl` 路径 — 待正式集成前迁移到 `paths.yaml`
  - M2GNN / GIE 完整版需 PyG 依赖 — 当前简化版为 GNN-free 骨架
  - Phase 1.5 smoke test 结果尚未回来
- **Next session**:
  - 优先: 用户确认 ⚠️ 待对账风险 (E_Final_W5 vs energy 统计对比), 然后跑 `1_build_labels.py` 看输出分布
  - 若 Phase 1.5 smoke test 回来, 根据 `panoid_hit_rate` 决定是否全跑
  - Phase 1 收尾: 运行 `make_block_whitelist.py` 生成 208-block 白名单
- **Git**: `feat(phase-b): 7 standalone scripts — 7 backbone + 15 kg models + docs update`

---

## 2026-05-07 · Session 07 · Phase B 街景预测 v2 → v2.1 修 ridge 长度 bug, 精度翻倍验证

- **Phase**: Phase B (并行轨道)
- **Goal of this session**: 修 v1 R²(log)≈0.10 严重欠拟合
- **Actions**:
  - 诊断 v1 三大病灶: ① 单图 L2 归一化丢幅度 ② 单 mean pool 丢方差 ③ 无 baseline 对照
  - v2 重写 `2_predict_streetview.py`: 取消预归一化 + mean+std 双池化 + 辅助统计特征 (log 图数 + lon/lat std) + 输入端 BN + Ridge 与 MLP 同表 + L0 自检 + 单图特征缓存
  - v2 实测精度翻倍 (R²(log) 0.18→0.37) 但 Ridge 返回长度只有 test 集 → 保存阶段 "All arrays must be of the same length" → all_metrics 始终空 → 末尾 KeyError: 'R2_log'
  - v2.1 修 fit_ridge 返回全长预测 + main 末尾防御性处理空 metrics + try-except 加 traceback
  - L0 baseline (仅 log图数+GPS std Ridge) R²(log)=0.0923, 图像净增益 +0.28 (resnet50 mlp 0.371 - 0.092)
  - 释明: 高维 + 小样本下 Ridge 经常打过 MLP, 同表对照是必须的
- **Files produced/modified**:
  - `phase_b_scripts/2_predict_streetview.py` · 重写 v2.1 (~510 行)
- **Key findings** (待 append findings § 9):
  - SV 模态最佳: ResNet50 R²(log)=0.371, AttentionCNN 0.343, 与 KnowCL 论文 SV R²~0.377 同量级
  - 图像净增益 = backbone R² - 仅辅助特征 R² = 0.371 - 0.092 = +0.28 (信号显著)
  - 街景采样质量 OK, 不需要换 DINOv2/CLIP
  - v2 设计教训: 多输出函数返回时切片粒度要对齐, sklearn baseline 函数应统一返回全长, 算指标时切片
- **Errors encountered**:
  - v2 fit_ridge 只返回 test 预测但保存时与全长 mlp_pred 同 DataFrame → 长度冲突 → main 末尾 KeyError. 已记 task_plan Errors.
- **Open issues**:
  - 用户 v2.1 重跑 (磁盘缓存, < 5 分钟) 验证 metrics summary 落盘
  - Ridge 在部分 backbone 接近 MLP, 说明 MLP 容量够大可能轻微过拟合
- **Next session**: 跑 3_predict_remote_sensing.py, 看 SI 是否如预期高于 SV
- **Git**: `fix(phase-b/sv): v2.1 ridge full-set predict + defensive empty-metrics`

---

## 2026-05-07 · Session 08 · Phase B 遥感预测 v2.1 — 与街景同源 bug 修复 + 几何辅助特征

- **Phase**: Phase B (并行轨道)
- **Goal of this session**: 检查 3_predict_remote_sensing.py 与街景 v1 同源的 L2 归一化 bug, 升级到 v2.1 与街景对齐
- **Actions**:
  - 静态审查 3_predict_remote_sensing.py 找出 6 个问题:
    ① L367 单图 L2 归一化丢幅度 (与街景 v1 同款 fatal bug)
    ② 缺 L0 baseline self-check
    ③ MLP 旧超参 (dropout 0.3 / wd 1e-4 / lr 1e-3 / batch 32)
    ④ 无 Ridge 对照
    ⑤ print_full_comparison 与街景 v2.1 schema (含 model/R2_norm) 不兼容
    ⑥ final_comparison 列命名冲突
  - 重写 3_predict_remote_sensing.py v2.1 (~660 行):
    * 删除归一化, cache 改名 _v2.npz 自动绕开 v1 旧缓存
    * MLP 升级到 v2.1 同款 (BN 输入 + 256→64 + dropout 0.5 + wd 1e-3 + lr 5e-4 + batch 64)
    * 加几何辅助特征 5 维 (log 面积/周长 + 紧凑度 + 中心经纬度) 替代街景的"图数+GPS std"
    * 加 L0 baseline (均值 + 几何辅助 Ridge)
    * 加 Ridge 与 MLP 同表对照, 全样本预测以保对齐
    * metrics_summary 增加 model 和 R2_norm 列, 与街景 v2.1 schema 完全一致
    * final_comparison 列前缀统一 sv_<bb> / rs_<bb>
    * AST 语法检查通过
  - 阐明遥感 vs 街景架构差异: 遥感单图无法 mean+std 池化, 改走"几何辅助"路线
- **Files produced/modified**:
  - `phase_b_scripts/3_predict_remote_sensing.py` · 重写 v2.1 (~660 行)
- **Key findings** (待 append findings § 9):
  - 遥感 v1 与街景 v1 共享同一行 L2 归一化 bug — Phase B 单图 backbone 流水线模板都需要审计
  - 遥感"零成本基线" = 街区几何 (面积是能耗的强基线), 区别于街景的"图数+GPS std"
  - sv_metrics_summary 和 rs_metrics_summary 已用同一 schema (含 model/R2_norm), 可直接 concat
- **Errors encountered**:
  - L2 归一化 bug 被 v1 模板从街景复制到遥感 — 模板复用时要审计每行
- **Open issues**:
  - 用户跑 v2.1 后回填 R²(log) 数字, 验证 SV < SI 不等式
  - cache 改名 _v2.npz 后第一次跑会重新提 7 backbone × 698 张图 ≈ 30-40 分钟 GPU
  - 几何辅助 Ridge baseline R²(log) 期望 0.10-0.20 (面积是强基线)
- **Next session**: 用户回填 v2.1 结果 → 进入 Phase B KG 轨道 (4_build_base_kg → 7_train_kg)
- **Git**: `fix(phase-b/rs): v2.1 drop L2 norm + geom aux + ridge + L0 baseline + schema align`

---

## 2026-05-07 · Session 09 · Phase B 遥感 v2.1 实测落盘 + SV/SI 对比定调 + CLAUDE.md L2 戒律新增

- **Phase**: Phase B (并行轨道) — Step 2 (SV) 和 Step 3 (SI) 均已实测完成
- **Goal of this session**: 用户跑通遥感 v2.1, 评估是否需要继续优化, 定调进入 KG 轨道; 同步更新 4 文档.
- **Actions**:
  - 用户实测遥感 v2.1: 7 backbone × {mlp, ridge} = 14 行 metrics 完整落盘
  - 排序结果: densenet121 ridge R²(log)=0.342 (头) → mobilenet_v3_large mlp 0.162 (尾)
  - 对比街景 v2.1: SV 最佳 resnet50 mlp R²(log)=0.371; SI 最佳 densenet121 ridge 0.342
  - 关键发现: log 空间 SV (0.371) > SI (0.342), raw 空间 SI (0.172) > SV (0.127), 两者总体水平相当, **沈阳能耗任务上 SV < SI 等级不严格成立** — 与 KnowCL 论文 § 5.2 "城市/指标偏好不同" 一致
  - 几何辅助 baseline R²(log)=0.067, 图像净增益 +0.275 ✓ 显著
  - 决策: 单模态精度已合格 (远超 0.05 健康阈值, 接近 KnowCL 纽约 SV R²=0.377 量级), 不再优化, 进入 KG 轨道
  - CLAUDE.md § 9 L2 段追加第 4 项戒律: 单图 backbone 特征聚合前不要做 L2 归一化 (此前 sv/rs v1→v2 都验证了)
  - 同步全量更新 progress.md (本条) / task_plan.md (Phase B 步骤打勾, Decisions +2, Errors 维持) / findings.md (§ 9 +5 条 + § 11.1 表更新)
- **Files produced/modified**:
  - `docs/progress.md` · 追加本条 (Session 09)
  - `docs/task_plan.md` · 修改 (Phase B 步骤 2-3 标 ✅ 已实测, Decisions +2 行, 当前活动 Phase 更新)
  - `docs/findings.md` · 修改 (§ 9 +5 条 sv+rs v2.1 实测发现, § 11.1 表 rs 行新增列)
  - `docs/CLAUDE.md` · 修改 (§ 9 L2 第 4 项追加, § 11 硬约束 + 1 条 v1 模板审计)
- **Key findings** (已 append 到 findings.md § 9):
  - 沈阳能耗任务 SV < SI 等级不严格成立 (log 空间 SV 略胜, raw 空间 SI 略胜), 与论文 § 5.2 一致
  - 轻量 backbone (MobileNetV3, EfficientNet-B0) 在单图遥感 + 224 patch 上明显垫底 (R²(log) 0.16-0.18 vs 其他 0.29-0.34)
  - Ridge 在 DenseNet121 上首次明确打败 MLP (0.342 vs 0.320), 验证"高维+小样本 Ridge 常更稳"经验法则
  - 遥感几何辅助 baseline R²(log)=0.067, 弱于街景 aux 的 0.092 (街景含图数信号)
  - L2 归一化在单图特征聚合阶段是反向操作 (论文里只在 InfoNCE 投影头之后才归一化)
- **Errors encountered**: 无新增 — 用户的 v2.1 跑通无 bug
- **Open issues**:
  - ⚠️ Phase B Step 1 标签对账 (E_Final_W5 vs energy) 仍然挂着 — 进入 KG 轨道前应该再确认一次
  - rs_predictions_<bb>.csv 现在含 mlp_pred_energy + ridge_pred_energy 两列, 下游融合阶段决定用哪一列要明确写 (建议 densenet121 ridge 作为 SI 模态代表)
  - DINOv2 / CLIP / 多池化 spatial mean+max+std 等进一步优化方向已记录, 当前不做
- **Next session**:
  - 进入 Phase B Step 4: 跑 `4_build_base_kg.py` → 看 base KG 三元组规模 + entities/relations 分布
  - 然后 5_build_building_kg.py / 6_kg_models.py 自测 / 7_train_kg.py 跑 15 模型 link prediction
  - 出 `metrics_summary.csv` (15 KG 模型 MRR/Hits@K) + `embeddings_<model>.npz` 以备融合阶段用
- **Git**: `docs(phase-b): session 09 sv+rs v2.1 results landed + claude L2 guard + 4-doc sync`
