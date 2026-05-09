# -*- coding: utf-8 -*-
r"""
============================================================
7_train_kg.py  v4.3   (2026-05-09)
  KG embedding 训练 + 下游能耗回归 · 分层归因评估
============================================================

★★ 相对 v4.2 的关键修订 (2026-05-09 用户驳回 v4.2 的"掺手工特征"做法) ★★

  v4.2 在下游评估时把 handcrafted features (POI 类目计数 + 建筑统计 + 几何)
  和 block embedding 一起 concat 进 Ridge/MLP, 这相当于在评估"KG 模型的下游
  能力"时偷渡了大量非 KG 信号 (即使把 KG 换成随机向量, 光靠 handcrafted 也
  能跑到 R²(log) ≈ 0.2-0.3). 这是不科学的, 论文里没法这么写.

  v4.3 改为 **分层归因**: 每个 KG 模型同时报 4 组特征下的下游 R²:

    A. kg_only            : block_emb (d=32)                          ← 纯 KG
    B. kg_nbr             : A + per-relation 1-hop neighbor mean      ← 纯 KG
    C. kg_nbr_topo        : B + per-relation log(1+deg)               ← 纯 KG
    D. kg_oracle_handcraft: C + handcrafted features (POI/建筑/几何)  ← 混合, 仅 ablation

  默认仅跑 A/B/C 三组 (纯 KG). D 组需 --include-oracle 显式开启,
  报告时必须标注 "oracle ablation, not pure KG".

  论文里 KG 模型的代表精度取 B 或 C, 与 SV/SI 单模态精度 (Phase B Step 2/3)
  公平对比 (SV/SI 单模态也不掺 handcrafted, 只用 image backbone 输出).

  保留 v4.2 已修好的部分:
    - line 591 numpy.detach() bug fix
    - _norm_id 系列 (BlockID 对齐, "1.0" / "1" 折叠)
    - 训一个预测一个 + 增量写盘 + KeyboardInterrupt 兜底

输出 (<KG_ROOT>/<base|building>/embeddings/):
  embeddings_<model>.npz         : ent_emb [, rel_emb], block_id, block_emb, kg_metrics
                                   (block_id / block_emb 是给 8_contrastive 用的对齐版)
  metrics_<model>.json           : 单模型 KGE + 4 组下游详细
  ../metrics_summary.csv (KG_ROOT 根目录)
    每行 = (kg, model, feature_set, kg_mrr, kg_h10, ridge_test_r2,
            mlp_test_r2, ridge_alpha, n_features, ...)

用法:
  python 7_train_kg.py                                  # base+building, 默认 4 模型, 仅 A/B/C
  python 7_train_kg.py --kgs base --models transe       # 单模型快速冒烟
  python 7_train_kg.py --include-oracle                 # 加做 oracle ablation 上界
  python 7_train_kg.py --skip-existing                  # 已存 npz 复用 KGE, 只重算下游
  python 7_train_kg.py --plain-block-emb --kgs base --models transe
                                                        # 随机 emb 健康检查 (R² 应 ≈ 0)
"""

from __future__ import annotations
import os, sys, json, argparse, time, math
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 加载 6_kg_models.py
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "6_kg_models.py")
if not os.path.exists(_MODEL_PATH):
    raise RuntimeError(f"找不到 {_MODEL_PATH}")
models_mod = _load_module("kg_models", _MODEL_PATH)
MODEL_REG = getattr(models_mod, "MODEL_REGISTRY", None) or models_mod.MODEL_REG

# ============================================================
# 路径
# ============================================================
ROOT      = os.environ.get("KNOWCL_ROOT", r"G:\Knowcl")
KG_ROOT   = os.path.join(ROOT, "999-输出成果文件", "003-知识图谱")
LABEL_CSV = os.path.join(ROOT, "999-输出成果文件", "002-能耗预测", "energy_labels.csv")

# ============================================================
# 默认超参
# ============================================================
DEFAULT_DIM    = 32
DEFAULT_EPOCHS = 100
BATCH_SIZE     = 1024
LR_KG          = 5e-4
KG_PATIENCE    = 15

# ============================================================
# ID 规范化 (跟 4_/5_ 完全一致)
# ============================================================
ID_CANDIDATES = ['BlockID', 'block_id', 'BLOCKID', 'blockid',
                 'LandID',  'land_id',  'LANDID',  'landid',
                 'RegionID','region_id','REGIONID',
                 'ID', 'id', 'index', '_id', 'OBJECTID', 'FID']

def _pick_id_col(columns, candidates=ID_CANDIDATES):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns: return cand
        if cand.lower() in cols_lower: return cols_lower[cand.lower()]
    return None

def _norm_id(v) -> str:
    if v is None: return ""
    try:
        if pd.isna(v): return ""
    except Exception:
        pass
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if math.isfinite(v) and float(v).is_integer():
            return str(int(v))
        return str(v).strip()
    s = str(v).strip()
    try:
        f = float(s)
        if math.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s

def _norm_id_series(s: pd.Series) -> pd.Series:
    return s.map(_norm_id)

# ============================================================
# KG / block_index / relations 加载
# ============================================================
def _load_tsv(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return out

def load_kg(kg_dir):
    train = _load_tsv(os.path.join(kg_dir, "train.tsv"))
    valid = _load_tsv(os.path.join(kg_dir, "valid.tsv")) if os.path.exists(os.path.join(kg_dir, "valid.tsv")) else []
    test  = _load_tsv(os.path.join(kg_dir, "test.tsv"))  if os.path.exists(os.path.join(kg_dir, "test.tsv"))  else []
    with open(os.path.join(kg_dir, "entities.json"), "r", encoding="utf-8") as f:
        ent2id = json.load(f)
    with open(os.path.join(kg_dir, "relations.json"), "r", encoding="utf-8") as f:
        rel2id = json.load(f)
    return {"train": train, "valid": valid, "test": test,
            "n_e": len(ent2id), "n_r": len(rel2id),
            "ent2id": ent2id, "rel2id": rel2id}

def load_block_index(kg_dir):
    """读 block_index.tsv, 兼容 (bid \\t eid) 与 (eid \\t bid) 两种顺序"""
    out = {}
    p = os.path.join(kg_dir, "block_index.tsv")
    if not os.path.exists(p):
        # 回退: 从 entities.json 找 'block:' 前缀
        p2 = os.path.join(kg_dir, "entities.json")
        if os.path.exists(p2):
            with open(p2, "r", encoding="utf-8") as f:
                ent2id = json.load(f)
            for ent_str, eid in ent2id.items():
                if ":" in ent_str:
                    typ, val = ent_str.split(":", 1)
                    if typ == "block":
                        out[_norm_id(val)] = int(eid)
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                a, b = parts[0], parts[1]
                # 优先 (bid, eid)
                try:
                    eid = int(b); out[_norm_id(a)] = eid
                except ValueError:
                    try:
                        eid = int(a); out[_norm_id(b)] = eid
                    except ValueError:
                        continue
    return out

def load_relation_meta(kg_dir):
    meta = {}
    rel_path = os.path.join(kg_dir, "relations.json")
    if not os.path.exists(rel_path): return meta
    with open(rel_path, "r", encoding="utf-8") as f:
        rel2id = json.load(f)
    SYM_KEYWORDS = ["nearby", "border", "similar", "competitive", "co_check"]
    for rname, rid in rel2id.items():
        is_sym = any(k in rname.lower() for k in SYM_KEYWORDS)
        meta[int(rid)] = (rname, is_sym)
    return meta

# ============================================================
# labels.csv 加载
# ============================================================
def load_labels(label_csv, id_col=None):
    if not os.path.exists(label_csv):
        raise RuntimeError(f"找不到 labels: {label_csv}")
    df = pd.read_csv(label_csv)
    if id_col is None:
        id_col = _pick_id_col(df.columns)
    if id_col is None:
        raise RuntimeError(
            f"labels.csv 没有识别出 ID 列, 字段 {list(df.columns)}. "
            "请加 --label-id-col <列名> 手动指定")
    print(f"    使用 '{id_col}' 作 block 主键 (规范化后跟 KG block_index 对齐)", flush=True)
    if id_col != "BlockID":
        df = df.rename(columns={id_col: "BlockID"})
    df["BlockID"] = _norm_id_series(df["BlockID"])
    if "split" not in df.columns:
        raise RuntimeError(f"{label_csv} 缺 split 列, 字段 {list(df.columns)}")

    energy_col = None
    for c in ("energy", "energy_log", "y", "label", "log_energy"):
        if c in df.columns:
            energy_col = c; break
    if energy_col is None:
        raise RuntimeError(f"{label_csv} 缺 energy 列, 字段 {list(df.columns)}")
    if energy_col != "energy":
        df = df.rename(columns={energy_col: "energy"})
    is_log = (energy_col == "energy_log") or (energy_col == "log_energy")
    if not is_log:
        e = df["energy"].astype(float).clip(lower=1e-6)
        if e.max() / max(e.min(), 1e-6) > 50:
            df["energy_log"] = np.log(e)
        else:
            df["energy_log"] = e
    else:
        df["energy_log"] = df["energy"].astype(float)

    train_df = df[df["split"] == "train"]
    log_mean = train_df["energy_log"].mean()
    log_std  = train_df["energy_log"].std()
    print(f"    train={int((df['split']=='train').sum())} "
          f"val={int((df['split']=='val').sum())} "
          f"test={int((df['split']=='test').sum())}  "
          f"log_mean={log_mean:.4f}  log_std={log_std:.4f}", flush=True)
    return df, float(log_mean), float(log_std)

# ============================================================
# ★★★ 关键新模块: per-relation 邻居 embedding & 度数 ★★★
# ============================================================
def build_block_neighbor_features(block_idx, ent_emb, kg_triples, n_r):
    """
    对每个 KG 中的 block 实体, 计算:
      M_nbr_per_rel[block_local, r, :] = 关系 r 下 1-hop 邻居 embedding 均值 (出+入)
      M_deg_per_rel[block_local, r]    = 关系 r 下 1-hop 度数 (出+入)
    """
    block_eids = sorted(set(int(v) for v in block_idx.values()))
    block_eid_to_local = {e: i for i, e in enumerate(block_eids)}
    n_blocks = len(block_eids)
    d = ent_emb.shape[1]

    sum_nbr = np.zeros((n_blocks, n_r, d), dtype=np.float32)
    cnt_nbr = np.zeros((n_blocks, n_r),    dtype=np.int32)

    for h, r, t in kg_triples:
        h, r, t = int(h), int(r), int(t)
        if h in block_eid_to_local:
            i = block_eid_to_local[h]
            sum_nbr[i, r] += ent_emb[t]
            cnt_nbr[i, r] += 1
        if t in block_eid_to_local:
            j = block_eid_to_local[t]
            sum_nbr[j, r] += ent_emb[h]
            cnt_nbr[j, r] += 1

    cnt_safe = np.maximum(cnt_nbr[..., None], 1).astype(np.float32)
    mean_nbr = sum_nbr / cnt_safe
    M_nbr = mean_nbr.reshape(n_blocks, n_r * d).astype(np.float32)
    M_deg = np.log1p(cnt_nbr.astype(np.float32))
    return block_eid_to_local, M_nbr, M_deg

def _zscore(M):
    if M.size == 0: return M
    mu = M.mean(0, keepdims=True); sd = M.std(0, keepdims=True) + 1e-6
    return (M - mu) / sd

def make_feature_sets(df_lab, block_idx, ent_emb,
                      M_nbr, M_deg, block_eid_to_local,
                      handcrafted_csv=None):
    """
    构造 4 组特征 (默认前 3 组):
      A. kg_only           : block_emb (d)
      B. kg_nbr            : A + per-relation 邻居均值 (d * n_r)
      C. kg_nbr_topo       : B + per-relation log(1+deg) (n_r)
      D. kg_oracle_handcraft: C + handcrafted (变长, 仅 oracle)
    """
    bids = df_lab["BlockID"].astype(str).values
    n = len(bids)
    d = ent_emb.shape[1]
    n_r = M_deg.shape[1]

    block_emb_arr = np.zeros((n, d), dtype=np.float32)
    nbr_arr       = np.zeros((n, n_r * d), dtype=np.float32)
    deg_arr       = np.zeros((n, n_r), dtype=np.float32)

    miss = 0; miss_examples = []
    for i, bid in enumerate(bids):
        eid = block_idx.get(bid, None)
        if eid is None or eid not in block_eid_to_local:
            miss += 1
            if len(miss_examples) < 3: miss_examples.append(bid)
            continue
        local = block_eid_to_local[eid]
        block_emb_arr[i] = ent_emb[eid]
        nbr_arr[i]       = M_nbr[local]
        deg_arr[i]       = M_deg[local]

    hit = n - miss
    print(f"  [build_X] block_idx 命中 {hit}/{n} ({100.0*hit/max(n,1):.1f}%)", flush=True)
    if miss > 0:
        print(f"  [build_X] miss 样例 BlockID: {miss_examples}", flush=True)
        if hit / max(n, 1) < 0.9:
            print(f"  [build_X] ⚠⚠⚠ 命中率 < 90%! 检查 BlockID 对齐", flush=True)

    X_A = _zscore(block_emb_arr)
    X_B = np.concatenate([X_A, _zscore(nbr_arr)], axis=1)
    X_C = np.concatenate([X_B, _zscore(deg_arr)], axis=1)
    sets = {"kg_only": X_A, "kg_nbr": X_B, "kg_nbr_topo": X_C}

    if handcrafted_csv and os.path.exists(handcrafted_csv):
        feats = pd.read_csv(handcrafted_csv)
        feat_id_col = _pick_id_col(feats.columns)
        if feat_id_col and feat_id_col != "BlockID":
            feats = feats.rename(columns={feat_id_col: "BlockID"})
        if "BlockID" in feats.columns:
            feats["BlockID"] = _norm_id_series(feats["BlockID"])
            df_for_join = pd.DataFrame({"BlockID": bids})
            merged = df_for_join.merge(feats, on="BlockID", how="left")
            num_cols = [c for c in merged.columns
                        if c != "BlockID" and pd.api.types.is_numeric_dtype(merged[c])]
            if num_cols:
                F_arr = merged[num_cols].fillna(0).values.astype(np.float32)
                X_D = np.concatenate([X_C, _zscore(F_arr)], axis=1)
                sets["kg_oracle_handcraft"] = X_D
                print(f"  [build_X] oracle 加载 handcrafted: {F_arr.shape[1]} 列 "
                      f"(⚠ 仅做 ablation, 不能算 KG 模型精度)", flush=True)
    return sets

# ============================================================
# KGE 训练 (1-vs-N CE 损失 + filtered MRR 早停)
# ============================================================
def _build_filter_dict(triples_list_of_lists):
    h2t = defaultdict(set); t2h = defaultdict(set)
    for triples in triples_list_of_lists:
        for h, r, t in triples:
            h2t[(int(h), int(r))].add(int(t))
            t2h[(int(t), int(r))].add(int(h))
    return h2t, t2h

@torch.no_grad()
def _eval_filtered_mrr(model, kg, batch=512, max_test=2000):
    device = next(model.parameters()).device
    h2t, t2h = _build_filter_dict([kg["train"], kg["valid"], kg["test"]])
    test = kg["test"]
    if len(test) > max_test:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(test), size=max_test, replace=False)
        test = [test[i] for i in idx]
    ranks = []
    for i in range(0, len(test), batch):
        bs = test[i:i+batch]
        h = torch.tensor([t[0] for t in bs], dtype=torch.long, device=device)
        r = torch.tensor([t[1] for t in bs], dtype=torch.long, device=device)
        t_ = torch.tensor([t[2] for t in bs], dtype=torch.long, device=device)
        scores = model.score_all_tails(h, r)
        for k, (hh, rr, tt) in enumerate(bs):
            for tf in h2t.get((hh, rr), set()):
                if tf != tt: scores[k, tf] = -1e9
        true_score = scores[torch.arange(len(bs), device=device), t_].unsqueeze(1)
        rank = (scores >= true_score).sum(dim=1).cpu().numpy()
        ranks.extend(rank.tolist())
        scores2 = model.score_all_heads(t_, r)
        for k, (hh, rr, tt) in enumerate(bs):
            for hf in t2h.get((tt, rr), set()):
                if hf != hh: scores2[k, hf] = -1e9
        true_score2 = scores2[torch.arange(len(bs), device=device), h].unsqueeze(1)
        rank2 = (scores2 >= true_score2).sum(dim=1).cpu().numpy()
        ranks.extend(rank2.tolist())
    ranks = np.array(ranks, dtype=np.float64)
    return {
        "mrr": float((1.0 / ranks).mean()),
        "h1":  float((ranks <= 1).mean()),
        "h3":  float((ranks <= 3).mean()),
        "h10": float((ranks <= 10).mean()),
    }

def train_kg_model(model_name, kg, dim, epochs, device,
                   batch=BATCH_SIZE, lr=LR_KG, patience=KG_PATIENCE):
    if model_name not in MODEL_REG:
        raise ValueError(f"未知模型 {model_name}, 可选 {list(MODEL_REG.keys())}")
    cls = MODEL_REG[model_name]
    use_dim = dim
    if model_name in ("rotate", "roth", "refh", "atth", "complex", "quate", "cone"):
        if use_dim % 2 != 0: use_dim += 1
    n_e, n_r = kg["n_e"], kg["n_r"]
    print(f"  loaded {len(kg['train']):,} / {len(kg['valid']):,} / {len(kg['test']):,} triples, "
          f"n_e={n_e:,} n_r={n_r}", flush=True)
    model = cls(n_e, n_r, use_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,} | 设备: {device}", flush=True)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    train_arr = np.asarray(kg["train"], dtype=np.int64)
    n_train = len(train_arr)

    best_mrr = -1.0; bad = 0; best_state = None
    last_metrics = {}
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0.0; n_batch = 0
        for i in range(0, n_train, batch):
            idx = perm[i:i+batch]
            pos = train_arr[idx]
            h = torch.from_numpy(pos[:, 0]).to(device)
            r = torch.from_numpy(pos[:, 1]).to(device)
            t = torch.from_numpy(pos[:, 2]).to(device)
            scores = model.score_all_tails(h, r)
            loss = F.cross_entropy(scores, t)
            scores2 = model.score_all_heads(t, r)
            loss = loss + F.cross_entropy(scores2, h)
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item(); n_batch += 1
        avg_loss = total_loss / max(n_batch, 1)

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            model.eval()
            m = _eval_filtered_mrr(model, kg)
            last_metrics = m
            print(f"    ep{ep:3d}  loss={avg_loss:.4f}  "
                  f"valid MRR={m['mrr']:.4f}  H@1={m['h1']:.4f}  "
                  f"H@10={m['h10']:.4f}", flush=True)
            if m["mrr"] > best_mrr:
                best_mrr = m["mrr"]; bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    print(f"    early stop at ep{ep}", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  [KG] {model_name} test MRR={last_metrics.get('mrr',float('nan')):.4f}", flush=True)
    return model, last_metrics

# ============================================================
# 回归头
# ============================================================
def metrics_reg(y_true, y_pred):
    e = y_true - y_pred
    mae  = float(np.abs(e).mean())
    rmse = float(np.sqrt((e ** 2).mean()))
    sst  = float(((y_true - y_true.mean()) ** 2).sum())
    sse  = float((e ** 2).sum())
    r2   = 1.0 - sse / max(sst, 1e-12)
    return {"r2": r2, "mae": mae, "rmse": rmse}

def fit_ridge(X_tr, y_tr, X_va, y_va, X_te, y_te):
    from sklearn.linear_model import Ridge
    best = None
    for alpha in [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        m = Ridge(alpha=alpha)
        m.fit(X_tr, y_tr)
        v = metrics_reg(y_va, m.predict(X_va))
        if best is None or v["r2"] > best["val_r2"]:
            best = {"val_r2": v["r2"], "alpha": alpha, "model": m}
    m = best["model"]
    return {"alpha": best["alpha"],
            "train": metrics_reg(y_tr, m.predict(X_tr)),
            "val":   metrics_reg(y_va, m.predict(X_va)),
            "test":  metrics_reg(y_te, m.predict(X_te))}

def fit_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te, device,
            hidden=128, epochs=300, lr=5e-4, dropout=0.5, patience=40,
            weight_decay=1e-3):
    Xt = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xv = torch.from_numpy(X_va.astype(np.float32)).to(device)
    yv = torch.from_numpy(y_va.astype(np.float32)).to(device)
    Xs = torch.from_numpy(X_te.astype(np.float32)).to(device)
    ys = torch.from_numpy(y_te.astype(np.float32)).to(device)
    D = X_tr.shape[1]
    net = nn.Sequential(
        nn.BatchNorm1d(D),
        nn.Linear(D, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1)
    ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf"); bad = 0; best_state = None
    for ep in range(1, epochs + 1):
        net.train()
        pred = net(Xt).squeeze(-1)
        loss = F.mse_loss(pred, yt)
        optim.zero_grad(); loss.backward(); optim.step()
        net.eval()
        with torch.no_grad():
            vloss = F.mse_loss(net(Xv).squeeze(-1), yv).item()
        if vloss < best_val:
            best_val = vloss; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pt = net(Xt).squeeze(-1).cpu().numpy()
        pv = net(Xv).squeeze(-1).cpu().numpy()
        ps = net(Xs).squeeze(-1).cpu().numpy()
    return {"train": metrics_reg(y_tr, pt),
            "val":   metrics_reg(y_va, pv),
            "test":  metrics_reg(y_te, ps)}

# ============================================================
# 增量写盘 helpers (每个 KG 模型产出多行, 每个 feature_set 一行)
# ============================================================
def _rows_from_result(r):
    if r is None: return []
    rows = []
    base_kw = {"kg": r["kg"], "model": r["model"]}
    km = r.get("kg_metrics", {}) or {}
    kg_cols = {f"kg_{k}": km.get(k, None) for k in ("mrr", "h1", "h3", "h10")}
    ds = r.get("downstream") or {}
    if not ds:
        rows.append({**base_kw, "feature_set": "_no_downstream_",
                     "error": r.get("error", ""), **kg_cols})
        return rows
    for set_name, content in ds.items():
        row = {**base_kw, "feature_set": set_name,
               "n_features": content.get("n_feat", None),
               **kg_cols}
        if content.get("ridge"):
            row["ridge_test_r2"]  = content["ridge"]["test"]["r2"]
            row["ridge_test_mae"] = content["ridge"]["test"]["mae"]
            row["ridge_alpha"]    = content["ridge"]["alpha"]
        if content.get("mlp"):
            row["mlp_test_r2"]  = content["mlp"]["test"]["r2"]
            row["mlp_test_mae"] = content["mlp"]["test"]["mae"]
        rows.append(row)
    return rows

def _append_to_summary(summary_csv, all_rows):
    flat = []
    for r in all_rows:
        flat.extend(_rows_from_result(r))
    if flat:
        os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
        pd.DataFrame(flat).to_csv(summary_csv, index=False, encoding="utf-8-sig")

def _save_per_model_json(out_dir, model_name, result):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"metrics_{model_name}.json")
    safe = {}
    for k, v in result.items():
        try:
            json.dumps(v, default=str); safe[k] = v
        except Exception:
            safe[k] = str(v)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [save] {path}", flush=True)

# ============================================================
# 单次实验
# ============================================================
def run_one(kg_name, model_name, args, device):
    print(f"\n========== {kg_name.upper()} | {model_name} ==========", flush=True)
    kg_dir = os.path.join(KG_ROOT, kg_name)
    if not os.path.exists(os.path.join(kg_dir, "train.tsv")):
        print(f"  [skip] {kg_dir} 没有 train.tsv", flush=True); return None

    out_dir = os.path.join(kg_dir, "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"embeddings_{model_name}.npz")

    kg = load_kg(kg_dir)

    # ----- KGE 训练 / 加载 -----
    if args.skip_existing and os.path.exists(npz):
        print(f"  [skip] {npz} 已存在, 直接加载", flush=True)
        d = np.load(npz, allow_pickle=True)
        ent_emb = d["ent_emb"]
        kg_metrics = d["metrics"].item() if "metrics" in d.files else {}
    elif args.plain_block_emb:
        ent_emb = np.random.RandomState(0).randn(kg["n_e"], args.dim).astype(np.float32) * 0.01
        kg_metrics = {"mrr": float("nan"), "_note": "plain random init for ablation"}
        np.savez_compressed(npz, ent_emb=ent_emb, metrics=kg_metrics)
    else:
        model, kg_metrics = train_kg_model(model_name, kg, args.dim, args.epochs, device)
        ent_emb = model.get_entity_embedding()
        if hasattr(ent_emb, "detach"):
            ent_emb = ent_emb.detach().cpu().numpy()
        ent_emb = np.asarray(ent_emb, dtype=np.float32)

        rel_emb = None
        if hasattr(model, "get_relation_embedding"):
            try:
                rel_emb = model.get_relation_embedding()
                if hasattr(rel_emb, "detach"):
                    rel_emb = rel_emb.detach().cpu().numpy()
                rel_emb = np.asarray(rel_emb, dtype=np.float32)
            except Exception:
                rel_emb = None

        kwargs = {"ent_emb": ent_emb, "metrics": kg_metrics}
        if rel_emb is not None: kwargs["rel_emb"] = rel_emb

        # ★★★ 给 8_contrastive_kg_image.py 用: 直接保存 (BlockID, block_emb) 对齐版 ★★★
        block_idx_now = load_block_index(kg_dir)
        if block_idx_now:
            sorted_bids = sorted(block_idx_now.keys(), key=lambda x: (len(x), x))
            sorted_eids = [block_idx_now[b] for b in sorted_bids]
            kwargs["block_id"]  = np.array(sorted_bids)
            kwargs["block_emb"] = ent_emb[np.asarray(sorted_eids, dtype=np.int64)]
        np.savez_compressed(npz, **kwargs)
        print(f"  [save] {npz}  ent_emb.shape={ent_emb.shape}", flush=True)

    if args.no_downstream:
        return {"kg": kg_name, "model": model_name, "kg_metrics": kg_metrics, "downstream": None}

    # ----- 下游 (4 组特征) -----
    df_lab, log_mean, log_std = load_labels(args.labels, id_col=args.label_id_col)
    block_idx = load_block_index(kg_dir)
    if not block_idx:
        print(f"  [warn] block_index 为空, 跳过下游", flush=True)
        return {"kg": kg_name, "model": model_name, "kg_metrics": kg_metrics, "downstream": None}
    print(f"  block_idx 大小 {len(block_idx)} (KG 中 block 实体数)", flush=True)

    block_eid_to_local, M_nbr, M_deg = build_block_neighbor_features(
        block_idx, ent_emb, kg["train"], kg["n_r"])

    handcrafted_csv = (os.path.join(kg_dir, "block_features.csv")
                       if args.include_oracle else None)
    feat_sets = make_feature_sets(df_lab, block_idx, ent_emb,
                                  M_nbr, M_deg, block_eid_to_local,
                                  handcrafted_csv=handcrafted_csv)

    idx_tr = df_lab.index[df_lab["split"] == "train"].values
    idx_va = df_lab.index[df_lab["split"] == "val"].values
    idx_te = df_lab.index[df_lab["split"] == "test"].values
    y_tr = df_lab.loc[idx_tr, "energy_log"].values
    y_va = df_lab.loc[idx_va, "energy_log"].values
    y_te = df_lab.loc[idx_te, "energy_log"].values

    ds_results = {}
    for set_name, X in feat_sets.items():
        X_tr, X_va, X_te = X[idx_tr], X[idx_va], X[idx_te]
        print(f"\n  ── feature_set='{set_name}'  X_tr={X_tr.shape}", flush=True)
        rid = fit_ridge(X_tr, y_tr, X_va, y_va, X_te, y_te)
        print(f"    Ridge α={rid['alpha']}: train R²={rid['train']['r2']:.4f}  "
              f"val R²={rid['val']['r2']:.4f}  test R²={rid['test']['r2']:.4f}", flush=True)
        mlp = fit_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te, device)
        print(f"    MLP:        train R²={mlp['train']['r2']:.4f}  "
              f"val R²={mlp['val']['r2']:.4f}  test R²={mlp['test']['r2']:.4f}", flush=True)
        ds_results[set_name] = {"ridge": rid, "mlp": mlp, "n_feat": X.shape[1]}

    return {"kg": kg_name, "model": model_name,
            "kg_metrics": kg_metrics, "downstream": ds_results}

# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kgs",    default="base,building")
    ap.add_argument("--models", default="transe,distmult,complex,rotate",
                    help=f"逗号分隔, 可选 {list(MODEL_REG.keys())}")
    ap.add_argument("--dim",    type=int, default=DEFAULT_DIM)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--labels", default=LABEL_CSV)
    ap.add_argument("--label-id-col", default=None)
    ap.add_argument("--no-downstream",   action="store_true")
    ap.add_argument("--include-oracle",  action="store_true",
                    help="加跑 oracle ablation 组 (block_emb+nbr+deg+handcrafted), "
                         "结果会显式标 'kg_oracle_handcraft', 论文里不能算 KG 模型精度")
    ap.add_argument("--plain-block-emb", action="store_true",
                    help="用随机 block emb (验证下游 R² 是否真来自 KG 而不是数据漏)")
    ap.add_argument("--skip-existing",   action="store_true")
    args = ap.parse_args()

    print(f"[7_train_kg v4.3] KG_ROOT = {KG_ROOT}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[7_train_kg v4.3] device  = {device}", flush=True)
    print(f"[7_train_kg v4.3] kgs     = {args.kgs}", flush=True)
    print(f"[7_train_kg v4.3] models  = {args.models}", flush=True)
    print(f"[7_train_kg v4.3] include_oracle = {args.include_oracle}", flush=True)

    if not args.no_downstream:
        try:
            _ = load_labels(args.labels, id_col=args.label_id_col)
        except Exception as e:
            print(f"\n[ERR] 加载 labels 失败: {e}", flush=True); sys.exit(1)

    summary_csv = os.path.join(KG_ROOT, "metrics_summary.csv")
    rows = []
    n_total = len(args.kgs.split(",")) * len(args.models.split(","))
    n_done = 0
    for kg_name in args.kgs.split(","):
        kg_name = kg_name.strip()
        for model_name in args.models.split(","):
            model_name = model_name.strip()
            n_done += 1
            print(f"\n--- 进度 {n_done}/{n_total}: {kg_name}/{model_name} ---", flush=True)
            try:
                result = run_one(kg_name, model_name, args, device)
                rows.append(result)
                if result is not None:
                    out_dir = os.path.join(KG_ROOT, kg_name, "embeddings")
                    _save_per_model_json(out_dir, model_name, result)
                _append_to_summary(summary_csv, rows)
                print(f"  [summary] 已更新 {summary_csv}", flush=True)
            except KeyboardInterrupt:
                print(f"\n[interrupt] 用户中断, 已完成的结果已写入 {summary_csv}", flush=True)
                _append_to_summary(summary_csv, rows); return
            except Exception as e:
                import traceback
                print(f"\n[ERR] {kg_name}/{model_name}: {e}", flush=True)
                traceback.print_exc()
                rows.append({"kg": kg_name, "model": model_name, "error": str(e)})
                _append_to_summary(summary_csv, rows)

    print(f"\n[summary] 全部完成, 最终结果 {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
