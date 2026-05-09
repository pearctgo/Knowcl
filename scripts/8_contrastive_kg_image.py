# -*- coding: utf-8 -*-
r"""
============================================================
8_contrastive_kg_image.py   v1.0 (2026-05-09)
  KnowCL 风格跨模态对比学习: image (SV/SI) ↔ KG block embedding
============================================================

本脚本是项目核心目的 (CLAUDE.md § 0): 把 KG 的语义注入图像表征,
对应 KnowCL 论文 § 4 的 Stage 1 (image-KG InfoNCE) + Stage 2 (下游回归).

【输入】
  - image features (per-block):
      SV: 由 2_predict_streetview.py v2.1 输出的 sv_image_features_<bb>.npz
          含逐图特征 (N_imgs, D); 我们做 mean+std 双池化得 (n_blocks, 2D)
      SI: 由 3_predict_remote_sensing.py v2.1 输出的 rs_features_block_<bb>_v2.npz
          已是 per-block (n_blocks, D)
  - KG block embedding:
      由 7_train_kg.py v4.3 输出的 embeddings_<kg_model>.npz
      (含 block_id 与 block_emb, 已按 BlockID 排序对齐)
  - labels: energy_labels.csv (含 split)

【流程】
  Stage 1 (无监督 InfoNCE):
    在 train+val 的 block 上, 把 img_feat ∈ R^{D_img} 和 kg_emb ∈ R^{D_kg}
    分别经 2-layer MLP 投影到公共空间 R^{128}, 做对称点积 InfoNCE.
    sim(I, e) = <I_proj, e_proj>      (论文 § 4.4 用点积, 不用 cosine)
    L = - log[exp(sim_ii / τ) / Σ_j exp(sim_ij / τ)]
        - log[exp(sim_ii / τ) / Σ_j exp(sim_ji / τ)]
    τ = 0.07 (KnowCL 默认), batch_size = 32 (小样本)
    监控: train loss + val loss + alignment + uniformity

  Stage 2 (有监督下游):
    fixed img_proj (Stage1 学到的) → energy_log 回归
    报 4 行 (per backbone × kg_model × {ridge, mlp}, 加 baseline 对照):
      a) baseline_raw_image     : raw image_feat 直接 ridge/mlp           ← 单模态 (= E1/E2)
      b) baseline_kg_only       : kg_emb 直接 ridge/mlp                   ← KG-only (= 7_kg_only)
      c) contrastive_image      : img_proj(image_feat) ridge/mlp          ← KG-aware image
      d) contrastive_concat     : concat(img_proj, kg_proj) ridge/mlp     ← 双模态融合

    报 R²(log) / R²(raw) / RMSE(raw) / MAE(log).

【输出】 (<KG_ROOT>/<base|building>/contrastive/)
  contrastive_<modality>_<bb>_<kg_model>.npz   : 投影头权重 + 投影后向量
  metrics_contrastive.csv                       : 每运行一次追加一组 4 行
                                                  (kg, modality, backbone, kg_model, head, ...)

【用法】
  # 街景 + KG (base, rotate)
  python 8_contrastive_kg_image.py --modality sv --backbone resnet50 \
                                   --kg base --kg-model rotate

  # 遥感 + KG (building, transe)
  python 8_contrastive_kg_image.py --modality rs --backbone densenet121 \
                                   --kg building --kg-model transe

  # 跑全套 (所有 backbone × 选定 kg_model)
  python 8_contrastive_kg_image.py --modality sv --backbone all --kg base --kg-model rotate
"""

from __future__ import annotations
import os, sys, json, argparse, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 路径
# ============================================================
ROOT      = os.environ.get("KNOWCL_ROOT", r"G:\Knowcl")
ENERGY    = os.path.join(ROOT, "999-输出成果文件", "002-能耗预测")
KG_ROOT   = os.path.join(ROOT, "999-输出成果文件", "003-知识图谱")
LABEL_CSV = os.path.join(ENERGY, "energy_labels.csv")

# ============================================================
# ID 规范化 (与 4_/5_/7_ 一致)
# ============================================================
def _norm_id(v) -> str:
    if v is None: return ""
    try:
        if pd.isna(v): return ""
    except Exception:
        pass
    if isinstance(v, (int, np.integer)): return str(int(v))
    if isinstance(v, (float, np.floating)):
        if math.isfinite(v) and float(v).is_integer(): return str(int(v))
        return str(v).strip()
    s = str(v).strip()
    try:
        f = float(s)
        if math.isfinite(f) and f.is_integer(): return str(int(f))
    except Exception:
        pass
    return s

def _norm_id_array(a):
    return np.array([_norm_id(x) for x in a])

# ============================================================
# 加载 image features (per-block, BlockID 对齐)
# ============================================================
def load_image_features(modality, backbone, label_df):
    """
    返回:
        bids_img : np.array of str, BlockID, 长度 n_blocks_img
        feats    : np.array (n_blocks_img, D)  per-block 池化后的特征
    与 label_df 通过 BlockID inner join 对齐留给上层.
    """
    if modality == "sv":
        path = os.path.join(ENERGY, f"sv_image_features_{backbone}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到 SV 特征文件: {path}\n"
                f"请先跑 2_predict_streetview.py --backbone {backbone}")
        d = np.load(path, allow_pickle=True)
        # SV 特征通常是 per-image. 兼容多种 schema:
        if "block_id" in d.files and "feats" in d.files:
            feats_per_img = d["feats"]                                 # (N_imgs, D)
            bids_per_img  = _norm_id_array(d["block_id"])              # (N_imgs,)
        elif "block_ids" in d.files and "feats" in d.files:
            feats_per_img = d["feats"]
            bids_per_img  = _norm_id_array(d["block_ids"])
        elif "feats" in d.files and "ids" in d.files:
            feats_per_img = d["feats"]
            bids_per_img  = _norm_id_array(d["ids"])
        else:
            # 已是 per-block (no aggregation needed)
            if "block_emb" in d.files:
                bids = _norm_id_array(d["block_id"]) if "block_id" in d.files else np.arange(d["block_emb"].shape[0]).astype(str)
                return bids, d["block_emb"].astype(np.float32)
            raise RuntimeError(f"无法识别 sv npz schema, files={d.files}")
        # mean + std 双池化
        unique_bids = sorted(set(bids_per_img.tolist()))
        D = feats_per_img.shape[1]
        out = np.zeros((len(unique_bids), 2 * D), dtype=np.float32)
        for i, b in enumerate(unique_bids):
            mask = bids_per_img == b
            if mask.sum() == 0: continue
            seg = feats_per_img[mask]
            mean = seg.mean(axis=0)
            std  = seg.std(axis=0) if mask.sum() > 1 else np.zeros_like(mean)
            out[i] = np.concatenate([mean, std])
        return np.array(unique_bids), out

    elif modality == "rs":
        path = os.path.join(ENERGY, f"rs_features_block_{backbone}_v2.npz")
        if not os.path.exists(path):
            # 兼容老命名
            alt = os.path.join(ENERGY, f"rs_features_block_{backbone}.npz")
            if os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(
                    f"找不到 SI 特征文件: {path}\n"
                    f"请先跑 3_predict_remote_sensing.py --backbone {backbone}")
        d = np.load(path, allow_pickle=True)
        # RS 已是 per-block
        feats_key = "feats" if "feats" in d.files else ("features" if "features" in d.files else None)
        bid_key   = "block_id" if "block_id" in d.files else ("block_ids" if "block_ids" in d.files else "ids")
        if feats_key is None or bid_key not in d.files:
            raise RuntimeError(f"无法识别 rs npz schema, files={d.files}")
        return _norm_id_array(d[bid_key]), d[feats_key].astype(np.float32)
    else:
        raise ValueError(f"未知 modality: {modality}")

# ============================================================
# 加载 KG block embedding
# ============================================================
def load_kg_block_emb(kg_name, kg_model_name):
    """
    返回 (bids_kg, kg_emb)
    优先读 7_train_kg v4.3 写入的 block_id / block_emb 两个 key.
    若没有 (旧版 npz), 回退用 ent_emb + block_index.tsv 还原.
    """
    npz = os.path.join(KG_ROOT, kg_name, "embeddings", f"embeddings_{kg_model_name}.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(
            f"找不到 KG embedding: {npz}\n"
            f"请先跑 python 7_train_kg.py --kgs {kg_name} --models {kg_model_name}")
    d = np.load(npz, allow_pickle=True)
    if "block_id" in d.files and "block_emb" in d.files:
        return _norm_id_array(d["block_id"]), d["block_emb"].astype(np.float32)
    # 回退
    print("  [warn] npz 没有 block_id / block_emb, 从 ent_emb + block_index.tsv 还原", flush=True)
    ent_emb = d["ent_emb"]
    block_idx = {}
    bi_path = os.path.join(KG_ROOT, kg_name, "block_index.tsv")
    if os.path.exists(bi_path):
        with open(bi_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    a, b = parts[0], parts[1]
                    try:
                        eid = int(b); block_idx[_norm_id(a)] = eid
                    except ValueError:
                        try:
                            eid = int(a); block_idx[_norm_id(b)] = eid
                        except ValueError:
                            continue
    if not block_idx:
        raise RuntimeError("block_index.tsv 也找不到, 无法还原 block_emb")
    bids = sorted(block_idx.keys(), key=lambda x: (len(x), x))
    eids = [block_idx[b] for b in bids]
    return np.array(bids), ent_emb[np.asarray(eids, dtype=np.int64)].astype(np.float32)

# ============================================================
# labels
# ============================================================
def load_labels(label_csv):
    df = pd.read_csv(label_csv)
    id_col = None
    for c in ("BlockID", "block_id", "blockid", "id"):
        if c in df.columns: id_col = c; break
    if id_col is None:
        raise RuntimeError(f"labels.csv 没有 BlockID 列, 字段 {list(df.columns)}")
    if id_col != "BlockID":
        df = df.rename(columns={id_col: "BlockID"})
    df["BlockID"] = df["BlockID"].map(_norm_id)

    energy_col = None
    for c in ("energy", "energy_log", "log_energy"):
        if c in df.columns: energy_col = c; break
    if energy_col is None:
        raise RuntimeError(f"labels.csv 没有 energy 列, 字段 {list(df.columns)}")
    if energy_col != "energy": df = df.rename(columns={energy_col: "energy"})
    is_log = energy_col != "energy"
    if not is_log:
        e = df["energy"].astype(float).clip(lower=1e-6)
        df["energy_log"] = np.log(e) if e.max() / max(e.min(), 1e-6) > 50 else e
    else:
        df["energy_log"] = df["energy"].astype(float)
    return df

# ============================================================
# 投影头
# ============================================================
class ProjHead(nn.Module):
    """KnowCL 论文 § 4.3.2 投影头: 2-layer MLP + ReLU"""
    def __init__(self, in_d, hidden=256, out_d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden), nn.ReLU(),
            nn.Linear(hidden, out_d)
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# Stage 1: InfoNCE 对比预训练
# ============================================================
def train_infonce(img_train, kg_train, img_val, kg_val, device,
                  d_proj=128, hidden=256, batch=32, epochs=200,
                  lr=3e-4, tau=0.07, patience=30, seed=42):
    """
    返回训练好的 (img_proj, kg_proj) + 训练 log
    """
    torch.manual_seed(seed); np.random.seed(seed)
    img_train_t = torch.from_numpy(img_train).to(device)
    kg_train_t  = torch.from_numpy(kg_train ).to(device)
    img_val_t   = torch.from_numpy(img_val  ).to(device)
    kg_val_t    = torch.from_numpy(kg_val   ).to(device)

    img_proj = ProjHead(img_train.shape[1], hidden=hidden, out_d=d_proj).to(device)
    kg_proj  = ProjHead(kg_train.shape[1],  hidden=hidden, out_d=d_proj).to(device)

    optim = torch.optim.Adam(
        list(img_proj.parameters()) + list(kg_proj.parameters()),
        lr=lr, weight_decay=1e-5)

    n = len(img_train_t)
    best_val = float("inf"); bad = 0; best_state = None
    history = []
    for ep in range(1, epochs + 1):
        img_proj.train(); kg_proj.train()
        perm = torch.randperm(n, device=device)
        ep_losses = []
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            if len(idx) < 4: continue   # 太小的 batch 跳过, InfoNCE 无意义
            a = img_proj(img_train_t[idx])         # (B, d_proj)
            b = kg_proj(kg_train_t[idx])
            # 对称点积 InfoNCE (KnowCL §4.4)
            logits = (a @ b.t()) / tau              # (B, B)
            labels = torch.arange(len(idx), device=device)
            loss = (F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.t(), labels)) / 2.0
            optim.zero_grad(); loss.backward(); optim.step()
            ep_losses.append(loss.item())
        ep_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")

        # val
        img_proj.eval(); kg_proj.eval()
        with torch.no_grad():
            a_v = img_proj(img_val_t); b_v = kg_proj(kg_val_t)
            logits_v = (a_v @ b_v.t()) / tau
            n_v = len(img_val_t)
            if n_v >= 4:
                labels_v = torch.arange(n_v, device=device)
                vloss = ((F.cross_entropy(logits_v, labels_v) +
                          F.cross_entropy(logits_v.t(), labels_v)) / 2.0).item()
            else:
                vloss = ep_loss
            # alignment & uniformity
            a_n = F.normalize(a_v, dim=1); b_n = F.normalize(b_v, dim=1)
            align = ((a_n - b_n) ** 2).sum(1).mean().item()
            unif  = (torch.cdist(a_n, a_n) ** 2).mul(-2).exp().mean().log().item() if n_v > 1 else 0.0

        history.append({"ep": ep, "train_loss": ep_loss, "val_loss": vloss,
                        "align": align, "uniformity": unif})
        if ep % 10 == 0 or ep == 1:
            print(f"    ep{ep:3d}  train={ep_loss:.4f}  val={vloss:.4f}  "
                  f"align={align:.4f}  unif={unif:.4f}", flush=True)
        if vloss < best_val - 1e-4:
            best_val = vloss; bad = 0
            best_state = {
                "img": {k: v.detach().cpu().clone() for k, v in img_proj.state_dict().items()},
                "kg":  {k: v.detach().cpu().clone() for k, v in kg_proj.state_dict().items()},
            }
        else:
            bad += 1
            if bad >= patience:
                print(f"    early stop at ep{ep}", flush=True); break

    if best_state is not None:
        img_proj.load_state_dict(best_state["img"])
        kg_proj.load_state_dict(best_state["kg"])
    img_proj.eval(); kg_proj.eval()
    return img_proj, kg_proj, history, best_val

# ============================================================
# Stage 2: 下游回归 (ridge + mlp)
# ============================================================
def metrics_reg(y_true, y_pred):
    e = y_true - y_pred
    sst = ((y_true - y_true.mean()) ** 2).sum()
    sse = (e ** 2).sum()
    return {"r2": float(1.0 - sse / max(sst, 1e-12)),
            "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean()))}

def fit_ridge(X_tr, y_tr, X_va, y_va, X_te, y_te):
    from sklearn.linear_model import Ridge
    best = None
    for alpha in [0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        m = Ridge(alpha=alpha); m.fit(X_tr, y_tr)
        v = metrics_reg(y_va, m.predict(X_va))
        if best is None or v["r2"] > best["val_r2"]:
            best = {"val_r2": v["r2"], "alpha": alpha, "model": m}
    m = best["model"]
    return {"alpha": best["alpha"],
            "test": metrics_reg(y_te, m.predict(X_te))}

def fit_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te, device,
            hidden=128, epochs=300, lr=5e-4, dropout=0.5,
            patience=40, weight_decay=1e-3):
    Xt = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xv = torch.from_numpy(X_va.astype(np.float32)).to(device)
    yv = torch.from_numpy(y_va.astype(np.float32)).to(device)
    Xs = torch.from_numpy(X_te.astype(np.float32)).to(device)
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
        loss = F.mse_loss(net(Xt).squeeze(-1), yt)
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
    if best_state is not None: net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        ps = net(Xs).squeeze(-1).cpu().numpy()
    return {"test": metrics_reg(y_te, ps)}

# ============================================================
# 主流程: 单 (modality, backbone, kg_model) 实验
# ============================================================
def run_one(modality, backbone, kg_name, kg_model_name, args, device):
    print(f"\n========== [contrastive] {modality.upper()} | {backbone} | {kg_name}/{kg_model_name} ==========", flush=True)

    # 1) 加载所有数据
    df_lab = load_labels(args.labels)
    bids_img, img_feat = load_image_features(modality, backbone, df_lab)
    bids_kg,  kg_emb   = load_kg_block_emb(kg_name, kg_model_name)
    print(f"  image: n_blocks={len(bids_img)}, dim={img_feat.shape[1]}", flush=True)
    print(f"  kg   : n_blocks={len(bids_kg)},  dim={kg_emb.shape[1]}", flush=True)

    # 2) 三方 inner join: labels ∩ image ∩ kg
    set_lab = set(df_lab["BlockID"].astype(str))
    set_img = set(bids_img.tolist())
    set_kg  = set(bids_kg.tolist())
    common  = sorted(set_lab & set_img & set_kg)
    print(f"  三方交集 BlockID: {len(common)}  "
          f"(label={len(set_lab)}, img={len(set_img)}, kg={len(set_kg)})", flush=True)
    if len(common) < 100:
        raise RuntimeError(f"三方交集只有 {len(common)} 块, 太少, 检查 BlockID 是否对齐")

    img_id_to_row = {b: i for i, b in enumerate(bids_img.tolist())}
    kg_id_to_row  = {b: i for i, b in enumerate(bids_kg.tolist())}

    sub = df_lab[df_lab["BlockID"].isin(common)].drop_duplicates("BlockID").copy()
    sub = sub.sort_values("BlockID").reset_index(drop=True)
    img_aligned = np.stack([img_feat[img_id_to_row[b]] for b in sub["BlockID"]])
    kg_aligned  = np.stack([kg_emb[ kg_id_to_row[ b]] for b in sub["BlockID"]])
    y           = sub["energy_log"].values.astype(np.float32)

    # 3) 标准化 (Stage1 InfoNCE 对幅度敏感)
    def _zsc(M):
        mu = M.mean(0, keepdims=True); sd = M.std(0, keepdims=True) + 1e-6
        return ((M - mu) / sd).astype(np.float32)
    img_aligned = _zsc(img_aligned)
    kg_aligned  = _zsc(kg_aligned)

    # 4) split
    tr_mask = (sub["split"] == "train").values
    va_mask = (sub["split"] == "val").values
    te_mask = (sub["split"] == "test").values
    print(f"  split: train={tr_mask.sum()}  val={va_mask.sum()}  test={te_mask.sum()}", flush=True)

    img_tr_va = img_aligned[tr_mask | va_mask]
    kg_tr_va  = kg_aligned[ tr_mask | va_mask]
    # 在 train+val 内部再切 80/20 给 InfoNCE 自己做早停
    n_tv = len(img_tr_va)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_tv)
    n_v_inner = max(8, n_tv // 5)
    val_idx = perm[:n_v_inner]
    tr_idx  = perm[n_v_inner:]
    img_train_in = img_tr_va[tr_idx]; kg_train_in = kg_tr_va[tr_idx]
    img_val_in   = img_tr_va[val_idx]; kg_val_in  = kg_tr_va[val_idx]

    # 5) Stage 1: InfoNCE
    print(f"\n  ── Stage 1: InfoNCE 对比预训练 (n_train_inner={len(tr_idx)}, n_val_inner={n_v_inner})", flush=True)
    img_proj, kg_proj, hist, best_val = train_infonce(
        img_train_in, kg_train_in, img_val_in, kg_val_in, device,
        d_proj=args.d_proj, hidden=args.proj_hidden, batch=args.batch,
        epochs=args.epochs, lr=args.lr, tau=args.tau, patience=args.patience)

    # 6) Stage 2: 下游
    print(f"\n  ── Stage 2: 下游能耗回归", flush=True)
    img_t = torch.from_numpy(img_aligned).to(device)
    kg_t  = torch.from_numpy(kg_aligned ).to(device)
    with torch.no_grad():
        img_proj_arr = img_proj(img_t).cpu().numpy()
        kg_proj_arr  = kg_proj( kg_t ).cpu().numpy()

    y_tr = y[tr_mask]; y_va = y[va_mask]; y_te = y[te_mask]

    feature_sets = {
        "baseline_raw_image": img_aligned,
        "baseline_kg_only":   kg_aligned,
        "contrastive_image":  img_proj_arr,
        "contrastive_concat": np.concatenate([img_proj_arr, kg_proj_arr], axis=1),
    }

    rows = []
    for name, X in feature_sets.items():
        Xtr, Xva, Xte = X[tr_mask], X[va_mask], X[te_mask]
        rid = fit_ridge(Xtr, y_tr, Xva, y_va, Xte, y_te)
        mlp = fit_mlp(  Xtr, y_tr, Xva, y_va, Xte, y_te, device)
        print(f"    {name:25s}  ridge α={rid['alpha']:>6}  R²={rid['test']['r2']:.4f}  "
              f"|  mlp R²={mlp['test']['r2']:.4f}", flush=True)
        rows.append({
            "kg": kg_name, "modality": modality, "backbone": backbone,
            "kg_model": kg_model_name, "head": name,
            "n_features": X.shape[1],
            "ridge_test_r2":  rid["test"]["r2"],
            "ridge_test_mae": rid["test"]["mae"],
            "ridge_alpha":    rid["alpha"],
            "mlp_test_r2":    mlp["test"]["r2"],
            "mlp_test_mae":   mlp["test"]["mae"],
            "infonce_best_val": best_val,
        })

    # 7) 保存
    out_dir = os.path.join(KG_ROOT, kg_name, "contrastive")
    os.makedirs(out_dir, exist_ok=True)
    npz_out = os.path.join(out_dir, f"contrastive_{modality}_{backbone}_{kg_model_name}.npz")
    np.savez_compressed(
        npz_out,
        block_id=np.array(common),
        img_aligned=img_aligned, kg_aligned=kg_aligned,
        img_proj=img_proj_arr, kg_proj=kg_proj_arr,
        infonce_history=np.array(hist, dtype=object),
    )
    print(f"  [save] {npz_out}", flush=True)
    return rows

# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["sv", "rs"], required=True)
    ap.add_argument("--backbone", default="resnet50",
                    help="单个 backbone 名, 或 'all' 跑全部 7 个")
    ap.add_argument("--kg",       default="base",  choices=["base", "building"])
    ap.add_argument("--kg-model", default="rotate",
                    help="单个 KG 模型名, 或 'all' 跑全部")
    ap.add_argument("--labels",   default=LABEL_CSV)
    ap.add_argument("--d-proj",   type=int,   default=128)
    ap.add_argument("--proj-hidden", type=int, default=256)
    ap.add_argument("--batch",    type=int,   default=32)
    ap.add_argument("--epochs",   type=int,   default=200)
    ap.add_argument("--lr",       type=float, default=3e-4)
    ap.add_argument("--tau",      type=float, default=0.07)
    ap.add_argument("--patience", type=int,   default=30)
    args = ap.parse_args()

    print(f"[8_contrastive v1.0] KG_ROOT  = {KG_ROOT}", flush=True)
    print(f"[8_contrastive v1.0] modality = {args.modality}", flush=True)
    print(f"[8_contrastive v1.0] backbone = {args.backbone}", flush=True)
    print(f"[8_contrastive v1.0] kg/model = {args.kg}/{args.kg_model}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbones = (["resnet50", "densenet121", "convnext_tiny", "vit_b_16",
                  "mobilenet_v3_large", "attention_cnn", "efficientnet_b0"]
                 if args.backbone == "all" else [args.backbone])
    kg_models = (["transe", "distmult", "complex", "rotate"]
                 if args.kg_model == "all" else [args.kg_model])

    out_csv = os.path.join(KG_ROOT, "metrics_contrastive.csv")
    all_rows = []
    if os.path.exists(out_csv):
        try:
            old = pd.read_csv(out_csv)
            all_rows = old.to_dict("records")
        except Exception:
            pass

    for bb in backbones:
        for km in kg_models:
            try:
                rows = run_one(args.modality, bb, args.kg, km, args, device)
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
                print(f"  [summary] 已更新 {out_csv} ({len(all_rows)} 行)", flush=True)
            except KeyboardInterrupt:
                pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
                print(f"\n[interrupt] 已保存当前结果到 {out_csv}", flush=True); return
            except Exception as e:
                import traceback
                print(f"\n[ERR] {bb}/{km}: {e}", flush=True); traceback.print_exc()

    print(f"\n[done] 全部完成, 最终结果 {out_csv}", flush=True)


if __name__ == "__main__":
    main()
