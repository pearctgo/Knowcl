# -*- coding: utf-8 -*-
"""
==============================================================
 2_predict_streetview.py  —  街景预测街区能耗 (v2.1 bugfix)
==============================================================
 v2.1 vs v2 改动 (仅修两处 bug, 其余完全保留):
   1. fit_ridge 返回全样本预测 (而非仅 test 集), 修复保存阶段
      "All arrays must be of the same length" 异常
   2. main 末尾防御性处理 all_metrics 为空的边界 (KeyError: 'R2_log')

 由于 7 backbone 的 sv_image_features_<bb>.npz 缓存已存在,
 重跑会跳过 GPU 提取阶段, 全程不到 5 分钟.
==============================================================
"""
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ============== 路径配置 ==============
SV_INDEX_CSV = r"G:\Knowcl\999-输出成果文件\001-街景重采_baidu\tables\streetview_index.csv"
OUT_DIR      = r"G:\Knowcl\999-输出成果文件\002-能耗预测"
LABELS_CSV   = os.path.join(OUT_DIR, "energy_labels.csv")
STATS_JSON   = os.path.join(OUT_DIR, "label_stats.json")
BASELINE_CSV = os.path.join(OUT_DIR, "sv_baseline_metrics.csv")
SUMMARY_CSV  = os.path.join(OUT_DIR, "sv_metrics_summary.csv")
ALL_PRED_CSV = os.path.join(OUT_DIR, "sv_all_models.csv")

# ============== 设备 / 提取 ==============
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACT_BATCH  = 64
NUM_WORKERS    = 4

# ============== MLP 训练超参 ==============
DROPOUT        = 0.5
LR             = 5e-4
WD             = 1e-3
EPOCHS         = 200
BATCH_SIZE     = 64
EARLY_STOP_PAT = 30
SEED           = 42

RIDGE_ALPHAS   = [0.1, 1.0, 10.0, 100.0, 1000.0]

BACKBONES = [
    "resnet50",
    "densenet121",
    "convnext_tiny",
    "vit_b_16",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "attention_cnn",
]


# --------------------------------------------------------------
# 1. CBAM 与 backbone 工厂
# --------------------------------------------------------------
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sig_c = nn.Sigmoid()
        pad = (kernel_size - 1) // 2
        self.conv_s = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.sig_s = nn.Sigmoid()

    def forward(self, x):
        ca = self.sig_c(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        x = x * ca
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sig_s(self.conv_s(torch.cat([avg, mx], dim=1)))
        return x * sa


def _build_attention_cnn():
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    base.layer4 = nn.Sequential(base.layer4, CBAM(2048))
    base.fc = nn.Identity()
    return base


def build_backbone(name: str):
    name = name.lower()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity(); dim = 2048
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity(); dim = 1024
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Identity(); dim = 768
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity(); dim = 768
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m.classifier = nn.Identity(); dim = 960
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity(); dim = 1280
    elif name == "attention_cnn":
        m = _build_attention_cnn(); dim = 2048
    else:
        raise ValueError(f"Unknown backbone: {name}")
    m.eval().to(DEVICE)
    return m, dim, transform


# --------------------------------------------------------------
# 2. 数据集与每图特征提取
# --------------------------------------------------------------
class SVDataset(Dataset):
    def __init__(self, df, transform):
        self.df, self.tf = df.reset_index(drop=True), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.df.iloc[idx]["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.tf(img), idx


def load_sv_index() -> pd.DataFrame:
    df = pd.read_csv(SV_INDEX_CSV, encoding="utf-8")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    labels = pd.read_csv(LABELS_CSV)
    valid_bids = set(labels["block_id"].astype(int))
    df = df[df["BlockID"].astype(int).isin(valid_bids)].reset_index(drop=True)
    return df


def extract_image_features(backbone_name: str, force=False):
    cache = os.path.join(OUT_DIR, f"sv_image_features_{backbone_name}.npz")
    df = load_sv_index()

    if os.path.exists(cache) and not force:
        print(f"  [{backbone_name}] 读取图层特征缓存: {cache}")
        d = np.load(cache, allow_pickle=False)
        return d["features"], d["bid_arr"], df

    print(f"  [{backbone_name}] 开始提取 {len(df)} 张图像特征...")
    model, dim, tf = build_backbone(backbone_name)
    loader = DataLoader(SVDataset(df, tf), batch_size=EXTRACT_BATCH,
                        shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=(DEVICE == "cuda"))

    feats = np.zeros((len(df), dim), dtype=np.float32)
    with torch.no_grad():
        for imgs, idxs in tqdm(loader, desc=f"{backbone_name}"):
            f = model(imgs.to(DEVICE, non_blocking=True))
            if f.dim() > 2:
                f = f.flatten(1)
            feats[idxs.numpy()] = f.cpu().numpy()

    bid_arr = df["BlockID"].astype(int).values
    np.savez_compressed(cache, features=feats, bid_arr=bid_arr)
    print(f"  [{backbone_name}] 缓存已写: {cache}, shape={feats.shape}")
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return feats, bid_arr, df


# --------------------------------------------------------------
# 3. 街区级聚合 (多池化 + 辅助统计)
# --------------------------------------------------------------
def pool_block_features(img_feats, bid_arr, block_ids, pool="mean_std"):
    dim = img_feats.shape[1]
    if pool == "mean":
        out = np.zeros((len(block_ids), dim), dtype=np.float32)
    elif pool == "mean_std":
        out = np.zeros((len(block_ids), dim * 2), dtype=np.float32)
    elif pool == "mean_max_std":
        out = np.zeros((len(block_ids), dim * 3), dtype=np.float32)
    else:
        raise ValueError(f"Unknown pool: {pool}")

    for i, bid in enumerate(block_ids):
        chunk = img_feats[bid_arr == int(bid)]
        if len(chunk) == 0:
            continue
        mu = chunk.mean(axis=0).astype(np.float32)
        if pool == "mean":
            out[i] = mu
        elif pool == "mean_std":
            sd = chunk.std(axis=0).astype(np.float32) if len(chunk) > 1 \
                else np.zeros(dim, dtype=np.float32)
            out[i] = np.concatenate([mu, sd])
        elif pool == "mean_max_std":
            mx = chunk.max(axis=0).astype(np.float32)
            sd = chunk.std(axis=0).astype(np.float32) if len(chunk) > 1 \
                else np.zeros(dim, dtype=np.float32)
            out[i] = np.concatenate([mu, mx, sd])
    return out


def build_aux_features(sv_df: pd.DataFrame, block_ids):
    has_lon = "lon" in sv_df.columns
    has_lat = "lat" in sv_df.columns
    n_aux = 1 + int(has_lon) + int(has_lat)
    aux = np.zeros((len(block_ids), n_aux), dtype=np.float32)
    grp = sv_df.groupby("BlockID")
    for i, bid in enumerate(block_ids):
        bid = int(bid)
        if bid not in grp.groups:
            continue
        g = grp.get_group(bid)
        feats = [float(np.log1p(len(g)))]
        if has_lon:
            feats.append(float(g["lon"].std()) if len(g) > 1 else 0.0)
        if has_lat:
            feats.append(float(g["lat"].std()) if len(g) > 1 else 0.0)
        aux[i] = feats
    aux_names = ["log_n_imgs"]
    if has_lon: aux_names.append("lon_std")
    if has_lat: aux_names.append("lat_std")
    return aux, aux_names


# --------------------------------------------------------------
# 4. L0 baseline self-check
# --------------------------------------------------------------
def baseline_metrics_row(name, y_te_norm, p_te_norm, y_te_raw, p_te_raw):
    return {
        "model":     name,
        "R2_norm":   r2_score(y_te_norm, p_te_norm),
        "R2_raw":    r2_score(y_te_raw, p_te_raw),
        "R2_log":    r2_score(np.log1p(np.clip(y_te_raw, 0, None)),
                              np.log1p(np.clip(p_te_raw, 0, None))),
        "RMSE_raw":  np.sqrt(mean_squared_error(y_te_raw, p_te_raw)),
        "MAE_raw":   mean_absolute_error(y_te_raw, p_te_raw),
    }


def to_raw(pred_norm, mu, sigma):
    return np.expm1(pred_norm * sigma + mu)


def run_baselines(merged, aux_feats, mu, sigma):
    splits = merged["split"].values
    y_norm = merged["energy_norm"].values.astype(np.float32)
    y_raw  = merged["energy_raw"].values.astype(np.float32)

    tr, va, te = splits == "train", splits == "val", splits == "test"
    rows = []

    mean_norm = float(y_norm[tr].mean())
    p_norm = np.full(te.sum(), mean_norm)
    p_raw  = to_raw(p_norm, mu, sigma)
    rows.append(baseline_metrics_row("mean_predictor",
                                     y_norm[te], p_norm, y_raw[te], p_raw))

    Xtr, Xva, Xte = aux_feats[tr], aux_feats[va], aux_feats[te]
    ytr_norm, yva_norm = y_norm[tr], y_norm[va]
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best_alpha, best_rmse = None, float("inf")
    for a in RIDGE_ALPHAS:
        rg = Ridge(alpha=a, random_state=SEED).fit(Xtr_s, ytr_norm)
        rmse_va = np.sqrt(mean_squared_error(yva_norm, rg.predict(Xva_s)))
        if rmse_va < best_rmse:
            best_rmse, best_alpha = rmse_va, a
    rg = Ridge(alpha=best_alpha, random_state=SEED).fit(Xtr_s, ytr_norm)
    p_norm = rg.predict(Xte_s)
    p_raw  = to_raw(p_norm, mu, sigma)
    rows.append(baseline_metrics_row(
        f"ridge_aux_only(alpha={best_alpha})",
        y_norm[te], p_norm, y_raw[te], p_raw))

    df = pd.DataFrame(rows)
    df.to_csv(BASELINE_CSV, index=False, encoding="utf-8-sig")
    print("\n[L0 baseline] (lower R² 越接近 0 / 负, 说明该 baseline 越差)")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  → 写入 {BASELINE_CSV}\n")
    return df


# --------------------------------------------------------------
# 5. MLP 与训练
# --------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_va, y_va, in_dim):
    torch.manual_seed(SEED)
    model = MLP(in_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.SmoothL1Loss()

    tr_ld = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(),
                                     torch.from_numpy(y_tr).float()),
                       batch_size=BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(TensorDataset(torch.from_numpy(X_va).float(),
                                     torch.from_numpy(y_va).float()),
                       batch_size=BATCH_SIZE)

    best, best_state, patience = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        ps, gs = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                ps.append(model(xb.to(DEVICE)).cpu().numpy())
                gs.append(yb.numpy())
        val_rmse = float(np.sqrt(np.mean((np.concatenate(ps) -
                                          np.concatenate(gs)) ** 2)))
        if val_rmse < best - 1e-5:
            best, best_state, patience = val_rmse, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if ep % 20 == 0 or ep == EPOCHS - 1:
            print(f"        ep {ep:3d}: val_rmse={val_rmse:.4f}  best={best:.4f}")
        if patience >= EARLY_STOP_PAT:
            print(f"        early stop @ ep {ep}, best val_rmse={best:.4f}")
            break
    model.load_state_dict(best_state)
    return model, best


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).float().to(DEVICE)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


# === v2.1 BUGFIX: 返回全样本预测, 不只是 test ===
def fit_ridge(X_tr, y_tr, X_va, y_va, X_all):
    """Ridge baseline. 在 train 上 fit, val 上调 alpha, X_all 上 predict.

    返回 (best_alpha, val_rmse, pred_all_norm)
       pred_all_norm 长度 = len(X_all), 与 MLP 预测对齐, 便于一并保存.
    """
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_all_s = sc.transform(X_all)
    best_alpha, best_rmse, best_model = None, float("inf"), None
    for a in RIDGE_ALPHAS:
        rg = Ridge(alpha=a, random_state=SEED).fit(X_tr_s, y_tr)
        rmse = np.sqrt(mean_squared_error(y_va, rg.predict(X_va_s)))
        if rmse < best_rmse:
            best_rmse, best_alpha, best_model = rmse, a, rg
    return best_alpha, best_rmse, best_model.predict(X_all_s)


# --------------------------------------------------------------
# 6. 单 backbone 完整流程
# --------------------------------------------------------------
def run_one_backbone(bb_name: str, sv_df: pd.DataFrame, labels: pd.DataFrame,
                     mu: float, sigma: float, pool: str, force_extract: bool):
    print(f"\n{'='*66}\n 街景 backbone: {bb_name}\n{'='*66}")

    img_feats, bid_arr, sv_df_ok = extract_image_features(
        bb_name, force=force_extract)

    block_ids = sorted(set(int(b) for b in bid_arr).intersection(
        set(int(b) for b in labels["block_id"])))
    df_block = pd.DataFrame({"block_id": block_ids})
    merged = labels.merge(df_block, on="block_id", how="inner") \
                   .reset_index(drop=True)

    pooled = pool_block_features(img_feats, bid_arr,
                                 merged["block_id"].values, pool=pool)
    aux, aux_names = build_aux_features(sv_df_ok, merged["block_id"].values)
    X = np.concatenate([pooled, aux], axis=1)
    y = merged["energy_norm"].values.astype(np.float32)
    splits = merged["split"].values
    tr, va, te = splits == "train", splits == "val", splits == "test"

    print(f"  [{bb_name}] 池化={pool}, 总维度={X.shape[1]} "
          f"(图特征 {pooled.shape[1]} + 辅助 {len(aux_names)})")
    print(f"  [{bb_name}] 对齐样本: {len(X)} "
          f"(train={tr.sum()} val={va.sum()} test={te.sum()})")

    # === v2.1: 全样本 Ridge 预测, 与 MLP 对齐 ===
    best_a, ridge_val_rmse, ridge_pred_norm_all = fit_ridge(
        X[tr], y[tr], X[va], y[va], X)

    print(f"  [{bb_name}] 训练 MLP (in_dim={X.shape[1]})...")
    mlp, val_rmse = train_mlp(X[tr], y[tr], X[va], y[va], X.shape[1])

    pred_norm_all = predict(mlp, X)

    y_raw = merged["energy_raw"].values
    pred_raw_all       = to_raw(pred_norm_all, mu, sigma)
    ridge_pred_raw_all = to_raw(ridge_pred_norm_all, mu, sigma)

    y_te_norm, y_te_raw = y[te], y_raw[te]
    p_te_raw_mlp   = pred_raw_all[te]
    p_te_raw_ridge = ridge_pred_raw_all[te]
    p_te_norm_mlp   = pred_norm_all[te]
    p_te_norm_ridge = ridge_pred_norm_all[te]

    metrics_mlp = {
        "backbone":  bb_name,
        "model":     "mlp",
        "feat_dim":  X.shape[1],
        "val_rmse":  val_rmse,
        "R2_norm":   r2_score(y_te_norm, p_te_norm_mlp),
        "R2_raw":    r2_score(y_te_raw, p_te_raw_mlp),
        "R2_log":    r2_score(np.log1p(y_te_raw),
                              np.log1p(np.clip(p_te_raw_mlp, 0, None))),
        "RMSE_raw":  np.sqrt(mean_squared_error(y_te_raw, p_te_raw_mlp)),
        "MAE_raw":   mean_absolute_error(y_te_raw, p_te_raw_mlp),
    }
    metrics_ridge = {
        "backbone":  bb_name,
        "model":     f"ridge(a={best_a})",
        "feat_dim":  X.shape[1],
        "val_rmse":  ridge_val_rmse,
        "R2_norm":   r2_score(y_te_norm, p_te_norm_ridge),
        "R2_raw":    r2_score(y_te_raw, p_te_raw_ridge),
        "R2_log":    r2_score(np.log1p(y_te_raw),
                              np.log1p(np.clip(p_te_raw_ridge, 0, None))),
        "RMSE_raw":  np.sqrt(mean_squared_error(y_te_raw, p_te_raw_ridge)),
        "MAE_raw":   mean_absolute_error(y_te_raw, p_te_raw_ridge),
    }
    print(f"  ► MLP   R²(log)={metrics_mlp['R2_log']:.4f}  "
          f"R²(raw)={metrics_mlp['R2_raw']:.4f}  RMSE={metrics_mlp['RMSE_raw']:.2f}")
    print(f"  ► Ridge R²(log)={metrics_ridge['R2_log']:.4f}  "
          f"R²(raw)={metrics_ridge['R2_raw']:.4f}  RMSE={metrics_ridge['RMSE_raw']:.2f}")

    # 保存 (v2.1: 现在所有数组长度都是 len(X), 不再不一致)
    pred_csv = os.path.join(OUT_DIR, f"sv_predictions_{bb_name}.csv")
    pd.DataFrame({
        "block_id":           merged["block_id"].values,
        "split":              splits,
        "true_energy":        y_raw,
        "mlp_pred_energy":    pred_raw_all,
        "ridge_pred_energy":  ridge_pred_raw_all,
    }).to_csv(pred_csv, index=False, encoding="utf-8-sig")
    torch.save(mlp.state_dict(), os.path.join(OUT_DIR, f"sv_mlp_{bb_name}.pt"))

    del mlp
    if DEVICE == "cuda": torch.cuda.empty_cache()

    return [metrics_mlp, metrics_ridge], merged["block_id"].values, pred_raw_all


# --------------------------------------------------------------
# 7. main
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=BACKBONES,
                        choices=BACKBONES + ["all"],
                        help="待跑 backbone 列表")
    parser.add_argument("--pool", default="mean_std",
                        choices=["mean", "mean_std", "mean_max_std"],
                        help="街区级池化策略")
    parser.add_argument("--force-extract", action="store_true",
                        help="强制重提单图特征 (清缓存)")
    args = parser.parse_args()

    if "all" in args.backbones:
        args.backbones = BACKBONES

    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError("先运行 1_build_labels.py")

    print(f"将跑 {len(args.backbones)} 个 backbone: {args.backbones}")
    print(f"运行设备: {DEVICE}, 池化: {args.pool}\n")

    labels = pd.read_csv(LABELS_CSV)
    with open(STATS_JSON, encoding="utf-8") as f:
        s = json.load(f)
    mu, sigma = s["mu"], s["sigma"]
    sv_df = load_sv_index()
    print(f"读 streetview_index: {len(sv_df)} 张图, "
          f"覆盖 {sv_df['BlockID'].nunique()} 街区\n")

    block_ids_all = sorted(set(sv_df["BlockID"].astype(int)).intersection(
        set(labels["block_id"].astype(int))))
    df_block = pd.DataFrame({"block_id": block_ids_all})
    merged_all = labels.merge(df_block, on="block_id", how="inner") \
                       .reset_index(drop=True)
    aux_all, _ = build_aux_features(sv_df, merged_all["block_id"].values)
    print("=" * 66 + "\n L0 BASELINE SELF-CHECK\n" + "=" * 66)
    run_baselines(merged_all, aux_all, mu, sigma)

    all_metrics = []
    all_pred_dict = {"block_id": None}
    for bb in args.backbones:
        try:
            rows, bids, pred_raw = run_one_backbone(
                bb, sv_df, labels, mu, sigma, args.pool, args.force_extract)
            all_metrics.extend(rows)
            if all_pred_dict["block_id"] is None:
                all_pred_dict["block_id"] = bids
            all_pred_dict[f"{bb}_mlp"] = pred_raw
        except Exception as e:
            print(f"  ⚠ {bb} 失败: {e}")
            import traceback; traceback.print_exc()
            continue

    # === v2.1 BUGFIX: 防御性处理空 metrics ===
    if not all_metrics:
        print("\n⚠ 所有 backbone 都失败了, 跳过汇总.")
        return

    sm = pd.DataFrame(all_metrics).sort_values(by="R2_log", ascending=False) \
                                   .reset_index(drop=True)
    sm.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 78)
    print("  街景模型测试集指标对比 (按 R²(log) 降序)")
    print("=" * 78)
    print(sm.to_string(index=False,
        float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.2f}"))
    print("=" * 78)
    print(f"\n[汇总] {SUMMARY_CSV}")

    if all_pred_dict["block_id"] is not None:
        pd.DataFrame(all_pred_dict).to_csv(
            ALL_PRED_CSV, index=False, encoding="utf-8-sig")
        print(f"[汇总] backbone 预测合并: {ALL_PRED_CSV}")

    # 诊断
    base = pd.read_csv(BASELINE_CSV)
    aux_r2_log = base.loc[base["model"].str.startswith("ridge_aux"), "R2_log"].values[0]
    best_bb_r2_log = sm["R2_log"].max()
    gain = best_bb_r2_log - aux_r2_log
    print("\n" + "=" * 78)
    print(f"  诊断: 仅辅助特征 R²(log) = {aux_r2_log:.4f}")
    print(f"        最佳 backbone R²(log) = {best_bb_r2_log:.4f}")
    print(f"        图像 backbone 净增益 = {gain:+.4f}")
    if gain < 0.02:
        print("  ⚠ 增益 < 0.02 → 图像 backbone 几乎没贡献.")
    elif gain < 0.05:
        print("  ◐ 图像有微弱信号, 远未达论文级 R²~0.4.")
    else:
        print("  ✓ 图像信号显著, 进入 KnowCL 主流程后还能再涨.")
    print("=" * 78)


if __name__ == "__main__":
    main()

