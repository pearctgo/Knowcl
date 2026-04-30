# -*- coding: utf-8 -*-
"""
==============================================================
 2_predict_streetview.py  ——  街景预测街区能耗
==============================================================
流程  :
  ① 读 streetview_index.csv → 每张图过 ResNet50 → 2048 维特征
  ② 同一 BlockID 的所有图 mean pooling（图片不足 16 张也兼容）
  ③ MLP 回归 (训练 → 验证早停 → 测试集评估)
  ④ 输出 sv_predictions.csv 与控制台指标

输出  :
  G:\\Knowcl\\999-输出成果文件\\002-能耗预测\\
      ├ sv_features_block.npz   特征缓存
      ├ sv_mlp.pt               训练好的 MLP
      └ sv_predictions.csv      block_id, split, true_energy, sv_pred_energy
==============================================================
"""
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============== 配置 ==============
SV_INDEX_CSV = r"G:\Knowcl\999-输出成果文件\001-街景重采_baidu\tables\streetview_index.csv"
OUT_DIR      = r"G:\Knowcl\999-输出成果文件\002-能耗预测"
LABELS_CSV   = os.path.join(OUT_DIR, "energy_labels.csv")
STATS_JSON   = os.path.join(OUT_DIR, "label_stats.json")
FEAT_CACHE   = os.path.join(OUT_DIR, "sv_features_block.npz")
MODEL_PATH   = os.path.join(OUT_DIR, "sv_mlp.pt")
PRED_CSV     = os.path.join(OUT_DIR, "sv_predictions.csv")

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACT_BATCH  = 64
NUM_WORKERS    = 4
MLP_HIDDEN     = [512, 128]
DROPOUT        = 0.3
LR             = 1e-3
WD             = 1e-4
EPOCHS         = 200
BATCH_SIZE     = 32
EARLY_STOP_PAT = 30
SEED           = 42


# --------------------------------------------------------------
# 1. ResNet50 特征提取
# --------------------------------------------------------------
class SVDataset(Dataset):
    def __init__(self, df, transform):
        self.df, self.transform = df.reset_index(drop=True), transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.df.iloc[idx]["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.transform(img), idx


def extract_block_features(force=False):
    """
    返回 (block_features, block_ids)
    block_features.shape = (n_block, 2048)
    """
    if os.path.exists(FEAT_CACHE) and not force:
        print(f"[Step2] 读取特征缓存: {FEAT_CACHE}")
        d = np.load(FEAT_CACHE)
        return d["features"], d["block_id"]

    print(f"[Step2] 读取街景索引: {SV_INDEX_CSV}")
    df = pd.read_csv(SV_INDEX_CSV, encoding="utf-8")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    labels = pd.read_csv(LABELS_CSV)
    valid = set(labels["block_id"].astype(int))
    df = df[df["BlockID"].astype(int).isin(valid)].reset_index(drop=True)
    print(f"        待提取图像: {len(df)} 张, 覆盖街区 {df['BlockID'].nunique()}")

    # 不足 16 张的街区统计
    cnt = df.groupby("BlockID").size()
    print(f"        每街区图数: 中位 {int(cnt.median())}, "
          f"min {int(cnt.min())}, max {int(cnt.max())}, "
          f"<16: {(cnt < 16).sum()} 个街区")

    # 模型
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Identity()
    net.eval().to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    loader = DataLoader(SVDataset(df, transform),
                        batch_size=EXTRACT_BATCH, shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=(DEVICE == "cuda"))

    feats = np.zeros((len(df), 2048), dtype=np.float32)
    with torch.no_grad():
        for imgs, idxs in tqdm(loader, desc="extract SV"):
            f = net(imgs.to(DEVICE, non_blocking=True)).cpu().numpy()
            feats[idxs.numpy()] = f
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    # mean pooling 到街区级（图片不足 16 张也兼容）
    block_ids = sorted(df["BlockID"].astype(int).unique())
    block_feats = np.zeros((len(block_ids), 2048), dtype=np.float32)
    bid_arr = df["BlockID"].astype(int).values
    for i, bid in enumerate(block_ids):
        block_feats[i] = feats[bid_arr == bid].mean(axis=0)

    np.savez_compressed(FEAT_CACHE, features=block_feats,
                        block_id=np.array(block_ids, dtype=np.int64))
    print(f"        特征缓存: {FEAT_CACHE}, shape={block_feats.shape}")
    return block_feats, np.array(block_ids)


# --------------------------------------------------------------
# 2. MLP 回归器
# --------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=MLP_HIDDEN, dropout=DROPOUT):
        super().__init__()
        dims, layers = [in_dim] + hidden, []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
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
            best = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if ep % 20 == 0 or ep == EPOCHS - 1:
            print(f"        ep {ep:3d}: val_rmse={val_rmse:.4f}  best={best:.4f}")
        if patience >= EARLY_STOP_PAT:
            print(f"        early stop @ ep {ep}")
            break
    model.load_state_dict(best_state)
    return model


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).float().to(DEVICE)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------
# main
# --------------------------------------------------------------
def main():
    np.random.seed(SEED)
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError("先运行 1_build_labels.py")

    feats, bids = extract_block_features()
    labels = pd.read_csv(LABELS_CSV)

    df_feat = pd.DataFrame({"block_id": bids,
                            "feat_idx": np.arange(len(bids))})
    merged = labels.merge(df_feat, on="block_id", how="inner") \
                   .reset_index(drop=True)

    X = feats[merged["feat_idx"].values]
    y = merged["energy_norm"].values.astype(np.float32)
    splits = merged["split"].values
    tr, va, te = splits == "train", splits == "val", splits == "test"
    print(f"[Step2] 对齐样本: {len(X)} (train={tr.sum()} val={va.sum()} test={te.sum()})")

    print("[Step2] 训练 MLP...")
    model = train_mlp(X[tr], y[tr], X[va], y[va], X.shape[1])

    pred_norm = predict(model, X)

    with open(STATS_JSON, encoding="utf-8") as f:
        s = json.load(f)
    mu, sigma = s["mu"], s["sigma"]
    pred_log  = pred_norm * sigma + mu
    pred_raw  = np.expm1(pred_log)
    true_raw  = merged["energy_raw"].values

    # 指标
    y_te, p_te = true_raw[te], pred_raw[te]
    print("\n[Step2] === 街景模型 测试集指标 ===")
    print(f"        R²  (raw): {r2_score(y_te, p_te):.4f}")
    print(f"        R²  (log): {r2_score(np.log1p(y_te), np.log1p(p_te)):.4f}")
    print(f"        RMSE(raw): {np.sqrt(mean_squared_error(y_te, p_te)):.2f}")
    print(f"        MAE (raw): {mean_absolute_error(y_te, p_te):.2f}")

    out = pd.DataFrame({
        "block_id":       merged["block_id"].values,
        "split":          splits,
        "true_energy":    true_raw,
        "sv_pred_energy": pred_raw,
    })
    out.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[Step2] 预测已保存: {PRED_CSV}")
    print(f"        模型已保存: {MODEL_PATH}")


if __name__ == "__main__":
    main()
