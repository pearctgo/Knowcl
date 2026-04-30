# -*- coding: utf-8 -*-
"""
==============================================================
 3_predict_remote_sensing.py  ——  遥感影像预测街区能耗
==============================================================
流程  :
  ① 对每个有标签的街区, 按几何 bbox 取影像
       (a) 优先 ESRI World Imagery (zoom=17)
       (b) 失败 fallback 到本地 TIF: G:\\Knowcl\\11-卫星数据\\影像下载_2503152313.tif
  ② ResNet50 → 2048 维, MLP 回归
  ③ 输出 rs_predictions.csv, 并打印与街景的对比

输出 :
  G:\\Knowcl\\999-输出成果文件\\002-能耗预测\\
      ├ rs_images\\Block_*.jpg
      ├ rs_features_block.npz
      ├ rs_mlp.pt
      ├ rs_predictions.csv
      └ comparison.csv     (合并 sv & rs 两路预测, 便于做散点图)
==============================================================
"""
import os
import io
import json
import math
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============== 配置 ==============
SHP_PATH    = r"G:\Knowcl\8-街区数据\沈阳L4能耗.shp"
LOCAL_TIF   = r"G:\Knowcl\11-卫星数据\影像下载_2503152313.tif"
OUT_DIR     = r"G:\Knowcl\999-输出成果文件\002-能耗预测"
LABELS_CSV  = os.path.join(OUT_DIR, "energy_labels.csv")
STATS_JSON  = os.path.join(OUT_DIR, "label_stats.json")
RS_IMG_DIR  = os.path.join(OUT_DIR, "rs_images")
FEAT_CACHE  = os.path.join(OUT_DIR, "rs_features_block.npz")
MODEL_PATH  = os.path.join(OUT_DIR, "rs_mlp.pt")
PRED_CSV    = os.path.join(OUT_DIR, "rs_predictions.csv")
SV_PRED_CSV = os.path.join(OUT_DIR, "sv_predictions.csv")
CMP_CSV     = os.path.join(OUT_DIR, "comparison.csv")
os.makedirs(RS_IMG_DIR, exist_ok=True)

ESRI_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
HEADERS  = {"User-Agent": "EnergyPipeline/1.0"}
ZOOM       = 17
BUFFER_M   = 50
PATCH_SIZE = 224

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACT_BATCH  = 32
NUM_WORKERS    = 4
MLP_HIDDEN     = [512, 128]
DROPOUT        = 0.3
LR             = 1e-3
WD             = 1e-4
EPOCHS         = 200
BATCH_SIZE     = 32
EARLY_STOP_PAT = 30
SEED           = 42


# ==============================================================
# 1. 影像准备 (ESRI 优先 + 本地 TIF fallback)
# ==============================================================
def deg2num(lat, lon, z):
    n = 2.0 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) +
                          1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def num2deg(x, y, z):
    n = 2.0 ** z
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def fetch_tile(z, x, y, retries=3):
    for a in range(retries):
        try:
            r = requests.get(ESRI_URL.format(z=z, x=x, y=y),
                             headers=HEADERS, timeout=15)
            if r.status_code == 200 and len(r.content) > 200:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            time.sleep(1.5 ** a)
    return None


def mosaic_esri(min_lon, min_lat, max_lon, max_lat, z=ZOOM):
    x0, y1 = deg2num(min_lat, min_lon, z)
    x1, y0 = deg2num(max_lat, max_lon, z)
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    if cols * rows > 64:
        return None
    canvas = Image.new("RGB", (cols * 256, rows * 256))
    ok = 0
    for i, x in enumerate(range(x0, x1 + 1)):
        for j, y in enumerate(range(y0, y1 + 1)):
            t = fetch_tile(z, x, y)
            if t is not None:
                canvas.paste(t, (i * 256, j * 256))
                ok += 1
    if ok == 0:
        return None
    nw_lat, nw_lon = num2deg(x0, y0, z)
    se_lat, se_lon = num2deg(x1 + 1, y1 + 1, z)
    px_lon = canvas.width / (se_lon - nw_lon)
    px_lat = canvas.height / (nw_lat - se_lat)
    L = max(0,             int((min_lon - nw_lon) * px_lon))
    R = min(canvas.width,  int((max_lon - nw_lon) * px_lon))
    T = max(0,             int((nw_lat - max_lat) * px_lat))
    B = min(canvas.height, int((nw_lat - min_lat) * px_lat))
    return canvas.crop((L, T, R, B))


def crop_local_tif(geom_4326):
    """从本地 TIF 按几何裁切, 返回 PIL.Image 或 None"""
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping
    from shapely.ops import transform as sh_transform
    import pyproj

    with rasterio.open(LOCAL_TIF) as src:
        # 投影到 TIF 的 CRS
        if src.crs is not None and str(src.crs).upper() != "EPSG:4326":
            project = pyproj.Transformer.from_crs(
                "EPSG:4326", src.crs, always_xy=True).transform
            geom_proj = sh_transform(project, geom_4326)
        else:
            geom_proj = geom_4326
        try:
            out_img, _ = rio_mask(src, [mapping(geom_proj)], crop=True)
        except Exception:
            return None

        if out_img.shape[0] >= 3:
            arr = np.transpose(out_img[:3], (1, 2, 0))
        elif out_img.shape[0] == 1:
            arr = np.transpose(np.repeat(out_img, 3, axis=0), (1, 2, 0))
        else:
            return None

        # 拉伸到 0-255
        if arr.dtype != np.uint8:
            valid = arr[arr > 0]
            if valid.size == 0:
                return None
            mn, mx = np.percentile(valid, [2, 98])
            arr = np.clip((arr - mn) / max(mx - mn, 1e-6) * 255,
                          0, 255).astype(np.uint8)
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            return None
        return Image.fromarray(arr)


def make_square(img, size=PATCH_SIZE):
    w, h = img.size
    s = max(w, h)
    bg = Image.new("RGB", (s, s), (0, 0, 0))
    bg.paste(img, ((s - w) // 2, (s - h) // 2))
    return bg.resize((size, size), Image.BILINEAR)


def prepare_rs_images():
    """对每个有标签的街区准备一张 PATCH_SIZE x PATCH_SIZE 的影像"""
    labels = pd.read_csv(LABELS_CSV)
    valid_ids = set(labels["block_id"].astype(int))

    print(f"[Step3] 读取街区: {SHP_PATH}")
    gdf = gpd.read_file(SHP_PATH, encoding="utf-8")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    # 米制 buffer
    gdf_m = gdf.to_crs(epsg=3857)
    gdf_m["buf"] = gdf_m.geometry.buffer(BUFFER_M)
    gdf_m = gdf_m.set_geometry("buf").to_crs(epsg=4326)

    has_local = os.path.exists(LOCAL_TIF)
    print(f"        本地 TIF 可用: {has_local}")

    records, esri_fails = [], []
    print("[Step3] 阶段 1: 从 ESRI 拉取...")
    for _, row in tqdm(gdf_m.iterrows(), total=len(gdf_m), desc="ESRI"):
        bid = int(row["BlockID"])
        if bid not in valid_ids:
            continue

        out_path = os.path.join(RS_IMG_DIR, f"Block_{bid}.jpg")
        if os.path.exists(out_path):
            records.append({"block_id": bid, "path": out_path, "src": "cached"})
            continue

        minx, miny, maxx, maxy = row["buf"].bounds
        img = mosaic_esri(minx, miny, maxx, maxy)
        if img is not None:
            sq = make_square(img)
            sq.save(out_path, "JPEG", quality=92)
            records.append({"block_id": bid, "path": out_path, "src": "esri"})
        else:
            esri_fails.append((bid, row["buf"]))

    if esri_fails and has_local:
        print(f"[Step3] 阶段 2: 用本地 TIF 补 {len(esri_fails)} 个...")
        for bid, geom in tqdm(esri_fails, desc="local TIF"):
            out_path = os.path.join(RS_IMG_DIR, f"Block_{bid}.jpg")
            try:
                img = crop_local_tif(geom)
                if img is not None:
                    make_square(img).save(out_path, "JPEG", quality=92)
                    records.append({"block_id": bid, "path": out_path,
                                    "src": "local_tif"})
                else:
                    records.append({"block_id": bid, "path": "", "src": "failed"})
            except Exception:
                records.append({"block_id": bid, "path": "", "src": "failed"})
    else:
        for bid, _ in esri_fails:
            records.append({"block_id": bid, "path": "", "src": "failed"})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RS_IMG_DIR, "_index.csv"),
              index=False, encoding="utf-8-sig")
    print(f"        esri={(df.src == 'esri').sum()} | "
          f"cached={(df.src == 'cached').sum()} | "
          f"local_tif={(df.src == 'local_tif').sum()} | "
          f"failed={(df.src == 'failed').sum()}")

    return df[df["src"] != "failed"].reset_index(drop=True)


# ==============================================================
# 2. ResNet50 特征
# ==============================================================
class RSDataset(Dataset):
    def __init__(self, df, transform):
        self.df, self.transform = df.reset_index(drop=True), transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        try:
            img = Image.open(self.df.iloc[i]["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE))
        return self.transform(img), i


def extract_features(force=False):
    if os.path.exists(FEAT_CACHE) and not force:
        print(f"[Step3] 读取特征缓存: {FEAT_CACHE}")
        d = np.load(FEAT_CACHE)
        return d["features"], d["block_id"]

    df = prepare_rs_images()
    print(f"[Step3] 提取 {len(df)} 张遥感影像特征...")
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Identity()
    net.eval().to(DEVICE)
    transform = transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(RSDataset(df, transform),
                        batch_size=EXTRACT_BATCH, shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=(DEVICE == "cuda"))
    feats = np.zeros((len(df), 2048), dtype=np.float32)
    with torch.no_grad():
        for imgs, idxs in tqdm(loader, desc="extract RS"):
            f = net(imgs.to(DEVICE)).cpu().numpy()
            feats[idxs.numpy()] = f
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    block_ids = df["block_id"].astype(int).values
    np.savez_compressed(FEAT_CACHE, features=feats, block_id=block_ids)
    return feats, block_ids


# ==============================================================
# 3. MLP 回归
# ==============================================================
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        dims, layers = [in_dim] + MLP_HIDDEN, []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]),
                       nn.BatchNorm1d(dims[i + 1]),
                       nn.ReLU(inplace=True), nn.Dropout(DROPOUT)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_predict(X, y, splits, in_dim):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tr, va = splits == "train", splits == "val"

    model = MLP(in_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.SmoothL1Loss()

    tr_ld = DataLoader(TensorDataset(torch.from_numpy(X[tr]).float(),
                                     torch.from_numpy(y[tr]).float()),
                       batch_size=BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(TensorDataset(torch.from_numpy(X[va]).float(),
                                     torch.from_numpy(y[va]).float()),
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

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).float().to(DEVICE)
            out.append(model(xb).cpu().numpy())
    return model, np.concatenate(out)


# ==============================================================
# main
# ==============================================================
def main():
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError("先运行 1_build_labels.py")

    feats, bids = extract_features()
    labels = pd.read_csv(LABELS_CSV)
    df_feat = pd.DataFrame({"block_id": bids,
                            "feat_idx": np.arange(len(bids))})
    merged = labels.merge(df_feat, on="block_id", how="inner") \
                   .reset_index(drop=True)

    X = feats[merged["feat_idx"].values]
    y = merged["energy_norm"].values.astype(np.float32)
    splits = merged["split"].values
    tr, va, te = splits == "train", splits == "val", splits == "test"
    print(f"[Step3] 对齐样本: {len(X)} (train={tr.sum()} val={va.sum()} test={te.sum()})")

    print("[Step3] 训练 MLP...")
    model, pred_norm = train_and_predict(X, y, splits, X.shape[1])

    with open(STATS_JSON, encoding="utf-8") as f:
        s = json.load(f)
    mu, sigma = s["mu"], s["sigma"]
    pred_log = pred_norm * sigma + mu
    pred_raw = np.expm1(pred_log)
    true_raw = merged["energy_raw"].values

    y_te, p_te = true_raw[te], pred_raw[te]
    print("\n[Step3] === 遥感模型 测试集指标 ===")
    print(f"        R²  (raw): {r2_score(y_te, p_te):.4f}")
    print(f"        R²  (log): {r2_score(np.log1p(y_te), np.log1p(p_te)):.4f}")
    print(f"        RMSE(raw): {np.sqrt(mean_squared_error(y_te, p_te)):.2f}")
    print(f"        MAE (raw): {mean_absolute_error(y_te, p_te):.2f}")

    out = pd.DataFrame({
        "block_id":       merged["block_id"].values,
        "split":          splits,
        "true_energy":    true_raw,
        "rs_pred_energy": pred_raw,
    })
    out.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[Step3] 预测已保存: {PRED_CSV}")

    # ===== 与街景对比 =====
    if os.path.exists(SV_PRED_CSV):
        sv = pd.read_csv(SV_PRED_CSV)
        cmp = sv.merge(out[["block_id", "rs_pred_energy"]],
                       on="block_id", how="inner")
        cmp.to_csv(CMP_CSV, index=False, encoding="utf-8-sig")
        print(f"\n========== 街景 vs 遥感 对比 (测试集) ==========")
        sv_te = cmp[cmp["split"] == "test"]
        if len(sv_te) > 0:
            r2_sv  = r2_score(sv_te["true_energy"], sv_te["sv_pred_energy"])
            r2_rs  = r2_score(sv_te["true_energy"], sv_te["rs_pred_energy"])
            mae_sv = mean_absolute_error(sv_te["true_energy"], sv_te["sv_pred_energy"])
            mae_rs = mean_absolute_error(sv_te["true_energy"], sv_te["rs_pred_energy"])
            r2_log_sv = r2_score(np.log1p(sv_te["true_energy"]),
                                 np.log1p(sv_te["sv_pred_energy"]))
            r2_log_rs = r2_score(np.log1p(sv_te["true_energy"]),
                                 np.log1p(sv_te["rs_pred_energy"]))
            print(f"  指标            街景            遥感")
            print(f"  R²    (raw)  : {r2_sv:>8.4f}     {r2_rs:>8.4f}")
            print(f"  R²    (log)  : {r2_log_sv:>8.4f}     {r2_log_rs:>8.4f}")
            print(f"  MAE   (raw)  : {mae_sv:>8.2f}     {mae_rs:>8.2f}")
            print(f"\n  对比 CSV: {CMP_CSV}")


if __name__ == "__main__":
    main()
