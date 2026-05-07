# -*- coding: utf-8 -*-
"""
==============================================================
 3_predict_remote_sensing.py  —  遥感预测街区能耗 (v2.1)
==============================================================

 v1 → v2.1 核心修复 (与 2_predict_streetview.py v2.1 同款诊断):
   1. 删除 feats /= np.linalg.norm(...) — 单图 L2 归一化丢幅度信号
   2. cache 加 _v2 后缀避免误用 v1 旧缓存
   3. MLP 升级: BN 输入 + 256→64 + dropout 0.5 + wd 1e-3 + lr 5e-4 + batch 64
   4. 加 L0 baseline self-check: 均值预测器 + 几何辅助特征 Ridge baseline
   5. 加 Ridge baseline 与 MLP 同表对照 (高维+小样本 Ridge 常打过 MLP)
   6. 加街区几何辅助特征: log(面积/周长), 紧凑度, 中心经纬度
   7. metrics_summary schema 增加 model 和 R2_norm 两列, 与街景 v2.1 对齐
   8. final_comparison 列前缀统一 sv_<bb> / rs_<bb>, 不再冲突

 与街景 v2 的关键差别:
   • 街景: 每街区多张图 → mean+std 池化能补方差信号
   • 遥感: 每街区一张图 → std 永远是 0, 故不做多池化
   • 改用"街区几何辅助特征" (面积/周长/形状/坐标) 作为遥感的零成本 baseline

 输出 (G:\\Knowcl\\999-输出成果文件\\002-能耗预测\\):
   rs_images\\Block_*.jpg                  影像缓存
   rs_features_block_<bb>_v2.npz           backbone 特征缓存 (无归一化)
   rs_predictions_<bb>.csv                 单 backbone 预测明细
   rs_mlp_<bb>.pt                          MLP 权重
   rs_baseline_metrics.csv                 L0 baseline 自检表
   rs_metrics_summary.csv                  7 backbone × {mlp, ridge} 排序表
   rs_all_models.csv                       预测合并: block × backbone
   final_comparison.csv                    sv × rs × 7 backbone 总对比
   metrics_summary_all.csv                 街景 + 遥感 metrics 总表
==============================================================
"""
import os
import io
import json
import math
import time
import argparse
import warnings
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

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ============== 路径 & 超参 ==============
SHP_PATH    = r"G:\Knowcl\8-街区数据\沈阳L4能耗.shp"
LOCAL_TIF   = r"G:\Knowcl\11-卫星数据\影像下载_2503152313.tif"
OUT_DIR     = r"G:\Knowcl\999-输出成果文件\002-能耗预测"
LABELS_CSV  = os.path.join(OUT_DIR, "energy_labels.csv")
STATS_JSON  = os.path.join(OUT_DIR, "label_stats.json")
RS_IMG_DIR  = os.path.join(OUT_DIR, "rs_images")
BASELINE_CSV = os.path.join(OUT_DIR, "rs_baseline_metrics.csv")
SUMMARY_CSV  = os.path.join(OUT_DIR, "rs_metrics_summary.csv")
ALL_PRED_CSV = os.path.join(OUT_DIR, "rs_all_models.csv")
FINAL_CSV    = os.path.join(OUT_DIR, "final_comparison.csv")
METRICS_ALL  = os.path.join(OUT_DIR, "metrics_summary_all.csv")
os.makedirs(RS_IMG_DIR, exist_ok=True)

ESRI_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
HEADERS  = {"User-Agent": "EnergyPipeline/2.1"}
ZOOM, BUFFER_M, PATCH_SIZE = 17, 50, 224

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACT_BATCH  = 32
NUM_WORKERS    = 4

# v2.1 训练超参 (与街景 v2.1 完全一致)
DROPOUT        = 0.5      # 0.3 → 0.5
LR             = 5e-4     # 1e-3 → 5e-4
WD             = 1e-3     # 1e-4 → 1e-3
EPOCHS         = 200
BATCH_SIZE     = 64       # 32 → 64
EARLY_STOP_PAT = 30
SEED           = 42

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]

ALL_BACKBONES = [
    "resnet50", "densenet121", "convnext_tiny", "vit_b_16",
    "mobilenet_v3_large", "efficientnet_b0", "attention_cnn",
]


# ==============================================================
# 1. CBAM + AttentionCNN
# ==============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hid = max(in_planes // ratio, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, in_planes, 1, bias=False),
        )
    def forward(self, x):
        return torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class AttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.cbam = CBAM(2048)
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.features(x); x = self.cbam(x)
        return self.pool(x).flatten(1)


# ==============================================================
# 2. Backbone 工厂 (与街景 v2.1 完全对齐)
# ==============================================================
def build_backbone(name):
    imagenet_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    default_tf = transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(), imagenet_norm,
    ])
    if name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=w);  net.fc = nn.Identity()
        return net.eval(), 2048, w.transforms()
    if name == "densenet121":
        w = models.DenseNet121_Weights.IMAGENET1K_V1
        net = models.densenet121(weights=w);  net.classifier = nn.Identity()
        return net.eval(), 1024, w.transforms()
    if name == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        net = models.convnext_tiny(weights=w);  net.classifier[2] = nn.Identity()
        return net.eval(), 768, w.transforms()
    if name == "vit_b_16":
        w = models.ViT_B_16_Weights.IMAGENET1K_V1
        net = models.vit_b_16(weights=w);  net.heads = nn.Identity()
        return net.eval(), 768, w.transforms()
    if name == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        net = models.mobilenet_v3_large(weights=w);  net.classifier = nn.Identity()
        return net.eval(), 960, w.transforms()
    if name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        net = models.efficientnet_b0(weights=w);  net.classifier = nn.Identity()
        return net.eval(), 1280, w.transforms()
    if name == "attention_cnn":
        return AttentionCNN().eval(), 2048, default_tf
    raise ValueError(name)


# ==============================================================
# 3. ESRI tile 下载 + 本地 TIF fallback
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
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    if cols * rows > 64: return None
    canvas = Image.new("RGB", (cols * 256, rows * 256))
    ok = 0
    for i, x in enumerate(range(x0, x1 + 1)):
        for j, y in enumerate(range(y0, y1 + 1)):
            t = fetch_tile(z, x, y)
            if t is not None:
                canvas.paste(t, (i * 256, j * 256))
                ok += 1
    if ok == 0: return None
    nw_lat, nw_lon = num2deg(x0, y0, z)
    se_lat, se_lon = num2deg(x1 + 1, y1 + 1, z)
    px_lon = canvas.width / (se_lon - nw_lon)
    px_lat = canvas.height / (nw_lat - se_lat)
    L = max(0, int((min_lon - nw_lon) * px_lon))
    R = min(canvas.width, int((max_lon - nw_lon) * px_lon))
    T = max(0, int((nw_lat - max_lat) * px_lat))
    B = min(canvas.height, int((nw_lat - min_lat) * px_lat))
    return canvas.crop((L, T, R, B))


def crop_local_tif(geom_4326):
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping
    from shapely.ops import transform as sh_transform
    import pyproj
    with rasterio.open(LOCAL_TIF) as src:
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
        if arr.dtype != np.uint8:
            valid = arr[arr > 0]
            if valid.size == 0: return None
            mn, mx = np.percentile(valid, [2, 98])
            arr = np.clip((arr - mn) / max(mx - mn, 1e-6) * 255,
                          0, 255).astype(np.uint8)
        if arr.shape[0] < 2 or arr.shape[1] < 2: return None
        return Image.fromarray(arr)


def make_square(img, size=PATCH_SIZE):
    w, h = img.size
    s = max(w, h)
    bg = Image.new("RGB", (s, s), (0, 0, 0))
    bg.paste(img, ((s - w) // 2, (s - h) // 2))
    return bg.resize((size, size), Image.BILINEAR)


def prepare_rs_images():
    """对每个有标签的街区准备一张影像 (一次性, 与 backbone 无关)"""
    idx_csv = os.path.join(RS_IMG_DIR, "_index.csv")
    if os.path.exists(idx_csv):
        df = pd.read_csv(idx_csv)
        df = df[df["src"] != "failed"].reset_index(drop=True)
        df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
        if len(df) > 0:
            print(f"[Step3] 复用已下载遥感影像: {len(df)} 张")
            return df

    labels = pd.read_csv(LABELS_CSV)
    valid_ids = set(labels["block_id"].astype(int))

    print(f"[Step3] 读取街区: {SHP_PATH}")
    gdf = gpd.read_file(SHP_PATH, encoding="utf-8")
    if gdf.crs is None: gdf = gdf.set_crs("EPSG:4326")
    gdf_m = gdf.to_crs(epsg=3857)
    gdf_m["buf"] = gdf_m.geometry.buffer(BUFFER_M)
    gdf_m = gdf_m.set_geometry("buf").to_crs(epsg=4326)

    has_local = os.path.exists(LOCAL_TIF)
    print(f"        本地 TIF 可用: {has_local}")

    records, esri_fails = [], []
    print("[Step3] 阶段 1: 从 ESRI 拉取...")
    for _, row in tqdm(gdf_m.iterrows(), total=len(gdf_m), desc="ESRI"):
        bid = int(row["BlockID"])
        if bid not in valid_ids: continue
        out_path = os.path.join(RS_IMG_DIR, f"Block_{bid}.jpg")
        if os.path.exists(out_path):
            records.append({"block_id": bid, "path": out_path, "src": "cached"})
            continue
        minx, miny, maxx, maxy = row["buf"].bounds
        img = mosaic_esri(minx, miny, maxx, maxy)
        if img is not None:
            make_square(img).save(out_path, "JPEG", quality=92)
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
    df.to_csv(idx_csv, index=False, encoding="utf-8-sig")
    print(f"        esri={(df.src == 'esri').sum()} | "
          f"cached={(df.src == 'cached').sum()} | "
          f"local_tif={(df.src == 'local_tif').sum()} | "
          f"failed={(df.src == 'failed').sum()}")
    return df[df["src"] != "failed"].reset_index(drop=True)


# ==============================================================
# 4. 几何辅助特征 (从 SHP 计算, 用于 L0 baseline 与 MLP concat)
# ==============================================================
def build_geom_aux_features(block_ids):
    """每街区的几何辅助特征:
       [log(area_m2), log(perim_m), compactness, lon_c, lat_c]

       compactness = 4πA / P² ∈ (0, 1], 圆形=1, 越扁越小
       coords 用 EPSG:4326 中心点 (相对位置信号)
    """
    print("[Step3] 计算街区几何辅助特征...")
    gdf = gpd.read_file(SHP_PATH, encoding="utf-8")
    if gdf.crs is None: gdf = gdf.set_crs("EPSG:4326")

    # UTM 51N 计算面积/周长
    gdf_m = gdf.to_crs(epsg=32651)
    gdf_m["area_m2"] = gdf_m.geometry.area
    gdf_m["perim_m"] = gdf_m.geometry.length
    gdf_m["compact"] = (4 * np.pi * gdf_m["area_m2"]) / \
                      (gdf_m["perim_m"] ** 2 + 1e-8)

    # 4326 取中心点
    centroids = gdf.geometry.centroid
    gdf_m["lon_c"] = centroids.x.values
    gdf_m["lat_c"] = centroids.y.values

    gdf_m["BlockID"] = gdf_m["BlockID"].astype(int)
    gdf_m = gdf_m.set_index("BlockID")

    n = len(block_ids)
    aux = np.zeros((n, 5), dtype=np.float32)
    miss = 0
    for i, bid in enumerate(block_ids):
        bid = int(bid)
        if bid not in gdf_m.index:
            miss += 1
            continue
        row = gdf_m.loc[bid]
        if isinstance(row, pd.DataFrame):  # 罕见: 重复 BlockID
            row = row.iloc[0]
        aux[i] = [
            float(np.log1p(row["area_m2"])),
            float(np.log1p(row["perim_m"])),
            float(row["compact"]),
            float(row["lon_c"]),
            float(row["lat_c"]),
        ]
    if miss > 0:
        print(f"        ⚠ 几何特征缺失 {miss} 块 (将填 0)")
    aux_names = ["log_area", "log_perim", "compact", "lon_c", "lat_c"]
    return aux, aux_names


# ==============================================================
# 5. 特征提取 (无 L2 归一化, _v2 缓存)
# ==============================================================
class RSDataset(Dataset):
    def __init__(self, df, transform):
        self.df, self.transform = df.reset_index(drop=True), transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        try:
            img = Image.open(self.df.iloc[i]["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE))
        return self.transform(img), i


def extract_features_for_backbone(name, df_imgs, force=False):
    """v2.1: cache 加 _v2 后缀, 避免误用 v1 归一化旧缓存."""
    cache = os.path.join(OUT_DIR, f"rs_features_block_{name}_v2.npz")
    if os.path.exists(cache) and not force:
        print(f"  [{name}] 读取特征缓存: {os.path.basename(cache)}")
        d = np.load(cache)
        return d["features"], d["block_id"]

    net, feat_dim, transform = build_backbone(name)
    net = net.to(DEVICE)
    loader = DataLoader(RSDataset(df_imgs, transform),
                        batch_size=EXTRACT_BATCH, shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=(DEVICE == "cuda"))
    feats = np.zeros((len(df_imgs), feat_dim), dtype=np.float32)
    with torch.no_grad():
        for imgs, idxs in tqdm(loader, desc=f"  {name}", leave=False):
            f = net(imgs.to(DEVICE))
            if f.dim() > 2:
                f = f.flatten(1)
            feats[idxs.numpy()] = f.cpu().numpy()
    # v2.1: 不再 L2 归一化, 保留幅度信号
    block_ids = df_imgs["block_id"].astype(int).values
    np.savez_compressed(cache, features=feats, block_id=block_ids)
    print(f"  [{name}] 特征已缓存: shape={feats.shape}")
    del net
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return feats, block_ids


# ==============================================================
# 6. L0 baseline self-check
# ==============================================================
def to_raw(pred_norm, mu, sigma):
    return np.expm1(pred_norm * sigma + mu)


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


def run_baselines(merged, aux_feats, mu, sigma):
    """L0 self-check: 均值预测器 + 仅几何辅助特征 Ridge.

    返回的几何 Ridge R² 是遥感模态的"零成本基线",
    backbone 必须显著超过它才算图像有信号.
    """
    splits = merged["split"].values
    y_norm = merged["energy_norm"].values.astype(np.float32)
    y_raw  = merged["energy_raw"].values.astype(np.float32)

    tr, va, te = splits == "train", splits == "val", splits == "test"
    rows = []

    # 1) mean predictor
    mean_norm = float(y_norm[tr].mean())
    p_norm = np.full(te.sum(), mean_norm)
    p_raw  = to_raw(p_norm, mu, sigma)
    rows.append(baseline_metrics_row("mean_predictor",
                                     y_norm[te], p_norm, y_raw[te], p_raw))

    # 2) Ridge on geometry only
    Xtr, Xva, Xte = aux_feats[tr], aux_feats[va], aux_feats[te]
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best_alpha, best_rmse = None, float("inf")
    for a in RIDGE_ALPHAS:
        rg = Ridge(alpha=a, random_state=SEED).fit(Xtr_s, y_norm[tr])
        rmse = np.sqrt(mean_squared_error(y_norm[va], rg.predict(Xva_s)))
        if rmse < best_rmse:
            best_rmse, best_alpha = rmse, a
    rg = Ridge(alpha=best_alpha, random_state=SEED).fit(Xtr_s, y_norm[tr])
    p_norm = rg.predict(Xte_s)
    p_raw  = to_raw(p_norm, mu, sigma)
    rows.append(baseline_metrics_row(
        f"ridge_geom_only(alpha={best_alpha})",
        y_norm[te], p_norm, y_raw[te], p_raw))

    df = pd.DataFrame(rows)
    df.to_csv(BASELINE_CSV, index=False, encoding="utf-8-sig")
    print("\n[L0 baseline]")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  → {BASELINE_CSV}\n")
    return df


# ==============================================================
# 7. MLP & 训练 (与街景 v2.1 完全一致)
# ==============================================================
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


def fit_ridge(X_tr, y_tr, X_va, y_va, X_all):
    """高维+小样本时 Ridge 常打过 MLP, 用作对照.
    返回全样本预测以便保存对齐.
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


# ==============================================================
# 8. 单 backbone 完整流程
# ==============================================================
def run_one_backbone(name, df_imgs, geom_aux_full, geom_block_ids):
    print(f"\n{'=' * 60}\n 遥感 backbone:  {name}\n{'=' * 60}")

    # 1) 提特征
    feats, bids = extract_features_for_backbone(name, df_imgs)

    # 2) 与标签对齐
    labels = pd.read_csv(LABELS_CSV)
    df_feat = pd.DataFrame({"block_id": bids, "feat_idx": np.arange(len(bids))})
    merged = labels.merge(df_feat, on="block_id", how="inner") \
                   .reset_index(drop=True)

    # 3) backbone 特征 + 几何辅助 concat
    Xb = feats[merged["feat_idx"].values]                 # backbone 特征
    geom_lookup = {int(b): i for i, b in enumerate(geom_block_ids)}
    aux_idx = np.array([geom_lookup[int(b)]
                        for b in merged["block_id"].values])
    Xa = geom_aux_full[aux_idx]                            # 几何辅助
    X = np.concatenate([Xb, Xa], axis=1)                   # (N, dim+5)
    y = merged["energy_norm"].values.astype(np.float32)
    splits = merged["split"].values
    tr, va, te = splits == "train", splits == "val", splits == "test"

    print(f"  [{name}] 总维度={X.shape[1]} (backbone {Xb.shape[1]} + 几何 {Xa.shape[1]})")
    print(f"  [{name}] 对齐样本: {len(X)} (train={tr.sum()} val={va.sum()} test={te.sum()})")

    # 4) Ridge baseline (全样本预测)
    best_a, ridge_val_rmse, ridge_pred_norm_all = fit_ridge(
        X[tr], y[tr], X[va], y[va], X)

    # 5) MLP
    print(f"  [{name}] 训练 MLP (in_dim={X.shape[1]})...")
    mlp, val_rmse = train_mlp(X[tr], y[tr], X[va], y[va], X.shape[1])
    pred_norm_all = predict(mlp, X)

    # 6) 转 raw 算指标
    with open(STATS_JSON, encoding="utf-8") as f:
        s = json.load(f)
    mu, sigma = s["mu"], s["sigma"]
    pred_raw_all = to_raw(pred_norm_all, mu, sigma)
    ridge_pred_raw_all = to_raw(ridge_pred_norm_all, mu, sigma)
    y_raw = merged["energy_raw"].values

    y_te_norm, y_te_raw = y[te], y_raw[te]
    p_te_raw_mlp   = pred_raw_all[te]
    p_te_raw_ridge = ridge_pred_raw_all[te]
    p_te_norm_mlp   = pred_norm_all[te]
    p_te_norm_ridge = ridge_pred_norm_all[te]

    metrics_mlp = {
        "backbone":  name,
        "model":     "mlp",
        "feat_dim":  int(X.shape[1]),
        "val_rmse":  round(float(val_rmse), 4),
        "R2_norm":   round(float(r2_score(y_te_norm, p_te_norm_mlp)), 4),
        "R2_raw":    round(float(r2_score(y_te_raw, p_te_raw_mlp)), 4),
        "R2_log":    round(float(r2_score(np.log1p(y_te_raw),
                            np.log1p(np.clip(p_te_raw_mlp, 0, None)))), 4),
        "RMSE_raw":  round(float(np.sqrt(mean_squared_error(y_te_raw, p_te_raw_mlp))), 2),
        "MAE_raw":   round(float(mean_absolute_error(y_te_raw, p_te_raw_mlp)), 2),
    }
    metrics_ridge = {
        "backbone":  name,
        "model":     f"ridge(a={best_a})",
        "feat_dim":  int(X.shape[1]),
        "val_rmse":  round(float(ridge_val_rmse), 4),
        "R2_norm":   round(float(r2_score(y_te_norm, p_te_norm_ridge)), 4),
        "R2_raw":    round(float(r2_score(y_te_raw, p_te_raw_ridge)), 4),
        "R2_log":    round(float(r2_score(np.log1p(y_te_raw),
                            np.log1p(np.clip(p_te_raw_ridge, 0, None)))), 4),
        "RMSE_raw":  round(float(np.sqrt(mean_squared_error(y_te_raw, p_te_raw_ridge))), 2),
        "MAE_raw":   round(float(mean_absolute_error(y_te_raw, p_te_raw_ridge)), 2),
    }
    print(f"  ► MLP   R²(log)={metrics_mlp['R2_log']:.4f}  "
          f"R²(raw)={metrics_mlp['R2_raw']:.4f}  RMSE={metrics_mlp['RMSE_raw']:.2f}")
    print(f"  ► Ridge R²(log)={metrics_ridge['R2_log']:.4f}  "
          f"R²(raw)={metrics_ridge['R2_raw']:.4f}  RMSE={metrics_ridge['RMSE_raw']:.2f}")

    # 7) 保存
    pred_csv = os.path.join(OUT_DIR, f"rs_predictions_{name}.csv")
    pd.DataFrame({
        "block_id":           merged["block_id"].values,
        "split":              splits,
        "true_energy":        y_raw,
        "mlp_pred_energy":    pred_raw_all,
        "ridge_pred_energy":  ridge_pred_raw_all,
    }).to_csv(pred_csv, index=False, encoding="utf-8-sig")
    torch.save(mlp.state_dict(), os.path.join(OUT_DIR, f"rs_mlp_{name}.pt"))

    del mlp
    if DEVICE == "cuda": torch.cuda.empty_cache()

    return [metrics_mlp, metrics_ridge], merged["block_id"].values, pred_raw_all


# ==============================================================
# 9. 汇总: rs_all_models + final_comparison + metrics_summary_all
# ==============================================================
def aggregate_and_compare(backbones_done, all_pred_dict):
    """rs_all_models.csv: 每 backbone 一列 rs_<bb> (统一前缀)
       final_comparison.csv: 与 sv_all_models.csv 合并, 列前缀 sv_/rs_
    """
    if all_pred_dict.get("block_id") is None:
        print("[aggregate] 无成功 backbone, 跳过.")
        return None

    # 1) rs_all_models.csv (统一 rs_ 前缀)
    rs_dict = {"block_id": all_pred_dict["block_id"]}
    if "split" in all_pred_dict:
        rs_dict["split"] = all_pred_dict["split"]
    if "true_energy" in all_pred_dict:
        rs_dict["true_energy"] = all_pred_dict["true_energy"]
    for bb in backbones_done:
        key = f"{bb}_pred"
        if key in all_pred_dict:
            rs_dict[f"rs_{bb}"] = all_pred_dict[key]
    pd.DataFrame(rs_dict).to_csv(ALL_PRED_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[汇总] 遥感 backbone 预测合并: {ALL_PRED_CSV}")

    # 2) final_comparison.csv: sv 与 rs 合并 (前缀化避免冲突)
    sv_all = os.path.join(OUT_DIR, "sv_all_models.csv")
    if os.path.exists(sv_all):
        sv = pd.read_csv(sv_all)
        # sv 列名形如 'resnet50_mlp', 改名为 'sv_resnet50' 与 rs_<bb> 对齐
        rename_map = {}
        for c in sv.columns:
            if c.endswith("_mlp"):
                rename_map[c] = "sv_" + c[:-4]
        sv = sv.rename(columns=rename_map)
        rs = pd.DataFrame(rs_dict)
        # 合并: 共享 block_id, sv 提供 split/true_energy, rs 提供 rs_<bb>
        common_cols = [c for c in ["split", "true_energy"]
                       if c in sv.columns and c in rs.columns]
        rs_join = rs.drop(columns=common_cols, errors="ignore")
        merged = sv.merge(rs_join, on="block_id", how="inner")
        merged.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
        n_sv = len([c for c in merged.columns if c.startswith("sv_")])
        n_rs = len([c for c in merged.columns if c.startswith("rs_")])
        print(f"[汇总] 最终对比表: {FINAL_CSV}  "
              f"({len(merged)} 街区 / sv {n_sv} / rs {n_rs})")
    else:
        print(f"[汇总] 跳过 final_comparison: 找不到 {sv_all}")


def print_full_comparison(metrics_rs):
    """合并街景 + 遥感 metrics, 兼容街景 v2.1 的新 schema (含 model 列)."""
    sv_metrics_path = os.path.join(OUT_DIR, "sv_metrics_summary.csv")
    rows = []

    if os.path.exists(sv_metrics_path):
        sv_df = pd.read_csv(sv_metrics_path)
        for _, r in sv_df.iterrows():
            row = {"modality": "streetview", **r.to_dict()}
            rows.append(row)

    for m in metrics_rs:
        rows.append({"modality": "remote_sensing", **m})

    if not rows:
        print("[print_full_comparison] 无数据可汇总.")
        return

    df = pd.DataFrame(rows)
    # 选取交集列, 容忍街景旧版本 schema
    cols_pref = ["modality", "backbone", "model", "feat_dim", "val_rmse",
                 "R2_norm", "R2_raw", "R2_log", "RMSE_raw", "MAE_raw"]
    cols = [c for c in cols_pref if c in df.columns]
    df = df[cols]
    df = df.sort_values(["modality", "R2_log"],
                        ascending=[True, False]).reset_index(drop=True)
    df.to_csv(METRICS_ALL, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("           街景 vs 遥感 × 各 backbone × {mlp, ridge}  测试集对比 (按 R²(log) 降序)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print(f"\n[汇总] 总指标表: {METRICS_ALL}")


# ==============================================================
# 10. main
# ==============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="all",
                        choices=ALL_BACKBONES + ["all"])
    parser.add_argument("--force-extract", action="store_true",
                        help="忽略特征缓存, 强制重提")
    args = parser.parse_args()

    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError("先运行 1_build_labels.py")

    np.random.seed(SEED)
    bbs = ALL_BACKBONES if args.backbone == "all" else [args.backbone]
    print(f"将依次跑 {len(bbs)} 个 backbone: {bbs}")
    print(f"运行设备: {DEVICE}\n")

    # 1) 准备遥感影像 (一次性, 与 backbone 无关)
    df_imgs = prepare_rs_images()

    # 2) 计算几何辅助特征
    geom_aux, geom_names = build_geom_aux_features(df_imgs["block_id"].values)
    geom_block_ids = df_imgs["block_id"].astype(int).values
    print(f"        几何特征列: {geom_names}\n")

    # 3) L0 baseline self-check
    labels = pd.read_csv(LABELS_CSV)
    with open(STATS_JSON, encoding="utf-8") as f:
        s = json.load(f)
    mu, sigma = s["mu"], s["sigma"]
    df_lookup = pd.DataFrame({
        "block_id":   geom_block_ids,
        "geom_idx":   np.arange(len(geom_block_ids))
    })
    merged_all = labels.merge(df_lookup, on="block_id", how="inner") \
                       .reset_index(drop=True)
    geom_for_baseline = geom_aux[merged_all["geom_idx"].values]
    print("=" * 60 + "\n L0 BASELINE SELF-CHECK\n" + "=" * 60)
    run_baselines(merged_all, geom_for_baseline, mu, sigma)

    # 4) 跑每个 backbone
    metrics_all = []
    all_pred_dict = {"block_id": None}
    backbones_done = []

    if args.force_extract:
        # 清旧 _v2 缓存
        for bb in bbs:
            cache = os.path.join(OUT_DIR, f"rs_features_block_{bb}_v2.npz")
            if os.path.exists(cache):
                os.remove(cache)
                print(f"[--force-extract] 已删 {os.path.basename(cache)}")

    for bb in bbs:
        try:
            rows, bids, pred_raw = run_one_backbone(
                bb, df_imgs, geom_aux, geom_block_ids)
            metrics_all.extend(rows)
            backbones_done.append(bb)

            if all_pred_dict["block_id"] is None:
                all_pred_dict["block_id"] = bids
                # 加载 split / true_energy 一并存 (供 rs_all_models 用)
                pred_csv = os.path.join(OUT_DIR, f"rs_predictions_{bb}.csv")
                pred_df = pd.read_csv(pred_csv)
                all_pred_dict["split"] = pred_df["split"].values
                all_pred_dict["true_energy"] = pred_df["true_energy"].values
            all_pred_dict[f"{bb}_pred"] = pred_raw
        except Exception as e:
            print(f"!! {bb} 失败: {e}")
            import traceback; traceback.print_exc()

    # 5) rs metrics summary
    if not metrics_all:
        print("\n⚠ 所有 backbone 都失败了, 跳过汇总.")
        return

    sm = pd.DataFrame(metrics_all).sort_values(
        by="R2_log", ascending=False).reset_index(drop=True)
    sm.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 90)
    print("           遥感模型测试集指标对比 (按 R²(log) 降序)")
    print("=" * 90)
    print(sm.to_string(index=False))
    print("=" * 90)
    print(f"\n[汇总] {SUMMARY_CSV}")

    # 6) 汇总到 rs_all_models / final_comparison
    aggregate_and_compare(backbones_done, all_pred_dict)

    # 7) 街景 + 遥感总对比表
    print_full_comparison(metrics_all)

    # 8) 诊断
    if os.path.exists(BASELINE_CSV):
        base = pd.read_csv(BASELINE_CSV)
        geom_r2_log = base.loc[base["model"].str.startswith("ridge_geom"),
                               "R2_log"].values[0]
        best_bb_r2_log = sm["R2_log"].max()
        gain = best_bb_r2_log - geom_r2_log
        print("\n" + "=" * 90)
        print(f"  诊断: 仅几何辅助特征 R²(log) = {geom_r2_log:.4f}")
        print(f"        最佳 backbone R²(log) = {best_bb_r2_log:.4f}")
        print(f"        遥感图像净增益 = {gain:+.4f}")
        if gain < 0.02:
            print("  ⚠ 增益 < 0.02 → 遥感 backbone 几乎没贡献.")
            print("    建议: 检查 ESRI 影像质量 (是否大量云/雪/糊) 或换 SI 图源.")
        elif gain < 0.05:
            print("  ◐ 遥感图像有微弱信号, 远未达论文级 R²~0.4.")
        else:
            print("  ✓ 遥感信号显著, 进入 KnowCL 主流程后还能再涨.")
        print("=" * 90)


if __name__ == "__main__":
    main()

