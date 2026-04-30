# -*- coding: utf-8 -*-
"""
==============================================================
 1_build_labels.py  ——  构建街区能耗标签
==============================================================
读取   : G:\\Knowcl\\8-街区数据\\沈阳L4能耗.shp
字段   : BlockID (主键) , E_Final_W5 (能耗)
处理   : log 变换 → Z-score → 5 分位分层抽样 → 7:1.5:1.5 划分
输出   : G:\\Knowcl\\999-输出成果文件\\002-能耗预测\\
            ├ energy_labels.csv     (block_id, energy_raw, energy_log,
            │                        energy_norm, split, lng, lat)
            ├ label_stats.json      (mu, sigma 用于反归一化)
            └ energy_distribution.png
==============================================================
"""
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============== 配置 ==============
SHP_PATH = r"G:\Knowcl\8-街区数据\沈阳L4能耗.shp"
OUT_DIR  = r"G:\Knowcl\999-输出成果文件\002-能耗预测"
SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print(f"[Step1] 读取 shp:  {SHP_PATH}")
    gdf = gpd.read_file(SHP_PATH, encoding="utf-8")
    print(f"        街区总数:  {len(gdf)}")
    print(f"        所有字段:  {list(gdf.columns)}")

    # ---- 必备字段检查 ----
    for fld in ["BlockID", "E_Final_W5"]:
        if fld not in gdf.columns:
            raise KeyError(f"shp 中缺少字段: {fld}")

    # ---- 投影到 WGS84 取质心 ----
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    centroids = gdf.to_crs("EPSG:4326").geometry.centroid

    df = pd.DataFrame({
        "block_id":   gdf["BlockID"].astype(int).values,
        "energy_raw": pd.to_numeric(gdf["E_Final_W5"], errors="coerce").values,
        "lng":        centroids.x.values,
        "lat":        centroids.y.values,
    })

    # ---- 清洗 ----
    n0 = len(df)
    df = df.dropna(subset=["energy_raw"])
    df = df[df["energy_raw"] > 0]
    df = df.drop_duplicates(subset=["block_id"]).reset_index(drop=True)
    print(f"        过滤无效后剩余: {len(df)} / {n0}")
    print(f"        能耗范围: {df.energy_raw.min():.2f}  ~  {df.energy_raw.max():.2f}")

    # ---- log 变换 ----
    df["energy_log"] = np.log1p(df["energy_raw"])
    df["bin"] = pd.qcut(df["energy_log"], q=5, labels=False, duplicates="drop")

    # ---- 分层划分 7:1.5:1.5 ----
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["bin"], random_state=SEED,
    )
    val_ratio_in_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_in_temp),
        stratify=temp_df["bin"], random_state=SEED,
    )

    df["split"] = "train"
    df.loc[df["block_id"].isin(val_df["block_id"]),  "split"] = "val"
    df.loc[df["block_id"].isin(test_df["block_id"]), "split"] = "test"

    # ---- 用 train 计算均值方差，做 Z-score ----
    mu    = float(train_df["energy_log"].mean())
    sigma = float(train_df["energy_log"].std())
    df["energy_norm"] = (df["energy_log"] - mu) / sigma

    print(f"        train={(df.split=='train').sum()}, "
          f"val={(df.split=='val').sum()}, test={(df.split=='test').sum()}")
    print(f"        log(E) μ={mu:.4f}, σ={sigma:.4f}")

    # ---- 保存 ----
    out_csv = os.path.join(OUT_DIR, "energy_labels.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with open(os.path.join(OUT_DIR, "label_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"mu": mu, "sigma": sigma, "n_total": len(df),
                   "field": "E_Final_W5"}, f, indent=2, ensure_ascii=False)

    # ---- 分布图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["energy_raw"], bins=50, color="#3b82f6", alpha=0.85)
    axes[0].set_title("Raw E_Final_W5"); axes[0].grid(alpha=0.3)
    axes[1].hist(df["energy_log"], bins=50, color="#10b981", alpha=0.85)
    axes[1].set_title("log(1+E_Final_W5)"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "energy_distribution.png"), dpi=150)
    plt.close(fig)

    print(f"\n[Step1] 完成。所有输出位于:\n        {OUT_DIR}")


if __name__ == "__main__":
    main()
