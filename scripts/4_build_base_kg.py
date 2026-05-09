# -*- coding: utf-8 -*-
r"""
4_build_base_kg.py  v4.2 (KnowSite-aligned, 沈阳数据)

★ v4.2 相对 v4.1 的改动:
  1. ★ 新增 _norm_id() 工具: 把 ID (BlockID/LandID) 统一规范化, 防止
     "1.0" / "1" / "1.00" 这种浮点字符串和整型字符串对不上 (跟 labels.csv 对齐)
  2. read_blocks/read_lands 内部全部走 _norm_id, block_index.tsv 写出的也是规范化后的字符串
  3. 其它逻辑不变

数据源:
  blocks:    G:\Knowcl\8-街区数据\沈阳L4能耗.shp   (Region 实体, 街区)
  lands:     G:\Knowcl\8-街区数据\沈阳L5.shp        (Land 实体, 5_ 用)
  landtype:  G:\Knowcl\16-地块数据\沈阳市.shp        (Level1_cn 来源)
  POI:       G:\Knowcl\6-POI数据\merged_poi.shp

base KG 关系 (KnowSite KDD'22 Table 1 子集):
  asym (加 inverse):
    block_locate_at_landtype, block_has_poi,
    poi_has_main_cat, poi_has_sub_cat, poi_has_sub_type,
    sub_cat_subcateof_main_cat, sub_type_subcateof_sub_cat
  sym:
    block_borderby_block (buffer 30m intersects),
    block_nearby_block   (≤1km),
    block_similarfunc_block (POI cos≥0.95, top-8),
    poi_competitive_poi  (同类 ≤500m, top-5)
"""

from __future__ import annotations
import os, sys, json, argparse, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# ============================================================
# 路径配置
# ============================================================
ROOT         = os.environ.get("KNOWCL_ROOT", r"G:\Knowcl")
BLOCKS_SHP   = os.path.join(ROOT, "8-街区数据",  "沈阳L4能耗.shp")
LAND_SHP     = os.path.join(ROOT, "8-街区数据",  "沈阳L5.shp")
LANDTYPE_SHP = os.path.join(ROOT, "16-地块数据", "沈阳市.shp")
POI_SHP      = os.path.join(ROOT, "6-POI数据",   "merged_poi.shp")
OUT_DIR      = os.path.join(ROOT, "999-输出成果文件", "003-知识图谱", "base")

# ============================================================
# 关系超参
# ============================================================
NEARBY_RADIUS_M  = 1000.0
BORDER_BUFFER_M  = 30.0
SIM_FUNC_THR     = 0.95
SIM_FUNC_TOPK    = 8
COMP_RADIUS_M    = 500.0
COMP_TOPK        = 5
UTM_CRS          = "EPSG:32651"

TRAIN_RATIO, VAL_RATIO = 0.90, 0.05

BLOCK_ID_CANDIDATES = ['BlockID', 'block_id', 'BLOCKID', 'blockid',
                       'LandID',  'land_id',  'LANDID',  'landid',
                       'RegionID','region_id','REGIONID',
                       'ID', 'id', 'OBJECTID', 'FID', 'index']
LAND_ID_CANDIDATES  = ['LandID', 'land_id', 'LANDID', 'landid',
                       'UUID', 'uuid', 'ID', 'id', 'OBJECTID', 'FID', 'index']


# ============================================================
# ★★★ ID 规范化: 防止 "1.0" 和 "1" 对不上 ★★★
# ============================================================
def _norm_id(v) -> str:
    """把 ID 转成规范化字符串. 整数型 (1, 1.0, '1', '1.0') 全部 → '1'.
    其它字符串保留 (uuid 等). NaN/None → ''."""
    if v is None: return ""
    try:
        if pd.isna(v): return ""
    except Exception:
        pass
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v): return ""
        if float(v).is_integer():
            return str(int(v))
        return repr(float(v))
    s = str(v).strip()
    if not s or s.lower() == "nan": return ""
    # 尝试 int / float 折叠
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


def _norm_id_series(s: pd.Series) -> pd.Series:
    return s.map(_norm_id)


# ============================================================
# 注册表
# ============================================================
class EntRegistry:
    def __init__(self):
        self.rows = []
        self._idx = {}
    def add(self, etype, name):
        key = (etype, str(name))
        if key in self._idx: return self._idx[key]
        eid = len(self.rows)
        self.rows.append((eid, etype, str(name)))
        self._idx[key] = eid
        return eid
    def get(self, etype, name): return self._idx.get((etype, str(name)), None)
    def __len__(self): return len(self.rows)


class RelRegistry:
    def __init__(self):
        self.rows = []
        self._idx = {}
    def add(self, name, sym=False):
        if name in self._idx: return self._idx[name]
        rid = len(self.rows)
        self.rows.append((rid, name, sym))
        self._idx[name] = rid
        return rid
    def get(self, name): return self._idx.get(name, None)
    def __len__(self): return len(self.rows)


def add_inverse_relations(triples, rel_reg):
    inv_map = {}
    for rid, rname, sym in list(rel_reg.rows):
        if not sym:
            inv_id = rel_reg.add(rname + "_INV", sym=False)
            inv_map[rid] = inv_id
    extra = []
    for h, r, t in triples:
        if r in inv_map:
            extra.append((t, inv_map[r], h))
    return triples + extra


# ============================================================
# 数据读取
# ============================================================
def _read_with_encoding(path):
    import geopandas as gpd
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            g = gpd.read_file(path, encoding=enc)
            return g, enc
        except Exception:
            continue
    return gpd.read_file(path), None


def _pick_id_col(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns: return cand
        if cand.lower() in cols_lower: return cols_lower[cand.lower()]
    return None


def read_blocks(path, id_col=None, std_name="BlockID"):
    g, enc = _read_with_encoding(path)
    print(f"[Base-KG] 读取 blocks: {path}", flush=True)
    if enc and enc != "utf-8": print(f"          [encoding] 使用 {enc}", flush=True)
    print(f"          字段 {list(g.columns)} | 数量 {len(g)}", flush=True)
    if id_col is None:
        id_col = _pick_id_col(g.columns, BLOCK_ID_CANDIDATES)
    if id_col is None:
        print(f"          ⚠ 未找到 ID 列, 用 0-based index 作 {std_name}", flush=True)
        g = g.copy()
        g[std_name] = g.index.astype(str)
        id_col = std_name
    elif id_col != std_name:
        g = g.rename(columns={id_col: std_name})
    print(f"          使用 '{id_col}' 作 block 主键 → 内部统一为 '{std_name}' (规范化)", flush=True)
    g = g[[std_name, "geometry"]].copy()
    g[std_name] = _norm_id_series(g[std_name])  # ★ 规范化
    # 打印一段样例, 让用户自己看一眼
    sample = g[std_name].head(3).tolist()
    print(f"          样例 BlockID: {sample} (规范化后, 共 {g[std_name].nunique()} 个唯一)", flush=True)
    return g, id_col


def read_lands(path, id_col=None, std_name="LandID"):
    g, enc = _read_with_encoding(path)
    print(f"[Base-KG] 读取 lands: {path}", flush=True)
    if enc and enc != "utf-8": print(f"          [encoding] 使用 {enc}", flush=True)
    print(f"          字段 {list(g.columns)} | 数量 {len(g)}", flush=True)
    if id_col is None:
        id_col = _pick_id_col(g.columns, LAND_ID_CANDIDATES)
    if id_col is None:
        g = g.copy()
        g[std_name] = g.index.astype(str)
        id_col = std_name
    elif id_col != std_name:
        g = g.rename(columns={id_col: std_name})
    print(f"          使用 '{id_col}' 作 land 主键 → 内部统一为 '{std_name}' (规范化)", flush=True)
    g = g[[std_name, "geometry"]].copy()
    g[std_name] = _norm_id_series(g[std_name])
    return g, id_col


def read_landtype(path):
    g, enc = _read_with_encoding(path)
    print(f"[Base-KG] 读取用地属性: {path}", flush=True)
    if enc and enc != "utf-8": print(f"          [encoding] 使用 {enc}", flush=True)
    print(f"          字段 {list(g.columns)} | 数量 {len(g)}", flush=True)
    keep = [c for c in ["Level1_cn", "Level1", "Level2", "Level2_cn", "geometry"] if c in g.columns]
    g = g[keep].copy()
    if "Level1_cn" in g.columns:
        g["Level1_cn"] = g["Level1_cn"].fillna("__NA__").astype(str)
    return g


def read_poi(path):
    g, enc = _read_with_encoding(path)
    print(f"[Base-KG] 读取 POI: {path}", flush=True)
    if enc and enc != "utf-8": print(f"          [encoding] 使用 {enc}", flush=True)
    print(f"          字段 {list(g.columns)} | 数量 {len(g)}", flush=True)
    keep = [c for c in ["name", "main_cat", "sub_cat", "sub_type", "geometry"] if c in g.columns]
    g = g[keep].copy()
    for c in ("main_cat", "sub_cat", "sub_type"):
        if c in g.columns:
            g[c] = g[c].fillna("__NA__").astype(str)
    return g


def sjoin_poi_to_block(poi, blocks, block_col="BlockID"):
    import geopandas as gpd
    if poi.crs is None: poi = poi.set_crs("EPSG:4326")
    if blocks.crs is None: blocks = blocks.set_crs("EPSG:4326")
    if poi.crs != blocks.crs:
        poi = poi.to_crs(blocks.crs)
    print(f"[Base-KG] sjoin: POI -> block ...", flush=True)
    j = gpd.sjoin(poi, blocks[[block_col, "geometry"]], how="left", predicate="within")
    if "index_right" in j.columns: j = j.drop(columns=["index_right"])
    # ★ 落入 block 的 BlockID 也走规范化 (sjoin 后可能变成 object/float)
    j[block_col] = _norm_id_series(j[block_col])
    j.loc[j[block_col] == "", block_col] = np.nan
    n_in = j[block_col].notna().sum()
    print(f"          {n_in:,} / {len(poi):,} 个 POI 落到某街区内", flush=True)
    return j


def sjoin_block_to_landtype(blocks, landtype, block_col="BlockID"):
    import geopandas as gpd
    if landtype is None or "Level1_cn" not in landtype.columns: return {}
    if blocks.crs != landtype.crs:
        landtype = landtype.to_crs(blocks.crs)
    print(f"[Base-KG] sjoin: block 重心 -> 用地多边形 ...", flush=True)
    centers = blocks.copy()
    centers["geometry"] = centers.geometry.representative_point()
    j = gpd.sjoin(centers[[block_col, "geometry"]],
                  landtype[["Level1_cn", "geometry"]],
                  how="left", predicate="within")
    j = j.dropna(subset=["Level1_cn"])
    return dict(zip(_norm_id_series(j[block_col]), j["Level1_cn"].astype(str)))


# ============================================================
# 关系: 基础属性
# ============================================================
def build_locate_at(poi_with_blk, ent, rel, block_col="BlockID"):
    rid = rel.add("block_has_poi", sym=False)
    triples = []
    valid = poi_with_blk[poi_with_blk[block_col].notna()].copy()
    for _, row in valid.iterrows():
        bid = _norm_id(row[block_col])
        if not bid: continue
        bid_e = ent.add("block", bid)
        pname = row.get("name", None)
        if pname is None or pd.isna(pname): continue
        pid_e = ent.add("poi", f"{bid}::{pname}")
        triples.append((bid_e, rid, pid_e))
    return triples, valid


def build_poi_to_cate(poi_in_blk, ent, rel, block_col="BlockID"):
    triples = []
    rels = {}
    if "main_cat" in poi_in_blk.columns: rels[1] = rel.add("poi_has_main_cat", sym=False)
    if "sub_cat"  in poi_in_blk.columns: rels[2] = rel.add("poi_has_sub_cat",  sym=False)
    if "sub_type" in poi_in_blk.columns: rels[3] = rel.add("poi_has_sub_type", sym=False)
    for _, row in poi_in_blk.iterrows():
        pname = row.get("name", None)
        if pname is None or pd.isna(pname): continue
        bid = _norm_id(row[block_col])
        if not bid: continue
        pid_e = ent.add("poi", f"{bid}::{pname}")
        for k, col in [(1, "main_cat"), (2, "sub_cat"), (3, "sub_type")]:
            if k in rels:
                c = row[col]
                if c and c != "__NA__":
                    etype = {1: "main_cat", 2: "sub_cat", 3: "sub_type"}[k]
                    triples.append((pid_e, rels[k], ent.add(etype, c)))
    return triples


def build_subcate_of(poi_in_blk, ent, rel):
    triples = []
    if "sub_cat" in poi_in_blk.columns and "main_cat" in poi_in_blk.columns:
        r12 = rel.add("sub_cat_subcateof_main_cat", sym=False)
        df = poi_in_blk.dropna(subset=["sub_cat", "main_cat"])
        df = df[(df["sub_cat"] != "__NA__") & (df["main_cat"] != "__NA__")]
        for _, row in df[["sub_cat", "main_cat"]].drop_duplicates().iterrows():
            triples.append((ent.add("sub_cat", row["sub_cat"]), r12,
                            ent.add("main_cat", row["main_cat"])))
    if "sub_type" in poi_in_blk.columns and "sub_cat" in poi_in_blk.columns:
        r23 = rel.add("sub_type_subcateof_sub_cat", sym=False)
        df = poi_in_blk.dropna(subset=["sub_type", "sub_cat"])
        df = df[(df["sub_type"] != "__NA__") & (df["sub_cat"] != "__NA__")]
        for _, row in df[["sub_type", "sub_cat"]].drop_duplicates().iterrows():
            triples.append((ent.add("sub_type", row["sub_type"]), r23,
                            ent.add("sub_cat", row["sub_cat"])))
    return triples


def build_block_landtype(blocks, blk2lt, ent, rel, block_col="BlockID"):
    if not blk2lt: return []
    rid = rel.add("block_locate_at_landtype", sym=False)
    triples = []
    for bid, lt in blk2lt.items():
        bid = _norm_id(bid)
        if not bid: continue
        if not lt or lt == "__NA__": continue
        triples.append((ent.add("block", bid), rid, ent.add("landtype", lt)))
    return triples


# ============================================================
# 关系: 区域间
# ============================================================
def build_region_borderby(blocks, ent, rel, block_col="BlockID"):
    print(f"[Base-KG] 计算 BorderBy: buffer({BORDER_BUFFER_M}m).intersects ...", flush=True)
    rid = rel.add("block_borderby_block", sym=True)
    blocks_p = blocks.to_crs(UTM_CRS).reset_index(drop=True)
    sindex = blocks_p.sindex
    triples = []
    for i in range(len(blocks_p)):
        gi = blocks_p.geometry.iloc[i]
        gi_buf = gi.buffer(BORDER_BUFFER_M)
        cand = list(sindex.query(gi_buf, predicate="intersects"))
        for j in cand:
            if j <= i: continue
            gj = blocks_p.geometry.iloc[j]
            if gi.contains(gj) or gj.contains(gi): continue
            triples.append((ent.add("block", _norm_id(blocks_p[block_col].iloc[i])), rid,
                            ent.add("block", _norm_id(blocks_p[block_col].iloc[j]))))
    print(f"          BorderBy 边: {len(triples):,}", flush=True)
    return triples


def build_region_nearby(blocks, ent, rel, block_col="BlockID"):
    print(f"[Base-KG] 计算 NearBy: 中心距离 ≤ {NEARBY_RADIUS_M:.1f}m ...", flush=True)
    rid = rel.add("block_nearby_block", sym=True)
    blocks_p = blocks.to_crs(UTM_CRS).reset_index(drop=True)
    centers = blocks_p.geometry.centroid
    coords = np.array([[p.x, p.y] for p in centers])
    n = len(coords)
    centers_gdf = blocks_p.copy()
    centers_gdf["geometry"] = centers
    sindex = centers_gdf.sindex
    triples = []
    for i in range(n):
        cx, cy = coords[i]
        bbox = (cx - NEARBY_RADIUS_M, cy - NEARBY_RADIUS_M,
                cx + NEARBY_RADIUS_M, cy + NEARBY_RADIUS_M)
        for j in sindex.intersection(bbox):
            if j <= i: continue
            d = np.hypot(coords[j, 0] - cx, coords[j, 1] - cy)
            if d > NEARBY_RADIUS_M: continue
            triples.append((ent.add("block", _norm_id(blocks_p[block_col].iloc[i])), rid,
                            ent.add("block", _norm_id(blocks_p[block_col].iloc[j]))))
    print(f"          NearBy 边: {len(triples):,}", flush=True)
    return triples


def build_region_similarfunc(poi_with_blk, ent, rel, block_col="BlockID"):
    print(f"[Base-KG] 计算 SimilarFunction: cos≥{SIM_FUNC_THR}, top-{SIM_FUNC_TOPK}/block ...", flush=True)
    rid = rel.add("block_similarfunc_block", sym=True)
    if "main_cat" not in poi_with_blk.columns:
        print("          (无 main_cat 字段, 跳过)", flush=True); return []
    poi_in = poi_with_blk[poi_with_blk[block_col].notna()].copy()
    if len(poi_in) == 0:
        print("          (无 POI 落入街区, 跳过)", flush=True); return []
    poi_in[block_col] = _norm_id_series(poi_in[block_col])
    poi_in["main_cat"] = poi_in["main_cat"].fillna("__NA__").astype(str)
    cat_counts = (poi_in.groupby([block_col, "main_cat"]).size().unstack(fill_value=0))
    if cat_counts.shape[0] < 2: return []
    M = cat_counts.values.astype(np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    Mn = M / norms
    S = Mn @ Mn.T
    np.fill_diagonal(S, -1.0)
    triples = []
    bids = cat_counts.index.tolist()
    for i, bid_i in enumerate(bids):
        sims = S[i]
        idx = np.argsort(-sims)[:SIM_FUNC_TOPK]
        for j in idx:
            if sims[j] < SIM_FUNC_THR: break
            if j <= i: continue
            triples.append((ent.add("block", bid_i), rid, ent.add("block", bids[j])))
    print(f"          SimilarFunction 边: {len(triples):,}", flush=True)
    return triples


def build_poi_competitive(poi_in_blk, ent, rel, block_col="BlockID"):
    if "main_cat" not in poi_in_blk.columns: return []
    print(f"[Base-KG] 计算 Competitive: 同类 ≤{COMP_RADIUS_M}m, top-{COMP_TOPK}/POI ...", flush=True)
    rid = rel.add("poi_competitive_poi", sym=True)
    pi = poi_in_blk[poi_in_blk[block_col].notna()].copy()
    pi = pi[pi["main_cat"].fillna("__NA__") != "__NA__"]
    if len(pi) == 0: return []
    pi = pi.to_crs(UTM_CRS).reset_index(drop=True)
    pi[block_col] = _norm_id_series(pi[block_col])
    pi["__cx"] = pi.geometry.x
    pi["__cy"] = pi.geometry.y
    triples = []
    seen = set()
    for cat, sub in pi.groupby("main_cat"):
        if len(sub) < 2 or len(sub) > 20000: continue
        coords = sub[["__cx", "__cy"]].values
        cell = COMP_RADIUS_M
        gx = (coords[:, 0] // cell).astype(int)
        gy = (coords[:, 1] // cell).astype(int)
        bucket = defaultdict(list)
        for i in range(len(sub)): bucket[(gx[i], gy[i])].append(i)
        names = sub["name"].fillna("").astype(str).values
        bids = sub[block_col].astype(str).values
        for i in range(len(sub)):
            cands = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cands.extend(bucket.get((gx[i] + dx, gy[i] + dy), []))
            scored = []
            for j in cands:
                if j == i: continue
                d = np.hypot(coords[j, 0] - coords[i, 0], coords[j, 1] - coords[i, 1])
                if d <= COMP_RADIUS_M: scored.append((d, j))
            scored.sort()
            for _, j in scored[:COMP_TOPK]:
                a, b = sorted([i, j])
                key = (a, b, cat)
                if key in seen: continue
                seen.add(key)
                if not names[a] or not names[b]: continue
                triples.append((ent.add("poi", f"{bids[a]}::{names[a]}"), rid,
                                ent.add("poi", f"{bids[b]}::{names[b]}")))
    print(f"          Competitive 边: {len(triples):,}", flush=True)
    return triples


# ============================================================
# block 手工特征
# ============================================================
def compute_block_features(blocks, poi_with_blk, blk2lt, out_csv, block_col="BlockID"):
    print(f"[Base-KG] 计算 block_features.csv ...", flush=True)
    blocks_p = blocks.to_crs(UTM_CRS).copy()
    area = blocks_p.geometry.area
    feats = pd.DataFrame({block_col: _norm_id_series(blocks[block_col]).values,
                          "area_m2": area.values})
    valid = poi_with_blk[poi_with_blk[block_col].notna()].copy()
    valid[block_col] = _norm_id_series(valid[block_col])
    cnt = valid.groupby(block_col).size().rename("poi_count")
    feats = feats.merge(cnt.reset_index(), on=block_col, how="left").fillna({"poi_count": 0})
    feats["poi_density"] = feats["poi_count"] / (feats["area_m2"] + 1e-6) * 1e4
    if "main_cat" in valid.columns:
        cat_counts = (valid.groupby([block_col, "main_cat"]).size().unstack(fill_value=0))
        p = cat_counts.div(cat_counts.sum(axis=1).replace(0, 1), axis=0)
        ent_h = -(p * np.log(p.replace(0, 1))).sum(axis=1).rename("cat_diversity")
        feats = feats.merge(ent_h.reset_index(), on=block_col, how="left").fillna({"cat_diversity": 0})
        top_cats = cat_counts.sum(axis=0).sort_values(ascending=False).head(15).index.tolist()
        for c in top_cats:
            col = f"cat_{c}_share"
            share = (cat_counts[c] / cat_counts.sum(axis=1).replace(0, 1)).rename(col)
            feats = feats.merge(share.reset_index(), on=block_col, how="left").fillna({col: 0.0})
    if blk2lt:
        lt_series = pd.Series(blk2lt, name="landtype")
        top_lts = lt_series.value_counts().head(10).index.tolist()
        df_lt = pd.DataFrame({block_col: list(blk2lt.keys()),
                              "landtype": list(blk2lt.values())})
        for lt in top_lts:
            col = f"landtype_{lt}"
            df_lt[col] = (df_lt["landtype"] == lt).astype(int)
        df_lt = df_lt.drop(columns=["landtype"])
        feats = feats.merge(df_lt, on=block_col, how="left").fillna(0)
    feats.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"          → {out_csv}  ({feats.shape[0]} rows × {feats.shape[1]} cols)", flush=True)


# ============================================================
# 写出
# ============================================================
def split_and_write(triples, ent, rel, out_dir, seed=42, extra_meta=None):
    os.makedirs(out_dir, exist_ok=True)
    arr = np.array(sorted(set(triples)), dtype=np.int64)
    rng = np.random.default_rng(seed); rng.shuffle(arr)
    n = len(arr)
    n_tr = int(n * TRAIN_RATIO); n_va = int(n * VAL_RATIO)
    train, valid, test = arr[:n_tr], arr[n_tr:n_tr+n_va], arr[n_tr+n_va:]
    np.savetxt(os.path.join(out_dir, "train.tsv"), train, fmt="%d", delimiter="\t")
    np.savetxt(os.path.join(out_dir, "valid.tsv"), valid, fmt="%d", delimiter="\t")
    np.savetxt(os.path.join(out_dir, "test.tsv"),  test,  fmt="%d", delimiter="\t")
    with open(os.path.join(out_dir, "entities.tsv"), "w", encoding="utf-8") as f:
        for eid, etype, name in ent.rows:
            f.write(f"{eid}\t{etype}\t{name}\n")
    with open(os.path.join(out_dir, "relations.tsv"), "w", encoding="utf-8") as f:
        for rid, rname, sym in rel.rows:
            f.write(f"{rid}\t{rname}\t{int(sym)}\n")
    with open(os.path.join(out_dir, "block_index.tsv"), "w", encoding="utf-8") as f:
        for eid, etype, name in ent.rows:
            if etype == "block":
                f.write(f"{name}\t{eid}\n")
    stats = {
        "n_entities": len(ent),
        "n_relations": len(rel),
        "n_triples_total": int(n),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test":  int(len(test)),
        "entity_types": dict(Counter(et for _, et, _ in ent.rows)),
        "relation_list": [{"id": rid, "name": rname, "sym": bool(sym)}
                          for rid, rname, sym in rel.rows],
    }
    if extra_meta: stats.update(extra_meta)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[Base-KG] 写出 {out_dir}", flush=True)
    print(f"          entities={len(ent):,}  relations={len(rel)}  triples={n:,}"
          f"  (train={len(train):,} val={len(valid):,} test={len(test):,})", flush=True)


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--no-extra-rels", action="store_true")
    ap.add_argument("--no-inverse",    action="store_true")
    ap.add_argument("--block-id-col",  default=None,
                    help="手动指定 blocks ID 列 (覆盖自动识别)")
    args = ap.parse_args()

    extra_rels = not args.no_extra_rels
    use_inv    = not args.no_inverse
    mode       = "features-only" if args.features_only else "full (KG + features)"
    print(f"[Base-KG v4.2] 模式: {mode} | extra_rels={extra_rels} | inverse={use_inv}", flush=True)
    t0 = time.time()

    blocks, blk_id_col = read_blocks(BLOCKS_SHP, id_col=args.block_id_col)
    pois     = read_poi(POI_SHP)
    landtype = read_landtype(LANDTYPE_SHP)

    poi_with_blk = sjoin_poi_to_block(pois, blocks)
    blk2lt       = sjoin_block_to_landtype(blocks, landtype)
    print(f"          {len(blk2lt):,} / {len(blocks):,} 个 block 拿到 Level1_cn", flush=True)

    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if args.features_only:
        compute_block_features(blocks, poi_with_blk, blk2lt,
                               os.path.join(out_dir, "block_features.csv"))
        print(f"[Base-KG] features-only 完成, 耗时 {time.time()-t0:.1f}s", flush=True)
        return

    ent = EntRegistry()
    rel = RelRegistry()
    triples = []

    print("[Base-KG] 构造 block-POI / POI-Cate ...", flush=True)
    t_loc, valid_poi = build_locate_at(poi_with_blk, ent, rel)
    triples += t_loc
    triples += build_poi_to_cate(valid_poi, ent, rel)
    triples += build_block_landtype(blocks, blk2lt, ent, rel)

    if extra_rels:
        triples += build_subcate_of(valid_poi, ent, rel)
        triples += build_region_borderby(blocks, ent, rel)
        triples += build_region_nearby(blocks, ent, rel)
        triples += build_region_similarfunc(poi_with_blk, ent, rel)
        triples += build_poi_competitive(valid_poi, ent, rel)

    print(f"[Base-KG] 加 inverse 前: {len(triples):,} 三元组, {len(rel)} 关系", flush=True)

    if use_inv:
        triples = add_inverse_relations(triples, rel)
        print(f"[Base-KG] 加 inverse 后: {len(triples):,} 三元组, {len(rel)} 关系", flush=True)

    split_and_write(triples, ent, rel, out_dir,
                    extra_meta={"block_id_col_in_shp": blk_id_col,
                                "block_shp": BLOCKS_SHP})

    compute_block_features(blocks, poi_with_blk, blk2lt,
                           os.path.join(out_dir, "block_features.csv"))

    print(f"[Base-KG v4.2] 全部完成, 耗时 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
