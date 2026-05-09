# -*- coding: utf-8 -*-
r"""
5_build_building_kg.py  v4.2 (Base + Land/Building 扩展, 沈阳数据)

★ v4.2 相对 v4.1 的改动:
  1. ★ 全部 block_id / land_id 走 base_mod._norm_id 规范化
     ("1.0" / "1" / "1.00" 都折叠成 "1"), 跟 labels.csv 对齐
  2. block_index.tsv 也写规范化后的字符串
  3. 其它逻辑不变

数据源:
  blocks:    沿用 base_mod.BLOCKS_SHP (L4能耗.shp)
  lands:     沿用 base_mod.LAND_SHP   (L5.shp)
  buildings: G:\Knowcl\9-建筑物数据\processed_shenyang20230318.shp
  landtype:  base_mod.LANDTYPE_SHP    (16-地块数据/沈阳市.shp)

新增 6 个关系:
  morphology       (Land,     Land)     sym  9 类标签 (A-I) 同类互连, cap=10/Land
  landFunction     (Land,     Land)     sym  Level1_cn 相同
  orientation      (Land,     Land)     sym  同 block 内主体建筑朝向 4 桶相同
  buildingFunction (Building, Building) sym  同 Land 内 Function 相同, top-5/楼
  belongsToRegion  (Land,     Region)   asym
  belongsToLand    (Building, Land)     asym
"""

from __future__ import annotations
import os, sys, json, argparse, time, math
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# 复用 4_
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "4_build_base_kg.py")
if not os.path.exists(_BASE_PATH):
    raise RuntimeError(f"找不到 {_BASE_PATH}, 请把 4_build_base_kg.py 放在同目录")
base_mod = _load_module("base_kg_v4", _BASE_PATH)

EntRegistry = base_mod.EntRegistry
RelRegistry = base_mod.RelRegistry
add_inverse_relations    = base_mod.add_inverse_relations
read_blocks              = base_mod.read_blocks
read_lands               = base_mod.read_lands
read_landtype            = base_mod.read_landtype
read_poi                 = base_mod.read_poi
sjoin_poi_to_block       = base_mod.sjoin_poi_to_block
sjoin_block_to_landtype  = base_mod.sjoin_block_to_landtype
build_locate_at          = base_mod.build_locate_at
build_poi_to_cate        = base_mod.build_poi_to_cate
build_subcate_of         = base_mod.build_subcate_of
build_block_landtype     = base_mod.build_block_landtype
build_region_borderby    = base_mod.build_region_borderby
build_region_nearby      = base_mod.build_region_nearby
build_region_similarfunc = base_mod.build_region_similarfunc
build_poi_competitive    = base_mod.build_poi_competitive
UTM_CRS                  = base_mod.UTM_CRS
TRAIN_RATIO              = base_mod.TRAIN_RATIO
VAL_RATIO                = base_mod.VAL_RATIO

# ★ ID 规范化(从 4_ 引)
_norm_id        = base_mod._norm_id
_norm_id_series = base_mod._norm_id_series

# ============================================================
# 路径
# ============================================================
ROOT         = base_mod.ROOT
BUILDING_SHP = os.path.join(ROOT, "9-建筑物数据", "processed_shenyang20230318.shp")
OUT_DIR      = os.path.join(ROOT, "999-输出成果文件", "003-知识图谱", "building")

# 5_ 专用超参
MORPH_CAP        = 10
LANDFUNC_CAP     = 20
BLDFUNC_TOPK     = 5

H_LOW_FLOORS  = 3.5
H_MID_FLOORS  = 9.5
M_PER_FLOOR   = 3.0
SLAB_ASPECT   = 2.5
ENCLOSED_NMIN = 4


# ============================================================
# 读建筑物
# ============================================================
def read_buildings(path):
    import geopandas as gpd
    g = None
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            g = gpd.read_file(path, encoding=enc)
            print(f"           [encoding] 使用 {enc} 成功读取", flush=True)
            break
        except Exception:
            continue
    if g is None: g = gpd.read_file(path)
    print(f"[Build-KG] 读取建筑物: {path}", flush=True)
    print(f"           字段 {list(g.columns)} | 数量 {len(g):,}", flush=True)
    keep = [c for c in ["merged_id", "Height", "Function", "Age", "Quality", "建筑ID", "geometry"]
            if c in g.columns]
    g = g[keep].copy()
    if "Function" in g.columns:
        g["Function"] = g["Function"].fillna("__NA__").astype(str)
    if "merged_id" not in g.columns:
        g["merged_id"] = np.arange(len(g))
    g["merged_id"] = _norm_id_series(g["merged_id"])
    return g


# ============================================================
# 1. belongsToRegion (Land within Region)
# ============================================================
def build_belongs_to_region(lands, blocks, ent, rel,
                            land_col="LandID", block_col="BlockID"):
    import geopandas as gpd
    print(f"[Build-KG] 计算 belongsToRegion (Land within Region) ...", flush=True)
    rid = rel.add("land_belongsto_region", sym=False)
    if lands.crs != blocks.crs:
        lands_p = lands.to_crs(blocks.crs)
    else:
        lands_p = lands.copy()
    j = gpd.sjoin(lands_p[[land_col, "geometry"]],
                  blocks[[block_col, "geometry"]],
                  how="left", predicate="within")
    j = j.dropna(subset=[block_col])
    triples = []
    land2blk = {}
    for _, row in j.iterrows():
        u = _norm_id(row[land_col]); b = _norm_id(row[block_col])
        if not u or not b: continue
        triples.append((ent.add("land", u), rid, ent.add("block", b)))
        land2blk[u] = b
    print(f"           belongsToRegion: {len(triples):,}", flush=True)
    return triples, land2blk


# ============================================================
# 2. belongsToLand (Building within Land)
# ============================================================
def build_belongs_to_land(buildings, lands, ent, rel, land_col="LandID"):
    import geopandas as gpd
    print(f"[Build-KG] 计算 belongsToLand (Building within Land) ...", flush=True)
    rid = rel.add("building_belongsto_land", sym=False)
    bldg = buildings.copy()
    if bldg.crs is None: bldg = bldg.set_crs("EPSG:4326")
    if bldg.crs != lands.crs: bldg = bldg.to_crs(lands.crs)
    centers = bldg.copy()
    centers["geometry"] = centers.geometry.representative_point()
    j = gpd.sjoin(centers[["merged_id", "geometry"]],
                  lands[[land_col, "geometry"]],
                  how="left", predicate="within")
    j = j.dropna(subset=[land_col])
    triples = []
    bldg2land = {}
    for _, row in j.iterrows():
        b = _norm_id(row["merged_id"]); u = _norm_id(row[land_col])
        if not b or not u: continue
        triples.append((ent.add("building", b), rid, ent.add("land", u)))
        bldg2land[b] = u
    print(f"           belongsToLand: {len(triples):,}", flush=True)
    return triples, bldg2land


# ============================================================
# 3. morphology — 9 类
# ============================================================
def _classify_morphology(buildings_in_land, land_geom_utm):
    bldg = buildings_in_land
    if bldg is None or len(bldg) == 0: return None

    if "Height" not in bldg.columns: return None
    h = pd.to_numeric(bldg["Height"], errors="coerce").dropna()
    if len(h) == 0: return None
    h_mean = h.mean()
    floors = h_mean / M_PER_FLOOR
    if   floors <= H_LOW_FLOORS:  ht = "low"
    elif floors <= H_MID_FLOORS:  ht = "mid"
    else:                          ht = "high"

    n = len(bldg)
    layout = "point"
    if n >= 1:
        try:
            areas = bldg.geometry.area
            i_max = areas.idxmax()
            main_b = bldg.loc[i_max]
            mbr = main_b.geometry.minimum_rotated_rectangle
            coords = list(mbr.exterior.coords)
            edges = [math.hypot(coords[i+1][0]-coords[i][0],
                                coords[i+1][1]-coords[i][1]) for i in range(4)]
            long_e = max(edges); short_e = max(min(edges), 1e-6)
            aspect = long_e / short_e
        except Exception:
            aspect = 1.0

        if aspect >= SLAB_ASPECT:
            layout = "slab"
        elif n >= ENCLOSED_NMIN:
            lc = land_geom_utm.centroid
            angles = []
            for _, b in bldg.iterrows():
                bc = b.geometry.centroid
                dx = bc.x - lc.x; dy = bc.y - lc.y
                if abs(dx) < 1e-6 and abs(dy) < 1e-6: continue
                angles.append(math.atan2(dy, dx))
            if len(angles) >= ENCLOSED_NMIN:
                angles.sort()
                gaps = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
                gaps.append(2 * math.pi + angles[0] - angles[-1])
                if max(gaps) < math.pi:
                    layout = "enclosed"

    code = {("low", "point"):    "A", ("low", "slab"):    "B", ("low", "enclosed"):    "C",
            ("mid", "point"):    "D", ("mid", "slab"):    "E", ("mid", "enclosed"):    "F",
            ("high","point"):    "G", ("high","slab"):    "H", ("high","enclosed"):    "I"}
    return code.get((ht, layout))


def build_land_morphology(lands, buildings, bldg2land, ent, rel, land_col="LandID"):
    print(f"[Build-KG] 计算 morphology (9 类: A 低-点式 .. I 高-围合) ...", flush=True)
    rid = rel.add("land_morphology_land", sym=True)
    if not bldg2land:
        print("           (没有 bldg2land, 跳过)", flush=True); return [], {}

    lands_p = lands.to_crs(UTM_CRS).reset_index(drop=True)
    bldg = buildings.to_crs(UTM_CRS).copy()
    bldg["__bid"] = _norm_id_series(bldg["merged_id"])
    bldg["__land"] = bldg["__bid"].map(bldg2land)
    bldg = bldg.dropna(subset=["__land"])
    bldg_by_land = dict(iter(bldg.groupby("__land")))

    land_morpho = {}
    for _, row in lands_p.iterrows():
        u = _norm_id(row[land_col])
        sub = bldg_by_land.get(u, None)
        if sub is None or len(sub) == 0: continue
        code = _classify_morphology(sub, row.geometry)
        if code is not None:
            land_morpho[u] = code

    print(f"           已分类 {len(land_morpho):,} / {len(lands_p):,} 个 land", flush=True)
    dist = Counter(land_morpho.values())
    desc = {"A": "低-点式", "B": "低-板式", "C": "低-围合",
            "D": "中-点式", "E": "中-板式", "F": "中-围合",
            "G": "高-点式", "H": "高-板式", "I": "高-围合"}
    for c in "ABCDEFGHI":
        if c in dist:
            print(f"             {c} ({desc[c]}): {dist[c]:,}", flush=True)

    grp = defaultdict(list)
    for u, c in land_morpho.items():
        grp[c].append(u)

    triples = []
    for code, uuids in grp.items():
        m = len(uuids)
        if m < 2: continue
        for i in range(m):
            for k in range(i + 1, min(i + 1 + MORPH_CAP, m)):
                triples.append((ent.add("land", uuids[i]), rid,
                                ent.add("land", uuids[k])))
    print(f"           morphology 边: {len(triples):,}", flush=True)

    rid_attr = rel.add("land_has_morphology_class", sym=False)
    for u, c in land_morpho.items():
        triples.append((ent.add("land", u), rid_attr, ent.add("morpho_class", c)))

    return triples, land_morpho


# ============================================================
# 4. landFunction
# ============================================================
def sjoin_land_to_landtype(lands, landtype, land_col="LandID"):
    import geopandas as gpd
    if landtype is None or "Level1_cn" not in landtype.columns: return {}
    if lands.crs != landtype.crs: landtype = landtype.to_crs(lands.crs)
    print(f"[Build-KG] sjoin: land 重心 -> 用地多边形 (Level1_cn) ...", flush=True)
    centers = lands.copy()
    centers["geometry"] = centers.geometry.representative_point()
    j = gpd.sjoin(centers[[land_col, "geometry"]],
                  landtype[["Level1_cn", "geometry"]],
                  how="left", predicate="within")
    j = j.dropna(subset=["Level1_cn"])
    return dict(zip(_norm_id_series(j[land_col]), j["Level1_cn"].astype(str)))


def build_land_function(land2lt, ent, rel):
    print(f"[Build-KG] 计算 landFunction: Level1_cn 一致 ...", flush=True)
    rid = rel.add("land_landfunction_land", sym=True)
    if not land2lt:
        print("           (无 land Level1_cn, 跳过)", flush=True); return []
    grp = defaultdict(list)
    for u, lt in land2lt.items():
        if lt and lt != "__NA__": grp[lt].append(u)
    triples = []
    for lt, uuids in grp.items():
        m = len(uuids)
        if m < 2: continue
        for i in range(m):
            for k in range(i + 1, min(i + 1 + LANDFUNC_CAP, m)):
                triples.append((ent.add("land", uuids[i]), rid,
                                ent.add("land", uuids[k])))
    print(f"           landFunction 边: {len(triples):,}", flush=True)
    return triples


# ============================================================
# 5. orientation
# ============================================================
def _build_orient_per_land(buildings, bldg2land):
    if buildings is None or "merged_id" not in buildings.columns: return {}
    print(f"[Build-KG]   计算每个 Land 主体建筑朝向 ...", flush=True)
    bldg = buildings.to_crs(UTM_CRS).copy()
    bldg["__area"] = bldg.geometry.area
    bldg["__bid"] = _norm_id_series(bldg["merged_id"])
    bldg["__land"] = bldg["__bid"].map(bldg2land)
    bldg = bldg.dropna(subset=["__land"])
    if len(bldg) == 0:
        print("           (建筑无 land 关联, 跳过)", flush=True)
        return {}
    idx_max = bldg.groupby("__land")["__area"].idxmax()
    rep = bldg.loc[idx_max]
    out = {}
    for _, row in rep.iterrows():
        try:
            mbr = row.geometry.minimum_rotated_rectangle
            coords = list(mbr.exterior.coords)
            edges = []
            for i in range(4):
                dx = coords[i+1][0] - coords[i][0]
                dy = coords[i+1][1] - coords[i][1]
                edges.append((math.hypot(dx, dy), dx, dy))
            edges.sort(reverse=True)
            _, dx, dy = edges[0]
            ang = math.degrees(math.atan2(dy, dx)) % 180
            bucket = int(((ang + 22.5) % 180) // 45)
            out[_norm_id(row["__land"])] = bucket
        except Exception:
            continue
    print(f"           已得到 {len(out):,} 个 land 朝向", flush=True)
    return out


def build_land_orientation(lands, blocks, buildings, bldg2land, ent, rel,
                           land_col="LandID", block_col="BlockID"):
    import geopandas as gpd
    print(f"[Build-KG] 计算 orientation: 同 block 内主体建筑朝向匹配 ...", flush=True)
    rid = rel.add("land_orientation_land", sym=True)
    if not bldg2land:
        print("           (无 bldg2land, 跳过)", flush=True); return []
    orient = _build_orient_per_land(buildings, bldg2land)
    if not orient:
        print("           (无可计算朝向, 跳过)", flush=True); return []

    if lands.crs != blocks.crs:
        lands_p = lands.to_crs(blocks.crs)
    else:
        lands_p = lands.copy()
    centers = lands_p.copy()
    centers["geometry"] = centers.geometry.representative_point()
    j = gpd.sjoin(centers[[land_col, "geometry"]],
                  blocks[[block_col, "geometry"]],
                  how="left", predicate="within")
    j = j.dropna(subset=[block_col])
    blk_groups = defaultdict(list)
    for _, row in j.iterrows():
        u = _norm_id(row[land_col])
        if u in orient:
            blk_groups[_norm_id(row[block_col])].append(u)
    print(f"           {len(blk_groups):,} 个 block 内有 ≥1 个含建筑的 land", flush=True)

    triples = []
    for bid, uuids in blk_groups.items():
        if len(uuids) < 2: continue
        for i in range(len(uuids)):
            for k in range(i + 1, len(uuids)):
                if orient[uuids[i]] == orient[uuids[k]]:
                    triples.append((ent.add("land", uuids[i]), rid,
                                    ent.add("land", uuids[k])))
    print(f"           orientation 边: {len(triples):,}", flush=True)
    return triples


# ============================================================
# 6. buildingFunction
# ============================================================
def build_building_function(buildings, bldg2land, ent, rel):
    print(f"[Build-KG] 计算 buildingFunction: 同 Land 内 Function 一致, top-{BLDFUNC_TOPK}/楼 ...", flush=True)
    if "Function" not in buildings.columns or not bldg2land:
        print("           (跳过)", flush=True); return []
    rid = rel.add("building_buildingfunction_building", sym=True)
    bldg = buildings.copy()
    bldg["__bid"]  = _norm_id_series(bldg["merged_id"])
    bldg["__land"] = bldg["__bid"].map(bldg2land)
    bldg = bldg.dropna(subset=["__land"])
    bldg = bldg[bldg["Function"].astype(str) != "__NA__"]
    triples = []
    for (land, fn), sub in bldg.groupby(["__land", "Function"]):
        bids = sub["__bid"].tolist()
        m = len(bids)
        if m < 2: continue
        for i in range(m):
            for k in range(i + 1, min(i + 1 + BLDFUNC_TOPK, m)):
                triples.append((ent.add("building", bids[i]), rid,
                                ent.add("building", bids[k])))
    print(f"           buildingFunction 边: {len(triples):,}", flush=True)
    return triples


# ============================================================
# building 属性关系
# ============================================================
def build_building_attrs(buildings, ent, rel,
                         bins_height=(10, 20, 30, 50, 100),
                         bins_age=(1980, 1990, 2000, 2010, 2020)):
    triples = []
    if "Function" in buildings.columns:
        rfn = rel.add("building_has_function", sym=False)
        for _, row in buildings.iterrows():
            v = row["Function"]
            bid = _norm_id(row["merged_id"])
            if not bid: continue
            if v and v != "__NA__":
                triples.append((ent.add("building", bid),
                                rfn, ent.add("bld_function", v)))
    if "Height" in buildings.columns:
        rh = rel.add("building_has_height_bin", sym=False)
        for _, row in buildings.iterrows():
            try: h = float(row["Height"])
            except Exception: continue
            if not np.isfinite(h): continue
            bid = _norm_id(row["merged_id"])
            if not bid: continue
            bn = "h_lt_{}".format(bins_height[0])
            for thr in bins_height:
                if h >= thr: bn = "h_ge_{}".format(thr)
            triples.append((ent.add("building", bid),
                            rh, ent.add("height_bin", bn)))
    if "Age" in buildings.columns:
        ra = rel.add("building_has_age_bin", sym=False)
        for _, row in buildings.iterrows():
            try: yr = float(row["Age"])
            except Exception: continue
            if not np.isfinite(yr): continue
            bid = _norm_id(row["merged_id"])
            if not bid: continue
            bn = "y_lt_{}".format(bins_age[0])
            for thr in bins_age:
                if yr >= thr: bn = "y_ge_{}".format(thr)
            triples.append((ent.add("building", bid),
                            ra, ent.add("age_bin", bn)))
    if "Quality" in buildings.columns:
        rq = rel.add("building_has_quality", sym=False)
        for _, row in buildings.iterrows():
            v = row["Quality"]
            if v is None or pd.isna(v): continue
            bid = _norm_id(row["merged_id"])
            if not bid: continue
            triples.append((ent.add("building", bid),
                            rq, ent.add("quality", str(v))))
    print(f"[Build-KG] 建筑物属性关系 (function/height/age/quality): {len(triples):,} 三元组", flush=True)
    return triples


# ============================================================
# block_features (含建筑统计)
# ============================================================
def compute_block_features_with_building(blocks, poi_with_blk, blk2lt,
                                         buildings, out_csv, block_col="BlockID"):
    import geopandas as gpd
    print(f"[Build-KG] 计算 block_features.csv (含建筑统计) ...", flush=True)
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
    print("           sjoin building -> block ...", flush=True)
    bldg = buildings.copy()
    if bldg.crs is None: bldg = bldg.set_crs("EPSG:4326")
    if bldg.crs != blocks.crs: bldg = bldg.to_crs(blocks.crs)
    bldg_c = bldg.copy()
    bldg_c["geometry"] = bldg_c.geometry.representative_point()
    j = gpd.sjoin(bldg_c, blocks[[block_col, "geometry"]], how="left", predicate="within")
    j = j.dropna(subset=[block_col])
    j[block_col] = _norm_id_series(j[block_col])
    bcnt = j.groupby(block_col).size().rename("bld_count")
    feats = feats.merge(bcnt.reset_index(), on=block_col, how="left").fillna({"bld_count": 0})
    if "Height" in j.columns:
        h = pd.to_numeric(j["Height"], errors="coerce")
        j["__h"] = h
        feats = feats.merge(j.groupby(block_col)["__h"].mean().rename("bld_height_mean").reset_index(),
                            on=block_col, how="left").fillna({"bld_height_mean": 0})
        feats = feats.merge(j.groupby(block_col)["__h"].std().rename("bld_height_std").reset_index(),
                            on=block_col, how="left").fillna({"bld_height_std": 0})
    if "Age" in j.columns:
        a = pd.to_numeric(j["Age"], errors="coerce")
        j["__a"] = a
        feats = feats.merge(j.groupby(block_col)["__a"].mean().rename("bld_age_mean").reset_index(),
                            on=block_col, how="left").fillna({"bld_age_mean": 0})
    if "Function" in j.columns:
        fn_counts = j.groupby([block_col, "Function"]).size().unstack(fill_value=0)
        for c in fn_counts.columns[:6]:
            col = f"bld_func_{c}_share"
            tot = fn_counts.sum(axis=1).replace(0, 1)
            share = (fn_counts[c] / tot).rename(col)
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
    print(f"           → {out_csv}  ({feats.shape[0]} rows × {feats.shape[1]} cols)", flush=True)


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
        "n_entities": len(ent), "n_relations": len(rel), "n_triples_total": int(n),
        "n_train": int(len(train)), "n_valid": int(len(valid)), "n_test": int(len(test)),
        "entity_types": dict(Counter(et for _, et, _ in ent.rows)),
        "relation_list": [{"id": rid, "name": rname, "sym": bool(sym)}
                          for rid, rname, sym in rel.rows],
    }
    if extra_meta: stats.update(extra_meta)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[Build-KG] 写出 {out_dir}", flush=True)
    print(f"           entities={len(ent):,}  relations={len(rel)}  triples={n:,}"
          f"  (train={len(train):,} val={len(valid):,} test={len(test):,})", flush=True)


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-only",      action="store_true")
    ap.add_argument("--no-extra-rels",      action="store_true")
    ap.add_argument("--no-inverse",         action="store_true")
    ap.add_argument("--no-building-attrs",  action="store_true")
    ap.add_argument("--block-id-col",       default=None)
    ap.add_argument("--land-id-col",        default=None)
    args = ap.parse_args()

    extra_rels = not args.no_extra_rels
    use_inv    = not args.no_inverse
    bld_attrs  = not args.no_building_attrs
    mode       = "features-only" if args.features_only else "full (KG + features)"
    print(f"[Build-KG v4.2] 模式: {mode} | extra_rels={extra_rels} | inverse={use_inv}"
          f" | bld_attrs={bld_attrs}", flush=True)
    t0 = time.time()

    blocks, blk_id_col = read_blocks(base_mod.BLOCKS_SHP, id_col=args.block_id_col)
    lands,  land_id_col = read_lands(base_mod.LAND_SHP, id_col=args.land_id_col)
    pois      = read_poi(base_mod.POI_SHP)
    buildings = read_buildings(BUILDING_SHP)
    landtype  = read_landtype(base_mod.LANDTYPE_SHP)

    poi_with_blk = sjoin_poi_to_block(pois, blocks)
    blk2lt       = sjoin_block_to_landtype(blocks, landtype)
    print(f"           {len(blk2lt):,} / {len(blocks):,} 个 block 拿到 Level1_cn", flush=True)
    land2lt      = sjoin_land_to_landtype(lands, landtype)
    print(f"           {len(land2lt):,} / {len(lands):,} 个 land 拿到 Level1_cn", flush=True)

    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if args.features_only:
        compute_block_features_with_building(blocks, poi_with_blk, blk2lt, buildings,
                                             os.path.join(out_dir, "block_features.csv"))
        print(f"[Build-KG] features-only 完成, 耗时 {time.time()-t0:.1f}s", flush=True)
        return

    ent = EntRegistry()
    rel = RelRegistry()
    triples = []

    print("[Build-KG] === A. 构造 base KG 关系 ===", flush=True)
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

    print("[Build-KG] === B. 构造 Land/Building 关系 (6 个新关系) ===", flush=True)
    t_btr, land2blk  = build_belongs_to_region(lands, blocks, ent, rel)
    triples += t_btr
    t_btl, bldg2land = build_belongs_to_land(buildings, lands, ent, rel)
    triples += t_btl
    t_morph, _       = build_land_morphology(lands, buildings, bldg2land, ent, rel)
    triples += t_morph
    triples += build_land_function(land2lt, ent, rel)
    triples += build_land_orientation(lands, blocks, buildings, bldg2land, ent, rel)
    triples += build_building_function(buildings, bldg2land, ent, rel)

    if bld_attrs:
        print("[Build-KG] === C. 建筑物属性关系 ===", flush=True)
        triples += build_building_attrs(buildings, ent, rel)

    print(f"[Build-KG] 加 inverse 前: {len(triples):,} 三元组, {len(rel)} 关系", flush=True)

    if use_inv:
        triples = add_inverse_relations(triples, rel)
        print(f"[Build-KG] 加 inverse 后: {len(triples):,} 三元组, {len(rel)} 关系", flush=True)

    split_and_write(triples, ent, rel, out_dir,
                    extra_meta={"block_id_col_in_shp": blk_id_col,
                                "land_id_col_in_shp": land_id_col,
                                "block_shp": base_mod.BLOCKS_SHP,
                                "land_shp": base_mod.LAND_SHP,
                                "morphology_classes": ["A:低-点式", "B:低-板式", "C:低-围合",
                                                       "D:中-点式", "E:中-板式", "F:中-围合",
                                                       "G:高-点式", "H:高-板式", "I:高-围合"]})

    compute_block_features_with_building(blocks, poi_with_blk, blk2lt, buildings,
                                         os.path.join(out_dir, "block_features.csv"))

    print(f"[Build-KG v4.2] 全部完成, 耗时 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
