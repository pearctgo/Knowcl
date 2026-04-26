# -*- coding: utf-8 -*-
"""
collect_streetview_baidu_full.py
================================

沈阳 L4 街区街景全量重采脚本。

本版本已针对你的本机目录做了修复：
- 项目根目录默认：G:/Knowcl
- L4 街区 shp 默认：G:/Knowcl/8-街区数据/沈阳L4能耗.shp
- 街区 ID 字段默认：BlockID
- 支持从 G:/Knowcl/.env、脚本上级目录 .env、当前工作目录 .env 读取 DATA_ROOT
- 修复原脚本 build_config 中 cfg_yaml["_DATA_ROOT"] 与 load_paths_config 中 "_data_root" 键名不一致的问题
- 若 config/paths.yaml 不存在，也会自动使用默认路径运行

用法：

    conda activate cs
    python G:/Knowcl/888-代码/scripts/街景采集.py --max-points 50

或：

    cd /d G:/Knowcl
    conda activate cs
    python 888-代码/scripts/街景采集.py --max-points 50

依赖：

    pip install geopandas pandas numpy requests pillow shapely pyproj tqdm pyyaml

注意：mapsv0.bdimg.com 是百度地图前端内部端点，非官方公开 API。请仅用于学术研究，不要商用或转分发下载图片。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import yaml
from PIL import Image, ImageStat
from shapely.geometry import Point
from tqdm import tqdm


# =============================================================================
# 0. 路径与 .env
# =============================================================================

# 你的本机项目根目录。即使 .env 没加载成功，也会回退到这里。
FALLBACK_DATA_ROOT = Path(r"G:/Knowcl")

# 假设脚本在 <repo>/888-代码/scripts/ 或 <repo>/scripts/ 下。
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

# 优先猜测项目根目录：
# 1) 如果脚本路径形如 G:/Knowcl/888-代码/scripts/街景采集.py，则 parents[2] 是 G:/Knowcl
# 2) 如果脚本路径形如 G:/Knowcl/scripts/xxx.py，则 parents[1] 是 G:/Knowcl
_possible_roots = []
try:
    _possible_roots.append(SCRIPT_PATH.parents[2])
except Exception:
    pass
try:
    _possible_roots.append(SCRIPT_PATH.parents[1])
except Exception:
    pass
_possible_roots.append(Path.cwd())
_possible_roots.append(FALLBACK_DATA_ROOT)

REPO_ROOT = next((p for p in _possible_roots if (p / ".env").exists() or (p / "config").exists()), FALLBACK_DATA_ROOT)
DEFAULT_CONFIG = REPO_ROOT / "config" / "paths.yaml"


def load_dotenv(path: Path) -> None:
    """极简 .env 读取器：只处理 KEY=VALUE，不覆盖已存在环境变量。"""
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8")

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def load_all_possible_dotenv() -> None:
    """从多个常见位置加载 .env，避免在 C:/Users/Administrator 下运行时找不到 .env。"""
    candidates = []
    for root in _possible_roots:
        candidates.append(root / ".env")
    candidates.append(FALLBACK_DATA_ROOT / ".env")
    candidates.append(Path.cwd() / ".env")

    seen = set()
    for p in candidates:
        p = p.resolve() if p.exists() else p
        if str(p) in seen:
            continue
        seen.add(str(p))
        load_dotenv(p)


def expand_env_vars(value: Any) -> Any:
    """展开 ${DATA_ROOT}、%DATA_ROOT% 等环境变量。"""
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def choose_data_root(cfg: Dict[str, Any]) -> Path:
    """确定 DATA_ROOT。优先级：配置文件 > 环境变量 > G:/Knowcl。"""
    raw = cfg.get("data_root") or os.environ.get("DATA_ROOT") or str(FALLBACK_DATA_ROOT)
    raw = expand_env_vars(raw)
    p = Path(str(raw)).expanduser()
    # 不使用 strict resolve，避免路径暂时不存在时报错。
    return p.resolve() if p.exists() else p


def load_paths_config(config_path: Path) -> Dict[str, Any]:
    """读取 config/paths.yaml；不存在也没关系，使用默认路径。"""
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        cfg = {}

    data_root = choose_data_root(cfg)

    # 同时写入两个键，兼容旧代码里大小写写错的问题。
    cfg["_data_root"] = data_root
    cfg["_DATA_ROOT"] = data_root
    return cfg


load_all_possible_dotenv()


# =============================================================================
# 1. 配置
# =============================================================================

@dataclass
class Config:
    # --- 数据路径 ---
    data_root: Path
    blocks_shp: Path
    output_root: Path
    output_image_dir: Path
    output_table_dir: Path

    # --- 可选相关路径：本脚本当前主要用 blocks_shp，但先保留，方便后续扩展 ---
    lands_shp: Path
    buildings_shp: Path
    poi_shp: Path
    remote_sensing_tif: Path
    streetview_source_dir: Path

    # --- 字段名 ---
    block_id_field: str = "BlockID"
    land_id_field: str = "LandID"
    energy_fields: Tuple[str, ...] = ("E_Final_W5", "Energy")

    # --- CRS ---
    storage_crs: str = "EPSG:4326"
    compute_crs: str = "EPSG:32651"  # 沈阳 UTM Zone 51N

    # --- 百度内部端点 ---
    qsdata_url: str = "https://mapsv0.bdimg.com/"
    pr3d_url: str = "https://mapsv0.bdimg.com/"
    referer: str = "https://map.baidu.com/"

    # --- 图片参数 ---
    image_width: int = 480
    image_height: int = 320
    fov: int = 90
    pitch: int = 0
    headings: Tuple[int, ...] = (0, 90, 180, 270)

    # --- 采样策略 ---
    max_candidate_points_per_block: int = 4
    grid_spacing_m: int = 200
    include_centroid: bool = True
    include_representative_point: bool = True

    # --- 反爬 ---
    sleep_between_points_sec: float = 2.0
    sleep_between_requests_sec: float = 0.3
    request_timeout: int = 20
    max_retries: int = 3
    retry_backoff_sec: float = 2.0

    # --- 上限保护 ---
    max_points_this_run: Optional[int] = None
    dry_run: bool = False

    random_seed: int = 2026


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="沈阳 L4 街景全量重采，百度内部端点 mapsv0.bdimg.com")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data-root", type=Path, default=None,
                   help="可选：手动指定 DATA_ROOT，例如 G:/Knowcl。优先级高于 .env 和 paths.yaml")
    p.add_argument("--blocks-shp", type=Path, default=None,
                   help="可选：手动指定 L4 街区 shp，例如 G:/Knowcl/8-街区数据/沈阳L4能耗.shp")
    p.add_argument("--block-id-field", type=str, default=None,
                   help="可选：街区 ID 字段，默认 BlockID")
    p.add_argument("--max-points", type=int, default=None,
                   help="本次最多处理多少个候选点。None = 全跑")
    p.add_argument("--dry-run", action="store_true",
                   help="只生成候选点，不真正请求百度街景")
    p.add_argument("--candidates-per-block", type=int, default=4)
    p.add_argument("--grid-spacing-m", type=int, default=200)
    p.add_argument("--sleep-between-points", type=float, default=2.0,
                   help="每个候选点之间暂停秒数，默认 2s，太快可能触发风控")
    p.add_argument("--blocks-filter-csv", type=Path, default=None,
                   help="可选：只采特定 block_id 列表，CSV 需含 BlockID 列")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg_yaml = load_paths_config(args.config)

    if args.data_root is not None:
        data_root = args.data_root.expanduser()
        data_root = data_root.resolve() if data_root.exists() else data_root
    else:
        data_root = cfg_yaml.get("_data_root") or cfg_yaml.get("_DATA_ROOT") or FALLBACK_DATA_ROOT

    def _resolve(rel_or_abs: Any) -> Path:
        rel_or_abs = expand_env_vars(rel_or_abs)
        p = Path(str(rel_or_abs)).expanduser()
        return p if p.is_absolute() else data_root / p

    blocks_shp = args.blocks_shp if args.blocks_shp is not None else _resolve(
        cfg_yaml.get("blocks_shp", "8-街区数据/沈阳L4能耗.shp")
    )
    blocks_shp = blocks_shp.expanduser()

    output_root = _resolve(cfg_yaml.get(
        "streetview_recollect_output",
        "999-输出成果文件/001-街景重采_baidu",
    ))

    # 下面这些路径当前采街景不一定用到，但根据你给出的数据目录写入配置，后续拼接多模态数据时可直接用。
    lands_shp = _resolve(cfg_yaml.get("lands_shp", "8-街区数据/沈阳L5.shp"))
    buildings_shp = _resolve(cfg_yaml.get("buildings_shp", "9-建筑物数据/沈阳建筑物三环.shp"))
    poi_shp = _resolve(cfg_yaml.get("poi_shp", "6-POI数据/merged_poi.shp"))
    remote_sensing_tif = _resolve(cfg_yaml.get("remote_sensing_tif", "11-卫星数据/影像下载_2503152313.tif"))
    streetview_source_dir = _resolve(cfg_yaml.get("streetview_source_dir", "12-街景文件"))

    block_id_field = args.block_id_field or cfg_yaml.get("block_id_field", "BlockID")
    land_id_field = cfg_yaml.get("land_id_field", "LandID")
    energy_fields = tuple(cfg_yaml.get("energy_fields", ["E_Final_W5", "Energy"]))

    return Config(
        data_root=data_root,
        blocks_shp=blocks_shp,
        output_root=output_root,
        output_image_dir=output_root / "images",
        output_table_dir=output_root / "tables",
        lands_shp=lands_shp,
        buildings_shp=buildings_shp,
        poi_shp=poi_shp,
        remote_sensing_tif=remote_sensing_tif,
        streetview_source_dir=streetview_source_dir,
        block_id_field=block_id_field,
        land_id_field=land_id_field,
        energy_fields=energy_fields,
        max_candidate_points_per_block=args.candidates_per_block,
        grid_spacing_m=args.grid_spacing_m,
        sleep_between_points_sec=args.sleep_between_points,
        max_points_this_run=args.max_points,
        dry_run=args.dry_run,
    )


# =============================================================================
# 2. 坐标转换：WGS84 -> GCJ02 -> BD09LL -> BD09MC，纯 Python，无 API 调用
# =============================================================================

_PI = 3.14159265358979324
_X_PI = _PI * 3000.0 / 180.0
_A = 6378245.0
_EE = 0.00669342162296594323


def _t_lat(x: float, y: float) -> float:
    r = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    r += (20.0 * math.sin(y * _PI) + 40.0 * math.sin(y / 3.0 * _PI)) * 2.0 / 3.0
    r += (160.0 * math.sin(y / 12.0 * _PI) + 320 * math.sin(y * _PI / 30.0)) * 2.0 / 3.0
    return r


def _t_lng(x: float, y: float) -> float:
    r = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    r += (20.0 * math.sin(x * _PI) + 40.0 * math.sin(x / 3.0 * _PI)) * 2.0 / 3.0
    r += (150.0 * math.sin(x / 12.0 * _PI) + 300.0 * math.sin(x / 30.0 * _PI)) * 2.0 / 3.0
    return r


def wgs84_to_gcj02(lng: float, lat: float) -> Tuple[float, float]:
    dlat = _t_lat(lng - 105.0, lat - 35.0)
    dlng = _t_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * _PI
    magic = 1 - _EE * math.sin(radlat) ** 2
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * _PI)
    dlng = (dlng * 180.0) / (_A / sqrtmagic * math.cos(radlat) * _PI)
    return lng + dlng, lat + dlat


def gcj02_to_bd09ll(lng: float, lat: float) -> Tuple[float, float]:
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * _X_PI)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * _X_PI)
    return z * math.cos(theta) + 0.0065, z * math.sin(theta) + 0.006


_LLBAND = [75.0, 60.0, 45.0, 30.0, 15.0, 0.0]
_LL2MC = [
    [-0.0015702102444, 111320.7020616939, 1704480524535203, -10338987376042340,
     26112667856603880, -35149669176653700, 26595700718403920, -10725012454188240,
     1800819912950474, 82.5],
    [0.0008277824516172526, 111320.7020463578, 647795574.6671607, -4082003173.641316,
     10774905663.51142, -15171875531.51559, 12053065338.62167, -5124939663.577472,
     913311935.9512032, 67.5],
    [0.00337398766765, 111320.7020202162, 4481351.045890365, -23393751.19931662,
     79682215.47186455, -115964993.2797253, 97236711.15602145, -43661946.33752821,
     8477230.501135234, 52.5],
    [0.00220636496208, 111320.7020209128, 51751.86112841131, 3796837.749470245,
     992013.7397791013, -1221952.21711287, 1340652.697009075, -620943.6990984312,
     144416.9293806241, 37.5],
    [-0.0003441963504368392, 111320.7020576856, 278.2353980772752, 2485758.690035394,
     6070.750963243378, 54821.18345352118, 9540.606633304236, -2710.55326746645,
     1405.483844121726, 22.5],
    [-0.0003218135878613132, 111320.7020701615, 0.00369383431289, 823725.6402795718,
     0.46104986909093, 2351.343141331292, 1.58060784298199, 8.77738589078284,
     0.37238884252424, 7.45],
]


def _convertor(x: float, y: float, cE: List[float]) -> Tuple[float, float]:
    xt = cE[0] + cE[1] * abs(x)
    cc = abs(y) / cE[9]
    yt = (cE[2] + cE[3] * cc + cE[4] * cc ** 2 + cE[5] * cc ** 3 +
          cE[6] * cc ** 4 + cE[7] * cc ** 5 + cE[8] * cc ** 6)
    sx = 1 if x >= 0 else -1
    sy = 1 if y >= 0 else -1
    return xt * sx, yt * sy


def bd09ll_to_bd09mc(lng: float, lat: float) -> Tuple[float, float]:
    cE: Optional[List[float]] = None
    for i, band in enumerate(_LLBAND):
        if lat >= band:
            cE = _LL2MC[i]
            break
    if cE is None:
        for i in range(len(_LLBAND) - 1, -1, -1):
            if lat <= -_LLBAND[i]:
                cE = _LL2MC[i]
                break
    if cE is None:
        cE = _LL2MC[-1]
    return _convertor(lng, lat, cE)


def wgs84_to_bd09mc(lng: float, lat: float) -> Tuple[float, float]:
    g_lng, g_lat = wgs84_to_gcj02(lng, lat)
    bd_lng, bd_lat = gcj02_to_bd09ll(g_lng, g_lat)
    return bd09ll_to_bd09mc(bd_lng, bd_lat)


# =============================================================================
# 3. 工具函数
# =============================================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def normalize_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def format_coord(x: float) -> str:
    return f"{x:.8f}".rstrip("0").rstrip(".")


def make_image_filename(lng: float, lat: float, heading: int, pitch: int) -> str:
    return f"{format_coord(lng)}_{format_coord(lat)}_{int(heading)}_{int(pitch)}.jpg"


def make_point_id(block_id: str, lng: float, lat: float, method: str, order: int) -> str:
    raw = f"{block_id}|{lng:.8f}|{lat:.8f}|{method}|{order}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"P_{normalize_id(block_id)}_{order:03d}_{h}"


def quick_image_is_valid(path: Path) -> Tuple[bool, str]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im2:
            w, h = im2.size
            if w < 10 or h < 10:
                return False, "image_too_small"
            gray = im2.convert("L").resize((64, 64))
            stat = ImageStat.Stat(gray)
            mean, std = float(stat.mean[0]), float(stat.stddev[0])
            if std < 1.0 or mean < 1.0 or mean > 254.0:
                return False, "blank_image"
        return True, "ok"
    except Exception as e:
        return False, f"image_open_failed:{type(e).__name__}"


def print_path_diagnostics(cfg: Config) -> None:
    """启动时打印关键路径，方便快速定位路径问题。"""
    checks = [
        ("DATA_ROOT", cfg.data_root),
        ("L4 街区 shp", cfg.blocks_shp),
        ("L5 地块 shp", cfg.lands_shp),
        ("建筑物 shp", cfg.buildings_shp),
        ("POI shp", cfg.poi_shp),
        ("遥感影像 tif", cfg.remote_sensing_tif),
        ("既有街景目录", cfg.streetview_source_dir),
        ("输出目录", cfg.output_root),
    ]
    print("路径检查:")
    for name, path in checks:
        exists = path.exists()
        mark = "OK" if exists or name == "输出目录" else "MISSING"
        print(f"  [{mark}] {name}: {path}")


# =============================================================================
# 4. 街区与候选点
# =============================================================================

def load_l4_blocks(cfg: Config) -> gpd.GeoDataFrame:
    if not cfg.blocks_shp.exists():
        raise FileNotFoundError(
            f"L4 街区 shp 不存在: {cfg.blocks_shp}\n"
            f"请检查：\n"
            f"  1. DATA_ROOT 是否为 G:/Knowcl，当前为: {cfg.data_root}\n"
            f"  2. .env 是否位于 G:/Knowcl/.env，且包含 DATA_ROOT=G:/Knowcl\n"
            f"  3. 或运行时加参数：--blocks-shp G:/Knowcl/8-街区数据/沈阳L4能耗.shp"
        )

    blocks = gpd.read_file(cfg.blocks_shp)
    if cfg.block_id_field not in blocks.columns:
        raise ValueError(
            f"L4 shp 缺少字段 {cfg.block_id_field}. 实际列: {list(blocks.columns)}"
        )
    if blocks.crs is None:
        raise ValueError("L4 shp 无 CRS，请先修复元数据")

    blocks = blocks.copy()
    blocks[cfg.block_id_field] = blocks[cfg.block_id_field].map(normalize_id)
    blocks = blocks[blocks.geometry.notna()].copy()
    invalid = ~blocks.geometry.is_valid
    if invalid.any():
        blocks.loc[invalid, "geometry"] = blocks.loc[invalid, "geometry"].buffer(0)
    return blocks


def regular_grid_within_polygon(poly, spacing_m: float) -> List[Point]:
    minx, miny, maxx, maxy = poly.bounds
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        return []
    xs = np.arange(minx, maxx + spacing_m, spacing_m)
    ys = np.arange(miny, maxy + spacing_m, spacing_m)
    points = []
    for x in xs:
        for y in ys:
            p = Point(float(x), float(y))
            if poly.contains(p):
                points.append(p)
    return points


def generate_candidate_points(blocks: gpd.GeoDataFrame, cfg: Config) -> pd.DataFrame:
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    blocks_metric = blocks.to_crs(cfg.compute_crs)
    rows: List[Dict[str, Any]] = []

    for _, row in tqdm(blocks_metric.iterrows(), total=len(blocks_metric), desc="Generate candidate points"):
        block_id = normalize_id(row[cfg.block_id_field])
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        cands: List[Tuple[Point, str]] = []
        if cfg.include_representative_point:
            try:
                cands.append((geom.representative_point(), "representative_point"))
            except Exception:
                pass
        if cfg.include_centroid:
            try:
                c = geom.centroid
                if geom.contains(c):
                    cands.append((c, "centroid"))
            except Exception:
                pass
        try:
            for p in regular_grid_within_polygon(geom, cfg.grid_spacing_m):
                cands.append((p, "grid"))
        except Exception:
            pass

        # 米级去重
        dedup: Dict[Tuple[int, int], Tuple[Point, str]] = {}
        for p, m in cands:
            key = (int(round(p.x)), int(round(p.y)))
            if key not in dedup:
                dedup[key] = (p, m)
        cands = list(dedup.values())

        priority = {"representative_point": 0, "centroid": 1, "grid": 2}
        cands.sort(key=lambda x: priority.get(x[1], 99))
        if len(cands) > cfg.max_candidate_points_per_block:
            must = [x for x in cands if x[1] in {"representative_point", "centroid"}]
            rest = [x for x in cands if x[1] == "grid"]
            random.shuffle(rest)
            cands = (must + rest)[: cfg.max_candidate_points_per_block]
        if not cands:
            continue

        cand_gdf = gpd.GeoDataFrame(
            [{"method": m, "geometry": p} for p, m in cands],
            crs=cfg.compute_crs,
        ).to_crs(cfg.storage_crs)

        for order, c in enumerate(cand_gdf.itertuples(), start=1):
            lng, lat = float(c.geometry.x), float(c.geometry.y)
            rows.append({
                "BlockID": block_id,
                "point_id": make_point_id(block_id, lng, lat, str(c.method), order),
                "lng_wgs": lng,
                "lat_wgs": lat,
                "method": str(c.method),
                "order_in_block": order,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 5. 百度内部端点请求
# =============================================================================

@dataclass
class PointResult:
    block_id: str
    point_id: str
    lng_wgs: float
    lat_wgs: float
    bd09mc_x: float
    bd09mc_y: float
    panoid: Optional[str]
    panoid_status: str
    images_attempted: int
    images_success: int
    images_skipped_existing: int
    images_failed: int
    elapsed_sec: float
    timestamp: str


_PANOID_PATTERN = re.compile(r'"id"\s*:\s*"([^"]+)"')


def _make_session(cfg: Config) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": cfg.referer,
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })
    return s


def get_panoid(session: requests.Session, cfg: Config, bd09mc_x: float, bd09mc_y: float) -> Tuple[Optional[str], str]:
    """查询百度全景 panoid。返回 (panoid, status)。"""
    params = {
        "qt": "qsdata",
        "x": f"{bd09mc_x:.2f}",
        "y": f"{bd09mc_y:.2f}",
        "l": "14",
        "action": "0",
        "mode": "day",
    }
    last_err = ""
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = session.get(cfg.qsdata_url, params=params, timeout=cfg.request_timeout)
            if resp.status_code != 200:
                last_err = f"http_{resp.status_code}"
                time.sleep(cfg.retry_backoff_sec * attempt)
                continue
            text = resp.text
            m = _PANOID_PATTERN.search(text)
            if m:
                return m.group(1), "ok"
            return None, "no_panoid"
        except Exception as e:
            last_err = f"{type(e).__name__}:{e}"
            time.sleep(cfg.retry_backoff_sec * attempt)
    return None, f"panoid_error:{last_err}"


def download_one_panorama(session: requests.Session, cfg: Config, panoid: str, heading: int, output_path: Path) -> Tuple[str, int, str]:
    """下载一张全景图。返回 (status, bytes, error)。"""
    if output_path.exists():
        ok, _ = quick_image_is_valid(output_path)
        if ok:
            return "skipped_existing", int(output_path.stat().st_size), ""
        try:
            output_path.unlink()
        except Exception:
            pass

    params = {
        "qt": "pr3d",
        "fovy": cfg.fov,
        "quality": 100,
        "panoid": panoid,
        "heading": heading,
        "pitch": cfg.pitch,
        "width": cfg.image_width,
        "height": cfg.image_height,
    }

    last_err = ""
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = session.get(cfg.pr3d_url, params=params, timeout=cfg.request_timeout)
            if resp.status_code != 200:
                last_err = f"http_{resp.status_code}"
                time.sleep(cfg.retry_backoff_sec * attempt)
                continue

            ctype = (resp.headers.get("Content-Type") or "").lower()
            content = resp.content or b""
            if "image" not in ctype:
                last_err = f"non_image:{ctype}"
                time.sleep(cfg.retry_backoff_sec * attempt)
                continue
            if len(content) < 1000:
                last_err = "too_small"
                time.sleep(cfg.retry_backoff_sec * attempt)
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = output_path.with_suffix(".tmp")
            tmp.write_bytes(content)
            ok, reason = quick_image_is_valid(tmp)
            if not ok:
                last_err = reason
                try:
                    tmp.unlink()
                except Exception:
                    pass
                time.sleep(cfg.retry_backoff_sec * attempt)
                continue
            tmp.replace(output_path)
            return "success", len(content), ""

        except Exception as e:
            last_err = f"{type(e).__name__}:{e}"
            time.sleep(cfg.retry_backoff_sec * attempt)

    return "failed", 0, last_err


def process_one_point(session: requests.Session, cfg: Config, row: pd.Series) -> Tuple[PointResult, List[Dict[str, Any]]]:
    """处理一个候选点，返回点级摘要和 4 张图的明细记录。"""
    t0 = time.time()
    block_id = normalize_id(row["BlockID"])
    point_id = str(row["point_id"])
    lng_wgs = float(row["lng_wgs"])
    lat_wgs = float(row["lat_wgs"])

    bd09mc_x, bd09mc_y = wgs84_to_bd09mc(lng_wgs, lat_wgs)

    image_records: List[Dict[str, Any]] = []

    block_dir = cfg.output_image_dir / f"Block_{block_id}"
    expected = {
        h: block_dir / make_image_filename(lng_wgs, lat_wgs, h, cfg.pitch)
        for h in cfg.headings
    }

    # 4 张图全已存在，跳过 panoid 查询。
    if all(p.exists() and quick_image_is_valid(p)[0] for p in expected.values()):
        for h, p in expected.items():
            image_records.append({
                "block_id": block_id,
                "point_id": point_id,
                "lng_wgs": lng_wgs,
                "lat_wgs": lat_wgs,
                "panoid": "(cached)",
                "heading": h,
                "output_path": str(p),
                "status": "skipped_existing",
                "bytes": int(p.stat().st_size),
                "error": "",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        return PointResult(
            block_id=block_id,
            point_id=point_id,
            lng_wgs=lng_wgs,
            lat_wgs=lat_wgs,
            bd09mc_x=bd09mc_x,
            bd09mc_y=bd09mc_y,
            panoid="(cached)",
            panoid_status="cached",
            images_attempted=0,
            images_success=0,
            images_skipped_existing=len(cfg.headings),
            images_failed=0,
            elapsed_sec=time.time() - t0,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ), image_records

    if cfg.dry_run:
        return PointResult(
            block_id=block_id,
            point_id=point_id,
            lng_wgs=lng_wgs,
            lat_wgs=lat_wgs,
            bd09mc_x=bd09mc_x,
            bd09mc_y=bd09mc_y,
            panoid=None,
            panoid_status="dry_run",
            images_attempted=0,
            images_success=0,
            images_skipped_existing=0,
            images_failed=0,
            elapsed_sec=time.time() - t0,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ), image_records

    panoid, status = get_panoid(session, cfg, bd09mc_x, bd09mc_y)
    if not panoid:
        return PointResult(
            block_id=block_id,
            point_id=point_id,
            lng_wgs=lng_wgs,
            lat_wgs=lat_wgs,
            bd09mc_x=bd09mc_x,
            bd09mc_y=bd09mc_y,
            panoid=None,
            panoid_status=status,
            images_attempted=0,
            images_success=0,
            images_skipped_existing=0,
            images_failed=0,
            elapsed_sec=time.time() - t0,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ), image_records

    n_success = n_skip = n_fail = 0
    for h in cfg.headings:
        out_path = expected[h]
        st, n_bytes, err = download_one_panorama(session, cfg, panoid, h, out_path)
        if st == "success":
            n_success += 1
        elif st == "skipped_existing":
            n_skip += 1
        else:
            n_fail += 1
        image_records.append({
            "block_id": block_id,
            "point_id": point_id,
            "lng_wgs": lng_wgs,
            "lat_wgs": lat_wgs,
            "panoid": panoid,
            "heading": h,
            "output_path": str(out_path),
            "status": st,
            "bytes": n_bytes,
            "error": err,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        if cfg.sleep_between_requests_sec > 0:
            time.sleep(cfg.sleep_between_requests_sec)

    return PointResult(
        block_id=block_id,
        point_id=point_id,
        lng_wgs=lng_wgs,
        lat_wgs=lat_wgs,
        bd09mc_x=bd09mc_x,
        bd09mc_y=bd09mc_y,
        panoid=panoid,
        panoid_status="ok",
        images_attempted=len(cfg.headings),
        images_success=n_success,
        images_skipped_existing=n_skip,
        images_failed=n_fail,
        elapsed_sec=time.time() - t0,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ), image_records


def collect_all(cfg: Config, candidates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """主采集循环。返回 (points_log, images_log)。"""
    if cfg.max_points_this_run is not None:
        plan = candidates.head(int(cfg.max_points_this_run)).copy()
    else:
        plan = candidates.copy()

    session = _make_session(cfg)
    point_records: List[Dict[str, Any]] = []
    image_records_all: List[Dict[str, Any]] = []

    pbar = tqdm(total=len(plan), desc="Collect points")
    for i, (_, row) in enumerate(plan.iterrows()):
        try:
            pres, imgs = process_one_point(session, cfg, row)
            point_records.append(asdict(pres))
            image_records_all.extend(imgs)
        except Exception as e:
            point_records.append({
                "block_id": normalize_id(row["BlockID"]),
                "point_id": str(row["point_id"]),
                "lng_wgs": float(row["lng_wgs"]),
                "lat_wgs": float(row["lat_wgs"]),
                "bd09mc_x": 0.0,
                "bd09mc_y": 0.0,
                "panoid": None,
                "panoid_status": f"exception:{type(e).__name__}:{e}",
                "images_attempted": 0,
                "images_success": 0,
                "images_skipped_existing": 0,
                "images_failed": 0,
                "elapsed_sec": 0.0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        pbar.update(1)
        if not cfg.dry_run and cfg.sleep_between_points_sec > 0:
            time.sleep(cfg.sleep_between_points_sec)

        if (i + 1) % 50 == 0:
            cfg.output_table_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(point_records).to_csv(
                cfg.output_table_dir / "points_log_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(image_records_all).to_csv(
                cfg.output_table_dir / "images_log_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )
            session.headers["User-Agent"] = random.choice(USER_AGENTS)

    pbar.close()
    return pd.DataFrame(point_records), pd.DataFrame(image_records_all)


# =============================================================================
# 6. 输出汇总
# =============================================================================

def build_streetview_index(cfg: Config, images_log: pd.DataFrame) -> pd.DataFrame:
    """成功的图片汇总成 streetview_index.csv，供后续建模阶段消费。"""
    cols = ["BlockID", "image_path", "lng", "lat", "heading", "pitch", "panoid", "source"]
    if images_log.empty:
        return pd.DataFrame(columns=cols)
    ok = images_log[images_log["status"].isin(["success", "skipped_existing"])].copy()
    if ok.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "BlockID": ok["block_id"].astype(str),
        "image_path": ok["output_path"].astype(str),
        "lng": ok["lng_wgs"].astype(float),
        "lat": ok["lat_wgs"].astype(float),
        "heading": ok["heading"].astype(int),
        "pitch": cfg.pitch,
        "panoid": ok["panoid"].astype(str),
        "source": "baidu_mapsv0_pr3d",
    })
    return out.sort_values(["BlockID", "image_path"]).reset_index(drop=True)


def summarize(cfg: Config, blocks: gpd.GeoDataFrame, candidates: pd.DataFrame,
              points_log: pd.DataFrame, images_log: pd.DataFrame,
              sv_index: pd.DataFrame) -> Dict[str, Any]:
    s: Dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(cfg.data_root),
        "blocks_shp": str(cfg.blocks_shp),
        "blocks_total": int(len(blocks)),
        "candidates_total": int(len(candidates)),
        "points_processed": int(len(points_log)),
        "images_records": int(len(images_log)),
        "dry_run": cfg.dry_run,
        "max_points_this_run": cfg.max_points_this_run,
    }
    if len(points_log):
        ps = points_log["panoid_status"].value_counts().to_dict()
        s["panoid_status_counts"] = {str(k): int(v) for k, v in ps.items()}
        ok_panoid = (points_log["panoid_status"].isin(["ok", "cached"])).sum()
        s["panoid_hit_rate"] = round(float(ok_panoid) / max(1, len(points_log)), 4)
    if len(images_log):
        ist = images_log["status"].value_counts().to_dict()
        s["image_status_counts"] = {str(k): int(v) for k, v in ist.items()}
        sent = images_log[~images_log["status"].isin(["skipped_existing"])]
        if len(sent):
            s["image_success_rate_excl_cached"] = round(
                float((sent["status"] == "success").sum()) / len(sent), 4
            )
    if len(sv_index):
        s["covered_blocks_after"] = int(sv_index["BlockID"].nunique())
        per_block = sv_index.groupby("BlockID").size()
        s["images_per_block_median"] = int(per_block.median())
        s["images_per_block_max"] = int(per_block.max())
        s["images_per_block_min"] = int(per_block.min())
    return s


def write_readme(cfg: Config, summary: Dict[str, Any]) -> None:
    lines = [
        f"# 沈阳 L4 街景全量重采记录 · {summary['generated_at']}",
        "",
        "## 方法",
        "",
        "使用百度地图前端内部端点 `mapsv0.bdimg.com`，不需要百度开放平台 AK。",
        "",
        "Pipeline:",
        "1. WGS84 → BD09MC，本地纯 Python 坐标转换",
        "2. mapsv0/qt=qsdata → panoid",
        "3. mapsv0/qt=pr3d?panoid=...&heading=... → JPEG × 4",
        "",
        "## 配置",
        "",
        f"- DATA_ROOT: `{cfg.data_root}`",
        f"- L4 街区 shp: `{cfg.blocks_shp}`",
        f"- 街区 ID 字段: `{cfg.block_id_field}`",
        f"- 输出目录: `{cfg.output_root}`",
        f"- 图片尺寸: `{cfg.image_width}×{cfg.image_height}` heading {list(cfg.headings)}",
        f"- 每 L4 候选点: `{cfg.max_candidate_points_per_block}` 网格 `{cfg.grid_spacing_m}m`",
        f"- 点间停顿: `{cfg.sleep_between_points_sec}s`",
        f"- dry_run: `{cfg.dry_run}`",
        "",
        "## 摘要",
        "",
    ]
    for k, v in summary.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        "",
        "## 输出文件",
        "",
        "- `tables/candidate_points_all_l4.csv`",
        "- `tables/points_log.csv`",
        "- `tables/images_log.csv`",
        "- `tables/streetview_index.csv`",
        "- `tables/collection_summary.json`",
        "",
        "## 注意",
        "",
        "- mapsv0.bdimg.com 是百度内部端点，非官方文档化，后端调整可能导致脚本失效。",
        "- `panoid_status=no_panoid` 表示该坐标无全景，常见于小区或园区内部，不一定是 bug。",
        "- 若大量 `http_403/418`，通常是触发风控，可增大 `--sleep-between-points` 或更换网络。",
        "- 仅供学术研究使用。",
    ]
    (cfg.output_root / "README_collection.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 7. 主程序
# =============================================================================

def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    cfg.output_image_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_table_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("沈阳 L4 街景全量重采 · 百度内部端点 mapsv0.bdimg.com")
    print(f"  开始时间   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  REPO_ROOT  : {REPO_ROOT}")
    print(f"  DATA_ROOT  : {cfg.data_root}")
    print(f"  L4 shp     : {cfg.blocks_shp}")
    print(f"  输出目录   : {cfg.output_root}")
    print(f"  DRY_RUN    : {cfg.dry_run}")
    print(f"  max_points : {cfg.max_points_this_run}")
    print(f"  sleep      : 点间 {cfg.sleep_between_points_sec}s · 图间 {cfg.sleep_between_requests_sec}s")
    print("=" * 90)
    print_path_diagnostics(cfg)
    print("=" * 90)

    print("[1/4] 读取 L4 街区 ...")
    blocks = load_l4_blocks(cfg)
    print(f"      L4 街区数: {len(blocks):,}")

    if args.blocks_filter_csv:
        wl = pd.read_csv(args.blocks_filter_csv)
        if "BlockID" not in wl.columns:
            raise ValueError("blocks_filter_csv 需含 BlockID 列")
        keep = set(wl["BlockID"].map(normalize_id))
        blocks = blocks[blocks[cfg.block_id_field].isin(keep)].copy()
        print(f"      过滤后: {len(blocks):,}")

    print("[2/4] 生成候选点 ...")
    candidates = generate_candidate_points(blocks, cfg)
    if candidates.empty:
        raise RuntimeError("没有生成任何候选点。请检查街区几何、CRS 或 grid_spacing_m。")

    candidates.to_csv(
        cfg.output_table_dir / "candidate_points_all_l4.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"      候选点总数: {len(candidates):,}，覆盖 {candidates['BlockID'].nunique()} 块")

    print("[3/4] 执行采集 ...")
    points_log, images_log = collect_all(cfg, candidates)
    points_log.to_csv(
        cfg.output_table_dir / "points_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    images_log.to_csv(
        cfg.output_table_dir / "images_log.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[4/4] 汇总 streetview_index ...")
    sv_index = build_streetview_index(cfg, images_log)
    sv_index.to_csv(
        cfg.output_table_dir / "streetview_index.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"      新覆盖街区: {sv_index['BlockID'].nunique() if len(sv_index) else 0}")
    print(f"      可用图像数: {len(sv_index):,}")

    summary = summarize(cfg, blocks, candidates, points_log, images_log, sv_index)
    (cfg.output_table_dir / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(cfg, summary)

    print("=" * 90)
    print("[DONE] 完成。摘要:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] 用户中断。partial log 已保留，重跑会自动续采。")
    except Exception:
        print("[FATAL] 运行失败:")
        print(traceback.format_exc())
        sys.exit(1)
