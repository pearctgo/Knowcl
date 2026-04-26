# -*- coding: utf-8 -*-
"""
check_data_rebuild.py
=====================

沈阳街区能耗预测 · 原始数据重构前检查脚本

目标：
1. 固定使用 L4 街区作为预测单元，主键字段 BlockID
2. 固定使用 E_Final_W5 作为能耗标签
3. 检查 L5 地块、建筑物、POI、整城遥感 TIF、街景 JPG 是否可用于后续重构
4. 只输出 1-2 个文件：
   - data_check_report.md
   - data_check_summary.json，可通过 WRITE_JSON_SUMMARY 控制

运行：
    python check_data_rebuild.py

依赖：
    pip install geopandas pandas numpy rasterio pillow shapely pyproj

说明：
    本脚本只做检查和报告，不修改原始数据，不裁剪影像，不生成训练集。
"""

from __future__ import annotations

import json
import math
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# =============================================================================
# 用户配置区
# =============================================================================

DATA_ROOT = Path(r"G:\Knowcl")

PATHS = {
    "lands_l5": DATA_ROOT / r"8-街区数据\沈阳L5.shp",
    "blocks_l4": DATA_ROOT / r"8-街区数据\沈阳L4能耗.shp",
    "buildings": DATA_ROOT / r"9-建筑物数据\沈阳建筑物三环.shp",
    "poi": DATA_ROOT / r"6-POI数据\merged_poi.shp",
    "satellite": DATA_ROOT / r"11-卫星数据\影像下载_2503152313.tif",
    "streetview_dir": DATA_ROOT / r"12-街景文件",
}

OUTPUT_DIR = DATA_ROOT / r"999-输出成果文件\001-dataset_rebuild\00_data_check"

# 只输出 1 个文件时改为 False
WRITE_JSON_SUMMARY = True

# 固定字段
BLOCK_ID_FIELD = "BlockID"
LAND_ID_FIELD = "LandID"
ENERGY_FIELD = "E_Final_W5"

# 空间计算 CRS。沈阳大致位于 UTM 51N，适合面积、距离、相交计算。
COMPUTE_CRS = "EPSG:32651"
STORAGE_CRS = "EPSG:4326"

# 街景文件名格式：lng_lat_heading_pitch.jpg，例如 123.323981_41.71932101_180_0.jpg
STREETVIEW_PATTERN = re.compile(
    r"^(?P<lng>-?\d+(?:\.\d+)?)_"
    r"(?P<lat>-?\d+(?:\.\d+)?)_"
    r"(?P<heading>-?\d+(?:\.\d+)?)_"
    r"(?P<pitch>-?\d+(?:\.\d+)?)$"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_PREVIEW_ROWS = 20


# =============================================================================
# 依赖导入
# =============================================================================

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("[FATAL] 缺少 pandas/numpy，请先执行: pip install pandas numpy")

try:
    import geopandas as gpd
except ImportError:
    sys.exit("[FATAL] 缺少 geopandas，请先执行: pip install geopandas")

try:
    import rasterio
except ImportError:
    sys.exit("[FATAL] 缺少 rasterio，请先执行: pip install rasterio")

try:
    from PIL import Image, ImageStat
except ImportError:
    sys.exit("[FATAL] 缺少 pillow，请先执行: pip install pillow")

try:
    from shapely.geometry import box
except ImportError:
    sys.exit("[FATAL] 缺少 shapely，请先执行: pip install shapely")


# =============================================================================
# 工具函数
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if math.isnan(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def pct(x: float) -> str:
    return f"{x:.1%}"


def format_num(x: Any, digits: int = 4) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def read_vector(path: Path, name: str, report: "Report") -> Optional[gpd.GeoDataFrame]:
    if not path.exists():
        report.error(f"{name} 文件不存在：`{path}`")
        return None

    try:
        gdf = gpd.read_file(path)
        report.ok(f"{name} 读取成功：{len(gdf):,} 条记录。")
        return gdf
    except Exception as e:
        report.error(f"{name} 读取失败：{e}")
        report.code(traceback.format_exc(), "text")
        return None


def repair_geometry(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, int, int]:
    """尽量修复 invalid geometry。"""
    out = gdf.copy()
    before_invalid = int((~out.geometry.is_valid).sum())
    empty_count = int(out.geometry.is_empty.sum())

    if before_invalid > 0:
        try:
            out["geometry"] = out.geometry.buffer(0)
        except Exception:
            pass

    after_invalid = int((~out.geometry.is_valid).sum())
    fixed = before_invalid - after_invalid
    return out, fixed, after_invalid + empty_count


def to_compute_crs(gdf: gpd.GeoDataFrame, report: "Report", layer_name: str) -> Optional[gpd.GeoDataFrame]:
    if gdf.crs is None:
        report.warn(f"{layer_name} CRS 为空，无法可靠转换到 {COMPUTE_CRS}。")
        return None
    try:
        return gdf.to_crs(COMPUTE_CRS)
    except Exception as e:
        report.error(f"{layer_name} 转换到 {COMPUTE_CRS} 失败：{e}")
        return None


def normalize_id_series(s: pd.Series) -> pd.Series:
    """统一 ID 为字符串，去空白；数值型 1.0 变 1。"""
    def _norm(x: Any) -> str:
        text = safe_str(x)
        if not text:
            return ""
        try:
            f = float(text)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
        return text

    return s.map(_norm)


def numeric_stats(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {
            "count": 0,
            "missing": int(series.isna().sum()),
        }

    return {
        "count": int(len(s)),
        "missing": int(series.isna().sum()),
        "min": float(s.min()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
        "skew": float(s.skew()) if len(s) > 2 else 0.0,
        "zero_count": int((s == 0).sum()),
        "negative_count": int((s < 0).sum()),
    }


def parse_streetview_filename(path: Path) -> Optional[Dict[str, Any]]:
    m = STREETVIEW_PATTERN.match(path.stem)
    if not m:
        return None
    try:
        return {
            "lng": float(m.group("lng")),
            "lat": float(m.group("lat")),
            "heading": float(m.group("heading")),
            "pitch": float(m.group("pitch")),
        }
    except Exception:
        return None


def image_quick_check(path: Path) -> Dict[str, Any]:
    """快速检查图片是否可打开、大小、是否近似全黑/全白。"""
    result = {
        "can_open": False,
        "width": None,
        "height": None,
        "mode": None,
        "mean": None,
        "stddev": None,
        "looks_blank": None,
        "error": None,
    }

    try:
        with Image.open(path) as im:
            result["can_open"] = True
            result["width"] = int(im.width)
            result["height"] = int(im.height)
            result["mode"] = im.mode
            small = im.convert("L").resize((64, 64))
            stat = ImageStat.Stat(small)
            mean = float(stat.mean[0])
            stddev = float(stat.stddev[0])
            result["mean"] = mean
            result["stddev"] = stddev
            result["looks_blank"] = bool(stddev < 3.0 or mean < 3.0 or mean > 252.0)
    except Exception as e:
        result["error"] = str(e)

    return result


# =============================================================================
# 报告类
# =============================================================================

@dataclass
class Report:
    output_dir: Path
    lines: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def h1(self, text: str) -> None:
        self.lines.extend(["", f"# {text}", ""])

    def h2(self, text: str) -> None:
        self.lines.extend(["", f"## {text}", ""])

    def h3(self, text: str) -> None:
        self.lines.extend(["", f"### {text}", ""])

    def p(self, text: str = "") -> None:
        self.lines.append(str(text))

    def ok(self, text: str) -> None:
        self.lines.append(f"- ✅ {text}")

    def warn(self, text: str) -> None:
        self.lines.append(f"- ⚠️ {text}")
        self.issues.append(f"[WARN] {text}")

    def error(self, text: str) -> None:
        self.lines.append(f"- ❌ {text}")
        self.issues.append(f"[ERROR] {text}")

    def code(self, text: str, lang: str = "") -> None:
        self.lines.append(f"```{lang}")
        self.lines.append(str(text))
        self.lines.append("```")
        self.lines.append("")

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        if not headers:
            return
        self.lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        self.lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            values = list(row)
            if len(values) < len(headers):
                values += [""] * (len(headers) - len(values))
            if len(values) > len(headers):
                values = values[: len(headers)]
            clean = [str(x).replace("\n", "<br>") for x in values]
            self.lines.append("| " + " | ".join(clean) + " |")
        self.lines.append("")

    def render(self) -> str:
        return "\n".join(self.lines)


# =============================================================================
# 检查模块
# =============================================================================

def check_paths(report: Report) -> None:
    report.h2("1. 路径与文件存在性")

    rows = []
    for key, path in PATHS.items():
        rows.append([key, str(path), "存在" if path.exists() else "缺失"])
        report.summary[f"path_{key}"] = str(path)
        report.summary[f"path_{key}_exists"] = path.exists()

    report.table(["数据项", "路径", "状态"], rows)

    missing = [key for key, path in PATHS.items() if not path.exists()]
    if missing:
        report.error(f"存在缺失路径：{missing}")
    else:
        report.ok("所有配置路径均存在。")


def check_l4_blocks(report: Report) -> Optional[gpd.GeoDataFrame]:
    report.h2("2. L4 街区与能耗标签检查")

    gdf = read_vector(PATHS["blocks_l4"], "L4 街区", report)
    if gdf is None:
        return None

    report.summary["l4_rows"] = int(len(gdf))
    report.summary["l4_crs"] = str(gdf.crs)
    report.summary["l4_columns"] = list(gdf.columns)

    report.p(f"- 字段列表：`{list(gdf.columns)}`")
    report.p(f"- CRS：`{gdf.crs}`")

    if BLOCK_ID_FIELD not in gdf.columns:
        report.error(f"L4 缺少街区 ID 字段 `{BLOCK_ID_FIELD}`。")
        return gdf

    block_ids = normalize_id_series(gdf[BLOCK_ID_FIELD])
    unique_count = int(block_ids.nunique())
    empty_count = int((block_ids == "").sum())
    duplicate_count = int(len(block_ids) - unique_count - empty_count)

    report.summary["l4_blockid_unique"] = unique_count
    report.summary["l4_blockid_empty"] = empty_count
    report.summary["l4_blockid_duplicate_count"] = duplicate_count

    if empty_count:
        report.error(f"L4 中 `{BLOCK_ID_FIELD}` 有 {empty_count} 个空值。")
    else:
        report.ok(f"L4 `{BLOCK_ID_FIELD}` 无空值。")

    if duplicate_count:
        duplicated = block_ids[block_ids.duplicated(keep=False)].value_counts().head(MAX_PREVIEW_ROWS)
        report.error(f"L4 `{BLOCK_ID_FIELD}` 存在重复。重复样例：{duplicated.to_dict()}")
    else:
        report.ok(f"L4 `{BLOCK_ID_FIELD}` 唯一，街区数：{unique_count:,}。")

    if ENERGY_FIELD not in gdf.columns:
        report.error(f"L4 缺少指定能耗字段 `{ENERGY_FIELD}`。后续不能构建标签。")
    else:
        stats = numeric_stats(gdf[ENERGY_FIELD])
        report.summary["energy_field"] = ENERGY_FIELD
        report.summary["energy_stats"] = stats

        rows = [[k, format_num(v)] for k, v in stats.items()]
        report.h3(f"能耗字段 `{ENERGY_FIELD}` 统计")
        report.table(["指标", "值"], rows)

        if stats.get("negative_count", 0) > 0:
            report.error(f"`{ENERGY_FIELD}` 存在负值：{stats['negative_count']} 条，不能直接 log1p。")
        else:
            report.ok(f"`{ENERGY_FIELD}` 不存在负值，可用于 log1p 标签变换。")

        if stats.get("missing", 0) > 0:
            report.warn(f"`{ENERGY_FIELD}` 存在缺失值：{stats['missing']} 条。")
        else:
            report.ok(f"`{ENERGY_FIELD}` 无缺失。")

        if stats.get("zero_count", 0) > 0:
            zero_ratio = stats["zero_count"] / max(stats.get("count", 1), 1)
            report.warn(f"`{ENERGY_FIELD}` 零值数量：{stats['zero_count']}，零值比例：{pct(zero_ratio)}。需确认是否真实零能耗。")

        if abs(float(stats.get("skew", 0.0))) > 3:
            report.warn(f"`{ENERGY_FIELD}` 偏度较高：{stats.get('skew'):.2f}。训练时建议使用 log1p。")

    fixed_gdf, fixed_count, remaining_bad = repair_geometry(gdf)
    report.summary["l4_geometry_fixed_count"] = fixed_count
    report.summary["l4_geometry_remaining_bad_count"] = remaining_bad

    if fixed_count:
        report.warn(f"L4 geometry 发现并尝试修复 {fixed_count} 个 invalid geometry。")
    if remaining_bad:
        report.error(f"L4 仍有 {remaining_bad} 个 invalid 或 empty geometry。")
    else:
        report.ok("L4 geometry 有效。")

    l4_compute = to_compute_crs(fixed_gdf, report, "L4")
    if l4_compute is not None:
        areas = l4_compute.geometry.area
        report.summary["l4_area_m2_stats"] = numeric_stats(areas)
        report.h3("L4 面积统计，单位 m²")
        report.table(["指标", "值"], [[k, format_num(v)] for k, v in numeric_stats(areas).items()])

    return fixed_gdf


def check_l5_lands(report: Report, l4: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    report.h2("3. L5 地块检查与 L5→L4 预匹配")

    gdf = read_vector(PATHS["lands_l5"], "L5 地块", report)
    if gdf is None:
        return None

    report.summary["l5_rows"] = int(len(gdf))
    report.summary["l5_crs"] = str(gdf.crs)
    report.summary["l5_columns"] = list(gdf.columns)
    report.p(f"- 字段列表：`{list(gdf.columns)}`")
    report.p(f"- CRS：`{gdf.crs}`")

    if LAND_ID_FIELD not in gdf.columns:
        report.error(f"L5 缺少地块 ID 字段 `{LAND_ID_FIELD}`。")
    else:
        land_ids = normalize_id_series(gdf[LAND_ID_FIELD])
        unique_count = int(land_ids.nunique())
        empty_count = int((land_ids == "").sum())
        duplicate_count = int(len(land_ids) - unique_count - empty_count)
        report.summary["l5_landid_unique"] = unique_count
        report.summary["l5_landid_empty"] = empty_count
        report.summary["l5_landid_duplicate_count"] = duplicate_count

        if empty_count:
            report.error(f"L5 `{LAND_ID_FIELD}` 有 {empty_count} 个空值。")
        else:
            report.ok(f"L5 `{LAND_ID_FIELD}` 无空值。")
        if duplicate_count:
            report.error(f"L5 `{LAND_ID_FIELD}` 存在 {duplicate_count} 个重复记录。")
        else:
            report.ok(f"L5 `{LAND_ID_FIELD}` 唯一，地块数：{unique_count:,}。")

    fixed_gdf, fixed_count, remaining_bad = repair_geometry(gdf)
    if fixed_count:
        report.warn(f"L5 geometry 发现并尝试修复 {fixed_count} 个 invalid geometry。")
    if remaining_bad:
        report.error(f"L5 仍有 {remaining_bad} 个 invalid 或 empty geometry。")
    else:
        report.ok("L5 geometry 有效。")

    l5_compute = to_compute_crs(fixed_gdf, report, "L5")
    if l5_compute is not None:
        areas = l5_compute.geometry.area
        report.summary["l5_area_m2_stats"] = numeric_stats(areas)

    if l4 is not None and BLOCK_ID_FIELD in l4.columns and LAND_ID_FIELD in fixed_gdf.columns:
        check_land_to_block(report, fixed_gdf, l4)

    return fixed_gdf


def check_land_to_block(report: Report, lands: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame) -> None:
    report.h3("L5→L4 空间预匹配")

    if lands.crs is None or blocks.crs is None:
        report.warn("L5 或 L4 CRS 为空，跳过 L5→L4 空间预匹配。")
        return

    try:
        lands_c = lands.to_crs(COMPUTE_CRS).copy()
        blocks_c = blocks.to_crs(COMPUTE_CRS).copy()

        lands_c["_land_tmp_id"] = np.arange(len(lands_c))
        lands_centroid = lands_c.copy()
        lands_centroid["geometry"] = lands_centroid.geometry.centroid

        blocks_small = blocks_c[[BLOCK_ID_FIELD, "geometry"]].copy()

        try:
            joined = gpd.sjoin(
                lands_centroid[[LAND_ID_FIELD, "_land_tmp_id", "geometry"]],
                blocks_small,
                how="left",
                predicate="within",
            )
        except TypeError:
            joined = gpd.sjoin(
                lands_centroid[[LAND_ID_FIELD, "_land_tmp_id", "geometry"]],
                blocks_small,
                how="left",
                op="within",
            )

        matched = joined.dropna(subset=[BLOCK_ID_FIELD])
        matched_land_count = int(matched["_land_tmp_id"].nunique())
        unmatched = int(len(lands_c) - matched_land_count)
        matched_block_count = int(normalize_id_series(matched[BLOCK_ID_FIELD]).nunique()) if len(matched) else 0

        report.summary["land_to_block_centroid_matched_lands"] = matched_land_count
        report.summary["land_to_block_centroid_unmatched_lands"] = unmatched
        report.summary["land_to_block_centroid_matched_blocks"] = matched_block_count

        report.ok(f"L5 centroid 落入 L4 的地块数：{matched_land_count:,}/{len(lands_c):,}。")
        report.ok(f"L5 覆盖到的 L4 街区数：{matched_block_count:,}。")
        if unmatched:
            report.warn(f"有 {unmatched:,} 个 L5 地块 centroid 未落入任何 L4 街区，后续需要用最大相交面积兜底。")

        if len(matched):
            counts = normalize_id_series(matched[BLOCK_ID_FIELD]).value_counts()
            report.summary["land_per_block_stats"] = numeric_stats(counts)
            report.ok(
                "每个 L4 内 L5 数量："
                f"min={counts.min()}, median={int(counts.median())}, mean={counts.mean():.1f}, max={counts.max()}。"
            )
    except Exception as e:
        report.error(f"L5→L4 空间预匹配失败：{e}")
        report.code(traceback.format_exc(), "text")


def check_buildings(report: Report, l4: Optional[gpd.GeoDataFrame], l5: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    report.h2("4. 建筑物检查与落区预匹配")

    gdf = read_vector(PATHS["buildings"], "建筑物", report)
    if gdf is None:
        return None

    report.summary["building_rows"] = int(len(gdf))
    report.summary["building_crs"] = str(gdf.crs)
    report.summary["building_columns"] = list(gdf.columns)
    report.p(f"- 字段列表：`{list(gdf.columns)}`")
    report.p(f"- CRS：`{gdf.crs}`")

    fixed_gdf, fixed_count, remaining_bad = repair_geometry(gdf)
    if fixed_count:
        report.warn(f"建筑物 geometry 发现并尝试修复 {fixed_count} 个 invalid geometry。")
    if remaining_bad:
        report.error(f"建筑物仍有 {remaining_bad} 个 invalid 或 empty geometry。")
    else:
        report.ok("建筑物 geometry 有效。")

    bldg_c = to_compute_crs(fixed_gdf, report, "建筑物")
    if bldg_c is not None:
        area_stats = numeric_stats(bldg_c.geometry.area)
        report.summary["building_footprint_area_m2_stats"] = area_stats
        report.h3("建筑物 footprint 面积统计，单位 m²")
        report.table(["指标", "值"], [[k, format_num(v)] for k, v in area_stats.items()])

        very_small = int((bldg_c.geometry.area < 5).sum())
        very_large = int((bldg_c.geometry.area > 100000).sum())
        report.summary["building_footprint_area_lt5_count"] = very_small
        report.summary["building_footprint_area_gt100000_count"] = very_large
        if very_small:
            report.warn(f"建筑物 footprint 面积 < 5 m² 的记录有 {very_small:,} 条，可能是碎片。")
        if very_large:
            report.warn(f"建筑物 footprint 面积 > 100,000 m² 的记录有 {very_large:,} 条，需人工复核。")

    height_cols = [
        c for c in fixed_gdf.columns
        if any(k in safe_str(c).lower() for k in ["height", "高度", "floor", "floors", "层", "h"])
        and c != "geometry"
    ]
    report.summary["building_candidate_height_columns"] = height_cols
    if height_cols:
        report.ok(f"疑似高度/楼层字段：{height_cols}")
    else:
        report.warn("未识别到明显的建筑高度或楼层字段。后续可先使用 footprint 面积和数量构建特征。")

    if l4 is not None:
        check_point_layer_to_polygon_layer(
            report=report,
            point_like_gdf=fixed_gdf,
            polygon_gdf=l4,
            polygon_id_field=BLOCK_ID_FIELD,
            layer_name="建筑物",
            target_name="L4",
            summary_prefix="building_to_l4",
            use_centroid=True,
        )

    if l5 is not None:
        check_point_layer_to_polygon_layer(
            report=report,
            point_like_gdf=fixed_gdf,
            polygon_gdf=l5,
            polygon_id_field=LAND_ID_FIELD,
            layer_name="建筑物",
            target_name="L5",
            summary_prefix="building_to_l5",
            use_centroid=True,
        )

    return fixed_gdf


def check_poi(report: Report, l4: Optional[gpd.GeoDataFrame], l5: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    report.h2("5. POI 检查与落区预匹配")

    gdf = read_vector(PATHS["poi"], "POI", report)
    if gdf is None:
        return None

    report.summary["poi_rows"] = int(len(gdf))
    report.summary["poi_crs"] = str(gdf.crs)
    report.summary["poi_columns"] = list(gdf.columns)
    report.p(f"- 字段列表：`{list(gdf.columns)}`")
    report.p(f"- CRS：`{gdf.crs}`")

    fixed_gdf, fixed_count, remaining_bad = repair_geometry(gdf)
    if fixed_count:
        report.warn(f"POI geometry 发现并尝试修复 {fixed_count} 个 invalid geometry。")
    if remaining_bad:
        report.warn(f"POI 有 {remaining_bad} 个 invalid 或 empty geometry。")
    else:
        report.ok("POI geometry 有效。")

    category_candidates = [
        c for c in fixed_gdf.columns
        if any(k in safe_str(c).lower() for k in ["type", "category", "class", "kind", "类别", "大类", "小类", "中类"])
        and c != "geometry"
    ]
    report.summary["poi_candidate_category_columns"] = category_candidates

    if category_candidates:
        report.ok(f"疑似 POI 类别字段：{category_candidates}")
        rows = []
        for col in category_candidates[:5]:
            vc = fixed_gdf[col].astype(str).value_counts(dropna=False).head(10)
            rows.append([col, dict(vc)])
        report.table(["字段", "Top10 取值"], rows)
    else:
        report.warn("未识别到明显 POI 类别字段。后续只能先做 POI 数量/密度特征。")

    if l4 is not None:
        check_point_layer_to_polygon_layer(
            report=report,
            point_like_gdf=fixed_gdf,
            polygon_gdf=l4,
            polygon_id_field=BLOCK_ID_FIELD,
            layer_name="POI",
            target_name="L4",
            summary_prefix="poi_to_l4",
            use_centroid=False,
        )

    if l5 is not None:
        check_point_layer_to_polygon_layer(
            report=report,
            point_like_gdf=fixed_gdf,
            polygon_gdf=l5,
            polygon_id_field=LAND_ID_FIELD,
            layer_name="POI",
            target_name="L5",
            summary_prefix="poi_to_l5",
            use_centroid=False,
        )

    return fixed_gdf


def check_point_layer_to_polygon_layer(
    report: Report,
    point_like_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    polygon_id_field: str,
    layer_name: str,
    target_name: str,
    summary_prefix: str,
    use_centroid: bool,
) -> None:
    report.h3(f"{layer_name}→{target_name} 空间预匹配")

    if polygon_id_field not in polygon_gdf.columns:
        report.warn(f"{target_name} 缺少 ID 字段 `{polygon_id_field}`，跳过。")
        return

    if point_like_gdf.crs is None or polygon_gdf.crs is None:
        report.warn(f"{layer_name} 或 {target_name} CRS 为空，跳过空间预匹配。")
        return

    try:
        src = point_like_gdf[point_like_gdf.geometry.notna()].copy().to_crs(COMPUTE_CRS)
        dst = polygon_gdf[polygon_gdf.geometry.notna()].copy().to_crs(COMPUTE_CRS)

        src["_src_tmp_id"] = np.arange(len(src))

        if use_centroid:
            src_join = src[["_src_tmp_id", "geometry"]].copy()
            src_join["geometry"] = src_join.geometry.centroid
        else:
            src_join = src[["_src_tmp_id", "geometry"]].copy()

        dst_small = dst[[polygon_id_field, "geometry"]].copy()

        try:
            joined = gpd.sjoin(src_join, dst_small, how="left", predicate="within")
        except TypeError:
            joined = gpd.sjoin(src_join, dst_small, how="left", op="within")

        matched = joined.dropna(subset=[polygon_id_field]).copy()
        matched_records = int(matched["_src_tmp_id"].nunique())
        target_ids = normalize_id_series(matched[polygon_id_field]) if len(matched) else pd.Series([], dtype=str)
        target_count = int(target_ids.nunique()) if len(target_ids) else 0
        unmatched = int(len(src) - matched_records)

        report.summary[f"{summary_prefix}_matched_records"] = matched_records
        report.summary[f"{summary_prefix}_unmatched_records"] = unmatched
        report.summary[f"{summary_prefix}_target_count"] = target_count

        report.ok(f"{layer_name} 匹配到 {target_name} 的记录数：{matched_records:,}/{len(src):,}。")
        report.ok(f"覆盖 {target_name} 数量：{target_count:,}。")

        if unmatched:
            report.warn(f"{layer_name} 有 {unmatched:,} 条记录未匹配到 {target_name}。")

        if len(target_ids):
            counts = target_ids.value_counts()
            report.summary[f"{summary_prefix}_records_per_target_stats"] = numeric_stats(counts)
            report.ok(
                f"每个 {target_name} 的 {layer_name} 数量："
                f"min={counts.min()}, median={int(counts.median())}, "
                f"mean={counts.mean():.1f}, max={counts.max()}。"
            )
    except Exception as e:
        report.error(f"{layer_name}→{target_name} 空间预匹配失败：{e}")
        report.code(traceback.format_exc(), "text")


def check_satellite(report: Report, l4: Optional[gpd.GeoDataFrame]) -> None:
    report.h2("6. 整城遥感影像检查")

    path = PATHS["satellite"]
    if not path.exists():
        report.error(f"遥感影像不存在：`{path}`")
        return

    try:
        with rasterio.open(path) as ds:
            report.summary["satellite_path"] = str(path)
            report.summary["satellite_crs"] = str(ds.crs)
            report.summary["satellite_width"] = int(ds.width)
            report.summary["satellite_height"] = int(ds.height)
            report.summary["satellite_bands"] = int(ds.count)
            report.summary["satellite_dtypes"] = list(ds.dtypes)
            report.summary["satellite_resolution"] = [float(ds.res[0]), float(ds.res[1])]
            report.summary["satellite_bounds"] = {
                "left": float(ds.bounds.left),
                "bottom": float(ds.bounds.bottom),
                "right": float(ds.bounds.right),
                "top": float(ds.bounds.top),
            }

            report.ok(f"遥感影像读取成功：{path.name}")
            report.table(
                ["指标", "值"],
                [
                    ["CRS", ds.crs],
                    ["尺寸", f"{ds.width} × {ds.height}"],
                    ["波段数", ds.count],
                    ["数据类型", ds.dtypes],
                    ["分辨率", ds.res],
                    ["范围", ds.bounds],
                ],
            )

            if l4 is not None and l4.crs is not None and ds.crs is not None:
                l4_raster_crs = l4.to_crs(ds.crs)
                img_box = box(ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
                covered = l4_raster_crs[l4_raster_crs.geometry.intersects(img_box)]
                covered_count = int(len(covered))
                total = int(len(l4_raster_crs))
                ratio = covered_count / total if total else 0.0
                report.summary["satellite_l4_covered_blocks"] = covered_count
                report.summary["satellite_l4_total_blocks"] = total
                report.summary["satellite_l4_covered_ratio"] = ratio

                report.ok(f"遥感影像范围与 L4 相交街区数：{covered_count:,}/{total:,}，覆盖率 {pct(ratio)}。")
                if covered_count < total:
                    report.warn(f"遥感影像没有覆盖全部 L4 街区，缺少 {total - covered_count:,} 个。")
            else:
                report.warn("L4 CRS 或遥感 CRS 缺失，无法检查遥感覆盖街区比例。")

    except Exception as e:
        report.error(f"遥感影像读取失败：{e}")
        report.code(traceback.format_exc(), "text")


def check_streetview(report: Report, l4: Optional[gpd.GeoDataFrame], l5: Optional[gpd.GeoDataFrame]) -> None:
    report.h2("7. 街景文件名解析、质量抽样与落区预匹配")

    sv_dir = PATHS["streetview_dir"]
    if not sv_dir.exists():
        report.error(f"街景目录不存在：`{sv_dir}`")
        return

    image_files = sorted([p for p in sv_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    report.summary["streetview_image_count"] = int(len(image_files))
    report.ok(f"街景图片数量，递归统计：{len(image_files):,}。")

    if not image_files:
        report.warn("街景目录下未找到图片。")
        return

    records = []
    parse_success = 0
    invalid_coord = 0

    for idx, path in enumerate(image_files):
        parsed = parse_streetview_filename(path)
        row = {
            "idx": idx,
            "filename": path.name,
            "path": str(path),
            "parse_success": parsed is not None,
            "lng": None,
            "lat": None,
            "heading": None,
            "pitch": None,
        }
        if parsed:
            parse_success += 1
            row.update(parsed)
            if not (120.0 <= parsed["lng"] <= 126.0 and 39.0 <= parsed["lat"] <= 44.0):
                invalid_coord += 1
        records.append(row)

    df = pd.DataFrame(records)
    report.summary["streetview_parse_success_count"] = int(parse_success)
    report.summary["streetview_parse_success_ratio"] = parse_success / len(image_files)
    report.summary["streetview_invalid_coord_count"] = int(invalid_coord)

    report.ok(f"文件名成功解析经纬度/方向：{parse_success:,}/{len(image_files):,}，比例 {pct(parse_success / len(image_files))}。")
    if invalid_coord:
        report.warn(f"有 {invalid_coord:,} 张街景坐标不在沈阳大致范围内。")

    if parse_success:
        valid_df = df[df["parse_success"]].copy()
        report.table(
            ["字段", "最小值", "中位数", "最大值"],
            [
                ["lng", format_num(valid_df["lng"].min()), format_num(valid_df["lng"].median()), format_num(valid_df["lng"].max())],
                ["lat", format_num(valid_df["lat"].min()), format_num(valid_df["lat"].median()), format_num(valid_df["lat"].max())],
                ["heading", format_num(valid_df["heading"].min()), format_num(valid_df["heading"].median()), format_num(valid_df["heading"].max())],
                ["pitch", format_num(valid_df["pitch"].min()), format_num(valid_df["pitch"].median()), format_num(valid_df["pitch"].max())],
            ],
        )

        heading_counts = valid_df["heading"].value_counts().head(20).to_dict()
        report.summary["streetview_heading_top20"] = heading_counts
        report.p(f"- heading Top20：`{heading_counts}`")

    # 抽样检查图片是否能打开，避免全量读取太慢
    sample_files = image_files[: min(200, len(image_files))]
    qc_rows = []
    bad_open = 0
    blank_like = 0
    sizes = []

    for p in sample_files:
        qc = image_quick_check(p)
        if not qc["can_open"]:
            bad_open += 1
        if qc.get("looks_blank"):
            blank_like += 1
        if qc.get("width") and qc.get("height"):
            sizes.append((qc["width"], qc["height"]))
        if len(qc_rows) < 5:
            qc_rows.append([
                p.name,
                qc["can_open"],
                f"{qc.get('width')}×{qc.get('height')}",
                qc.get("mode"),
                format_num(qc.get("mean")),
                format_num(qc.get("stddev")),
                qc.get("looks_blank"),
            ])

    report.summary["streetview_sample_checked_count"] = len(sample_files)
    report.summary["streetview_sample_bad_open_count"] = bad_open
    report.summary["streetview_sample_blank_like_count"] = blank_like
    report.h3("街景图片抽样质量检查，最多前 200 张")
    report.table(["文件", "可打开", "尺寸", "模式", "灰度均值", "灰度标准差", "疑似空白"], qc_rows)

    if bad_open:
        report.warn(f"抽样中有 {bad_open} 张图片无法打开。")
    else:
        report.ok("抽样图片均可打开。")
    if blank_like:
        report.warn(f"抽样中有 {blank_like} 张图片疑似纯黑/纯白/低信息量。")

    if sizes:
        size_counts = Counter(sizes).most_common(10)
        report.summary["streetview_sample_size_top10"] = [{"size": list(k), "count": v} for k, v in size_counts]
        report.p(f"- 抽样图片尺寸 Top10：`{size_counts}`")

    if parse_success and l4 is not None:
        streetview_spatial_join(report, df[df["parse_success"]].copy(), l4, BLOCK_ID_FIELD, "L4", "streetview_to_l4")
    if parse_success and l5 is not None:
        streetview_spatial_join(report, df[df["parse_success"]].copy(), l5, LAND_ID_FIELD, "L5", "streetview_to_l5")


def streetview_spatial_join(
    report: Report,
    sv_df: pd.DataFrame,
    polygons: gpd.GeoDataFrame,
    polygon_id_field: str,
    target_name: str,
    summary_prefix: str,
) -> None:
    report.h3(f"街景→{target_name} 空间预匹配")

    if polygon_id_field not in polygons.columns:
        report.warn(f"{target_name} 缺少 ID 字段 `{polygon_id_field}`，跳过街景落区。")
        return
    if polygons.crs is None:
        report.warn(f"{target_name} CRS 为空，跳过街景落区。")
        return

    try:
        points = gpd.GeoDataFrame(
            sv_df,
            geometry=gpd.points_from_xy(sv_df["lng"], sv_df["lat"]),
            crs=STORAGE_CRS,
        )
        polygons2 = polygons.to_crs(STORAGE_CRS)
        polygons_small = polygons2[[polygon_id_field, "geometry"]].copy()

        try:
            joined = gpd.sjoin(points, polygons_small, how="left", predicate="within")
        except TypeError:
            joined = gpd.sjoin(points, polygons_small, how="left", op="within")

        matched = joined.dropna(subset=[polygon_id_field]).copy()
        matched_points = int(len(matched))
        target_ids = normalize_id_series(matched[polygon_id_field]) if len(matched) else pd.Series([], dtype=str)
        target_count = int(target_ids.nunique()) if len(target_ids) else 0
        total_points = int(len(points))
        unmatched = total_points - matched_points

        report.summary[f"{summary_prefix}_matched_points"] = matched_points
        report.summary[f"{summary_prefix}_unmatched_points"] = unmatched
        report.summary[f"{summary_prefix}_target_count"] = target_count

        report.ok(f"街景点匹配到 {target_name}：{matched_points:,}/{total_points:,}。")
        report.ok(f"街景覆盖 {target_name} 数量：{target_count:,}。")
        if unmatched:
            report.warn(f"有 {unmatched:,} 个街景点未落入任何 {target_name}。")

        if len(target_ids):
            counts = target_ids.value_counts()
            report.summary[f"{summary_prefix}_points_per_target_stats"] = numeric_stats(counts)
            report.ok(
                f"每个 {target_name} 街景数：min={counts.min()}, "
                f"median={int(counts.median())}, mean={counts.mean():.1f}, max={counts.max()}。"
            )

            lt4 = int((counts < 4).sum())
            lt10 = int((counts < 10).sum())
            report.summary[f"{summary_prefix}_targets_lt4_points"] = lt4
            report.summary[f"{summary_prefix}_targets_lt10_points"] = lt10
            if lt4:
                report.warn(f"{target_name} 中有 {lt4:,} 个单元街景数 < 4，不适合直接做多方向街景训练。")
            if lt10:
                report.warn(f"{target_name} 中有 {lt10:,} 个单元街景数 < 10，训练时需采样/补充策略。")

    except Exception as e:
        report.error(f"街景→{target_name} 空间预匹配失败：{e}")
        report.code(traceback.format_exc(), "text")


def check_final_alignment(report: Report) -> None:
    report.h2("8. 后续重构关键结论")

    l4_count = report.summary.get("l4_blockid_unique")
    energy_stats = report.summary.get("energy_stats", {})
    sv_l4_count = report.summary.get("streetview_to_l4_target_count")
    sat_l4_count = report.summary.get("satellite_l4_covered_blocks")
    bldg_l4_count = report.summary.get("building_to_l4_target_count")
    poi_l4_count = report.summary.get("poi_to_l4_target_count")

    rows = [
        ["L4 街区数", l4_count],
        [f"有 `{ENERGY_FIELD}` 非空标签数", energy_stats.get("count") if isinstance(energy_stats, dict) else None],
        ["遥感覆盖 L4 数", sat_l4_count],
        ["建筑物覆盖 L4 数", bldg_l4_count],
        ["POI 覆盖 L4 数", poi_l4_count],
        ["街景覆盖 L4 数", sv_l4_count],
    ]
    report.table(["项目", "数量"], rows)

    if l4_count and energy_stats and energy_stats.get("count"):
        if int(energy_stats.get("count")) == int(l4_count):
            report.ok("L4 街区与 E_Final_W5 标签数量一致，可作为主预测单元。")
        else:
            report.warn("L4 街区数与 E_Final_W5 非空标签数不一致，导出训练集前需要剔除缺失标签街区。")

    if sat_l4_count is not None and l4_count is not None:
        if int(sat_l4_count) == int(l4_count):
            report.ok("整城遥感覆盖全部 L4，后续可按 BlockID 裁剪生成遥感数据集。")
        else:
            report.warn("整城遥感未覆盖全部 L4，裁剪前需要处理缺失街区。")

    if sv_l4_count is not None:
        report.p(
            "- 街景当前建议口径：使用文件名解析出的经纬度点，空间 join 到 L4 的结果；"
            "不要使用任何未验证的原始街区字段。"
        )

    report.p("")
    report.p("建议下一步脚本：")
    report.p("1. `01_standardize_blocks_lands.py`：标准化 L4/L5，生成 `land_to_block.csv`。")
    report.p("2. `02_build_block_features.py`：建筑物和 POI 落区，生成 block-level 表格特征。")
    report.p("3. `03_crop_satellite_by_block.py`：按 L4 BlockID 裁剪整城 TIF。")
    report.p("4. `04_index_streetview.py`：解析街景文件名、落区、质量筛选，生成 streetview_index.csv。")


# =============================================================================
# 输出
# =============================================================================

def write_outputs(report: Report) -> List[Path]:
    ensure_dir(report.output_dir)
    written: List[Path] = []

    md_path = report.output_dir / "data_check_report.md"
    md_path.write_text(report.render(), encoding="utf-8")
    written.append(md_path)

    if WRITE_JSON_SUMMARY:
        payload = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_root": str(DATA_ROOT),
            "paths": {k: str(v) for k, v in PATHS.items()},
            "issues": report.issues,
            "summary": report.summary,
        }
        json_path = report.output_dir / "data_check_summary.json"
        json_path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(json_path)

    return written


# =============================================================================
# 主程序
# =============================================================================

def main() -> None:
    print("=" * 88)
    print("沈阳街区能耗预测 · 原始数据重构前检查")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 88)

    report = Report(output_dir=OUTPUT_DIR)
    report.h1(f"数据检查报告 · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.p(f"> DATA_ROOT: `{DATA_ROOT}`  ")
    report.p(f"> 预测单元: L4 街区 `{BLOCK_ID_FIELD}`  ")
    report.p(f"> 固定能耗标签: `{ENERGY_FIELD}`  ")
    report.p(f"> 计算 CRS: `{COMPUTE_CRS}`  ")
    report.p(f"> 输出文件: `data_check_report.md`" + ("、`data_check_summary.json`" if WRITE_JSON_SUMMARY else ""))
    report.p("")

    l4 = None
    l5 = None

    steps = [
        ("路径检查", lambda: check_paths(report)),
        ("L4 街区与标签", lambda: globals().__setitem__("l4", check_l4_blocks(report))),
        ("L5 地块", lambda: globals().__setitem__("l5", check_l5_lands(report, globals().get("l4")))),
        ("建筑物", lambda: check_buildings(report, globals().get("l4"), globals().get("l5"))),
        ("POI", lambda: check_poi(report, globals().get("l4"), globals().get("l5"))),
        ("遥感", lambda: check_satellite(report, globals().get("l4"))),
        ("街景", lambda: check_streetview(report, globals().get("l4"), globals().get("l5"))),
        ("最终对齐结论", lambda: check_final_alignment(report)),
    ]

    # 这里不用 nonlocal，是为了脚本直接运行兼容；实际对象保存在 globals 中。
    globals()["l4"] = None
    globals()["l5"] = None

    for step_name, fn in steps:
        print(f"[RUN] {step_name}")
        try:
            fn()
            print(f"[OK]  {step_name}")
        except Exception:
            err = traceback.format_exc()
            report.error(f"步骤 `{step_name}` 运行异常。")
            report.code(err, "text")
            print(f"[ERR] {step_name}")

    report.h1("问题汇总")
    if report.issues:
        for issue in report.issues:
            report.p(f"- {issue}")
    else:
        report.ok("未发现 WARN/ERROR。")

    report.h1("关键数字摘要")
    report.code(json.dumps(jsonable(report.summary), ensure_ascii=False, indent=2), "json")

    report.h1("本次实际输出文件")
    expected = ["data_check_report.md"]
    if WRITE_JSON_SUMMARY:
        expected.append("data_check_summary.json")
    for name in expected:
        report.p(f"- `{OUTPUT_DIR / name}`")

    written = write_outputs(report)

    print("")
    print("=" * 88)
    print("[DONE] 数据检查完成")
    print(f"[DONE] 输出目录: {OUTPUT_DIR}")
    print(f"[DONE] 输出文件数量: {len(written)}")
    for p in written:
        print(f"       - {p}")
    print(f"[DONE] WARN/ERROR 数量: {len(report.issues)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
