# -*- coding: utf-8 -*-
"""
scripts/check_data.py
=====================

沈阳街区能耗预测 · Phase 1 数据检查脚本

默认输出文件数量:
    1. data_check_report.md
    2. data_check_summary.json

如只想输出 1 个文件:
    将 WRITE_JSON_SUMMARY = False

运行:
    python scripts/check_data.py
"""

from __future__ import annotations

import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# =============================================================================
# 输出控制
# =============================================================================

WRITE_JSON_SUMMARY = True

MAX_LIST_PREVIEW = 20
MAX_FILES_PREVIEW = 30


# =============================================================================
# 依赖导入
# =============================================================================

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("[FATAL] 缺少 python-dotenv，请先执行: pip install python-dotenv")

try:
    import yaml
except ImportError:
    sys.exit("[FATAL] 缺少 PyYAML，请先执行: pip install PyYAML")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("[FATAL] 缺少 pandas/numpy，请先执行: pip install pandas numpy")

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    gpd = None
    HAS_GPD = False

try:
    import rasterio
    HAS_RIO = True
except ImportError:
    rasterio = None
    HAS_RIO = False


# =============================================================================
# 通用工具
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def is_number_like(s: str) -> bool:
    s = safe_str(s)
    if not s:
        return False
    try:
        float(s)
        return True
    except Exception:
        return False


def normalize_numeric_string(s: Any) -> str:
    """
    将 1.0 / "1.0" / "001" 等尽量归一成 "1"。
    非数字则原样返回。
    """
    text = safe_str(s)
    if not text:
        return text

    try:
        f = float(text)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return text


def canonical_region_id(x: Any) -> str:
    """
    统一街区 ID 形式。

    规则:
        Region_1        -> Region_1
        region_1        -> Region_1
        Block_1         -> Region_1
        Area_1          -> Region_1
        1 / "1" / 1.0   -> Region_1

    这样可以解决:
        label: Region_1
        KG:    Region_1
        shp:   1
        csv:   1
    之间的直接匹配问题。
    """
    s = safe_str(x)
    if not s:
        return ""

    s = s.replace(" ", "")

    lower = s.lower()

    for prefix in ["region_", "block_", "area_"]:
        if lower.startswith(prefix):
            tail = s.split("_", 1)[1]
            tail = normalize_numeric_string(tail)
            return "Region_{}".format(tail)

    if is_number_like(s):
        return "Region_{}".format(normalize_numeric_string(s))

    return s


def clean_id_set(values: Any) -> Set[str]:
    out = set()

    if values is None:
        return out

    for v in values:
        cid = canonical_region_id(v)
        if cid:
            out.add(cid)

    return out


def jsonable(obj: Any) -> Any:
    """
    将 set / numpy 标量 / Path 等转换为 JSON 可序列化对象。
    """
    if isinstance(obj, set):
        return sorted(jsonable(x) for x in obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]

    try:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    return obj


def read_text_safely(path: Path) -> str:
    for enc in ["utf-8", "gbk", "gb18030"]:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def find_first_existing_file(folder: Path, suffixes: Set[str]) -> Optional[Path]:
    if not folder.exists():
        return None

    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            return p

    return None


def pick_id_column(columns: List[Any]) -> Optional[str]:
    keywords = [
        "blockid",
        "block_id",
        "region_id",
        "region",
        "block",
        "街区",
        "编号",
        "id",
    ]

    for kw in keywords:
        for c in columns:
            name = safe_str(c).lower()
            if kw in name:
                return str(c)

    return None


def pick_lon_lat_columns(columns: List[Any]) -> Tuple[Optional[str], Optional[str]]:
    lon_candidates = ["经度", "lon", "lng", "longitude", "x"]
    lat_candidates = ["纬度", "lat", "latitude", "y"]

    lon_col = None
    lat_col = None

    for kw in lon_candidates:
        for c in columns:
            if kw == safe_str(c).lower() or kw in safe_str(c).lower():
                lon_col = str(c)
                break
        if lon_col:
            break

    for kw in lat_candidates:
        for c in columns:
            if kw == safe_str(c).lower() or kw in safe_str(c).lower():
                lat_col = str(c)
                break
        if lat_col:
            break

    return lon_col, lat_col


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


# =============================================================================
# 配置加载
# =============================================================================

def load_config() -> Dict[str, Any]:
    """
    读取:
        G:/Knowcl/888-代码/.env
        G:/Knowcl/888-代码/config/paths.yaml

    并展开 ${DATA_ROOT}。
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    env_path = project_root / ".env"
    yaml_path = project_root / "config" / "paths.yaml"

    if not env_path.exists():
        sys.exit(
            "[FATAL] 未找到 .env 文件: {}\n"
            "请确认它位于 888-代码 根目录下。".format(env_path)
        )

    if not yaml_path.exists():
        sys.exit(
            "[FATAL] 未找到路径配置文件: {}\n"
            "请确认 config/paths.yaml 是否存在。".format(yaml_path)
        )

    load_dotenv(dotenv_path=env_path, override=True)

    data_root = os.environ.get("DATA_ROOT", "").strip()
    if not data_root:
        sys.exit("[FATAL] .env 中 DATA_ROOT 为空，请填写后重试。")

    data_root = data_root.replace("\\", "/")

    raw_yaml = yaml_path.read_text(encoding="utf-8")
    raw_yaml = raw_yaml.replace("${DATA_ROOT}", data_root)

    cfg = yaml.safe_load(raw_yaml)

    if not isinstance(cfg, dict):
        sys.exit("[FATAL] paths.yaml 读取结果不是 dict，请检查 YAML 格式。")

    cfg["_data_root"] = data_root
    cfg["_project_root"] = str(project_root)
    cfg["_script_dir"] = str(script_dir)

    return cfg


# =============================================================================
# 报告对象
# =============================================================================

class Report:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.lines = []
        self.issues = []
        self.summary = {}
        self.id_sets = {}

    def h1(self, text: str) -> None:
        self.lines.append("")
        self.lines.append("# {}".format(text))
        self.lines.append("")

    def h2(self, text: str) -> None:
        self.lines.append("")
        self.lines.append("## {}".format(text))
        self.lines.append("")

    def h3(self, text: str) -> None:
        self.lines.append("")
        self.lines.append("### {}".format(text))
        self.lines.append("")

    def p(self, text: str = "") -> None:
        self.lines.append(str(text))

    def ok(self, text: str) -> None:
        self.lines.append("- ✅ {}".format(text))

    def warn(self, text: str) -> None:
        self.lines.append("- ⚠️ {}".format(text))
        self.issues.append("[WARN] {}".format(text))

    def error(self, text: str) -> None:
        self.lines.append("- ❌ {}".format(text))
        self.issues.append("[ERROR] {}".format(text))

    def code(self, text: str, lang: str = "") -> None:
        self.lines.append("```{}".format(lang))
        self.lines.append(str(text))
        self.lines.append("```")
        self.lines.append("")

    def table(self, headers: List[str], rows: List[List[Any]]) -> None:
        if not headers:
            return

        self.lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        self.lines.append("| " + " | ".join("---" for _ in headers) + " |")

        for row in rows:
            row = list(row)
            if len(row) < len(headers):
                row += [""] * (len(headers) - len(row))
            if len(row) > len(headers):
                row = row[:len(headers)]

            self.lines.append("| " + " | ".join(str(x).replace("\n", "<br>") for x in row) + " |")

        self.lines.append("")

    def render(self) -> str:
        return "\n".join(self.lines)


# =============================================================================
# Shapefile 选择与读取
# =============================================================================

def list_shapefiles(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.shp"))


def choose_main_block_shp(cfg: Dict[str, Any], r: Optional[Report] = None) -> Optional[Path]:
    """
    优先选择 L4 街区文件。
    因为你的报告中:
        沈阳L4.shp = 757 个街区
        沈阳L5.shp = 2187 个地块/土地单元
    建模主街区应优先用 L4。
    """
    raw = cfg.get("raw", {})
    block_dir = Path(str(raw.get("blocks", "")))

    shps = list_shapefiles(block_dir)
    if not shps:
        return None

    for shp in shps:
        if "l4" in shp.name.lower() or "L4" in shp.name:
            return shp

    for shp in shps:
        try:
            gdf_head = gpd.read_file(shp, rows=5) if HAS_GPD else None
            if gdf_head is not None:
                cols = [safe_str(c).lower() for c in gdf_head.columns]
                if any("blockid" in c or "block" in c or "街区" in c for c in cols):
                    return shp
        except Exception:
            continue

    return shps[0]


def choose_main_building_shp(cfg: Dict[str, Any]) -> Optional[Path]:
    raw = cfg.get("raw", {})
    bldg_dir = Path(str(raw.get("buildings", "")))

    shps = list_shapefiles(bldg_dir)
    if not shps:
        return None

    for shp in shps:
        if "processed" in shp.name.lower():
            return shp

    for shp in shps:
        if "三环" in shp.name:
            return shp

    return shps[0]


def get_block_id_column(gdf: Any) -> Optional[str]:
    cols = list(gdf.columns)

    preferred = ["BlockID", "block_id", "blockid", "RegionID", "region_id", "街区ID", "街区id", "id", "Id", "index"]

    for name in preferred:
        if name in cols:
            return name

    return pick_id_column(cols)


# =============================================================================
# C01 配置与目录
# =============================================================================

def c01_config_and_directory(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C01 · 配置与目录存在性")

    data_root = cfg.get("_data_root", "")
    project_root = cfg.get("_project_root", "")

    r.p("- DATA_ROOT: `{}`".format(data_root))
    r.p("- Project root: `{}`".format(project_root))
    r.p("")

    rows = []
    missing = []

    for section in ["raw", "outputs"]:
        values = cfg.get(section, {})
        if not isinstance(values, dict):
            continue

        for key, path in sorted(values.items()):
            p = Path(str(path))
            exists = p.exists()
            rows.append([
                "{}.{}".format(section, key),
                str(p),
                "存在" if exists else "缺失",
            ])

            if not exists:
                missing.append(str(p))

    r.table(["配置项", "路径", "状态"], rows)

    r.summary["missing_dirs_count"] = len(missing)
    r.summary["missing_dirs"] = missing[:MAX_LIST_PREVIEW]

    if missing:
        r.warn("{} 个目录缺失。脚本不会自动创建除报告目录外的其他输出目录。".format(len(missing)))
    else:
        r.ok("配置中的目录均存在。")


# =============================================================================
# C02 文件清单
# =============================================================================

def c02_file_inventory(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C02 · 文件清单")

    raw = cfg.get("raw", {})
    if not isinstance(raw, dict):
        r.warn("paths.yaml 中未找到 raw 配置。")
        return

    rows = []

    for key, value in raw.items():
        folder = Path(str(value))

        if not folder.exists():
            rows.append([key, str(folder), "目录缺失", "—", "—"])
            continue

        files_top = [p for p in folder.iterdir() if p.is_file()]
        files_all = [p for p in folder.rglob("*") if p.is_file()]

        ext_counter = Counter(
            p.suffix.lower() if p.suffix else "<无扩展名>"
            for p in files_top
        )

        ext_text = ", ".join(
            "{}×{}".format(ext, cnt)
            for ext, cnt in ext_counter.most_common(8)
        )

        rows.append([
            key,
            str(folder),
            len(files_top),
            len(files_all),
            ext_text or "空",
        ])

    r.table(["数据项", "路径", "顶层文件数", "递归文件数", "顶层扩展名"], rows)


# =============================================================================
# C03 能耗标签
# =============================================================================

def load_json_as_df(path: Path, r: Report) -> Optional[pd.DataFrame]:
    try:
        text = read_text_safely(path)
        raw = json.loads(text)
    except Exception as e:
        r.error("JSON 读取失败: {}，错误: {}".format(path.name, e))
        return None

    if isinstance(raw, list):
        r.ok("JSON 结构: list。")
        return pd.DataFrame(raw)

    if isinstance(raw, dict):
        if all(isinstance(v, dict) for v in raw.values()):
            r.ok("JSON 结构: dict of dict；顶层 key 作为 block_id。")
            df = pd.DataFrame.from_dict(raw, orient="index")
            df.index.name = "block_id"
            df = df.reset_index()
            return df

        for k, v in raw.items():
            if isinstance(v, list):
                r.ok("JSON 结构: dict，其中 `{}` 是 list，作为记录表。".format(k))
                return pd.DataFrame(v)

        r.warn("JSON 结构为普通 dict，按单行 DataFrame 处理。")
        return pd.DataFrame([raw])

    r.error("不支持的 JSON 根类型: {}".format(type(raw)))
    return None


def analyze_label_df(df: pd.DataFrame, r: Report, source_name: str) -> None:
    r.p("- 行数: `{}`".format(len(df)))
    r.p("- 列数: `{}`".format(len(df.columns)))
    r.p("- 列名: `{}`".format(list(df.columns)))

    id_col = pick_id_column(list(df.columns))

    if id_col:
        ids = clean_id_set(df[id_col].dropna().tolist())
        r.id_sets["label_ids"] = ids
        r.summary["label_id_col"] = id_col
        r.summary["label_block_count"] = len(ids)
        r.summary["label_id_preview"] = sorted(ids)[:MAX_LIST_PREVIEW]
        r.ok("识别到标签 ID 列: `{}`，归一化后街区数: {}".format(id_col, len(ids)))
    else:
        r.warn("未识别到标签 ID 列。")

    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not num_cols:
        r.warn("标签文件中未发现数值列。")
        return

    energy_keywords = ["energy", "ec", "elec", "power", "kwh", "mj", "gj", "能耗", "用电"]
    energy_cols = [
        c for c in num_cols
        if any(k in safe_str(c).lower() for k in energy_keywords)
    ]

    if not energy_cols:
        energy_cols = num_cols
        r.warn("未通过关键字识别到能耗列，将展示所有数值列统计。")
    else:
        r.ok("疑似能耗列: {}".format(energy_cols))

    rows = []

    for col in energy_cols[:10]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(s) == 0:
            continue

        mean = float(s.mean())
        median = float(s.median())
        std = float(s.std()) if len(s) > 1 else 0.0
        skew = float(s.skew()) if len(s) > 2 else 0.0
        zero_ratio = float((s == 0).mean())
        min_v = float(s.min())
        max_v = float(s.max())
        log1p_ok = bool((s >= 0).all())

        rows.append([
            col,
            len(s),
            "{:.4f}".format(mean),
            "{:.4f}".format(median),
            "{:.4f}".format(std),
            "{:.4f}".format(min_v),
            "{:.4f}".format(max_v),
            "{:.2f}".format(skew),
            "{:.1%}".format(zero_ratio),
            "是" if log1p_ok else "否",
        ])

        prefix = "label_col_{}".format(col)
        r.summary[prefix + "_mean"] = mean
        r.summary[prefix + "_median"] = median
        r.summary[prefix + "_skew"] = skew
        r.summary[prefix + "_zero_ratio"] = zero_ratio
        r.summary[prefix + "_log1p_ok"] = log1p_ok

        if skew > 3:
            r.warn("列 `{}` 偏度 {:.2f} > 3，建模前建议使用 log1p 或稳健变换。".format(col, skew))
        if zero_ratio > 0.3:
            r.warn("列 `{}` 零值比例 {:.1%} > 30%，需确认是否为真实零值。".format(col, zero_ratio))
        if not log1p_ok:
            r.error("列 `{}` 存在负值，不能直接 log1p。".format(col))

    if rows:
        r.table(
            ["列名", "非空数", "均值", "中位数", "标准差", "最小值", "最大值", "偏度", "零值比", "log1p可用"],
            rows,
        )


def c03_energy_label(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C03 · 能耗标签检查")

    label_dir = Path(str(cfg.get("raw", {}).get("labels", "")))

    if not label_dir.exists():
        r.error("标签目录不存在: {}".format(label_dir))
        return

    files = []
    files += sorted(label_dir.glob("*.json"))
    files += sorted(label_dir.glob("*.csv"))
    files += sorted(label_dir.glob("*.xlsx"))
    files += sorted(label_dir.glob("*.xls"))

    if not files:
        r.error("标签目录下未找到 JSON / CSV / Excel 文件。")
        return

    r.p("候选标签文件: `{}`".format([p.name for p in files]))

    for path in files:
        r.h3("文件: {}".format(path.name))

        df = None

        if path.suffix.lower() == ".json":
            df = load_json_as_df(path, r)
        elif path.suffix.lower() == ".csv":
            try:
                df = read_csv_safely(path)
            except Exception as e:
                r.error("CSV 读取失败: {}".format(e))
        else:
            try:
                df = pd.read_excel(path)
            except Exception as e:
                r.error("Excel 读取失败: {}".format(e))

        if df is None:
            continue

        analyze_label_df(df, r, path.name)

        if "label_ids" in r.id_sets:
            break


# =============================================================================
# C04 街区数据
# =============================================================================

def c04_blocks(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C04 · 街区数据检查")

    if not HAS_GPD:
        r.warn("未安装 geopandas，跳过 Shapefile 深度检查。")
        return

    block_dir = Path(str(cfg.get("raw", {}).get("blocks", "")))

    if not block_dir.exists():
        r.error("街区目录不存在: {}".format(block_dir))
        return

    shps = list_shapefiles(block_dir)
    if not shps:
        r.error("街区目录下未找到 .shp 文件。")
        return

    main_shp = choose_main_block_shp(cfg, r)
    r.ok("主街区文件选择为: `{}`".format(main_shp.name if main_shp else "未识别"))

    rows = []

    for shp in shps:
        try:
            gdf = gpd.read_file(shp)
            id_col = get_block_id_column(gdf)
            rows.append([
                safe_relative(shp, block_dir),
                len(gdf),
                str(gdf.crs),
                id_col or "未识别",
                list(gdf.columns),
            ])

            if main_shp and shp.resolve() == main_shp.resolve():
                block_ids = set()

                if id_col:
                    block_ids = clean_id_set(gdf[id_col].dropna().tolist())
                else:
                    block_ids = clean_id_set(gdf.index.tolist())

                r.id_sets["block_ids"] = block_ids
                r.summary["main_block_shp"] = shp.name
                r.summary["main_block_count"] = len(gdf)
                r.summary["main_block_id_col"] = id_col
                r.summary["main_block_crs"] = str(gdf.crs)
                r.summary["main_block_id_preview"] = sorted(block_ids)[:MAX_LIST_PREVIEW]

        except Exception as e:
            rows.append([safe_relative(shp, block_dir), "读取失败", str(e), "", ""])

    r.table(["文件", "要素数", "CRS", "ID列", "字段"], rows)

    if "block_ids" in r.id_sets:
        r.ok("主街区归一化 ID 数: {}".format(len(r.id_sets["block_ids"])))
    else:
        r.warn("未能从主街区文件提取 block_id。")


# =============================================================================
# C05 建筑物数据
# =============================================================================

def c05_buildings(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C05 · 建筑物数据检查")

    if not HAS_GPD:
        r.warn("未安装 geopandas，跳过建筑物 Shapefile 检查。")
        return

    bldg_dir = Path(str(cfg.get("raw", {}).get("buildings", "")))

    if not bldg_dir.exists():
        r.error("建筑物目录不存在: {}".format(bldg_dir))
        return

    shps = list_shapefiles(bldg_dir)
    if not shps:
        r.error("建筑物目录下未找到 .shp 文件。")
        return

    main_shp = choose_main_building_shp(cfg)
    r.ok("主建筑物文件选择为: `{}`".format(main_shp.name if main_shp else "未识别"))

    rows = []

    for shp in shps:
        try:
            gdf = gpd.read_file(shp)

            height_cols = [
                c for c in gdf.columns
                if any(k in safe_str(c).lower() for k in ["height", "高度", "floor", "floors", "层数"])
            ]

            rows.append([
                safe_relative(shp, bldg_dir),
                len(gdf),
                str(gdf.crs),
                height_cols or "未识别",
                list(gdf.columns),
            ])

            if main_shp and shp.resolve() == main_shp.resolve():
                r.summary["main_building_shp"] = shp.name
                r.summary["main_building_count"] = len(gdf)
                r.summary["main_building_crs"] = str(gdf.crs)
                r.summary["main_building_height_cols"] = height_cols

        except Exception as e:
            rows.append([safe_relative(shp, bldg_dir), "读取失败", str(e), "", ""])

    r.table(["文件", "建筑物数", "CRS", "高度字段", "字段"], rows)


# =============================================================================
# C06 街景数据
# =============================================================================

def c06_streetview(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C06 · 街景文件检查")

    sv_dir = Path(str(cfg.get("raw", {}).get("streetview", "")))

    if not sv_dir.exists():
        r.error("街景目录不存在: {}".format(sv_dir))
        return

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    top_files = [p for p in sv_dir.iterdir() if p.is_file()]
    all_images = [p for p in sv_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]
    csv_files = sorted([p for p in top_files if p.suffix.lower() == ".csv"])

    r.ok("街景图片总数，递归统计: {}".format(len(all_images)))
    r.ok("街景 CSV 文件数: {}".format(len(csv_files)))

    r.summary["streetview_image_count_recursive"] = len(all_images)
    r.summary["streetview_csv_count"] = len(csv_files)

    if not csv_files:
        subdirs = [p for p in sv_dir.iterdir() if p.is_dir()]
        if subdirs:
            cnt_by_dir = {}
            for d in subdirs:
                cnt_by_dir[d.name] = len([
                    p for p in d.rglob("*")
                    if p.is_file() and p.suffix.lower() in image_exts
                ])

            raw_ids = clean_id_set(cnt_by_dir.keys())
            r.id_sets["streetview_raw_ids"] = raw_ids
            r.summary["streetview_block_count_by_subdir"] = len(raw_ids)

            counts = list(cnt_by_dir.values())
            if counts:
                r.ok("按子目录统计街景覆盖街区数: {}".format(len(counts)))
                r.ok("每街区图片数: min={}, median={}, mean={:.1f}, max={}".format(
                    min(counts),
                    int(np.median(counts)),
                    float(np.mean(counts)),
                    max(counts),
                ))
        else:
            r.warn("未找到 CSV 映射，也未找到按街区划分的子目录。")
        return

    # 只分析第一个主要 CSV；避免产生额外明细文件。
    csv_path = csv_files[0]
    r.h3("主街景 CSV: {}".format(csv_path.name))

    try:
        df = read_csv_safely(csv_path)
    except Exception as e:
        r.error("街景 CSV 读取失败: {}".format(e))
        return

    r.ok("CSV 行数: {}".format(len(df)))
    r.p("CSV 列名: `{}`".format(list(df.columns)))

    id_col = pick_id_column(list(df.columns))
    lon_col, lat_col = pick_lon_lat_columns(list(df.columns))

    if id_col:
        raw_ids = clean_id_set(df[id_col].dropna().tolist())
        r.id_sets["streetview_raw_ids"] = raw_ids
        r.summary["streetview_raw_id_col"] = id_col
        r.summary["streetview_raw_block_count"] = len(raw_ids)
        r.summary["streetview_raw_id_preview"] = sorted(raw_ids)[:MAX_LIST_PREVIEW]
        r.ok("识别到街景原始 block_id 列: `{}`，归一化后 ID 数: {}".format(id_col, len(raw_ids)))

        vc = df[id_col].dropna().map(canonical_region_id).value_counts()
        counts = vc.values.tolist()

        if counts:
            r.summary["streetview_points_count_from_csv"] = int(sum(counts))
            r.summary["streetview_per_block_min_raw"] = int(min(counts))
            r.summary["streetview_per_block_median_raw"] = int(np.median(counts))
            r.summary["streetview_per_block_max_raw"] = int(max(counts))

            r.ok("CSV 映射点数: {}".format(int(sum(counts))))
            r.ok("原始 ID 每街区点数: min={}, median={}, max={}".format(
                int(min(counts)),
                int(np.median(counts)),
                int(max(counts)),
            ))

            lt10 = int(sum(1 for x in counts if x < 10))
            lt40 = int(sum(1 for x in counts if x < 40))

            r.summary["streetview_blocks_lt10_raw"] = lt10
            r.summary["streetview_blocks_lt40_raw"] = lt40

            if lt10 > 0:
                r.warn("按原始 ID 统计，有 {} 个街区街景点数 < 10。".format(lt10))
            elif lt40 > 0:
                r.warn("按原始 ID 统计，有 {} 个街区街景点数 < 40。".format(lt40))
            else:
                r.ok("按原始 ID 统计，所有街区街景点数 >= 40。")
    else:
        r.warn("未识别到街景 CSV 中的 block_id 列。")

    if lon_col and lat_col:
        r.ok("识别到坐标列: lon=`{}`, lat=`{}`。".format(lon_col, lat_col))
        r.summary["streetview_lon_col"] = lon_col
        r.summary["streetview_lat_col"] = lat_col
    else:
        r.warn("未识别到街景经纬度列，无法做点落街区空间 join。")
        return

    if not HAS_GPD:
        r.warn("未安装 geopandas，无法做街景点落街区空间 join。")
        return

    main_block_shp = choose_main_block_shp(cfg, r)
    if main_block_shp is None:
        r.warn("未找到主街区 shp，无法做街景点落街区空间 join。")
        return

    try:
        blocks = gpd.read_file(main_block_shp)
        block_id_col = get_block_id_column(blocks)

        if block_id_col is None:
            r.warn("主街区 shp 未识别到 ID 列，无法做空间 join 后 ID 归一。")
            return

        df_xy = df[[lon_col, lat_col]].copy()
        df_xy[lon_col] = pd.to_numeric(df_xy[lon_col], errors="coerce")
        df_xy[lat_col] = pd.to_numeric(df_xy[lat_col], errors="coerce")
        df_xy = df_xy.dropna(subset=[lon_col, lat_col])

        if len(df_xy) == 0:
            r.warn("街景坐标列没有有效数值。")
            return

        points = gpd.GeoDataFrame(
            df_xy,
            geometry=gpd.points_from_xy(df_xy[lon_col], df_xy[lat_col]),
            crs="EPSG:4326",
        )

        if blocks.crs is None:
            r.warn("街区 shp CRS 为空，默认按 EPSG:4326 处理。")
            blocks = blocks.set_crs(epsg=4326)

        if str(blocks.crs) != str(points.crs):
            points = points.to_crs(blocks.crs)

        keep_cols = [block_id_col, "geometry"]
        blocks_small = blocks[keep_cols].copy()

        try:
            joined = gpd.sjoin(points, blocks_small, how="left", predicate="within")
        except TypeError:
            joined = gpd.sjoin(points, blocks_small, how="left", op="within")

        matched = joined.dropna(subset=[block_id_col]).copy()
        spatial_ids = clean_id_set(matched[block_id_col].tolist())

        r.id_sets["streetview_spatial_block_ids"] = spatial_ids
        r.summary["streetview_spatial_join_points_total"] = int(len(points))
        r.summary["streetview_spatial_join_points_matched"] = int(len(matched))
        r.summary["streetview_spatial_block_count"] = len(spatial_ids)
        r.summary["streetview_spatial_id_preview"] = sorted(spatial_ids)[:MAX_LIST_PREVIEW]

        r.ok("街景点空间 join 成功匹配点数: {}/{}".format(len(matched), len(points)))
        r.ok("空间 join 后覆盖主街区数: {}".format(len(spatial_ids)))

        vc2 = matched[block_id_col].map(canonical_region_id).value_counts()
        counts2 = vc2.values.tolist()

        if counts2:
            r.summary["streetview_per_block_min_spatial"] = int(min(counts2))
            r.summary["streetview_per_block_median_spatial"] = int(np.median(counts2))
            r.summary["streetview_per_block_max_spatial"] = int(max(counts2))

            r.ok("空间 join 后每街区点数: min={}, median={}, max={}".format(
                int(min(counts2)),
                int(np.median(counts2)),
                int(max(counts2)),
            ))

    except Exception as e:
        r.error("街景点空间 join 失败: {}".format(e))
        r.code(traceback.format_exc(), "text")


# =============================================================================
# C07 卫星影像
# =============================================================================

def c07_satellite(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C07 · 卫星影像检查")

    sat_dir = Path(str(cfg.get("raw", {}).get("satellite", "")))

    if not sat_dir.exists():
        r.error("卫星数据目录不存在: {}".format(sat_dir))
        return

    img_exts = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
    images = sorted([p for p in sat_dir.rglob("*") if p.is_file() and p.suffix.lower() in img_exts])

    r.ok("卫星影像文件数: {}".format(len(images)))
    r.summary["satellite_image_count"] = len(images)

    if not images:
        r.warn("未发现卫星影像文件。")
        return

    if not HAS_RIO:
        r.warn("未安装 rasterio，跳过影像尺寸、CRS 和范围检查。")
        return

    sample = images[0]
    r.h3("样例影像: {}".format(sample.name))

    try:
        with rasterio.open(sample) as ds:
            bounds = ds.bounds

            r.ok("尺寸: {} × {} px".format(ds.width, ds.height))
            r.ok("波段数: {}".format(ds.count))
            r.ok("CRS: {}".format(ds.crs))
            r.ok(
                "范围: left={:.4f}, bottom={:.4f}, right={:.4f}, top={:.4f}".format(
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top,
                )
            )

            r.summary["satellite_sample_file"] = sample.name
            r.summary["satellite_width"] = int(ds.width)
            r.summary["satellite_height"] = int(ds.height)
            r.summary["satellite_count"] = int(ds.count)
            r.summary["satellite_crs"] = str(ds.crs)
            r.summary["satellite_bounds"] = {
                "left": float(bounds.left),
                "bottom": float(bounds.bottom),
                "right": float(bounds.right),
                "top": float(bounds.top),
            }

            if HAS_GPD:
                main_block_shp = choose_main_block_shp(cfg, r)
                if main_block_shp:
                    blocks = gpd.read_file(main_block_shp)

                    if blocks.crs is None:
                        r.warn("主街区 CRS 为空，无法严谨判断卫星覆盖。")
                    else:
                        blocks2 = blocks.to_crs(ds.crs)

                        from shapely.geometry import box
                        img_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

                        covered = blocks2[blocks2.geometry.intersects(img_box)]
                        ratio = len(covered) / len(blocks2) if len(blocks2) else 0.0

                        r.ok("卫星范围覆盖主街区: {}/{} ({:.1%})".format(
                            len(covered),
                            len(blocks2),
                            ratio,
                        ))

                        r.summary["satellite_covered_blocks"] = int(len(covered))
                        r.summary["satellite_covered_ratio"] = float(ratio)

                        if len(covered) < len(blocks2):
                            r.warn("有 {} 个主街区不在卫星影像范围内。".format(len(blocks2) - len(covered)))

    except Exception as e:
        r.error("卫星影像读取失败: {}".format(e))


# =============================================================================
# C08 训练验证测试划分
# =============================================================================

def extract_ids_from_split_file(path: Path, r: Report) -> Set[str]:
    ids = set()

    try:
        if path.suffix.lower() == ".txt":
            text = read_text_safely(path)
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                first = line.split(",")[0].split("\t")[0].strip()
                cid = canonical_region_id(first)
                if cid:
                    ids.add(cid)
            return ids

        df = read_csv_safely(path)
        id_col = pick_id_column(list(df.columns))

        if id_col is None:
            id_col = str(df.columns[0])
            r.warn("{} 未找到明确 ID 列，使用第一列 `{}`。".format(path.name, id_col))
        else:
            r.ok("{} 使用 `{}` 作为 ID 列。".format(path.name, id_col))

        ids = clean_id_set(df[id_col].dropna().tolist())

    except Exception as e:
        r.error("{} 读取失败: {}".format(path.name, e))

    return ids


def c08_splits(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C08 · 训练/验证/测试划分检查")

    split_dir = Path(str(cfg.get("raw", {}).get("splits", "")))

    if not split_dir.exists():
        r.error("划分目录不存在: {}".format(split_dir))
        return

    files = sorted([p for p in split_dir.iterdir() if p.is_file()])
    r.p("目录文件: `{}`".format([p.name for p in files]))

    split_ids = {}

    candidates = [
        ("train", ["train"]),
        ("val", ["val", "valid", "validation"]),
        ("test", ["test"]),
    ]

    for label, keys in candidates:
        for p in files:
            stem = p.stem.lower()
            suffix = p.suffix.lower()

            if suffix not in {".csv", ".txt"}:
                continue

            if any(stem == k or stem.endswith("_" + k) or k in stem for k in keys):
                if label not in split_ids:
                    ids = extract_ids_from_split_file(p, r)
                    if ids:
                        split_ids[label] = ids
                        r.id_sets["split_" + label + "_ids"] = ids
                        r.summary["split_" + label + "_count"] = len(ids)
                        r.ok("{} -> {}: {} 个 ID".format(p.name, label, len(ids)))

    if len(split_ids) < 2:
        r.warn("识别到的划分集少于 2 个，无法检查数据泄漏。")
        return

    labels = list(split_ids.keys())
    leak_free = True

    rows = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = labels[i]
            b = labels[j]
            overlap = split_ids[a] & split_ids[b]

            rows.append([
                "{} ∩ {}".format(a, b),
                len(overlap),
                sorted(overlap)[:10],
            ])

            if overlap:
                leak_free = False
                r.error("{} 与 {} 存在 {} 个重复 ID。".format(a, b, len(overlap)))
            else:
                r.ok("{} 与 {} 无重叠。".format(a, b))

    r.table(["检查项", "重叠数量", "样例"], rows)

    total_unique = len(set().union(*split_ids.values()))
    total_sum = sum(len(v) for v in split_ids.values())

    r.summary["split_total_unique"] = total_unique
    r.summary["split_total_sum"] = total_sum
    r.summary["split_leak_free"] = leak_free

    ratio_rows = []
    for label in ["train", "val", "test"]:
        if label in split_ids:
            ratio_rows.append([
                label,
                len(split_ids[label]),
                "{:.1%}".format(len(split_ids[label]) / total_sum if total_sum else 0.0),
            ])

    r.table(["划分", "数量", "占比"], ratio_rows)


# =============================================================================
# C09 知识图谱
# =============================================================================

def c09_kg(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C09 · 知识图谱检查")

    kg_dir = Path(str(cfg.get("raw", {}).get("kg", "")))

    if not kg_dir.exists():
        r.error("KG 目录不存在: {}".format(kg_dir))
        return

    files = sorted([p for p in kg_dir.rglob("*") if p.is_file()])
    r.p("文件预览: `{}`".format([safe_relative(p, kg_dir) for p in files[:MAX_FILES_PREVIEW]]))

    triple_files = [
        p for p in files
        if p.suffix.lower() in {".txt", ".tsv", ".csv"}
    ]

    if not triple_files:
        r.warn("未发现 txt/tsv/csv 三元组文件。")
        return

    entities = set()
    relations = Counter()
    entity_type_counter = Counter()
    kg_region_ids = set()
    triple_count = 0

    for path in triple_files:
        try:
            text = read_text_safely(path)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            if not lines:
                continue

            valid_line_count = 0

            for line in lines:
                if "\t" in line:
                    parts = line.split("\t")
                elif "," in line and path.suffix.lower() == ".csv":
                    parts = line.split(",")
                else:
                    parts = line.split()

                if len(parts) < 3:
                    continue

                h = safe_str(parts[0])
                rel = safe_str(parts[1])
                t = safe_str(parts[2])

                if not h or not rel or not t:
                    continue

                valid_line_count += 1
                entities.add(h)
                entities.add(t)
                relations[rel] += 1

                for e in [h, t]:
                    if "_" in e:
                        prefix = e.split("_", 1)[0]
                        entity_type_counter[prefix] += 1

                        if prefix.lower() in {"region", "block", "area"}:
                            kg_region_ids.add(canonical_region_id(e))
                    else:
                        entity_type_counter["<raw>"] += 1

            triple_count += valid_line_count
            r.ok("{}: 有效三元组 {} 行。".format(path.name, valid_line_count))

        except Exception as e:
            r.warn("{} 读取失败: {}".format(path.name, e))

    r.id_sets["kg_region_ids"] = kg_region_ids

    r.summary["kg_triple_count"] = int(triple_count)
    r.summary["kg_entity_count"] = int(len(entities))
    r.summary["kg_relation_count"] = int(len(relations))
    r.summary["kg_region_count"] = int(len(kg_region_ids))
    r.summary["kg_region_preview"] = sorted(kg_region_ids)[:MAX_LIST_PREVIEW]

    r.ok("三元组总数: {:,}".format(triple_count))
    r.ok("实体总数: {:,}".format(len(entities)))
    r.ok("关系类型数: {}".format(len(relations)))
    r.ok("Region 实体数，归一化后: {}".format(len(kg_region_ids)))

    type_rows = [
        [k, v]
        for k, v in entity_type_counter.most_common(20)
    ]
    r.h3("实体类型分布")
    r.table(["实体类型前缀", "出现次数"], type_rows)

    rel_rows = [
        [k, v]
        for k, v in relations.most_common(30)
    ]
    r.h3("关系类型分布")
    r.table(["关系", "三元组数"], rel_rows)

    if len(relations) > 30:
        r.warn("KG 关系类型超过 30，后续 CompGCN/R-GCN 参数量可能偏大。")


# =============================================================================
# C10 多模态 ID 对齐
# =============================================================================

def c10_multimodal_id_alignment(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C10 · 多模态 ID 对齐与交集检查")

    label_ids = r.id_sets.get("label_ids", set())
    block_ids = r.id_sets.get("block_ids", set())
    kg_ids = r.id_sets.get("kg_region_ids", set())
    sv_raw_ids = r.id_sets.get("streetview_raw_ids", set())
    sv_spatial_ids = r.id_sets.get("streetview_spatial_block_ids", set())

    rows = [
        ["label_ids", len(label_ids), sorted(label_ids)[:10]],
        ["block_ids", len(block_ids), sorted(block_ids)[:10]],
        ["kg_region_ids", len(kg_ids), sorted(kg_ids)[:10]],
        ["streetview_raw_ids", len(sv_raw_ids), sorted(sv_raw_ids)[:10]],
        ["streetview_spatial_block_ids", len(sv_spatial_ids), sorted(sv_spatial_ids)[:10]],
    ]

    r.table(["ID 集合", "数量", "样例"], rows)

    if label_ids and block_ids:
        missing_in_blocks = label_ids - block_ids
        r.summary["label_missing_in_blocks_count"] = len(missing_in_blocks)

        if missing_in_blocks:
            r.warn("标签 ID 中有 {} 个不在主街区 shp 中。样例: {}".format(
                len(missing_in_blocks),
                sorted(missing_in_blocks)[:10],
            ))
        else:
            r.ok("标签 ID 与主街区 shp 可完全对齐。")

    if label_ids and kg_ids:
        label_kg = label_ids & kg_ids
        r.summary["label_kg_intersection"] = len(label_kg)

        if len(label_kg) == len(label_ids):
            r.ok("标签与 KG Region 完全对齐: {}/{}".format(len(label_kg), len(label_ids)))
        else:
            r.warn("标签与 KG Region 交集: {}/{}".format(len(label_kg), len(label_ids)))

    if label_ids and sv_raw_ids:
        label_sv_raw = label_ids & sv_raw_ids
        r.summary["label_sv_raw_intersection"] = len(label_sv_raw)

        if len(label_sv_raw) == 0:
            r.warn("标签与街景原始 ID 直接交集为 0；这通常说明街景 CSV 的 `街区ID` 不是主街区 Region ID。")
        else:
            r.ok("标签与街景原始 ID 交集: {}".format(len(label_sv_raw)))

    if label_ids and sv_spatial_ids:
        label_sv_spatial = label_ids & sv_spatial_ids
        r.summary["label_sv_spatial_intersection"] = len(label_sv_spatial)
        r.ok("标签与街景空间 join ID 交集: {}".format(len(label_sv_spatial)))

    # 推荐使用空间 join 后的街景 ID；若没有，则退回 raw ID。
    sv_recommended = sv_spatial_ids if sv_spatial_ids else sv_raw_ids

    modal_sets = []
    modal_names = []

    if label_ids:
        modal_sets.append(label_ids)
        modal_names.append("label")

    if kg_ids:
        modal_sets.append(kg_ids)
        modal_names.append("KG")

    if sv_recommended:
        modal_sets.append(sv_recommended)
        modal_names.append("SV_spatial" if sv_spatial_ids else "SV_raw")

    if len(modal_sets) >= 2:
        inter = set(modal_sets[0])
        for s in modal_sets[1:]:
            inter = inter & s

        r.summary["multimodal_intersection_recommended"] = len(inter)
        r.summary["multimodal_intersection_modal_names"] = modal_names
        r.summary["multimodal_intersection_preview"] = sorted(inter)[:MAX_LIST_PREVIEW]

        r.ok("推荐口径交集 `{}` = {} 个街区。".format(" ∩ ".join(modal_names), len(inter)))

        if len(inter) >= 500:
            r.ok("交集样本量 >= 500，可以进入后续裁切、聚合与建模准备。")
        else:
            r.warn("交集样本量 < 500。优先检查街景坐标空间 join 与 ID 映射，而不是直接训练模型。")
    else:
        r.warn("可用于交集的模态少于 2 个，无法计算多模态交集。")


# =============================================================================
# C11 CRS 汇总
# =============================================================================

def c11_crs_summary(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C11 · CRS 一致性汇总")

    rows = [
        ["主街区", r.summary.get("main_block_crs", "未知")],
        ["主建筑物", r.summary.get("main_building_crs", "未知")],
        ["卫星影像", r.summary.get("satellite_crs", "未知")],
    ]

    r.table(["数据层", "CRS"], rows)

    known = [x[1] for x in rows if x[1] and x[1] != "未知"]
    unique = sorted(set(known))

    r.summary["crs_unique"] = unique

    if len(unique) > 1:
        r.warn("多个空间数据 CRS 不一致。空间计算前建议统一到 EPSG:32651。")
    elif len(unique) == 1:
        r.ok("已读取到的空间数据 CRS 一致: {}".format(unique[0]))
    else:
        r.warn("未读取到足够 CRS 信息。")


# =============================================================================
# C12 建筑-街区空间关联
# =============================================================================

def c12_building_block_join(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C12 · 建筑物-街区空间关联检查")

    if not HAS_GPD:
        r.warn("未安装 geopandas，跳过建筑-街区空间 join。")
        return

    block_shp = choose_main_block_shp(cfg, r)
    bldg_shp = choose_main_building_shp(cfg)

    if block_shp is None:
        r.warn("未找到主街区 shp，跳过。")
        return

    if bldg_shp is None:
        r.warn("未找到主建筑物 shp，跳过。")
        return

    r.p("- 主街区文件: `{}`".format(block_shp.name))
    r.p("- 主建筑物文件: `{}`".format(bldg_shp.name))

    try:
        blocks = gpd.read_file(block_shp)
        buildings = gpd.read_file(bldg_shp)

        block_id_col = get_block_id_column(blocks)

        if block_id_col is None:
            r.warn("主街区未识别到 ID 列，无法统计每街区建筑数。")
            return

        if blocks.crs is None:
            r.warn("主街区 CRS 为空，无法可靠做空间 join。")
            return

        if buildings.crs is None:
            r.warn("建筑物 CRS 为空，无法可靠做空间 join。")
            return

        blocks = blocks.to_crs(epsg=32651)
        buildings = buildings.to_crs(epsg=32651)

        buildings = buildings[buildings.geometry.notna()].copy()
        blocks = blocks[blocks.geometry.notna()].copy()

        if len(buildings) == 0 or len(blocks) == 0:
            r.warn("街区或建筑物有效 geometry 为空。")
            return

        keep_cols = [block_id_col, "geometry"]
        blocks_small = blocks[keep_cols].copy()

        try:
            joined = gpd.sjoin(buildings[["geometry"]], blocks_small, how="left", predicate="within")
        except TypeError:
            joined = gpd.sjoin(buildings[["geometry"]], blocks_small, how="left", op="within")

        matched = joined.dropna(subset=[block_id_col]).copy()

        vc = matched[block_id_col].map(canonical_region_id).value_counts()
        counts = vc.values.tolist()

        r.summary["building_join_matched_buildings"] = int(len(matched))
        r.summary["building_join_blocks_with_buildings"] = int(len(vc))
        r.summary["building_join_blocks_total"] = int(len(blocks))

        r.ok("成功关联到街区的建筑物数: {}/{}".format(len(matched), len(buildings)))
        r.ok("有建筑物的街区数: {}/{}".format(len(vc), len(blocks)))

        if counts:
            r.summary["building_per_block_min"] = int(min(counts))
            r.summary["building_per_block_median"] = int(np.median(counts))
            r.summary["building_per_block_mean"] = float(np.mean(counts))
            r.summary["building_per_block_max"] = int(max(counts))

            r.ok("每街区建筑物数: min={}, median={}, mean={:.1f}, max={}".format(
                int(min(counts)),
                int(np.median(counts)),
                float(np.mean(counts)),
                int(max(counts)),
            ))

        zero_blocks = len(blocks) - len(vc)
        r.summary["building_join_zero_building_blocks"] = int(zero_blocks)

        if zero_blocks > 0:
            r.warn("{} 个街区内没有建筑物关联结果。".format(zero_blocks))

    except Exception as e:
        r.error("建筑-街区空间 join 失败: {}".format(e))
        r.code(traceback.format_exc(), "text")


# =============================================================================
# C13 其他基础数据
# =============================================================================

def c13_auxiliary_data(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C13 · 其他基础数据可用性检查")

    raw = cfg.get("raw", {})

    keys = [
        "energy",
        "bldg_height",
        "nightlight",
        "population",
        "edgar",
        "poi",
        "pretrained",
        "rs_raw",
        "code_ref",
    ]

    rows = []

    for key in keys:
        if key not in raw:
            continue

        folder = Path(str(raw[key]))

        if not folder.exists():
            rows.append([key, str(folder), "目录缺失", "—", "—"])
            continue

        files = [p for p in folder.rglob("*") if p.is_file()]
        ext = Counter(p.suffix.lower() if p.suffix else "<无扩展名>" for p in files)

        ext_text = ", ".join(
            "{}×{}".format(k, v)
            for k, v in ext.most_common(8)
        )

        note = ""

        if key == "nightlight" and len(files) == 0:
            note = "夜光目录为空；模型不用夜光时可忽略。"

        if key == "pretrained" and not any(p.suffix.lower() in {".npz", ".pt", ".pth", ".ckpt"} for p in files):
            note = "未发现常见预训练权重格式。"

        rows.append([
            key,
            str(folder),
            len(files),
            ext_text or "空",
            note,
        ])

        r.summary["aux_{}_file_count".format(key)] = len(files)

    r.table(["数据项", "路径", "文件数", "扩展名", "备注"], rows)


# =============================================================================
# C14 下一步建议
# =============================================================================

def c14_actionable_feedback(cfg: Dict[str, Any], r: Report) -> None:
    r.h2("C14 · 下一步建议")

    label_ids = r.id_sets.get("label_ids", set())
    kg_ids = r.id_sets.get("kg_region_ids", set())
    sv_raw_ids = r.id_sets.get("streetview_raw_ids", set())
    sv_spatial_ids = r.id_sets.get("streetview_spatial_block_ids", set())

    recommendations = []

    if label_ids and kg_ids:
        if len(label_ids & kg_ids) == len(label_ids):
            recommendations.append("标签与 KG 的 Region ID 已对齐，可以作为后续统一 ID 基准。")
        else:
            recommendations.append("标签与 KG 未完全对齐，需要先建立 Region ID 映射表。")

    if sv_spatial_ids:
        recommendations.append("街景建议优先使用经纬度空间 join 后的街区 ID，不要直接使用 CSV 原始 `街区ID`。")
    elif sv_raw_ids and label_ids and len(sv_raw_ids & label_ids) == 0:
        recommendations.append("街景原始 ID 与标签 ID 无交集；下一步应做街景点落区空间 join。")

    inter = r.summary.get("multimodal_intersection_recommended")

    if inter is not None:
        if int(inter) >= 500:
            recommendations.append("推荐口径多模态交集已达到 500，可进入卫星裁切、街景聚合和训练样本构建。")
        else:
            recommendations.append("推荐口径多模态交集不足 500，优先排查街景覆盖与 ID 映射。")

    missing_dirs_count = int(r.summary.get("missing_dirs_count", 0))
    if missing_dirs_count > 0:
        recommendations.append("部分输出目录缺失。建模前建议手动创建，但本脚本只创建数据检查报告目录。")

    if not recommendations:
        recommendations.append("未发现明确阻塞项。建议按报告中的 WARN/ERROR 顺序处理。")

    for i, item in enumerate(recommendations, 1):
        r.p("{}. {}".format(i, item))


# =============================================================================
# 写出文件
# =============================================================================

def write_outputs(r: Report) -> List[Path]:
    """
    只写 1-2 个文件:
        data_check_report.md
        data_check_summary.json
    """
    ensure_dir(r.output_dir)

    written = []

    report_path = r.output_dir / "data_check_report.md"
    report_path.write_text(r.render(), encoding="utf-8")
    written.append(report_path)

    if WRITE_JSON_SUMMARY:
        summary_path = r.output_dir / "data_check_summary.json"

        payload = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "issues": r.issues,
            "summary": r.summary,
            "id_set_counts": {
                k: len(v)
                for k, v in r.id_sets.items()
                if isinstance(v, set)
            },
            "id_set_previews": {
                k: sorted(v)[:MAX_LIST_PREVIEW]
                for k, v in r.id_sets.items()
                if isinstance(v, set)
            },
        }

        summary_path.write_text(
            json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(summary_path)

    return written


# =============================================================================
# 主程序
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("沈阳街区能耗预测 · 数据检查")
    print("运行时间: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 80)

    cfg = load_config()

    data_root = cfg.get("_data_root", "")

    output_dir = Path(
        str(
            cfg.get("outputs", {}).get(
                "check",
                Path(data_root) / "999-输出成果文件" / "00-数据检查报告",
            )
        )
    )

    r = Report(output_dir=output_dir)

    r.h1("数据检查报告 · {}".format(datetime.now().strftime("%Y-%m-%d %H:%M")))

    r.p("> 项目: 沈阳街区能耗预测  ")
    r.p("> DATA_ROOT: `{}`  ".format(data_root))
    r.p("> 脚本: `scripts/check_data.py`  ")
    r.p("> 输出控制: 本脚本默认只输出 `data_check_report.md` 和 `data_check_summary.json`。")
    r.p("")

    checks = [
        c01_config_and_directory,
        c02_file_inventory,
        c03_energy_label,
        c04_blocks,
        c05_buildings,
        c06_streetview,
        c07_satellite,
        c08_splits,
        c09_kg,
        c10_multimodal_id_alignment,
        c11_crs_summary,
        c12_building_block_join,
        c13_auxiliary_data,
        c14_actionable_feedback,
    ]

    for fn in checks:
        print("[RUN] {}".format(fn.__name__))

        try:
            fn(cfg, r)
            print("[OK]  {}".format(fn.__name__))
        except Exception:
            err = traceback.format_exc()
            r.error("{} 运行异常。".format(fn.__name__))
            r.code(err, "text")
            print("[ERR] {}".format(fn.__name__))

    r.h1("问题汇总")

    if r.issues:
        for issue in r.issues:
            r.p("- {}".format(issue))
    else:
        r.ok("未发现 WARN/ERROR。")

    r.h1("关键数字摘要")

    if r.summary:
        lines = []
        for k in sorted(r.summary.keys()):
            lines.append("{}: {}".format(k, jsonable(r.summary[k])))
        r.code("\n".join(lines), "yaml")
    else:
        r.p("暂无摘要。")

    r.h1("本次实际输出文件")

    expected_files = ["data_check_report.md"]
    if WRITE_JSON_SUMMARY:
        expected_files.append("data_check_summary.json")

    for name in expected_files:
        r.p("- `{}`".format(output_dir / name))

    written = write_outputs(r)

    print("")
    print("=" * 80)
    print("[DONE] 数据检查完成")
    print("[DONE] 输出目录: {}".format(output_dir))
    print("[DONE] 实际输出文件数量: {}".format(len(written)))

    for p in written:
        print("       - {}".format(p))

    print("[DONE] WARN/ERROR 数量: {}".format(len(r.issues)))

    if r.issues:
        print("")
        print("[问题列表]")
        for issue in r.issues:
            print("  {}".format(issue))

    print("=" * 80)


if __name__ == "__main__":
    main()
