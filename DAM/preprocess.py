"""
preprocess.py

將 wafer 邊緣環帶拉直，再切成重疊 patch，
輸出 Ultralytics YOLO segmentation 或 detection 資料集。

預期輸入：
input_dataset/
├── train/
│   ├── images/
│   │   ├── wafer_001.png
│   │   └── ...
│   └── masks/
│       ├── wafer_001.png
│       └── ...
├── val/
└── test/

Mask 格式：
- 單通道 PNG
- 0：背景
- 1：第 0 類
- 2：第 1 類
- 依此類推

例如只有一種「點膠異常」：
- 背景 = 0
- 異常 = 1
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Config:
    # 原始資料與輸出資料夾
    input_root: str = "input_dataset"
    output_root: str = "wafer_edge_dataset"

    # wafer 圓心與半徑。
    # 若每張圖不同，可改用 center_json。
    center_x: float = 2048.0
    center_y: float = 2048.0
    inner_radius: float = 1800.0
    outer_radius: float = 2020.0

    # 可選：每張 wafer 個別的圓心和半徑
    # JSON 格式請參考 README。
    center_json: str | None = None

    # 拉直後長條圖大小
    unwrap_width: int = 8192
    unwrap_height: int = 256

    # 沿圓周方向切 patch
    patch_width: int = 512
    stride: int = 384

    # 輸出模式：segment 或 detect
    task: str = "segment"

    # 每類名稱。Mask value 1 對應 names[0]。
    names: tuple[str, ...] = ("glue_defect",)

    # 過小區域可能只是插值雜訊
    min_area: float = 8.0

    # segmentation polygon 簡化程度
    polygon_epsilon_ratio: float = 0.002

    # 負樣本抽樣率。1.0 表示全部保留。
    negative_keep_ratio: float = 1.0

    # 隨機種子
    seed: int = 42


def load_config(path: str | Path) -> Config:
    """從 YAML 讀取設定。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "names" in raw:
        raw["names"] = tuple(raw["names"])

    return Config(**raw)


def load_geometry_table(path: str | None) -> dict:
    """讀取每張影像個別的 wafer 幾何資料。"""
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_geometry(
    image_path: Path,
    cfg: Config,
    geometry_table: dict,
) -> tuple[float, float, float, float]:
    """
    回傳 cx, cy, inner_radius, outer_radius。

    center_json 可用完整檔名或不含副檔名的 stem 當 key。
    """
    item = geometry_table.get(image_path.name)
    if item is None:
        item = geometry_table.get(image_path.stem)

    if item is None:
        return (
            cfg.center_x,
            cfg.center_y,
            cfg.inner_radius,
            cfg.outer_radius,
        )

    cx = float(item["center_x"])
    cy = float(item["center_y"])

    # 可直接提供 inner/outer radius。
    if "inner_radius" in item and "outer_radius" in item:
        inner = float(item["inner_radius"])
        outer = float(item["outer_radius"])
    else:
        # 或提供 wafer_radius，再搭配內外寬度。
        radius = float(item["wafer_radius"])
        inner = radius - float(item.get("inner_margin", 200))
        outer = radius + float(item.get("outer_margin", 20))

    return cx, cy, inner, outer


def build_polar_maps(
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    建立極座標展開用 map。

    水平方向：0～360 度
    垂直方向：inner_radius～outer_radius
    """
    cx, cy = center

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        output_width,
        endpoint=False,
        dtype=np.float32,
    )
    radius = np.linspace(
        inner_radius,
        outer_radius,
        output_height,
        dtype=np.float32,
    )

    theta_grid, radius_grid = np.meshgrid(theta, radius)

    map_x = cx + radius_grid * np.cos(theta_grid)
    map_y = cy + radius_grid * np.sin(theta_grid)

    return map_x.astype(np.float32), map_y.astype(np.float32)


def unwrap_image(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    is_mask: bool = False,
) -> np.ndarray:
    """使用相同幾何映射展開影像或 mask。"""
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def circular_crop(
    array: np.ndarray,
    start_x: int,
    patch_width: int,
) -> np.ndarray:
    """
    從長條圖切 patch。
    超過右側時會從左側接回來，處理 360°/0° 接縫。
    """
    width = array.shape[1]
    indices = (np.arange(start_x, start_x + patch_width) % width).astype(int)
    return array[:, indices].copy()


def iter_patch_starts(total_width: int, stride: int) -> Iterable[int]:
    """產生每個 patch 的起始 x。"""
    return range(0, total_width, stride)


def contour_to_segment_line(
    contour: np.ndarray,
    class_id: int,
    image_width: int,
    image_height: int,
    epsilon_ratio: float,
) -> str | None:
    """將 contour 轉成 YOLO segmentation polygon 格式。"""
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = max(0.5, epsilon_ratio * perimeter)
    polygon = cv2.approxPolyDP(contour, epsilon, closed=True)

    points = polygon.reshape(-1, 2)
    if len(points) < 3:
        return None

    coords: list[str] = [str(class_id)]
    for x, y in points:
        # 避免座標剛好超出 0～1。
        nx = np.clip(float(x) / image_width, 0.0, 1.0)
        ny = np.clip(float(y) / image_height, 0.0, 1.0)
        coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

    return " ".join(coords)


def contour_to_detect_line(
    contour: np.ndarray,
    class_id: int,
    image_width: int,
    image_height: int,
) -> str:
    """將 contour 外接矩形轉成 YOLO detection 格式。"""
    x, y, w, h = cv2.boundingRect(contour)

    cx = (x + w / 2.0) / image_width
    cy = (y + h / 2.0) / image_height
    nw = w / image_width
    nh = h / image_height

    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def mask_to_yolo_lines(
    patch_mask: np.ndarray,
    cfg: Config,
) -> list[str]:
    """
    將 patch mask 轉成 YOLO label lines。

    Mask value：
    0 = background
    1 = class 0
    2 = class 1
    """
    lines: list[str] = []
    height, width = patch_mask.shape[:2]

    for mask_value, _class_name in enumerate(cfg.names, start=1):
        class_id = mask_value - 1
        binary = (patch_mask == mask_value).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < cfg.min_area:
                continue

            if cfg.task == "segment":
                line = contour_to_segment_line(
                    contour=contour,
                    class_id=class_id,
                    image_width=width,
                    image_height=height,
                    epsilon_ratio=cfg.polygon_epsilon_ratio,
                )
                if line is not None:
                    lines.append(line)

            elif cfg.task == "detect":
                lines.append(
                    contour_to_detect_line(
                        contour=contour,
                        class_id=class_id,
                        image_width=width,
                        image_height=height,
                    )
                )
            else:
                raise ValueError("task 必須是 'segment' 或 'detect'")

    return lines


def find_matching_mask(mask_dir: Path, image_path: Path) -> Path | None:
    """尋找和影像同名的 mask。"""
    direct = mask_dir / f"{image_path.stem}.png"
    if direct.exists():
        return direct

    for suffix in IMAGE_SUFFIXES:
        candidate = mask_dir / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return candidate

    return None


def prepare_output(output_root: Path, reset: bool) -> None:
    """建立 YOLO 資料夾。"""
    if reset and output_root.exists():
        shutil.rmtree(output_root)

    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "metadata" / split).mkdir(parents=True, exist_ok=True)


def process_split(
    split: str,
    cfg: Config,
    geometry_table: dict,
    rng: np.random.Generator,
) -> dict:
    """處理 train、val 或 test。"""
    input_root = Path(cfg.input_root)
    output_root = Path(cfg.output_root)

    image_dir = input_root / split / "images"
    mask_dir = input_root / split / "masks"

    if not image_dir.exists():
        print(f"[略過] 找不到 {image_dir}")
        return {"images": 0, "positive": 0, "negative": 0}

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

    stats = {"images": 0, "positive": 0, "negative": 0}

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[警告] 無法讀取影像：{image_path}")
            continue

        mask_path = find_matching_mask(mask_dir, image_path)
        if mask_path is None:
            # 沒有 mask 時視為全背景。
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[警告] 無法讀取 mask：{mask_path}")
                continue

        if mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"影像與 mask 尺寸不同：{image_path.name}, "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )

        cx, cy, inner_radius, outer_radius = get_geometry(
            image_path=image_path,
            cfg=cfg,
            geometry_table=geometry_table,
        )

        map_x, map_y = build_polar_maps(
            center=(cx, cy),
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            output_width=cfg.unwrap_width,
            output_height=cfg.unwrap_height,
        )

        unwrapped_image = unwrap_image(image, map_x, map_y, is_mask=False)
        unwrapped_mask = unwrap_image(mask, map_x, map_y, is_mask=True)

        for patch_index, start_x in enumerate(
            iter_patch_starts(cfg.unwrap_width, cfg.stride)
        ):
            image_patch = circular_crop(
                unwrapped_image,
                start_x=start_x,
                patch_width=cfg.patch_width,
            )
            mask_patch = circular_crop(
                unwrapped_mask,
                start_x=start_x,
                patch_width=cfg.patch_width,
            )

            label_lines = mask_to_yolo_lines(mask_patch, cfg)
            is_positive = len(label_lines) > 0

            # 訓練集可抽掉部分正常 patch。
            # val/test 建議完整保留，才能正確計算誤報。
            if (
                split == "train"
                and not is_positive
                and rng.random() > cfg.negative_keep_ratio
            ):
                continue

            sample_name = f"{image_path.stem}_p{patch_index:04d}"
            out_image = output_root / "images" / split / f"{sample_name}.png"
            out_label = output_root / "labels" / split / f"{sample_name}.txt"
            out_meta = output_root / "metadata" / split / f"{sample_name}.json"

            ok = cv2.imwrite(str(out_image), image_patch)
            if not ok:
                raise IOError(f"無法寫入：{out_image}")

            out_label.write_text(
                "\n".join(label_lines),
                encoding="utf-8",
            )

            # metadata 用於推論後映射回原始 wafer。
            metadata = {
                "source_image": str(image_path),
                "source_name": image_path.name,
                "patch_index": patch_index,
                "start_x": start_x,
                "patch_width": cfg.patch_width,
                "unwrap_width": cfg.unwrap_width,
                "unwrap_height": cfg.unwrap_height,
                "center_x": cx,
                "center_y": cy,
                "inner_radius": inner_radius,
                "outer_radius": outer_radius,
            }
            out_meta.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            stats["positive" if is_positive else "negative"] += 1

        stats["images"] += 1
        print(f"[{split}] 完成：{image_path.name}")

    return stats


def write_dataset_yaml(cfg: Config) -> None:
    """產生 Ultralytics YOLO dataset.yaml。"""
    output_root = Path(cfg.output_root).resolve()

    data = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(cfg.names)},
    }

    with open(output_root / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定檔路徑",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="先刪除舊輸出資料",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg.patch_width > cfg.unwrap_width:
        raise ValueError("patch_width 不可大於 unwrap_width")
    if cfg.stride <= 0:
        raise ValueError("stride 必須大於 0")
    if not 0.0 <= cfg.negative_keep_ratio <= 1.0:
        raise ValueError("negative_keep_ratio 必須介於 0 和 1")

    output_root = Path(cfg.output_root)
    prepare_output(output_root, reset=args.reset)

    geometry_table = load_geometry_table(cfg.center_json)
    rng = np.random.default_rng(cfg.seed)

    all_stats = {}
    for split in ("train", "val", "test"):
        all_stats[split] = process_split(
            split=split,
            cfg=cfg,
            geometry_table=geometry_table,
            rng=rng,
        )

    write_dataset_yaml(cfg)

    (output_root / "preprocess_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n完成。統計：")
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))
    print(f"\nDataset YAML：{output_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
