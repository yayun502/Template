from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from wafer_edge_pipeline.annotations.factory import load_instances
from wafer_edge_pipeline.annotations.loaders import load_coco_index
from wafer_edge_pipeline.config import load_config
from wafer_edge_pipeline.exporters.yolo import (
    instance_patch_mask_to_yolo_lines,
)
from wafer_edge_pipeline.geometry.polar import build_polar_maps
from wafer_edge_pipeline.geometry.transform import (
    circular_crop,
    unwrap_image,
    unwrap_instance_to_mask,
)
from wafer_edge_pipeline.utils.geometry_loader import (
    load_geometry_table,
    resolve_geometry,
)
from wafer_edge_pipeline.utils.report import DatasetReport


IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff",
}


def prepare_output(root: Path, reset: bool) -> None:
    if reset and root.exists():
        shutil.rmtree(root)

    for split in ("train", "val", "test"):
        for folder in (
            "images",
            "labels",
            "metadata",
            "debug",
        ):
            (root / folder / split).mkdir(
                parents=True,
                exist_ok=True,
            )

    (root / "reports").mkdir(parents=True, exist_ok=True)


def save_dataset_yaml(cfg) -> None:
    root = Path(cfg.output_root).resolve()

    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            index: name
            for index, name in enumerate(cfg.names)
        },
    }

    with open(root / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def draw_debug(
    image: np.ndarray,
    instance_masks: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    canvas = image.copy()

    for class_id, mask in instance_masks:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            canvas,
            contours,
            -1,
            (0, 0, 255),
            2,
        )

    return canvas


def process_split(split: str, cfg, rng, geometry_table) -> dict:
    input_root = Path(cfg.input_root)
    output_root = Path(cfg.output_root)

    image_dir = input_root / split / "images"
    annotation_dir = input_root / split / "annotations"

    report = DatasetReport(cfg.names)

    if not image_dir.exists():
        report.save(output_root / "reports", split)
        return report.to_dict()

    coco_index = None
    if cfg.annotation_format == "coco":
        coco_relative = {
            "train": cfg.coco_train_json,
            "val": cfg.coco_val_json,
            "test": cfg.coco_test_json,
        }[split]

        coco_index = load_coco_index(
            input_root / coco_relative
        )

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
    )

    debug_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"無法讀取：{image_path}")

        geometry = resolve_geometry(
            image_path,
            cfg,
            geometry_table,
        )

        map_x, map_y = build_polar_maps(
            geometry,
            cfg.unwrap_width,
            cfg.unwrap_height,
        )

        unwrapped_image = unwrap_image(
            image,
            map_x,
            map_y,
        )

        instances = load_instances(
            annotation_format=cfg.annotation_format,
            image_path=image_path,
            image_shape=image.shape[:2],
            annotation_dir=annotation_dir,
            class_names=cfg.names,
            min_area=cfg.min_area,
            coco_index=coco_index,
        )

        for instance in instances:
            if instance.class_id >= len(cfg.names):
                raise ValueError(
                    f"{image_path.name} 有超出 names 範圍的 class_id="
                    f"{instance.class_id}"
                )
            report.add_instance(instance.class_id)

        unwrapped_instance_masks: list[tuple[int, np.ndarray]] = []

        for instance in instances:
            mask = unwrap_instance_to_mask(
                instance,
                image.shape[:2],
                map_x,
                map_y,
            )
            unwrapped_instance_masks.append(
                (instance.class_id, mask)
            )

        if (
            cfg.save_debug_images
            and debug_count < cfg.debug_max_per_split
        ):
            debug = draw_debug(
                unwrapped_image,
                unwrapped_instance_masks,
            )
            cv2.imwrite(
                str(
                    output_root
                    / "debug"
                    / split
                    / f"{image_path.stem}_unwrapped_instances.png"
                ),
                debug,
            )
            debug_count += 1

        for patch_index, start_x in enumerate(
            range(0, cfg.unwrap_width, cfg.stride)
        ):
            image_patch = circular_crop(
                unwrapped_image,
                start_x,
                cfg.patch_width,
            )

            label_lines: list[str] = []

            for class_id, full_mask in unwrapped_instance_masks:
                patch_mask = circular_crop(
                    full_mask,
                    start_x,
                    cfg.patch_width,
                )

                lines = instance_patch_mask_to_yolo_lines(
                    patch_mask=patch_mask,
                    class_id=class_id,
                    task=cfg.task,
                    min_area=cfg.min_area,
                    epsilon_ratio=cfg.polygon_epsilon_ratio,
                )
                label_lines.extend(lines)

            is_positive = bool(label_lines)

            if (
                split == "train"
                and not is_positive
                and rng.random() > cfg.negative_keep_ratio
            ):
                continue

            sample_name = f"{image_path.stem}_p{patch_index:04d}"

            out_image = (
                output_root
                / "images"
                / split
                / f"{sample_name}.png"
            )
            out_label = (
                output_root
                / "labels"
                / split
                / f"{sample_name}.txt"
            )
            out_meta = (
                output_root
                / "metadata"
                / split
                / f"{sample_name}.json"
            )

            cv2.imwrite(str(out_image), image_patch)
            out_label.write_text(
                "\n".join(label_lines),
                encoding="utf-8",
            )

            metadata = {
                "source_name": image_path.name,
                "patch_index": patch_index,
                "start_x": start_x,
                "unwrap_width": cfg.unwrap_width,
                "unwrap_height": cfg.unwrap_height,
                "patch_width": cfg.patch_width,
                "center_x": geometry.center_x,
                "center_y": geometry.center_y,
                "inner_radius": geometry.inner_radius,
                "outer_radius": geometry.outer_radius,
            }

            out_meta.write_text(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            report.add_patch(is_positive)

        report.add_source_image()
        print(
            f"[{split}] {image_path.name}: "
            f"{len(instances)} instances"
        )

    report.save(output_root / "reports", split)
    return report.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg.output_root)

    prepare_output(output_root, args.reset)

    rng = np.random.default_rng(cfg.seed)
    geometry_table = load_geometry_table(cfg.center_json)

    summary = {}
    for split in ("train", "val", "test"):
        summary[split] = process_split(
            split,
            cfg,
            rng,
            geometry_table,
        )

    save_dataset_yaml(cfg)

    (output_root / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
