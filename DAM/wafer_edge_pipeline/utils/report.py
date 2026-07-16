from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


class DatasetReport:
    def __init__(self, class_names: tuple[str, ...]) -> None:
        self.class_names = class_names
        self.source_images = 0
        self.positive_patches = 0
        self.negative_patches = 0
        self.class_instances = Counter()

    def add_source_image(self) -> None:
        self.source_images += 1

    def add_patch(self, positive: bool) -> None:
        if positive:
            self.positive_patches += 1
        else:
            self.negative_patches += 1

    def add_instance(self, class_id: int) -> None:
        self.class_instances[class_id] += 1

    def to_dict(self) -> dict:
        return {
            "source_images": self.source_images,
            "positive_patches": self.positive_patches,
            "negative_patches": self.negative_patches,
            "class_instances": {
                self.class_names[class_id]: count
                for class_id, count
                in sorted(self.class_instances.items())
            },
        }

    def save(self, output_dir: Path, split: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"{split}_report.json"
        json_path.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        csv_path = output_dir / f"{split}_class_counts.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "instance_count"])

            for class_id, class_name in enumerate(self.class_names):
                writer.writerow([
                    class_id,
                    class_name,
                    self.class_instances.get(class_id, 0),
                ])
