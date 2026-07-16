from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="wafer_edge_dataset/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="wafer_edge_dataset/reports",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    merged_counts: dict[str, int] = {}

    for split_data in data.values():
        for class_name, count in split_data.get(
            "class_instances",
            {}
        ).items():
            merged_counts[class_name] = (
                merged_counts.get(class_name, 0) + count
            )

    if merged_counts:
        names = list(merged_counts.keys())
        counts = [merged_counts[name] for name in names]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(names, counts)
        ax.set_title("Class Instance Distribution")
        ax.set_ylabel("Instance Count")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(
            output_dir / "class_distribution.png",
            dpi=150,
        )
        plt.close(fig)

    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
