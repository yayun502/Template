from ultralytics import YOLO
from pathlib import Path
import csv
import json

# =========================
# Config
# =========================
DATA_ROOT = Path("/home/yayun502/TSMC/TEM/datasets/mnist160")
VAL_PATH = DATA_ROOT / "val"

CONF_LOG_ROOT = Path("conf_logs")
PRED_DIR = CONF_LOG_ROOT / "predictions"
SUMMARY_DIR = CONF_LOG_ROOT / "summaries"

INTERVAL = 5  # every N epochs save one prediction csv
IMG_SUFFIXES = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


# =========================
# Helpers
# =========================
def ensure_dirs() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def collect_image_files(root: Path) -> list[Path]:
    image_files: list[Path] = []
    for pattern in IMG_SUFFIXES:
        image_files.extend(root.rglob(pattern))
    image_files = sorted(image_files)
    return image_files


# =========================
# Callback
# =========================
def on_fit_epoch_end(trainer):
    epoch = trainer.epoch + 1  # human-readable epoch index

    if epoch % INTERVAL != 0:
        return

    ensure_dirs()

    print(f"\n[INFO] Running predict at epoch {epoch}...\n")

    weight_path = trainer.save_dir / "weights" / "last.pt"
    if not weight_path.exists():
        print(f"[WARN] weight not found: {weight_path}")
        return

    image_files = collect_image_files(VAL_PATH)
    if not image_files:
        print(f"[WARN] no image files found under {VAL_PATH}")
        return

    image_files_str = [str(p) for p in image_files]

    yolo_model = YOLO(str(weight_path))
    results = list(yolo_model(image_files_str, stream=True, verbose=False))

    if len(results) != len(image_files):
        print(
            f"[WARN] len(results)={len(results)} != len(image_files)={len(image_files)}. "
            "Only zipped pairs will be written."
        )

    csv_path = PRED_DIR / f"epoch_{epoch}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "input_path",
            "result_path",
            "true_cls",
            "top1_cls",
            "top1_conf",
            "is_correct",
            "probs_json",
        ])

        for img_path, r in zip(image_files, results):
            probs = r.probs.data.detach().cpu()

            top1_conf = float(probs.max().item())
            top1_cls = int(probs.argmax().item())
            true_cls = int(img_path.parent.name)
            is_correct = int(top1_cls == true_cls)

            writer.writerow([
                epoch,
                str(img_path),
                str(r.path),
                true_cls,
                top1_cls,
                top1_conf,
                is_correct,
                json.dumps([float(x) for x in probs.tolist()]),
            ])

    print(f"[INFO] Saved predictions to: {csv_path}\n")


# =========================
# Train
# =========================
def train():
    ensure_dirs()

    model = YOLO("yolo26n-cls.pt")
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    model.train(
        data=str(DATA_ROOT),
        epochs=100,
        imgsz=64,
        batch=16,
        val=True,
        workers=8,
    )


if __name__ == "__main__":
    train()
