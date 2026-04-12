from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
CONF_LOG_ROOT = Path("conf_logs")
PRED_DIR = CONF_LOG_ROOT / "predictions"
SUMMARY_DIR = CONF_LOG_ROOT / "summaries"
CM_DIR = SUMMARY_DIR / "confusion_matrices"
FIG_DIR = CONF_LOG_ROOT / "figures"

THRESHOLD_METRICS_PATH = SUMMARY_DIR / "threshold_metrics_multiclass.csv"

# 你可以自行修改
THRESHOLDS = np.linspace(0.0, 1.0, 201)   # 0.000 ~ 1.000
SAVE_CM_FIGURES = True                    # True: 額外輸出 confusion matrix 圖
NORMALIZE_CM_FOR_FIGURE = False            # True: 圖上顯示 row-normalized confusion matrix


# =========================
# Helpers
# =========================
def ensure_dirs() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_CM_FIGURES:
        FIG_DIR.mkdir(parents=True, exist_ok=True)


def is_epoch_prediction_csv(path: Path) -> bool:
    return re.fullmatch(r"epoch_\d+\.csv", path.name) is not None


def extract_epoch(path: Path) -> int:
    m = re.fullmatch(r"epoch_(\d+)\.csv", path.name)
    if not m:
        raise ValueError(f"Cannot parse epoch from {path}")
    return int(m.group(1))


def get_epoch_prediction_csvs() -> list[Path]:
    csvs = [p for p in PRED_DIR.glob("epoch_*.csv") if is_epoch_prediction_csv(p)]
    return sorted(csvs, key=extract_epoch)


def load_prediction_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["epoch"] = df["epoch"].astype(int)
    df["true_cls"] = df["true_cls"].astype(int)
    df["top1_cls"] = df["top1_cls"].astype(int)
    df["top1_conf"] = df["top1_conf"].astype(float)
    return df


def get_num_classes(prediction_csvs: list[Path]) -> int:
    max_cls = -1
    for csv_path in prediction_csvs:
        df = load_prediction_csv(csv_path)
        max_cls = max(max_cls, int(df["true_cls"].max()), int(df["top1_cls"].max()))
    return max_cls + 1


def build_confusion_matrix(
    true_cls: np.ndarray,
    pred_cls: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Standard confusion matrix shape = [num_classes, num_classes]
    rows: true class
    cols: predicted top1 class
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(true_cls, pred_cls):
        cm[int(t), int(p)] += 1

    return cm


def save_confusion_matrix_csv(
    cm: np.ndarray,
    num_classes: int,
    out_csv_path: Path,
) -> None:
    row_labels = [str(i) for i in range(num_classes)]
    col_labels = [str(i) for i in range(num_classes)]

    cm_df = pd.DataFrame(cm, index=row_labels, columns=col_labels)
    cm_df.index.name = "true_cls"
    cm_df.to_csv(out_csv_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    num_classes: int,
    epoch: int,
    out_png_path: Path,
    normalize: bool = False,
) -> None:
    row_labels = [str(i) for i in range(num_classes)]
    col_labels = [str(i) for i in range(num_classes)]

    plot_cm = cm.astype(float)
    if normalize:
        row_sums = plot_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        plot_cm = plot_cm / row_sums

    plt.figure(figsize=(8, 6))
    plt.imshow(plot_cm, interpolation="nearest")
    plt.colorbar()

    plt.xticks(range(len(col_labels)), col_labels, rotation=45)
    plt.yticks(range(len(row_labels)), row_labels)

    plt.xlabel("Predicted top1 class")
    plt.ylabel("True class")
    plt.title(f"Confusion Matrix | epoch={epoch}")

    for i in range(plot_cm.shape[0]):
        for j in range(plot_cm.shape[1]):
            val = plot_cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(j, i, text, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()


# =========================
# Part 1: threshold metrics
# =========================
def build_threshold_metrics_multiclass():
    ensure_dirs()

    prediction_csvs = get_epoch_prediction_csvs()
    if not prediction_csvs:
        print(f"[WARN] No epoch prediction CSV found in {PRED_DIR}")
        return

    rows = []

    for csv_path in prediction_csvs:
        epoch = extract_epoch(csv_path)
        df = load_prediction_csv(csv_path)

        true_cls = df["true_cls"].to_numpy(dtype=int)
        pred_cls = df["top1_cls"].to_numpy(dtype=int)
        conf = df["top1_conf"].to_numpy(dtype=float)

        n = len(df)

        for th in THRESHOLDS:
            correct_mask = (conf >= th) & (pred_cls == true_cls)
            accuracy = float(np.sum(correct_mask) / n) if n > 0 else np.nan

            rows.append({
                "epoch": int(epoch),
                "threshold": float(th),
                "accuracy": float(accuracy),
            })

            print(f"[INFO] epoch={epoch}, th={th:.3f}, acc={accuracy:.4f}")

    out_df = pd.DataFrame(rows).sort_values(["epoch", "threshold"])
    out_df.to_csv(THRESHOLD_METRICS_PATH, index=False)
    print(f"[INFO] Saved threshold metrics to: {THRESHOLD_METRICS_PATH}")


# =========================
# Part 2: confusion matrix per epoch
# =========================
def build_confusion_matrices():
    ensure_dirs()

    prediction_csvs = get_epoch_prediction_csvs()
    if not prediction_csvs:
        print(f"[WARN] No epoch prediction CSV found in {PRED_DIR}")
        return

    num_classes = get_num_classes(prediction_csvs)

    for csv_path in prediction_csvs:
        epoch = extract_epoch(csv_path)
        df = load_prediction_csv(csv_path)

        true_cls = df["true_cls"].to_numpy(dtype=int)
        pred_cls = df["top1_cls"].to_numpy(dtype=int)

        cm = build_confusion_matrix(
            true_cls=true_cls,
            pred_cls=pred_cls,
            num_classes=num_classes,
        )

        cm_csv_path = CM_DIR / f"epoch_{epoch}.csv"
        save_confusion_matrix_csv(cm, num_classes, cm_csv_path)
        print(f"[INFO] Saved confusion matrix CSV to: {cm_csv_path}")

        if SAVE_CM_FIGURES:
            cm_png_path = FIG_DIR / f"epoch_{epoch}.png"
            plot_confusion_matrix(
                cm=cm,
                num_classes=num_classes,
                epoch=epoch,
                out_png_path=cm_png_path,
                normalize=NORMALIZE_CM_FOR_FIGURE,
            )
            print(f"[INFO] Saved confusion matrix figure to: {cm_png_path}")


# =========================
# Main
# =========================
def main():
    build_threshold_metrics_multiclass()
    build_confusion_matrices()


if __name__ == "__main__":
    main()
