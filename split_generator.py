import os
import random
import shutil
from pathlib import Path

"""
This script converts your dataset into YOLOv8 classification format
(two classes: normal / defect)

Input structure:

dataset/
├── Normal/
│   ├── xxx.jpg
├── Solder ring residue/
│   ├── xxx.jpg
│   ├── xxx.json (optional, ignored)

Output structure:
output_dataset/
├── train/
│   ├── normal/
│   ├── defect/
├── val/
│   ├── normal/
│   ├── defect/

Two modes:
1. downsample normal
2. no downsample (for class weight / augmentation usage)
"""

# =========================
# CONFIG
# =========================
INPUT_DIR = "dataset"
OUTPUT_DIR = "output_dataset"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

# Mode 1: downsample normal
DOWNSAMPLE_NORMAL = True
NORMAL_KEEP_RATIO = 0.3   # keep 30% of normal images

# =========================
# UTILS
# =========================

def get_images(folder):
    return [str(p) for p in Path(folder).glob("*.jpg")]


def split_list(data, train_ratio):
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def copy_files(file_list, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for f in file_list:
        shutil.copy(f, dest_folder)


# =========================
# MAIN
# =========================

def main():
    random.seed(42)

    normal_dir = os.path.join(INPUT_DIR, "Normal")
    defect_dir = os.path.join(INPUT_DIR, "Solder ring residue")

    normal_imgs = get_images(normal_dir)
    defect_imgs = get_images(defect_dir)

    print(f"Original normal: {len(normal_imgs)}")
    print(f"Original defect: {len(defect_imgs)}")

    # =========================
    # Mode 1: Downsample normal
    # =========================
    if DOWNSAMPLE_NORMAL:
        keep_num = int(len(normal_imgs) * NORMAL_KEEP_RATIO)
        normal_imgs = random.sample(normal_imgs, keep_num)
        print(f"After downsample normal: {len(normal_imgs)}")

    # =========================
    # Split
    # =========================
    normal_train, normal_val = split_list(normal_imgs, TRAIN_RATIO)
    defect_train, defect_val = split_list(defect_imgs, TRAIN_RATIO)

    # =========================
    # Copy to output
    # =========================
    for split, normal_set, defect_set in [
        ("train", normal_train, defect_train),
        ("val", normal_val, defect_val)
    ]:
        copy_files(normal_set, os.path.join(OUTPUT_DIR, split, "normal"))
        copy_files(defect_set, os.path.join(OUTPUT_DIR, split, "defect"))

    print("Dataset conversion done!")


if __name__ == "__main__":
    main()
