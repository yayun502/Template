import os
import cv2
import numpy as np


# ==============================
# CONFIG
# ==============================

INPUT_IMAGE_DIR = "images"
INPUT_LABEL_DIR = "labels"

OUTPUT_IMAGE_DIR = "patch_images"
OUTPUT_LABEL_DIR = "patch_labels"

PATCH_SIZE = 640
OVERLAP = 128

EDGE_BAND_WIDTH = 200


# ==============================
# UTILITY: read YOLO label
# ==============================

def load_yolo_labels(label_path, img_w, img_h):
    """
    YOLO format:
    class cx cy w h  (normalized)
    """
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path) as f:
        for line in f.readlines():
            cls, cx, cy, w, h = map(float, line.strip().split())

            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h

            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2

            boxes.append([int(cls), x1, y1, x2, y2])

    return boxes


# ==============================
# UTILITY: save YOLO label
# ==============================

def save_yolo_labels(label_path, boxes, patch_w, patch_h):

    with open(label_path, "w") as f:

        for cls, x1, y1, x2, y2 in boxes:

            cx = (x1 + x2) / 2 / patch_w
            cy = (y1 + y2) / 2 / patch_h
            w = (x2 - x1) / patch_w
            h = (y2 - y1) / patch_h

            f.write(f"{cls} {cx} {cy} {w} {h}\n")


# ==============================
# Step1: Find wafer ellipse
# ==============================

def detect_wafer_ellipse(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold segmentation
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # choose largest contour (wafer)
    cnt = max(contours, key=cv2.contourArea)

    cnt = cv2.convexHull(cnt)

    ellipse = cv2.fitEllipse(cnt)

    return ellipse


# ==============================
# Step2: Extract edge band
# ==============================

def extract_edge_band(image, ellipse, band_width):

    h, w = image.shape[:2]

    (cx, cy), (major, minor), angle = ellipse

    mask_outer = np.zeros((h, w), dtype=np.uint8)
    mask_inner = np.zeros((h, w), dtype=np.uint8)

    # outer ellipse
    cv2.ellipse(mask_outer, ellipse, 255, -1)

    # inner ellipse
    inner_axes = (int(major/2 - band_width), int(minor/2 - band_width))

    inner_ellipse = ((cx, cy), (inner_axes[0]*2, inner_axes[1]*2), angle)

    cv2.ellipse(mask_inner, inner_ellipse, 255, -1)

    edge_mask = cv2.subtract(mask_outer, mask_inner)

    edge_band = image.copy()
    edge_band[edge_mask == 0] = 0

    return edge_band, edge_mask


# ==============================
# Step3: Patch generation
# ==============================

def generate_patches(image, boxes, filename):

    h, w = image.shape[:2]

    stride = PATCH_SIZE - OVERLAP

    patch_id = 0

    for y in range(0, h - PATCH_SIZE + 1, stride):
        for x in range(0, w - PATCH_SIZE + 1, stride):

            patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            patch_boxes = []

            for cls, x1, y1, x2, y2 in boxes:

                # intersection
                ix1 = max(x1, x)
                iy1 = max(y1, y)
                ix2 = min(x2, x + PATCH_SIZE)
                iy2 = min(y2, y + PATCH_SIZE)

                if ix1 >= ix2 or iy1 >= iy2:
                    continue

                # convert to patch coords
                px1 = ix1 - x
                py1 = iy1 - y
                px2 = ix2 - x
                py2 = iy2 - y

                patch_boxes.append([cls, px1, py1, px2, py2])

            # save patch
            patch_name = f"{filename}_{patch_id}.jpg"

            cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, patch_name), patch)

            label_name = patch_name.replace(".jpg", ".txt")

            save_yolo_labels(
                os.path.join(OUTPUT_LABEL_DIR, label_name),
                patch_boxes,
                PATCH_SIZE,
                PATCH_SIZE
            )

            patch_id += 1


# ==============================
# MAIN PIPELINE
# ==============================

def process_image(img_path):

    filename = os.path.splitext(os.path.basename(img_path))[0]

    label_path = os.path.join(INPUT_LABEL_DIR, filename + ".txt")

    image = cv2.imread(img_path)

    h, w = image.shape[:2]

    boxes = load_yolo_labels(label_path, w, h)

    # detect wafer ellipse
    ellipse = detect_wafer_ellipse(image)

    if ellipse is None:
        print("No wafer detected")
        return

    # edge band
    edge_band, mask = extract_edge_band(image, ellipse, EDGE_BAND_WIDTH)

    # patch generation
    generate_patches(edge_band, boxes, filename)


# ==============================
# RUN
# ==============================

def main():

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    for img_name in os.listdir(INPUT_IMAGE_DIR):

        img_path = os.path.join(INPUT_IMAGE_DIR, img_name)

        process_image(img_path)

        print("processed:", img_name)


if __name__ == "__main__":
    main()
