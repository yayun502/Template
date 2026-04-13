import os
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms

from configs.config import (
    LABEL_MAP, IDX2LABEL,
    IMAGE_SIZE, MAX_LOCAL_VIEWS, MAX_GLOBAL_VIEWS, LOCAL_FOV_THRESHOLD,
    BACKBONE_NAME, FEAT_DIM, NUM_CLASSES, NUM_ATTRS, DEVICE
)
from models.defect_model import DefectClassifier


def load_image(path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)


def pad_images(imgs, max_num):
    if len(imgs) == 0:
        raise ValueError("No images to pad.")

    c, h, w = imgs[0].shape
    padded = torch.zeros(max_num, c, h, w)
    mask = torch.zeros(max_num, dtype=torch.long)

    actual_num = min(len(imgs), max_num)
    for i in range(actual_num):
        padded[i] = imgs[i]
        mask[i] = 1

    return padded, mask


def build_sample_tensors(sample_dir):
    meta_path = os.path.join(sample_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    local_imgs = []
    global_imgs = []

    for item in meta["images"]:
        img_path = os.path.join(sample_dir, item["file"])
        fov = item["fov"]
        img = load_image(img_path, IMAGE_SIZE)

        if fov <= LOCAL_FOV_THRESHOLD:
            local_imgs.append(img)
        else:
            global_imgs.append(img)

    if len(local_imgs) == 0 and len(global_imgs) > 0:
        local_imgs.append(global_imgs[0].clone())
    if len(global_imgs) == 0 and len(local_imgs) > 0:
        global_imgs.append(local_imgs[0].clone())

    local_imgs, local_mask = pad_images(local_imgs, MAX_LOCAL_VIEWS)
    global_imgs, global_mask = pad_images(global_imgs, MAX_GLOBAL_VIEWS)

    # add batch dimension
    local_imgs = local_imgs.unsqueeze(0)
    global_imgs = global_imgs.unsqueeze(0)
    local_mask = local_mask.unsqueeze(0)
    global_mask = global_mask.unsqueeze(0)

    return local_imgs, global_imgs, local_mask, global_mask


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model = DefectClassifier(
        backbone_name=BACKBONE_NAME,
        feat_dim=FEAT_DIM,
        num_classes=NUM_CLASSES,
        num_attrs=NUM_ATTRS,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    local_imgs, global_imgs, local_mask, global_mask = build_sample_tensors(args.sample_dir)
    local_imgs = local_imgs.to(device)
    global_imgs = global_imgs.to(device)
    local_mask = local_mask.to(device)
    global_mask = global_mask.to(device)

    outputs = model(
        local_imgs=local_imgs,
        global_imgs=global_imgs,
        local_mask=local_mask,
        global_mask=global_mask
    )

    probs = torch.softmax(outputs["logits"], dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    pred_name = IDX2LABEL[pred_idx]

    attr_probs = torch.sigmoid(outputs["attr_logits"])[0].cpu().tolist()
    local_attn = outputs["local_attn"][0].cpu().tolist()
    global_attn = outputs["global_attn"][0].cpu().tolist()

    print("===== Prediction =====")
    print(f"Predicted class: {pred_name}")
    print("Class probabilities:")
    for i in range(len(probs)):
        print(f"  {IDX2LABEL[i]}: {probs[i].item():.4f}")

    print("\nAttribute probabilities:")
    print(f"  has_np_pattern:    {attr_probs[0]:.4f}")
    print(f"  is_repetitive:     {attr_probs[1]:.4f}")
    print(f"  has_breakpoint:    {attr_probs[2]:.4f}")
    print(f"  is_single_struct:  {attr_probs[3]:.4f}")

    print("\nLocal attention weights:")
    print(local_attn)

    print("\nGlobal attention weights:")
    print(global_attn)


if __name__ == "__main__":
    main()
