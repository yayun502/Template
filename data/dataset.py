import os
import json
from typing import List

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DefectSampleDataset(Dataset):
    def __init__(
        self,
        sample_dirs: List[str],
        label_map: dict,
        image_size=224,
        max_local=6,
        max_global=6,
        local_fov_threshold=29,
        is_train=True
    ):
        self.sample_dirs = sample_dirs
        self.label_map = label_map
        self.image_size = image_size
        self.max_local = max_local
        self.max_global = max_global
        self.local_fov_threshold = local_fov_threshold
        self.is_train = is_train

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.sample_dirs)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def _pad_images(self, imgs, max_num):
        if len(imgs) == 0:
            raise ValueError("Branch has zero images.")

        c, h, w = imgs[0].shape
        padded = torch.zeros(max_num, c, h, w)
        mask = torch.zeros(max_num, dtype=torch.long)

        actual_num = min(len(imgs), max_num)
        for i in range(actual_num):
            padded[i] = imgs[i]
            mask[i] = 1

        return padded, mask

    def build_attr_labels(self, label_name):
        """
        [has_np_pattern, is_repetitive, has_breakpoint, is_single_structure]
        先用簡化版規則；未來可換成人工標註
        """
        if label_name == "Single":
            return [0, 1, 0, 1]
        elif label_name == "NP":
            return [1, 1, 0, 0]
        elif label_name == "定點":
            return [0, 0, 1, 0]
        elif label_name == "Multi":
            return [0, 0, 0, 0]
        else:
            raise ValueError(f"Unknown label: {label_name}")

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        label_path = os.path.join(sample_dir, "label.txt")
        meta_path = os.path.join(sample_dir, "meta.json")

        with open(label_path, "r", encoding="utf-8") as f:
            label_name = f.read().strip()

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        local_imgs = []
        global_imgs = []

        for item in meta["images"]:
            img_path = os.path.join(sample_dir, item["file"])
            fov = item["fov"]
            img = self._load_image(img_path)

            if fov <= self.local_fov_threshold:
                local_imgs.append(img)
            else:
                global_imgs.append(img)

        # 如果某 branch 沒圖，先用另一 branch 的第一張補
        if len(local_imgs) == 0 and len(global_imgs) > 0:
            local_imgs.append(global_imgs[0].clone())
        if len(global_imgs) == 0 and len(local_imgs) > 0:
            global_imgs.append(local_imgs[0].clone())

        if len(local_imgs) == 0 or len(global_imgs) == 0:
            raise ValueError(f"Sample {sample_dir} has no valid images.")

        local_imgs, local_mask = self._pad_images(local_imgs, self.max_local)
        global_imgs, global_mask = self._pad_images(global_imgs, self.max_global)

        label = self.label_map[label_name]
        attr_labels = self.build_attr_labels(label_name)

        return {
            "local_imgs": local_imgs,
            "global_imgs": global_imgs,
            "local_mask": local_mask,
            "global_mask": global_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "attr_labels": torch.tensor(attr_labels, dtype=torch.float32),
            "sample_dir": sample_dir
        }


def get_sample_dirs(root_dir):
    sample_dirs = []
    if not os.path.exists(root_dir):
        return sample_dirs

    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            sample_dirs.append(path)
    return sample_dirs
