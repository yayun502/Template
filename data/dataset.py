import os
from typing import List

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from utils.sample_ordering import load_and_prepare_sample_items


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

        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
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

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        label_path = os.path.join(sample_dir, "label.txt")
        with open(label_path, "r", encoding="utf-8") as f:
            label_name = f.read().strip()

        local_items, global_items = load_and_prepare_sample_items(
            sample_dir=sample_dir,
            local_fov_threshold=self.local_fov_threshold,
            max_local=self.max_local,
            max_global=self.max_global
        )

        if len(local_items) == 0 or len(global_items) == 0:
            raise ValueError(f"Sample {sample_dir} has no valid images after preparation.")

        local_imgs = [
            self._load_image(os.path.join(sample_dir, item["image_name"]))
            for item in local_items
        ]
        global_imgs = [
            self._load_image(os.path.join(sample_dir, item["image_name"]))
            for item in global_items
        ]

        local_imgs, local_mask = self._pad_images(local_imgs, self.max_local)
        global_imgs, global_mask = self._pad_images(global_imgs, self.max_global)

        label = self.label_map[label_name]

        return {
            "local_imgs": local_imgs,
            "global_imgs": global_imgs,
            "local_mask": local_mask,
            "global_mask": global_mask,
            "label": torch.tensor(label, dtype=torch.long),
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
