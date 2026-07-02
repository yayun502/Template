from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np


class BalancedClassificationTrainer(ClassificationTrainer):
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        # 先用原本 YOLO 的方式建立 dataset
        dataset = self.build_dataset(dataset_path, mode)

        if mode == "train":
            # Ultralytics classification dataset 通常有 samples: [(path, cls), ...]
            targets = [s[1] for s in dataset.samples]
            targets = np.array(targets)

            class_counts = np.bincount(targets)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[targets]

            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )

            shuffle = False
        else:
            sampler = None
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.args.workers,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fn", None),
        )


model = YOLO("yolov8m-cls.pt")

model.train(
    data="your_dataset",
    trainer=BalancedClassificationTrainer,
    epochs=50,
    imgsz=224,
    batch=16,
)
# ====================================
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch


class MyClsDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # [(img_path, label), ...]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {
            "img": img,
            "cls": torch.tensor(label, dtype=torch.long),
        }


class MyClassificationTrainer(ClassificationTrainer):
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = MyClsDataset(
            samples=self.train_samples if mode == "train" else self.val_samples,
            transform=None,  # 這裡放你的 transform
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == "train"),
            num_workers=self.args.workers,
            pin_memory=True,
        )


args = dict(
    model="yolov8m-cls.pt",
    data="dummy_path",   # 還是需要給，但可在 trainer 內部不用它
    epochs=50,
    imgsz=224,
    batch=16,
)

trainer = MyClassificationTrainer(overrides=args)
trainer.train()
