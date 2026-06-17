from pathlib import Path
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.loss import v8ClassificationLoss
from torchvision.datasets import ImageFolder

DATA = "your_dataset"        # 裡面要有 train/, val/
MODEL = "yolov8m-cls.pt"

GAMMA = 0.0                  # 0.0 = weighted CE；1.0/2.0 = weighted focal loss
MAX_WEIGHT = 5.0             # 先用 5，比較穩


def compute_class_weights(train_dir):
    dataset = ImageFolder(train_dir)

    num_classes = len(dataset.classes)

    counts = torch.zeros(num_classes)

    for _, cls in dataset.samples:
        counts[cls] += 1

    weights = counts.sum() / counts
    weights = weights / weights.mean()
    weights = torch.clamp(weights, max=5.0)

    print("\nYOLO class mapping:")
    for name, idx in dataset.class_to_idx.items():
        print(
            f"idx={idx:02d}, "
            f"class={name}, "
            f"count={int(counts[idx])}, "
            f"weight={weights[idx]:.3f}"
        )

    return weights


class WeightedFocalLoss:
    def __init__(self, weights, gamma=0.0):
        self.weights = weights
        self.gamma = gamma

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        targets = batch["cls"].long().view(-1)

        weights = self.weights.to(preds.device)

        if self.gamma == 0.0:
            loss = F.cross_entropy(preds, targets, weight=weights)
        else:
            ce = F.cross_entropy(preds, targets, reduction="none")
            pt = torch.exp(-ce)

            ce_weighted = F.cross_entropy(preds, targets, weight=weights, reduction="none")
            loss = ((1 - pt) ** self.gamma * ce_weighted).mean()

        return loss, loss.detach()


if __name__ == "__main__":
    weights = compute_class_weights(Path(DATA) / "train")

    # monkey patch YOLO classification loss
    v8ClassificationLoss.__call__ = lambda self, preds, batch: WeightedFocalLoss(
        weights=weights,
        gamma=GAMMA,
    )(preds, batch)

    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=100,
        imgsz=224,
        batch=64,
        device=0,
        project="runs/classify",
        name=f"weighted_gamma{GAMMA}",
    )
# ----------------------------------------
from pathlib import Path
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.nn.tasks import ClassificationModel
from torchvision.datasets import ImageFolder


DATA = "your_dataset"          # dataset/train, dataset/val
MODEL = "yolov8m-cls.pt"

GAMMA = 0.0                    # 0.0 = weighted CE；1.0/2.0 = weighted focal
WEIGHT_METHOD = "effective"    # "effective", "inverse", "inverse_sqrt"
BETA = 0.9999                  # only used when WEIGHT_METHOD="effective"
MAX_WEIGHT = 5.0
MIN_WEIGHT = 0.05


def compute_weights(train_dir):
    dataset = ImageFolder(train_dir)

    num_classes = len(dataset.classes)
    counts = torch.zeros(num_classes, dtype=torch.float32)

    for _, cls in dataset.samples:
        counts[cls] += 1

    if WEIGHT_METHOD == "effective":
        weights = (1.0 - BETA) / (1.0 - torch.pow(BETA, counts.clamp(min=1)))

    elif WEIGHT_METHOD == "inverse":
        weights = 1.0 / counts.clamp(min=1)

    elif WEIGHT_METHOD == "inverse_sqrt":
        weights = 1.0 / torch.sqrt(counts.clamp(min=1))

    else:
        raise ValueError(f"Unknown WEIGHT_METHOD: {WEIGHT_METHOD}")

    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=MIN_WEIGHT, max=MAX_WEIGHT)

    print("\nYOLO class mapping / class weights:")
    for name, idx in dataset.class_to_idx.items():
        print(
            f"idx={idx:02d}, "
            f"class={name}, "
            f"count={int(counts[idx])}, "
            f"weight={weights[idx]:.4f}"
        )

    return weights


class WeightedClsLoss:
    def __init__(self, weights, gamma=0.0):
        self.weights = weights.float()
        self.gamma = float(gamma)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        targets = batch["cls"].long().view(-1)

        weights = self.weights.to(device=preds.device, dtype=preds.dtype)

        if self.gamma == 0.0:
            loss = F.cross_entropy(
                preds,
                targets,
                weight=weights,
                reduction="mean",
            )
        else:
            ce = F.cross_entropy(
                preds,
                targets,
                reduction="none",
            )

            pt = torch.exp(-ce)

            ce_w = F.cross_entropy(
                preds,
                targets,
                weight=weights,
                reduction="none",
            )

            loss = (((1.0 - pt) ** self.gamma) * ce_w).mean()

        return loss, loss.detach()


class WeightedClassificationModel(ClassificationModel):
    def __init__(self, cfg, ch=3, nc=None, verbose=True, weights=None, gamma=0.0):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.custom_weights = weights
        self.gamma = gamma

    def init_criterion(self):
        return WeightedClsLoss(
            weights=self.custom_weights,
            gamma=self.gamma,
        )


class WeightedTrainer(ClassificationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        train_dir = Path(self.args.data) / "train"
        class_weights = compute_weights(train_dir)

        model = WeightedClassificationModel(
            cfg=cfg,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose,
            weights=class_weights,
            gamma=GAMMA,
        )

        if weights:
            model.load(weights)

        return model


if __name__ == "__main__":
    model = YOLO(MODEL)

    model.train(
        trainer=WeightedTrainer,
        data=DATA,
        epochs=100,
        imgsz=224,
        batch=64,
        device=0,
        project="runs/classify",
        name=f"{WEIGHT_METHOD}_gamma{GAMMA}",
    )
  
# #check
# from torchvision.datasets import ImageFolder

# dataset = ImageFolder("dataset/train")

# print(dataset.class_to_idx)
