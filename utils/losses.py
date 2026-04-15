import torch
import torch.nn.functional as F


def multiclass_focal_loss(
    logits,
    targets,
    gamma=2.0,
    alpha=None,
    reduction="mean"
):
    """
    logits:  [B, C]
    targets: [B]
    alpha:   [C] or None
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)

    targets = targets.view(-1, 1)
    log_pt = log_probs.gather(1, targets).squeeze(1)
    pt = probs.gather(1, targets).squeeze(1)

    focal_term = (1 - pt) ** gamma

    if alpha is not None:
        at = alpha.gather(0, targets.squeeze(1))
        loss = -at * focal_term * log_pt
    else:
        loss = -focal_term * log_pt

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def compute_total_loss(
    outputs,
    class_labels,
    attr_labels,
    attr_loss_weight=0.5,
    class_weights=None,
    cls_loss_type="ce",
    focal_gamma=2.0
):
    if cls_loss_type == "ce":
        cls_loss = F.cross_entropy(
            outputs["logits"],
            class_labels,
            weight=class_weights
        )

    elif cls_loss_type == "focal":
        cls_loss = multiclass_focal_loss(
            outputs["logits"],
            class_labels,
            gamma=focal_gamma,
            alpha=class_weights,
            reduction="mean"
        )

    else:
        raise ValueError(f"Unsupported cls_loss_type: {cls_loss_type}")

    attr_loss = F.binary_cross_entropy_with_logits(
        outputs["attr_logits"],
        attr_labels.float()
    )

    total_loss = cls_loss + attr_loss_weight * attr_loss

    return total_loss, cls_loss, attr_loss
