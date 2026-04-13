import torch.nn.functional as F


def compute_total_loss(outputs, class_labels, attr_labels, attr_loss_weight=0.5):
    cls_loss = F.cross_entropy(outputs["logits"], class_labels)

    attr_loss = F.binary_cross_entropy_with_logits(
        outputs["attr_logits"],
        attr_labels.float()
    )

    total_loss = cls_loss + attr_loss_weight * attr_loss

    return total_loss, cls_loss, attr_loss
