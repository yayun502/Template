import torch
import torch.nn.functional as F


def hierarchical_probs_from_logits(hier_logits):
    """
    hier_logits: [B, 3]
      [:, 0] -> np gate
      [:, 1] -> single gate
      [:, 2] -> breakpoint gate

    return:
      probs_4cls: [B, 4]
      order = [Single, NP, 定點, Multi]
    """
    p_np = torch.sigmoid(hier_logits[:, 0])
    p_single = torch.sigmoid(hier_logits[:, 1])
    p_bp = torch.sigmoid(hier_logits[:, 2])

    p_cls_np = p_np
    p_cls_single = (1 - p_np) * p_single
    p_cls_point = (1 - p_np) * (1 - p_single) * p_bp
    p_cls_multi = (1 - p_np) * (1 - p_single) * (1 - p_bp)

    probs_4cls = torch.stack(
        [p_cls_single, p_cls_np, p_cls_point, p_cls_multi],
        dim=1
    )
    return probs_4cls


def hierarchical_nll_loss(hier_logits, targets, class_weights=None, eps=1e-8):
    probs = hierarchical_probs_from_logits(hier_logits)  # [B, 4]
    probs = probs.clamp(min=eps, max=1.0)

    p_true = probs.gather(1, targets.view(-1, 1)).squeeze(1)
    loss = -torch.log(p_true)

    if class_weights is not None:
        w = class_weights.gather(0, targets)
        loss = loss * w

    return loss.mean()


def build_gate_targets(class_labels):
    """
    class order:
      0 = Single
      1 = NP
      2 = 定點
      3 = Multi

    return:
      [B, 3] -> [is_np, is_single, has_breakpoint]
    """
    is_np = (class_labels == 1).float()
    is_single = (class_labels == 0).float()
    has_bp = (class_labels == 2).float()

    gate_targets = torch.stack([is_np, is_single, has_bp], dim=1)
    return gate_targets
