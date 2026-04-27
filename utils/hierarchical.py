import torch
import torch.nn.functional as F


def hierarchical_probs_from_logits(hier_logits):
    """
    hier_logits: [B, 3]
      [:, 0] -> NP gate
      [:, 1] -> Single gate
      [:, 2] -> Breakpoint gate

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
    probs = hierarchical_probs_from_logits(hier_logits)
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


def vanilla_gate_loss(
    hier_logits,
    class_labels,
    np_weight=1.0,
    single_weight=1.0,
    bp_weight=1.0
):
    """
    一般 gate loss：
    所有樣本都會計算三個 gate。

    這個版本比較像 label extension。
    """
    gate_targets = build_gate_targets(class_labels)

    np_loss = F.binary_cross_entropy_with_logits(
        hier_logits[:, 0],
        gate_targets[:, 0],
        reduction="mean"
    )

    single_loss = F.binary_cross_entropy_with_logits(
        hier_logits[:, 1],
        gate_targets[:, 1],
        reduction="mean"
    )

    bp_loss = F.binary_cross_entropy_with_logits(
        hier_logits[:, 2],
        gate_targets[:, 2],
        reduction="mean"
    )

    total_weight = np_weight + single_weight + bp_weight
    gate_loss = (
        np_weight * np_loss +
        single_weight * single_loss +
        bp_weight * bp_loss
    ) / total_weight

    return gate_loss, np_loss, single_loss, bp_loss


def conditional_gate_loss(
    hier_logits,
    class_labels,
    np_weight=0.5,
    single_weight=1.0,
    bp_weight=1.0
):
    """
    Conditional gate loss，較符合 flow chart：

    class order:
      0 = Single
      1 = NP
      2 = 定點
      3 = Multi

    Gate 1:
      NP vs non-NP
      所有樣本都算

    Gate 2:
      Single vs non-Single
      只在 non-NP 樣本上算

    Gate 3:
      Breakpoint / 定點 vs Multi
      只在 non-NP 且 non-Single 的樣本上算
      也就是只在 定點 / Multi 上算
    """
    device = class_labels.device

    np_logit = hier_logits[:, 0]
    single_logit = hier_logits[:, 1]
    bp_logit = hier_logits[:, 2]

    # Gate 1: NP vs non-NP, all samples
    np_target = (class_labels == 1).float()
    np_loss = F.binary_cross_entropy_with_logits(
        np_logit,
        np_target,
        reduction="mean"
    )

    # Gate 2: Single vs non-Single, only non-NP samples
    non_np_mask = class_labels != 1
    if non_np_mask.sum() > 0:
        single_target = (class_labels[non_np_mask] == 0).float()
        single_loss = F.binary_cross_entropy_with_logits(
            single_logit[non_np_mask],
            single_target,
            reduction="mean"
        )
    else:
        single_loss = torch.tensor(0.0, device=device)

    # Gate 3: Breakpoint, only 定點 / Multi samples
    bp_mask = (class_labels != 1) & (class_labels != 0)
    if bp_mask.sum() > 0:
        bp_target = (class_labels[bp_mask] == 2).float()
        bp_loss = F.binary_cross_entropy_with_logits(
            bp_logit[bp_mask],
            bp_target,
            reduction="mean"
        )
    else:
        bp_loss = torch.tensor(0.0, device=device)

    total_weight = np_weight + single_weight + bp_weight

    gate_loss = (
        np_weight * np_loss +
        single_weight * single_loss +
        bp_weight * bp_loss
    ) / total_weight

    return gate_loss, np_loss, single_loss, bp_loss
