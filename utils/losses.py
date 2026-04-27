import torch
import torch.nn.functional as F

from utils.hierarchical import (
    hierarchical_nll_loss,
    vanilla_gate_loss,
    conditional_gate_loss
)


def multiclass_focal_loss(
    logits,
    targets,
    gamma=2.0,
    alpha=None,
    reduction="mean"
):
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
    return loss


def branch_entropy_regularization(branch_weights):
    """
    branch_weights: [B, 2]

    回傳 -entropy。
    加到 total loss 後，會鼓勵 entropy 變大，避免 branch gate 太早 collapse。
    """
    eps = 1e-8
    entropy = -torch.sum(
        branch_weights * torch.log(branch_weights + eps),
        dim=1
    ).mean()

    return -entropy


def compute_total_loss(
    outputs,
    class_labels,
    class_weights=None,
    cls_loss_type="ce",
    focal_gamma=2.0,
    use_hierarchical=False,
    hier_loss_weight=0.0,
    gate_loss_weight=0.3,
    use_conditional_gate_loss=True,
    gate_np_weight=0.5,
    gate_single_weight=1.0,
    gate_bp_weight=1.0,
    use_branch_entropy_reg=False,
    branch_entropy_weight=0.01
):
    # ----- Main 4-class loss -----
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

    total_loss = cls_loss

    device = class_labels.device

    hier_loss = torch.tensor(0.0, device=device)
    gate_loss = torch.tensor(0.0, device=device)
    np_gate_loss = torch.tensor(0.0, device=device)
    single_gate_loss = torch.tensor(0.0, device=device)
    bp_gate_loss = torch.tensor(0.0, device=device)
    branch_entropy_loss = torch.tensor(0.0, device=device)

    if use_hierarchical:
        # Hierarchical NLL loss
        # 若 hier_loss_weight = 0，這項仍會被算出來方便 log，
        # 但不會影響 total loss。
        hier_loss = hierarchical_nll_loss(
            outputs["hier_logits"],
            class_labels,
            class_weights=class_weights
        )

        # Gate loss
        if use_conditional_gate_loss:
            gate_loss, np_gate_loss, single_gate_loss, bp_gate_loss = conditional_gate_loss(
                hier_logits=outputs["hier_logits"],
                class_labels=class_labels,
                np_weight=gate_np_weight,
                single_weight=gate_single_weight,
                bp_weight=gate_bp_weight
            )
        else:
            gate_loss, np_gate_loss, single_gate_loss, bp_gate_loss = vanilla_gate_loss(
                hier_logits=outputs["hier_logits"],
                class_labels=class_labels,
                np_weight=gate_np_weight,
                single_weight=gate_single_weight,
                bp_weight=gate_bp_weight
            )

        total_loss = (
            total_loss
            + hier_loss_weight * hier_loss
            + gate_loss_weight * gate_loss
        )

    if use_branch_entropy_reg:
        branch_entropy_loss = branch_entropy_regularization(
            outputs["branch_weights"]
        )
        total_loss = total_loss + branch_entropy_weight * branch_entropy_loss

    return {
        "total_loss": total_loss,
        "cls_loss": cls_loss,
        "hier_loss": hier_loss,
        "gate_loss": gate_loss,
        "np_gate_loss": np_gate_loss,
        "single_gate_loss": single_gate_loss,
        "bp_gate_loss": bp_gate_loss,
        "branch_entropy_loss": branch_entropy_loss
    }
