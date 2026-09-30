"""Soft-label BCE plus an in-batch pairwise AUC surrogate (lever L5, cf. LibAUC)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_soft_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * w).sum() / w.sum().clamp_min(1e-6)


def pairwise_auc_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Squared-hinge ranking loss over (positive, negative) pairs within each target column.

    Positives are cells with soft label >= 0.5, negatives < 0.5; zero-weight cells are ignored.
    Pairs are weighted by the product of cell weights and by how far apart the soft labels are,
    so uncertain (≈0.5) labels contribute little.
    """
    total, norm = logits.new_zeros(()), logits.new_zeros(())
    for k in range(logits.shape[1]):
        s, t, c = logits[:, k], y[:, k], w[:, k]
        pos, neg = (t >= 0.5) & (c > 0), (t < 0.5) & (c > 0)
        if pos.any() and neg.any():
            diff = s[pos][:, None] - s[neg][None, :]
            pw = (c[pos][:, None] * c[neg][None, :]) * (t[pos][:, None] - t[neg][None, :]).abs()
            total = total + (pw * F.relu(margin - diff) ** 2).sum()
            norm = norm + pw.sum()
    return total / norm.clamp_min(1e-6)


def composite_loss(logits, y, w, auc_weight: float = 0.1) -> torch.Tensor:
    loss = weighted_soft_bce(logits, y, w)
    if auc_weight > 0:
        loss = loss + auc_weight * pairwise_auc_loss(logits, y, w)
    return loss
