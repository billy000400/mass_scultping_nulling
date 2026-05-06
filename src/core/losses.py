"""Loss functions and metrics."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def distance_correlation(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Compute squared distance correlation between x and y.

    Uses double-centered pairwise distance matrices so that dCorr² == 0
    if and only if X and Y are statistically independent.
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    a = torch.cdist(x, x, p=2)
    b = torch.cdist(y, y, p=2)

    A = a - a.mean(dim=1, keepdim=True) - a.mean(dim=0, keepdim=True) + a.mean()
    B = b - b.mean(dim=1, keepdim=True) - b.mean(dim=0, keepdim=True) + b.mean()

    dcov_xy = (A * B).mean()
    dcov_xx = (A * A).mean()
    dcov_yy = (B * B).mean()

    denom = torch.sqrt(dcov_xx * dcov_yy).clamp(min=eps)
    return dcov_xy / denom


def compute_class_weights(
    y: np.ndarray, method: str = "balanced"
) -> Optional[torch.Tensor]:
    if method is None or method == "uniform":
        return None

    y = np.asarray(y).reshape(-1).astype(int)
    unique, counts = np.unique(y, return_counts=True)
    num_classes = len(unique)

    if num_classes == 0:
        return None

    if method == "balanced":
        total = len(y)
        weights = np.zeros(num_classes, dtype=np.float32)
        for i, cls in enumerate(unique):
            if counts[i] > 0:
                weights[cls] = total / (num_classes * counts[i])
        return torch.from_numpy(weights).float()
    raise ValueError(f"Unknown weight method: {method}")


def make_loss_and_metrics(
    loss: str,
    num_classes: Optional[int],
    class_weights: Optional[torch.Tensor] = None,
):
    if loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        def metrics(logits, target_idx):
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == target_idx).float().mean().item()
                balanced_acc = None
                if num_classes is not None and num_classes > 1:
                    class_correct = torch.zeros(num_classes, device=logits.device)
                    class_total = torch.zeros(num_classes, device=logits.device)
                    for c in range(num_classes):
                        mask = target_idx == c
                        if mask.any():
                            class_total[c] = mask.sum().item()
                            class_correct[c] = (
                                (preds == target_idx) & mask
                            ).sum().item()
                    valid_classes = class_total > 0
                    if valid_classes.any():
                        per_class_acc = (
                            class_correct[valid_classes]
                            / class_total[valid_classes].clamp(min=1)
                        )
                        balanced_acc = per_class_acc.mean().item()
                result = {"acc": acc}
                if balanced_acc is not None:
                    result["balanced_acc"] = balanced_acc
                return result

        return criterion, metrics
    elif loss == "mse":
        criterion = nn.MSELoss()

        def metrics(pred, y):
            with torch.no_grad():
                mse = F.mse_loss(pred, y.view_as(pred)).item()
                return {"mse": mse}

        return criterion, metrics
    else:
        raise ValueError("loss must be 'ce' or 'mse'")
