import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cb_weights(counts, beta: float = 0.9999) -> np.ndarray:
    """Class-balanced weights via effective number of samples (Cui et al., CVPR 2019)."""
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * len(weights)
    return weights.astype(np.float32)


def compute_ldam_margins(counts, max_m: float = 0.5) -> np.ndarray:
    """Per-class margins ∝ 1 / n^0.25, normalized to max_m."""
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    margins = 1.0 / np.sqrt(np.sqrt(counts))
    margins = margins * (max_m / margins.max())
    return margins.astype(np.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss. Ignores targets == -1."""

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != -1
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            focal_weight = self.alpha[targets] * focal_weight

        loss = -focal_weight * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class PerClassFocalLoss(nn.Module):
    """Focal loss with per-class alpha and gamma (for weak-class targeting)."""

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.register_buffer("gamma", gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != -1
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        gamma_t = self.gamma[targets]
        focal_weight = (1.0 - pt) ** gamma_t
        focal_weight = self.alpha[targets] * focal_weight

        loss = -focal_weight * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_weak_class_loss_params(
    counts: np.ndarray,
    class_names: list,
    *,
    prior_f1: dict | None = None,
    f1_threshold: float = 0.45,
    base_gamma: float = 2.5,
    gamma_boost: float = 1.0,
    max_gamma: float = 4.5,
    max_weight: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-class focal alpha/gamma from counts + optional prior validation F1.

    Weak classes (F1 < threshold) get higher gamma (harder example focus).
    Classes that are over-predicted (precision < recall) get *reduced* alpha to
    curb false positives; under-predicted weak classes get a mild alpha boost.
    """
    num_classes = len(counts)
    counts = np.where(counts == 0, 1, counts).astype(float)

    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    min_w = weights[weights > 0].min()
    weights = np.clip(weights, min_w, min_w * max_weight)

    gammas = np.full(num_classes, base_gamma, dtype=np.float32)

    if prior_f1:
        for i, name in enumerate(class_names):
            stats = prior_f1.get(name)
            if not stats or not isinstance(stats, dict):
                continue
            f1 = float(stats.get("f1-score", 1.0))
            prec = float(stats.get("precision", 1.0))
            rec = float(stats.get("recall", 1.0))
            if f1 >= f1_threshold:
                continue

            weakness = (f1_threshold - f1) / max(f1_threshold, 1e-8)
            gammas[i] = min(base_gamma + gamma_boost * weakness, max_gamma)

            if prec < rec:
                # Over-predicted: dampen loss weight to reduce false-positive pressure.
                weights[i] *= max(prec / max(rec, 1e-8), 0.5)
            else:
                # Under-predicted weak class: mild boost.
                weights[i] *= 1.0 + 0.5 * weakness

    return weights.astype(np.float32), gammas.astype(np.float32)


def compute_weak_class_sampler_multipliers(
    class_names: list,
    *,
    prior_f1: dict | None = None,
    f1_threshold: float = 0.45,
    weak_boost: float = 1.8,
) -> np.ndarray:
    """
    Per-class sampler multipliers for weak classes.

  Only boosts under-predicted weak classes (precision >= recall); skips
  over-predicted ones where extra exposure worsens precision.
    """
    multipliers = np.ones(len(class_names), dtype=np.float64)
    if not prior_f1:
        return multipliers

    for i, name in enumerate(class_names):
        stats = prior_f1.get(name)
        if not stats or not isinstance(stats, dict):
            continue
        f1 = float(stats.get("f1-score", 1.0))
        prec = float(stats.get("precision", 1.0))
        rec = float(stats.get("recall", 1.0))
        if f1 >= f1_threshold:
            continue
        if prec >= rec:
            weakness = (f1_threshold - f1) / max(f1_threshold, 1e-8)
            multipliers[i] = 1.0 + (weak_boost - 1.0) * weakness
    return multipliers


class LabelSmoothingCrossEntropy(nn.Module):
    """Label-smoothing CE as an alternative to FocalLoss."""

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != -1
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss: effective-number reweighting + focal hard-example
    focus. Optional label smoothing stabilises minority-class boundaries.
    """

    def __init__(
        self,
        samples_per_class: np.ndarray,
        beta: float = 0.999,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        max_weight: float = 20.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        weights = compute_cb_weights(samples_per_class, beta=beta)
        min_w = weights[weights > 0].min()
        weights = np.clip(weights, min_w, min_w * max_weight)
        self.register_buffer("alpha", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != -1
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        targets = targets[valid_mask]
        n_classes = logits.size(-1)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / max(n_classes - 1, 1)
            one_hot = F.one_hot(targets, n_classes).float()
            soft_targets = one_hot * (1.0 - self.label_smoothing) + smooth
            ce = -(soft_targets * log_probs).sum(dim=-1)
            pt = (probs * one_hot).sum(dim=-1)
        else:
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            ce = -log_pt
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt.clamp(min=1e-8)) ** self.gamma
        loss = self.alpha[targets] * focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss (Cao et al., NeurIPS 2019).
    Adds class-dependent angular margins; defers re-weighting via DRW optional.
    """

    def __init__(
        self,
        samples_per_class: np.ndarray,
        max_m: float = 0.5,
        s: float = 30.0,
        weight: torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.s = s
        self.reduction = reduction
        margins = compute_ldam_margins(samples_per_class, max_m=max_m)
        self.register_buffer("margins", torch.tensor(margins, dtype=torch.float32))
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != -1
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        margins = self.margins[targets]
        logits_adj = logits.clone()
        logits_adj[torch.arange(len(targets)), targets] -= margins

        loss = F.cross_entropy(self.s * logits_adj, targets, weight=self.weight)
        return loss


def build_category_loss(
    loss_type: str,
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    *,
    focal_gamma: float = 2.0,
    cb_beta: float = 0.999,
    label_smoothing: float = 0.05,
    ldam_max_m: float = 0.5,
    ldam_s: float = 30.0,
    class_names: list | None = None,
    prior_f1: dict | None = None,
    weak_f1_threshold: float = 0.45,
    weak_gamma_boost: float = 1.0,
) -> nn.Module:
    """Factory for category classification losses."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)

    if loss_type == "focal":
        weights = 1.0 / counts
        weights = weights / weights.sum() * num_classes
        min_w = weights[weights > 0].min()
        weights = np.clip(weights, min_w, min_w * 20)
        alpha = torch.tensor(weights, dtype=torch.float32, device=device)
        return FocalLoss(alpha=alpha, gamma=focal_gamma)

    if loss_type == "weak_focal":
        names = class_names or [str(i) for i in range(num_classes)]
        alphas, gammas = compute_weak_class_loss_params(
            counts,
            names,
            prior_f1=prior_f1,
            f1_threshold=weak_f1_threshold,
            base_gamma=focal_gamma,
            gamma_boost=weak_gamma_boost,
        )
        alpha_t = torch.tensor(alphas, dtype=torch.float32, device=device)
        gamma_t = torch.tensor(gammas, dtype=torch.float32, device=device)
        return PerClassFocalLoss(alpha=alpha_t, gamma=gamma_t)

    if loss_type == "cb_focal":
        return ClassBalancedFocalLoss(
            samples_per_class=counts,
            beta=cb_beta,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        ).to(device)

    if loss_type == "ldam":
        cb_w = compute_cb_weights(counts, beta=cb_beta)
        weight = torch.tensor(cb_w, dtype=torch.float32, device=device)
        return LDAMLoss(
            samples_per_class=counts,
            max_m=ldam_max_m,
            s=ldam_s,
            weight=weight,
        ).to(device)

    raise ValueError(f"Unknown loss_type: {loss_type}")
