"""Per-class threshold calibration for category classification."""

import numpy as np
from sklearn.metrics import f1_score


def threshold_predict(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Predict class with highest margin above per-class threshold."""
    margins = probs - thresholds[np.newaxis, :]
    return margins.argmax(axis=1)


def calibrate_thresholds(
    probs: np.ndarray,
    true_labels: np.ndarray,
    num_classes: int,
    search_grid: np.ndarray = None,
) -> np.ndarray:
    if search_grid is None:
        search_grid = np.arange(0.01, 0.99, 0.02)

    thresholds = np.full(num_classes, 1.0 / num_classes)

    for c in range(num_classes):
        class_mask = true_labels == c
        n_class = class_mask.sum()

        if n_class < 3:
            class_probs = probs[:, c]
            thresholds[c] = max(0.01, float(np.percentile(class_probs, 5)))
            continue

        best_t = 1.0 / num_classes
        best_f1 = 0.0
        for t in search_grid:
            margins = probs - thresholds[np.newaxis, :]
            margins[:, c] = probs[:, c] - t
            preds = margins.argmax(axis=1)
            class_f1 = f1_score(
                true_labels, preds, labels=[c], average="macro", zero_division=0
            )
            if class_f1 > best_f1:
                best_f1 = class_f1
                best_t = t
        thresholds[c] = best_t

    return thresholds
