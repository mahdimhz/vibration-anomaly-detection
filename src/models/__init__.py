import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FEMTO_FS, OVERLAP, WINDOW_SIZE


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    degraded_alarm_indices = np.flatnonzero((y_true == 1) & (y_pred == 1))
    step_seconds = WINDOW_SIZE * (1 - OVERLAP) / FEMTO_FS
    time_to_detection = np.nan

    if len(degraded_alarm_indices) > 0:
        first_alarm_idx = degraded_alarm_indices[0]
        time_to_detection = (len(y_true) - 1 - first_alarm_idx) * step_seconds / 60

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "time_to_detection": float(time_to_detection),
    }
