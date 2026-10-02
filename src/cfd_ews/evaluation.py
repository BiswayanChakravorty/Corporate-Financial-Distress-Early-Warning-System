from __future__ import annotations
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

def binary_metrics(y_true: pd.Series, p_distress: pd.Series) -> dict[str, float]:
    """Calculate probability-sensitive binary classification metrics."""
    return {
        "roc_auc": float(roc_auc_score(y_true, p_distress)),
        "pr_auc": float(average_precision_score(y_true, p_distress)),
        "brier_score": float(brier_score_loss(y_true, p_distress)),
        "log_loss": float(log_loss(y_true, p_distress, labels=[0, 1])),
    }
