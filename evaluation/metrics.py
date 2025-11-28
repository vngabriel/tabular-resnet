import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy"""
    return float(accuracy_score(y_true, y_pred))


def calculate_auc_ovo(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_classes: int
) -> float:
    """
    Calculate AUC using One-vs-One strategy

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        n_classes: Number of classes

    Returns:
        AUC score (float)
    """
    try:
        if n_classes > 2:
            auc = roc_auc_score(
                y_true, y_pred_proba, multi_class="ovo", average="macro"
            )
        else:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        return float(auc)
    except Exception:
        return np.nan


def calculate_gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate G-Mean (geometric mean of sensitivities/recalls)

    This is the geometric mean of recall for each class.
    Useful for imbalanced datasets.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        G-Mean score (float)
    """
    cm = confusion_matrix(y_true, y_pred)

    # Calculate sensitivity (recall) for each class
    sensitivities = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)

    # Geometric mean
    gmean = np.power(np.prod(sensitivities), 1.0 / len(sensitivities))
    return float(gmean)


def calculate_cross_entropy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate cross-entropy loss (log loss)

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)

    Returns:
        Cross-entropy loss (float)
    """
    return float(log_loss(y_true, y_pred_proba))


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """
    Calculate all metrics at once

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        n_classes: Number of classes

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "auc_ovo": calculate_auc_ovo(y_true, y_pred_proba, n_classes),
        "gmean": calculate_gmean(y_true, y_pred),
        "cross_entropy": calculate_cross_entropy(y_true, y_pred_proba),
    }

    return metrics
