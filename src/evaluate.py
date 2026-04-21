"""Classification metrics, reports, and confusion matrix plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    y_true: Any,
    y_pred: Any,
    average: str = "weighted",
) -> dict[str, float]:
    """Compute common classification metrics.

    Parameters
    ----------
    y_true
        Ground-truth labels.
    y_pred
        Predicted labels (same length as ``y_true``).
    average
        Averaging strategy for precision, recall, and F1 when there are 2+
        classes (e.g. ``"weighted"``, ``"macro"``, ``"micro"``). For binary
        classification, ``"binary"`` is also valid.

    Returns
    -------
    dict[str, float]
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def print_classification_report(y_true: Any, y_pred: Any) -> None:
    """Print sklearn's classification report to stdout."""
    print(classification_report(y_true, y_pred, zero_division=0))


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: list[str],
    save_path: str | Path,
) -> Path:
    """Plot and save a confusion matrix figure.

    The figure is written to ``save_path`` (typically under ``outputs/figures``).

    Parameters
    ----------
    y_true
        Ground-truth labels.
    y_pred
        Predicted labels.
    labels
        Class names in the order used for matrix rows/columns.
    save_path
        Output file path (``.png`` recommended).

    Returns
    -------
    Path
        Resolved path where the figure was saved.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        ax=ax,
        colorbar=True,
        values_format="d",
    )
    ax.set_title("Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path.resolve()
