"""Train Logistic Regression and Random Forest; predict labels."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X_train: np.ndarray | spmatrix,
    y_train: Any,
    random_state: int = 42,
) -> LogisticRegression:
    """Fit multinomial Logistic Regression (interpretable linear model).

    Uses the ``lbfgs`` solver with ``multi_class='multinomial'``, which supports
    multi-class problems and works with sparse TF-IDF matrices.

    Parameters
    ----------
    X_train
        Training feature matrix (dense or sparse).
    y_train
        Training labels (any array-like accepted by sklearn).
    random_state
        Seed for reproducibility.

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray | spmatrix,
    y_train: Any,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit a Random Forest classifier for comparison with Logistic Regression.

    Parameters
    ----------
    X_train
        Training feature matrix (dense or sparse).
    y_train
        Training labels.
    random_state
        Seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_with_model(model: BaseEstimator, X: np.ndarray | spmatrix) -> np.ndarray:
    """Return class predictions for feature matrix ``X``."""
    if not hasattr(model, "predict"):
        raise TypeError(f"Model has no predict() method: {type(model).__name__}")
    y_pred = model.predict(X)
    return np.asarray(y_pred)
