"""Save and load trained models and vectorizer with joblib."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import config

_LR_NAME = "logistic_regression.joblib"
_RF_NAME = "random_forest.joblib"
_VEC_NAME = "tfidf_vectorizer.joblib"


def save_training_artifacts(
    logistic_regression: LogisticRegression,
    random_forest: RandomForestClassifier,
    vectorizer: TfidfVectorizer,
    directory: Path | None = None,
) -> dict[str, Path]:
    """Persist LR, RF, and TF-IDF vectorizer under ``outputs/models/`` (by default).

    Returns
    -------
    dict[str, Path]
        Keys: ``logistic_regression``, ``random_forest``, ``vectorizer``.
    """
    out_dir = directory if directory is not None else config.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "logistic_regression": out_dir / _LR_NAME,
        "random_forest": out_dir / _RF_NAME,
        "vectorizer": out_dir / _VEC_NAME,
    }
    joblib.dump(logistic_regression, paths["logistic_regression"])
    joblib.dump(random_forest, paths["random_forest"])
    joblib.dump(vectorizer, paths["vectorizer"])
    return paths


def load_training_artifacts(directory: Path | None = None) -> dict[str, Any]:
    """Load LR, RF, and vectorizer saved by :func:`save_training_artifacts`."""
    out_dir = directory if directory is not None else config.MODELS_DIR
    return {
        "logistic_regression": joblib.load(out_dir / _LR_NAME),
        "random_forest": joblib.load(out_dir / _RF_NAME),
        "vectorizer": joblib.load(out_dir / _VEC_NAME),
    }
