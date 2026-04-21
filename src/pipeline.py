"""Orchestrate load → preprocess → TF-IDF → train → evaluate → interpret."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

from . import config
from . import data_io
from . import evaluate
from . import features
from . import interpret
from . import models
from . import preprocess


def _preprocess_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``combined_text`` (and cleaned columns) to a split."""
    return preprocess.preprocess_dataframe(
        df,
        resume_col=config.COL_RESUME,
        jd_col=config.COL_JD,
    )


def _evaluate_and_plot(
    model: BaseEstimator,
    X,
    y_true,
    model_slug: str,
    split_name: str,
) -> tuple[dict[str, float], Path]:
    """Compute metrics, print a classification report, and save a confusion matrix."""
    y_pred = models.predict_with_model(model, X)
    metrics = evaluate.evaluate_model(y_true, y_pred, average="weighted")
    class_labels = [str(c) for c in model.classes_]

    print(f"\n=== {model_slug} — {split_name} ===")
    evaluate.print_classification_report(y_true, y_pred)

    save_path = config.FIGURES_DIR / f"confusion_matrix_{model_slug}_{split_name}.png"
    saved = evaluate.plot_confusion_matrix(
        y_true,
        y_pred,
        labels=class_labels,
        save_path=save_path,
    )
    return metrics, saved


def run_training_pipeline(csv_path: str) -> dict[str, Any]:
    """Run the full training workflow and return models, metrics, and figure paths.

    Steps
    -----
    1. Load and split the CSV (stratified train / validation / test).
    2. Preprocess each split (``combined_text`` for TF-IDF).
    3. Fit ``TfidfVectorizer`` on **training** ``combined_text`` only; transform
       validation and test.
    4. Train Logistic Regression and Random Forest on the training matrix.
    5. Evaluate both models on validation and test; save confusion matrices under
       ``outputs/figures``.

    Returns
    -------
    dict
        Common keys:

        - ``train_df``, ``val_df``, ``test_df``: preprocessed DataFrames
        - ``vectorizer``: fitted ``TfidfVectorizer``
        - ``label_classes``: list of class labels (from the trained LR model)
        - ``models``: ``{"logistic_regression": lr, "random_forest": rf}``
        - ``metrics``: nested dict by model slug, then ``"validation"`` /
          ``"test"`` with accuracy / precision / recall / f1
        - ``figures``: nested dict of saved confusion matrix paths (Path objects)
    """
    raw = data_io.load_dataset(csv_path)
    train_df, val_df, test_df = data_io.split_dataset(raw)

    train_df = _preprocess_split(train_df)
    val_df = _preprocess_split(val_df)
    test_df = _preprocess_split(test_df)

    vectorizer = features.build_tfidf_vectorizer()
    X_train = features.fit_transform_features(
        train_df[preprocess.COL_COMBINED], vectorizer
    )
    X_val = features.transform_features(val_df[preprocess.COL_COMBINED], vectorizer)
    X_test = features.transform_features(test_df[preprocess.COL_COMBINED], vectorizer)

    y_train = train_df[config.COL_LABEL]
    y_val = val_df[config.COL_LABEL]
    y_test = test_df[config.COL_LABEL]

    lr = models.train_logistic_regression(X_train, y_train)
    rf = models.train_random_forest(X_train, y_train)

    metrics: dict[str, dict[str, dict[str, float]]] = {
        "logistic_regression": {},
        "random_forest": {},
    }
    figure_paths: dict[str, dict[str, Path]] = {
        "logistic_regression": {},
        "random_forest": {},
    }

    for name, clf in (
        ("logistic_regression", lr),
        ("random_forest", rf),
    ):
        m_val, p_val = _evaluate_and_plot(clf, X_val, y_val, name, "validation")
        m_test, p_test = _evaluate_and_plot(clf, X_test, y_test, name, "test")
        metrics[name]["validation"] = m_val
        metrics[name]["test"] = m_test
        figure_paths[name]["validation"] = p_val
        figure_paths[name]["test"] = p_test

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "vectorizer": vectorizer,
        "label_classes": [str(c) for c in lr.classes_],
        "models": {
            "logistic_regression": lr,
            "random_forest": rf,
        },
        "metrics": metrics,
        "figures": figure_paths,
    }


def run_single_prediction(
    resume_text: str,
    job_description: str,
    trained_model: BaseEstimator,
    vectorizer: Any,
    skill_list: list[str],
) -> dict[str, Any]:
    """Predict job-fit label for one resume + JD pair and add simple skill overlap.

    ``trained_model`` should be a fitted sklearn classifier (e.g. Logistic
    Regression or Random Forest) with ``predict`` and ideally ``predict_proba``.
    ``vectorizer`` must be the same fitted ``TfidfVectorizer`` used in training.

    Parameters
    ----------
    resume_text
        Raw resume text.
    job_description
        Raw job description text.
    trained_model
        Fitted classifier from training.
    vectorizer
        Fitted ``TfidfVectorizer``.
    skill_list
        Skills/phrases passed to :func:`interpret.compare_resume_to_job`.

    Returns
    -------
    dict
        ``predicted_label``, optional ``class_probabilities``, and skill overlap
        fields (``matched_skills``, ``missing_skills``, ``resume_only_skills``,
        ``match_score``).
    """
    resume = resume_text if resume_text is not None else ""
    jd = job_description if job_description is not None else ""

    row = pd.DataFrame(
        {
            config.COL_RESUME: [resume],
            config.COL_JD: [jd],
        }
    )
    row = preprocess.preprocess_dataframe(
        row,
        resume_col=config.COL_RESUME,
        jd_col=config.COL_JD,
    )
    text = row[preprocess.COL_COMBINED].iloc[0]
    X = features.transform_features([text], vectorizer)

    pred = models.predict_with_model(trained_model, X)[0]

    out: dict[str, Any] = {
        "predicted_label": pred,
        "matched_skills": [],
        "missing_skills": [],
        "resume_only_skills": [],
        "match_score": 0.0,
    }

    if hasattr(trained_model, "predict_proba"):
        probs = trained_model.predict_proba(X)[0]
        classes = getattr(trained_model, "classes_", None)
        if classes is not None:
            out["class_probabilities"] = {
                str(classes[i]): float(probs[i]) for i in range(len(classes))
            }
        else:
            out["class_probabilities"] = {
                str(i): float(p) for i, p in enumerate(probs)
            }

    skills = interpret.compare_resume_to_job(resume, jd, skill_list)
    out["matched_skills"] = skills["matched_skills"]
    out["missing_skills"] = skills["missing_skills"]
    out["resume_only_skills"] = skills["resume_only_skills"]
    out["match_score"] = skills["match_score"]

    return out
