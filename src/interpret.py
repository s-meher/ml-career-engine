"""Interpretability: Logistic Regression coefficients and skill overlap."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _per_class_weight_vectors(model: LogisticRegression) -> dict[Any, np.ndarray]:
    """Map each class label to the coefficient vector used for that class.

    For binary logistic regression, sklearn stores a single row ``coef_[0]``
    aligned with ``classes_[1]`` as the positive direction; the opposite
    direction corresponds to ``classes_[0]``.
    """
    coef = np.asarray(model.coef_)
    classes = np.asarray(model.classes_)

    if coef.ndim != 2:
        raise ValueError(f"Unexpected coef_ shape: {coef.shape}")

    # Binary: one row, two classes
    if coef.shape[0] == 1 and classes.size == 2:
        w = coef[0]
        return {classes[0]: -w, classes[1]: w}

    if coef.shape[0] != classes.size:
        raise ValueError(
            "coef_ rows do not match number of classes. "
            f"Got coef_.shape={coef.shape}, classes={classes}."
        )

    return {classes[i]: coef[i] for i in range(classes.size)}


def get_top_terms_per_class(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    top_n: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    """List the strongest *positive* TF-IDF features per class for an LR model.

    For each class, terms are ranked by coefficient (largest first). Only
    strictly positive coefficients are included (up to ``top_n`` terms).

    Parameters
    ----------
    model
        A fitted multinomial / multiclass or binary ``LogisticRegression``.
    vectorizer
        The same fitted ``TfidfVectorizer`` used to build ``X``.
    top_n
        Maximum number of terms to keep per class.

    Returns
    -------
    dict[str, list[tuple[str, float]]]
        Keys are stringified class labels; values are ``(term, coefficient)``
        pairs for that class.

    Raises
    ------
    TypeError
        If ``model`` is not a ``LogisticRegression`` instance.
    ValueError
        If ``top_n`` is invalid or the model is not fitted.
    """
    if not isinstance(model, LogisticRegression):
        raise TypeError(f"Expected LogisticRegression, got {type(model).__name__}.")
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(f"Expected TfidfVectorizer, got {type(vectorizer).__name__}.")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n!r}.")

    if not hasattr(model, "coef_") or model.coef_ is None:
        raise ValueError("LogisticRegression model is not fitted (missing coef_).")

    feature_names = vectorizer.get_feature_names_out()
    if feature_names.shape[0] != model.coef_.shape[1]:
        raise ValueError(
            "Feature count mismatch between vectorizer and model coefficients."
        )

    out: dict[str, list[tuple[str, float]]] = {}
    for cls_label, weights in _per_class_weight_vectors(model).items():
        positive = [
            (str(feature_names[j]), float(weights[j]))
            for j in range(weights.shape[0])
            if weights[j] > 0
        ]
        positive.sort(key=lambda t: t[1], reverse=True)
        out[str(cls_label)] = positive[:top_n]

    return out


def extract_skills(text: str, skill_list: list[str]) -> set[str]:
    """Find which skills from ``skill_list`` appear in ``text`` (substring match).

    Matching is **case-insensitive**. The returned strings are the entries
    from ``skill_list`` that matched (including their original spelling).

    Parameters
    ----------
    text
        Resume or job description text.
    skill_list
        Candidate skills or phrases to search for.

    Returns
    -------
    set[str]
        Skills detected in ``text``.
    """
    if text is None:
        text = ""
    haystack = text.lower()
    found: set[str] = set()
    for skill in skill_list:
        if skill is None:
            continue
        s = str(skill).strip()
        if not s:
            continue
        if s.lower() in haystack:
            found.add(skill)
    return found


def compare_resume_to_job(
    resume_text: str,
    job_text: str,
    skill_list: list[str],
) -> dict[str, Any]:
    """Compare extracted skills between a resume and a job description.

    **Matched skills** appear in both texts. **Missing skills** appear in the
    job description but not the resume (a simple skill-gap view).
    **Resume-only skills** appear in the resume but not the job text.

    **Match score** is the fraction of job-description skills that are also
    present in the resume: ``|matched| / |job_skills|``. If no skills are
    detected in the job text, the score is ``0.0`` (no signal to match against).

    Parameters
    ----------
    resume_text
        Raw or cleaned resume text.
    job_text
        Raw or cleaned job description text.
    skill_list
        Shared vocabulary of skills/phrases to search for.

    Returns
    -------
    dict
        ``matched_skills``, ``missing_skills``, ``resume_only_skills`` (sorted
        lists), and ``match_score`` (float).
    """
    resume_skills = extract_skills(resume_text, skill_list)
    job_skills = extract_skills(job_text, skill_list)

    matched = resume_skills & job_skills
    missing = job_skills - resume_skills
    resume_only = resume_skills - job_skills

    if job_skills:
        match_score = len(matched) / len(job_skills)
    else:
        match_score = 0.0

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "resume_only_skills": sorted(resume_only),
        "match_score": float(match_score),
    }
