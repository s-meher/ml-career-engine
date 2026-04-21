"""NLP preprocessing for resume and job description text (lightweight, no transformers)."""

from __future__ import annotations

import re
import string
from functools import lru_cache

import pandas as pd

from . import config

# New column names produced by preprocess_dataframe
COL_CLEANED_RESUME = "cleaned_resume_text"
COL_CLEANED_JD = "cleaned_job_description"
COL_COMBINED = "combined_text"

# Optional stopword removal (kept as a simple module flag for reproducibility).
# If True, English stopwords are removed using NLTK after clean_text().
REMOVE_STOPWORDS = False


def clean_text(text: str) -> str:
    """Lowercase, remove ASCII punctuation, and collapse extra whitespace.

    Stopword removal is **not** applied here. If you want stopwords removed,
    set :data:`REMOVE_STOPWORDS = True` so :func:`preprocess_dataframe` applies
    that extra step after cleaning.

    Parameters
    ----------
    text
        Raw text. ``None`` and non-strings are treated safely: ``None`` becomes
        ``""``, other values are converted with ``str()`` before cleaning.

    Returns
    -------
    str
        Cleaned text, or ``""`` if the input is empty/whitespace-only after
        cleaning.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    lowered = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    no_punct = lowered.translate(translator)
    collapsed = re.sub(r"\s+", " ", no_punct).strip()
    return collapsed


@lru_cache(maxsize=1)
def _nltk_english_stopwords() -> frozenset[str]:
    """Return NLTK English stopwords (downloads the ``stopwords`` corpus if needed)."""
    try:
        from nltk.corpus import stopwords as nltk_stopwords
    except ModuleNotFoundError as exc:
        raise ValueError(
            "The nltk package is required for stopword removal. "
            "Install it with: pip install nltk"
        ) from exc

    try:
        words = nltk_stopwords.words("english")
    except LookupError:
        import nltk

        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as nltk_stopwords

        words = nltk_stopwords.words("english")

    return frozenset(words)


def _remove_stopwords(text: str) -> str:
    """Remove NLTK English stopwords from whitespace-tokenized text."""
    if not text:
        return ""
    stop = _nltk_english_stopwords()
    tokens = text.split()
    kept = [t for t in tokens if t not in stop]
    return " ".join(kept)


def _combine_resume_jd(resume: str, jd: str) -> str:
    """Join two cleaned strings with a single space; drops empty parts."""
    parts = [p for p in (resume.strip(), jd.strip()) if p]
    return " ".join(parts)


def preprocess_dataframe(
    df: pd.DataFrame,
    resume_col: str = "resume_text",
    jd_col: str = "job_description",
) -> pd.DataFrame:
    """Add cleaned text columns plus one combined document per row.

    Adds:

    - ``cleaned_resume_text``
    - ``cleaned_job_description``
    - ``combined_text`` (resume, then job description, separated by a space)

    The original frame is not modified; a copy is returned.

    Parameters
    ----------
    df
        Input table.
    resume_col
        Name of the resume text column.
    jd_col
        Name of the job description text column.
    Notes
    -----
    - Stopword removal can be enabled by setting :data:`REMOVE_STOPWORDS = True`.
      This uses NLTK English stopwords **after** :func:`clean_text`.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the three new columns.

    Raises
    ------
    ValueError
        If ``df`` is not a DataFrame, required columns are missing, or NLTK
        is unavailable while :data:`REMOVE_STOPWORDS` is enabled.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas.DataFrame, got {type(df).__name__}.")

    missing = [c for c in (resume_col, jd_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing column(s): {missing}. "
            f"Available columns: {list(df.columns)}."
        )

    out = df.copy()

    resume_series = out[resume_col].fillna("").astype(str)
    jd_series = out[jd_col].fillna("").astype(str)

    cleaned_resume = resume_series.map(clean_text)
    cleaned_jd = jd_series.map(clean_text)

    if REMOVE_STOPWORDS:
        cleaned_resume = cleaned_resume.map(_remove_stopwords)
        cleaned_jd = cleaned_jd.map(_remove_stopwords)

    out[COL_CLEANED_RESUME] = cleaned_resume
    out[COL_CLEANED_JD] = cleaned_jd
    out[COL_COMBINED] = [
        _combine_resume_jd(r, j) for r, j in zip(cleaned_resume, cleaned_jd, strict=True)
    ]

    return out


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            config.COL_RESUME: [
                "  Python, Pandas!!!  ",
                "",
                None,
            ],
            config.COL_JD: [
                "Seeking:   SQL & teamwork.",
                "Only JD here.",
                None,
            ],
        }
    )
    out = preprocess_dataframe(sample)
    print(out[[COL_CLEANED_RESUME, COL_CLEANED_JD, COL_COMBINED]])
