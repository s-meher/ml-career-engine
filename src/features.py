"""TF-IDF feature extraction (fit on train, transform val/test/inference)."""

from __future__ import annotations

from typing import Iterable, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer for the project.

    Parameters
    ----------
    max_features
        Max size of the vocabulary. Larger values can improve accuracy but
        increase memory usage.
    ngram_range
        N-gram range. Default (1, 2) uses unigrams and bigrams.

    Returns
    -------
    TfidfVectorizer
        An unfitted sklearn TF-IDF vectorizer.

    Raises
    ------
    ValueError
        If arguments are invalid.
    """
    if not isinstance(max_features, int) or isinstance(max_features, bool) or max_features <= 0:
        raise ValueError(f"max_features must be a positive int, got {max_features!r}.")
    if (
        not isinstance(ngram_range, tuple)
        or len(ngram_range) != 2
        or not all(isinstance(x, int) for x in ngram_range)
    ):
        raise ValueError("ngram_range must be a tuple of two ints, e.g. (1, 2).")
    if ngram_range[0] <= 0 or ngram_range[1] < ngram_range[0]:
        raise ValueError(f"Invalid ngram_range: {ngram_range!r}.")

    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=False,  # preprocessing already lowercases
        strip_accents=None,
    )


def fit_transform_features(
    train_texts: Sequence[str] | Iterable[str],
    vectorizer: TfidfVectorizer,
):
    """Fit the vectorizer on training texts and return the TF-IDF matrix.

    Important: this should only be called on the training split.

    Parameters
    ----------
    train_texts
        Training documents (use the `combined_text` column).
    vectorizer
        An unfitted TF-IDF vectorizer.

    Returns
    -------
    scipy.sparse.spmatrix
        TF-IDF feature matrix for training data.
    """
    if vectorizer is None:
        raise ValueError("vectorizer must be a fitted/unfitted TfidfVectorizer, got None.")
    return vectorizer.fit_transform(train_texts)


def transform_features(
    texts: Sequence[str] | Iterable[str],
    vectorizer: TfidfVectorizer,
):
    """Transform texts into TF-IDF features using a fitted vectorizer.

    Parameters
    ----------
    texts
        Documents to transform (validation/test/inference).
    vectorizer
        A TF-IDF vectorizer fitted on training data.

    Returns
    -------
    scipy.sparse.spmatrix
        TF-IDF feature matrix for the provided texts.
    """
    if vectorizer is None:
        raise ValueError("vectorizer must be a fitted TfidfVectorizer, got None.")
    return vectorizer.transform(texts)


def get_feature_names(vectorizer: TfidfVectorizer) -> list[str]:
    """Return the learned vocabulary terms in feature index order.

    Parameters
    ----------
    vectorizer
        A fitted TF-IDF vectorizer.

    Returns
    -------
    list[str]
        Feature names aligned with columns of the TF-IDF matrix.
    """
    if vectorizer is None:
        raise ValueError("vectorizer must be a fitted TfidfVectorizer, got None.")
    return list(vectorizer.get_feature_names_out())
