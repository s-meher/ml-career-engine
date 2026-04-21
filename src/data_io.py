"""Load CSV data and create stratified train/validation/test splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

REQUIRED_COLUMNS: tuple[str, ...] = (
    config.COL_RESUME,
    config.COL_JD,
    config.COL_LABEL,
)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load a labeled dataset from a CSV file and return a cleaned DataFrame.

    The file must exist and contain the columns ``resume_text``,
    ``job_description``, and ``label``. Rows with missing values in those
    columns are dropped; text fields are stripped of leading/trailing whitespace.

    Parameters
    ----------
    csv_path
        Path to a ``.csv`` file on disk.

    Returns
    -------
    pd.DataFrame
        Cleaned data containing at least the required columns (extra columns
        are preserved).

    Raises
    ------
    ValueError
        If the path is invalid, the file is missing, or the data fails
        validation.
    """
    path = Path(csv_path)
    if not csv_path or not str(csv_path).strip():
        raise ValueError("csv_path must be a non-empty string.")
    if not path.is_file():
        raise ValueError(f"CSV file not found: {path.resolve()}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {path.resolve()}") from exc
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {path}") from exc

    return validate_dataset(df)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns, drop incomplete rows, and strip text fields.

    Parameters
    ----------
    df
        Raw table loaded from CSV or built in memory.

    Returns
    -------
    pd.DataFrame
        A copy with only rows that have all required fields present; text
        columns are whitespace-stripped.

    Raises
    ------
    ValueError
        If ``df`` is not a DataFrame, required columns are missing, or no rows
        remain after cleaning.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas.DataFrame, got {type(df).__name__}.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "DataFrame is missing required columns: "
            f"{missing}. Expected all of {list(REQUIRED_COLUMNS)}. "
            f"Found columns: {list(df.columns)}."
        )

    out = df.copy()
    out[config.COL_RESUME] = out[config.COL_RESUME].astype("string").str.strip()
    out[config.COL_JD] = out[config.COL_JD].astype("string").str.strip()

    label_series = out[config.COL_LABEL]
    if not pd.api.types.is_numeric_dtype(label_series):
        out[config.COL_LABEL] = label_series.astype("string").str.strip()

    out = out.dropna(subset=list(REQUIRED_COLUMNS))

    if out.empty:
        raise ValueError(
            "No rows left after dropping missing values in "
            f"{list(REQUIRED_COLUMNS)}."
        )

    return out


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into stratified train, validation, and test sets.

    ``test_size`` and ``val_size`` are fractions of the **full** dataset
    (before splitting). For example, ``test_size=0.2`` and ``val_size=0.1``
    yields about 70% train, 10% validation, and 20% test.

    Parameters
    ----------
    df
        Cleaned labeled data (see :func:`validate_dataset`).
    test_size
        Fraction of rows to place in the test set (0 < x < 1).
    val_size
        Fraction of rows to place in the validation set (0 < x < 1).
    random_state
        Seed passed to scikit-learn for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)``.

    Raises
    ------
    ValueError
        If split sizes are invalid, the frame is empty, or stratification is
        not possible for the label distribution.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    for name, size in (("test_size", test_size), ("val_size", val_size)):
        if not isinstance(size, (int, float)) or isinstance(size, bool):
            raise ValueError(f"{name} must be a number, got {type(size).__name__}.")
        if not (0 < float(size) < 1):
            raise ValueError(f"{name} must be strictly between 0 and 1, got {size!r}.")

    if test_size + val_size >= 1:
        raise ValueError(
            "test_size + val_size must be less than 1 so the training set is "
            f"non-empty (got {test_size} + {val_size} >= 1)."
        )

    labels = df[config.COL_LABEL]
    counts = labels.value_counts()
    if (counts < 2).any():
        bad = counts[counts < 2].index.tolist()
        raise ValueError(
            "Stratified split requires at least 2 rows per label class. "
            f"These labels appear fewer than 2 times: {bad}"
        )

    stratify = labels

    try:
        train_val, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
        # Reserve ``val_size`` of the original rows from the train+val pool.
        val_fraction_of_train_val = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val,
            test_size=val_fraction_of_train_val,
            random_state=random_state,
            shuffle=True,
            stratify=train_val[config.COL_LABEL],
        )
    except ValueError as exc:
        raise ValueError(
            "Stratified train/validation/test split failed. This often happens "
            "when the dataset is small: each split still needs enough rows per "
            "label. Try adding more data, merging rare labels, or using smaller "
            "test_size / val_size values."
        ) from exc

    return train_df, val_df, test_df


if __name__ == "__main__":
    # Run from project root: python -m src.data_io
    import tempfile

    # Enough rows per label so two stratified splits succeed with default sizes.
    parts: list[dict[str, str]] = []
    for i in range(10):
        parts.append(
            {
                config.COL_RESUME: f"  Python skills resume {i}  ",
                config.COL_JD: f"Seeking Python developer {i}",
                config.COL_LABEL: "good fit",
            }
        )
        parts.append(
            {
                config.COL_RESUME: f"Retail experience {i}",
                config.COL_JD: f"Senior Python backend {i}",
                config.COL_LABEL: "poor fit",
            }
        )
        parts.append(
            {
                config.COL_RESUME: f"Python and SQL {i}",
                config.COL_JD: f"Data analyst Python {i}",
                config.COL_LABEL: "moderate fit",
            }
        )

    raw = pd.DataFrame(parts)
    raw.loc[len(raw)] = {config.COL_RESUME: None, config.COL_JD: "x", config.COL_LABEL: "good fit"}
    raw.loc[len(raw)] = {
        config.COL_RESUME: " ok ",
        config.COL_JD: None,
        config.COL_LABEL: "good fit",
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    ) as tmp:
        tmp_path = tmp.name
        raw.to_csv(tmp_path, index=False)

    try:
        loaded = load_dataset(tmp_path)
        print("load_dataset rows:", len(loaded))
        train_df, val_df, test_df = split_dataset(loaded, test_size=0.2, val_size=0.1)
        print("split sizes:", len(train_df), len(val_df), len(test_df))
        print("label counts (train):\n", train_df[config.COL_LABEL].value_counts())
    finally:
        Path(tmp_path).unlink(missing_ok=True)
