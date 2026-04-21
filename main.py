#!/usr/bin/env python3
"""CLI entry point: train on CSV (TF-IDF + Logistic Regression + Random Forest)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.interpret import get_top_terms_per_class
from src.pipeline import run_single_prediction, run_training_pipeline


# Skills used only for the post-training substring demo (not model features).
_DEMO_SKILL_LIST = [
    "python",
    "sql",
    "pandas",
    "machine learning",
    "kubernetes",
    "docker",
    "java",
    "aws",
    "communication",
]

_DEMO_RESUME = (
    "Data analyst with 3 years of experience. Strong Python and Pandas skills, "
    "daily SQL for reporting, and teamwork in agile environments."
)

_DEMO_JOB = (
    "Looking for a data analyst who knows Python, SQL, and Kubernetes. "
    "Docker and AWS are a plus. Excellent communication required."
)


def _print_metric_block(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for key in ("accuracy", "precision", "recall", "f1"):
        value = metrics.get(key)
        if value is not None:
            print(f"  {key:12s} {value:.4f}")


def _run_train(csv_path: str) -> None:
    path = Path(csv_path)
    print(f"Training with CSV: {path.resolve()}")
    out = run_training_pipeline(str(path))

    metrics = out["metrics"]
    figures = out["figures"]

    print("\n" + "=" * 60)
    print("VALIDATION METRICS (weighted average)")
    print("=" * 60)
    _print_metric_block("Logistic Regression", metrics["logistic_regression"]["validation"])
    print()
    _print_metric_block("Random Forest", metrics["random_forest"]["validation"])

    print("\n" + "=" * 60)
    print("TEST METRICS (weighted average)")
    print("=" * 60)
    _print_metric_block("Logistic Regression", metrics["logistic_regression"]["test"])
    print()
    _print_metric_block("Random Forest", metrics["random_forest"]["test"])

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX FILES")
    print("=" * 60)
    for model_name, splits in figures.items():
        for split_name, file_path in splits.items():
            print(f"  {model_name} / {split_name}: {file_path}")

    lr = out["models"]["logistic_regression"]
    vec = out["vectorizer"]
    print("\n" + "=" * 60)
    print("TOP POSITIVE TF-IDF TERMS (Logistic Regression, per class)")
    print("=" * 60)
    top = get_top_terms_per_class(lr, vec, top_n=10)
    for class_label, terms in top.items():
        print(f"\nClass: {class_label}")
        if not terms:
            print("  (no positive-weight terms in the top list)")
        else:
            for term, coef in terms:
                print(f"  {term:30s}  {coef:+.4f}")

    demo = run_single_prediction(
        _DEMO_RESUME,
        _DEMO_JOB,
        lr,
        vec,
        _DEMO_SKILL_LIST,
    )
    print("\n" + "=" * 60)
    print("DEMO — single prediction (Logistic Regression, in-memory)")
    print("=" * 60)
    print(f"  predicted_label:    {demo['predicted_label']}")
    probs = demo.get("class_probabilities")
    if probs:
        print("  class_probabilities:")
        for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {p:.4f}")
    else:
        print("  class_probabilities: (not available for this model)")
    print(f"  matched_skills:     {demo['matched_skills']}")
    print(f"  missing_skills:     {demo['missing_skills']}")
    print(f"  match_score:        {demo['match_score']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML-Powered Career Decision Engine (TF-IDF + LR/RF)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train models on a labeled CSV (resume_text, job_description, label).",
    )
    train_parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to training CSV, e.g. data/raw/job_fit_dataset.csv",
    )

    args = parser.parse_args()

    if args.command == "train":
        _run_train(args.csv_path)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    main()
