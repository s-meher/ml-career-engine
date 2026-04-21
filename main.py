#!/usr/bin/env python3
"""CLI entry point: train on CSV or run inference on resume + JD text."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML-Powered Career Decision Engine (TF-IDF + LR/RF)"
    )
    parser.add_argument("mode", choices=["train", "predict"], help="train or predict")
    # More arguments added when pipeline is implemented
    args = parser.parse_args()
    raise SystemExit(f"Mode {args.mode!r} not wired yet — implement src/pipeline.py next.")


if __name__ == "__main__":
    main()
