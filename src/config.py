"""Project-wide constants: paths, seeds, column names."""

from pathlib import Path

# Repository root (parent of `src/`)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

RANDOM_STATE = 42

# Expected CSV columns
COL_RESUME = "resume_text"
COL_JD = "job_description"
COL_LABEL = "label"

# Ordered label names for metrics and plots (adjust to match your CSV)
LABEL_CLASSES = ("poor fit", "moderate fit", "good fit")
