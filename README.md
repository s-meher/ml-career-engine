# ML-Powered Career Decision Engine (TF‑IDF + Logistic Regression / Random Forest)

Proof-of-concept ML project that predicts candidate–job “fit” from paired resume text and job description text. The pipeline uses TF‑IDF features with classic classifiers (Logistic Regression and Random Forest) and includes evaluation artifacts and lightweight interpretability outputs.

## Features

- Train/evaluate TF‑IDF + **Logistic Regression** and **Random Forest**
- Save/load trained artifacts (**vectorizer + models**)
- Save confusion matrices under `outputs/figures/`
- Streamlit demo app for interactive inference (uses saved artifacts)

## Tech stack

- Python
- scikit-learn
- pandas, numpy
- matplotlib
- Streamlit

## Project structure

- `src/`: pipeline modules (data loading, preprocessing, features, models, evaluation, persistence)
- `main.py`: CLI training entry point
- `app.py`: Streamlit demo app (inference using saved artifacts)
- `data/raw/`: training dataset location (see note below)
- `outputs/models/`: saved models and TF‑IDF vectorizer (`.joblib`)
- `outputs/figures/`: saved confusion matrix images (`.png`)

## Quickstart: demo app (no retraining required)

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The demo loads saved artifacts from `outputs/models/` (TF‑IDF vectorizer + Logistic Regression model).

## Training (requires dataset)

Place the dataset at `data/raw/job_fit_dataset.csv` with required columns:

- `resume_text`
- `job_description`
- `label`

Then run:

```bash
python3 main.py train --csv_path data/raw/job_fit_dataset.csv
```

## Dataset note (submission package)

The `data/raw/` folder may be **empty** in the submission package. Retraining requires `data/raw/job_fit_dataset.csv`, but the saved artifacts in `outputs/models/` allow the Streamlit demo app to run **without** retraining.

## Limitations

- Dataset is small/curated and results are intended as a proof-of-concept.
- This is a TF‑IDF + classic ML baseline (no deep learning / transformers).