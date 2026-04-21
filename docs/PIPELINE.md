# Main pipeline plan

## Goal

Read paired **resume** + **job description** texts and a **label** from CSV, build TF-IDF features, train **Logistic Regression** (primary, interpretable) and **Random Forest** (comparison), then report fit prediction, match score, matched/missing skills, and evaluation metrics.

## Data flow (high level)

1. **Load** CSV with columns: `resume_text`, `job_description`, `label`.
2. **Split** into train / validation / test (stratified by label).
3. **Preprocess** both text fields (lowercase, normalize whitespace, optional stopword removal / lemmatization via NLTK or spaCy).
4. **Combine** resume and JD into one document per row (e.g. concatenation with a separator) *or* build separate vectorizers—implementation choice documented in code; default: single TF-IDF on concatenated text for simplicity and one matrix for sklearn.
5. **Vectorize** with `TfidfVectorizer` (fit on train only).
6. **Train** Logistic Regression and Random Forest on training TF-IDF matrix.
7. **Evaluate** both models on validation/test: accuracy, precision, recall, F1, confusion matrix (matplotlib).
8. **Interpret**
   - **Logistic Regression**: largest positive/negative TF-IDF feature weights (mapped back to n-grams/tokens).
   - **Skills**: extract skill-like tokens/phrases from resume vs JD (simple keyword / noun-phrase overlap—no deep learning), report intersection and JD-only gaps.

## Module responsibilities

| Module | Role |
|--------|------|
| `config.py` | Paths, random seed, class order, file names |
| `data_io.py` | Read CSV, validate columns, stratified split |
| `preprocess.py` | Cleaning and optional NLP normalization |
| `features.py` | Fit/transform TF-IDF |
| `models.py` | Fit/predict LR and RF, optional calibration for “match score” |
| `evaluate.py` | Classification report, confusion matrix plot |
| `interpret.py` | Top LR features; matched/missing skills helper |
| `pipeline.py` | Wire steps 1–8 for train and for single-row inference |

## Entry point

- `main.py`: arguments for mode (`train` / `predict`), paths to CSV or single texts, output directory for figures and optional saved models (`joblib`).

## Outputs

- Printed metrics and saved confusion matrix images under `outputs/figures/`.
- Optional: saved `vectorizer`, `model`, and label encoder for reuse.

## Constraints (project rules)

- No transformers / deep learning.
- Interpretability via LR weights + explicit skill overlap, not SHAP-on-trees as a requirement (optional later).
