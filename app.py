#!/usr/bin/env python3
"""
Streamlit demo: ML-Powered Career Decision Engine
(job fit label + skill overlap using saved Logistic Regression + TF-IDF).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.interpret import get_top_terms_per_class
from src.persistence import load_logistic_regression_and_vectorizer
from src.pipeline import run_single_prediction

# Skills for substring overlap (same idea as the CLI demo; edit freely).
DEFAULT_SKILL_LIST = [
    "python",
    "java",
    "sql",
    "pandas",
    "numpy",
    "machine learning",
    "deep learning",
    "kubernetes",
    "docker",
    "aws",
    "azure",
    "react",
    "node",
    "communication",
    "leadership",
    "agile",
    "scrum",
]

_PAGE_STYLE = """
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 960px; }
    h1 { font-weight: 700; letter-spacing: -0.02em; }
    div[data-testid="stExpander"] { border: 1px solid rgba(49, 51, 63, 0.12); border-radius: 0.5rem; }
</style>
"""


@st.cache_resource
def _cached_lr_and_vectorizer():
    """Load once per app session; avoids reloading large objects on every click."""
    return load_logistic_regression_and_vectorizer()


def _fit_label_message(label: str) -> tuple[str, str]:
    """Return (style_key, short caption) for st.success / info / warning."""
    s = str(label).lower()
    if "good" in s:
        return "success", "Strong alignment with the job text (model view)."
    if "poor" in s or "weak" in s:
        return "warning", "Weaker alignment with the job text (model view)."
    return "info", "Moderate alignment with the job text (model view)."


def main() -> None:
    st.set_page_config(
        page_title="Career Fit Analyzer",
        page_icon="briefcase",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_PAGE_STYLE, unsafe_allow_html=True)

    st.title("ML-Powered Career Decision Engine")
    st.caption(
        "Interpretable NLP job fit + skill gap preview · Logistic Regression + TF-IDF"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        resume = st.text_area(
            "Resume text",
            height=220,
            placeholder="Paste your resume here…",
            label_visibility="visible",
        )
    with col2:
        jd = st.text_area(
            "Job description",
            height=220,
            placeholder="Paste the job description here…",
            label_visibility="visible",
        )

    with st.expander("Skill list used for overlap (substring match)", expanded=False):
        st.caption(
            "These strings are searched in your resume and the job description. "
            "Adjust the list in `app.py` if you want different skills."
        )
        skills_text = st.text_area(
            "One skill per line",
            value="\n".join(DEFAULT_SKILL_LIST),
            height=160,
            label_visibility="collapsed",
        )

    analyze = st.button("Analyze fit", type="primary", use_container_width=True)

    if not analyze:
        st.info("Enter a resume and job description, then click **Analyze fit**.")
        return

    skill_list = [s.strip() for s in skills_text.splitlines() if s.strip()]
    if not skill_list:
        st.warning("Add at least one skill in the expander (one per line).")
        return

    models_dir = Path(__file__).resolve().parent / "outputs" / "models"
    if not (models_dir / "logistic_regression.joblib").exists() or not (
        models_dir / "tfidf_vectorizer.joblib"
    ).exists():
        st.error(
            "Saved model files were not found under `outputs/models/`. "
            "Train first, for example:  \n`python main.py train --csv_path data/raw/job_fit_dataset.csv`"
        )
        return

    try:
        lr, vectorizer = _cached_lr_and_vectorizer()
    except FileNotFoundError as exc:
        st.error(f"Could not load model files: {exc}")
        return
    except Exception as exc:
        st.error(f"Error loading model or vectorizer: {exc}")
        return

    with st.spinner("Preprocessing and scoring…"):
        result = run_single_prediction(resume, jd, lr, vectorizer, skill_list)

    label = str(result["predicted_label"])
    style, caption = _fit_label_message(label)

    st.markdown("### Results")
    if style == "success":
        st.success(f"**Predicted fit:** `{label}`  \n{caption}")
    elif style == "warning":
        st.warning(f"**Predicted fit:** `{label}`  \n{caption}")
    else:
        st.info(f"**Predicted fit:** `{label}`  \n{caption}")

    probs = result.get("class_probabilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        if probs:
            top_cls = max(probs, key=probs.get)
            st.metric("Top class probability", f"{probs[top_cls]:.1%}", help=f"Highest: {top_cls}")
        else:
            st.metric("Top class probability", "—", help="Model has no predict_proba")
    with c2:
        st.metric("Skill match score (JD coverage)", f"{result['match_score']:.1%}")
    with c3:
        st.metric("Matched skills (count)", len(result["matched_skills"]))

    st.markdown("#### Skill overlap")
    st.caption("Based on your skill list and simple substring matching (not the TF-IDF model).")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.write("**Matched**")
        if result["matched_skills"]:
            for s in result["matched_skills"]:
                st.markdown(f"- `{s}`")
        else:
            st.caption("None")
    with sc2:
        st.write("**Missing (in JD, not resume)**")
        if result["missing_skills"]:
            for s in result["missing_skills"]:
                st.markdown(f"- `{s}`")
        else:
            st.caption("None")
    with sc3:
        st.write("**Resume-only**")
        if result["resume_only_skills"]:
            for s in result["resume_only_skills"]:
                st.markdown(f"- `{s}`")
        else:
            st.caption("None")

    if probs:
        with st.expander("All class probabilities", expanded=False):
            for cls in sorted(probs, key=probs.get, reverse=True):
                st.write(f"**{cls}** — {probs[cls]:.1%}")
                st.progress(min(max(float(probs[cls]), 0.0), 1.0))

    with st.expander("Top positive TF-IDF terms for the predicted class (Logistic Regression)", expanded=False):
        try:
            top_by_class = get_top_terms_per_class(lr, vectorizer, top_n=12)
            key = str(label)
            terms = top_by_class.get(key)
            if not terms:
                st.caption(
                    "No positive-weight terms listed for this class label, or label key mismatch. "
                    f"Available class keys: {', '.join(top_by_class)}"
                )
            else:
                st.caption("Terms with the largest positive coefficients for this class (interpretability hint).")
                for term, coef in terms:
                    st.markdown(f"- `{term}` · coefficient **{coef:+.4f}**")
        except Exception as exc:
            st.warning(f"Could not compute top terms: {exc}")


if __name__ == "__main__":
    main()
