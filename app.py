#!/usr/bin/env python3
"""
Streamlit demo: Career Decision Engine
(job fit + skill gap using saved Logistic Regression + TF-IDF).
"""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.interpret import get_top_terms_per_class
from src.persistence import load_logistic_regression_and_vectorizer
from src.pipeline import run_single_prediction

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
    "typescript",
    "communication",
    "leadership",
    "agile",
    "scrum",
]

SAMPLE_SWE_RESUME = """Software Engineer — 4 years building backend services and APIs.
Python (FastAPI, Django), PostgreSQL, Redis, Docker, and AWS (ECS, Lambda).
Experience with CI/CD (GitHub Actions), code review, and mentoring interns.
Computer Science B.S.; contributed to internal observability tooling."""

SAMPLE_SWE_JD = """Senior Backend Engineer — We need strong Python, REST API design,
PostgreSQL, and cloud experience (AWS preferred). Docker/Kubernetes a plus.
You will design scalable services, write tests, and collaborate with product."""

SAMPLE_DA_RESUME = """Data Analyst with SQL, Excel, and dashboard experience.
Built weekly reports for marketing; comfortable with stakeholder presentations.
Familiar with Python basics and data cleaning; eager to grow in analytics."""

SAMPLE_DA_JD = """Data Analyst — Seeking SQL, Python, Pandas, and experience with BI tools.
Machine learning exposure is a plus. You will support forecasting models,
maintain dashboards, and partner with finance on KPI definitions."""

_PAGE_CSS = """
<style>
  html, body, [class*="css"]  {
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
  }

  .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 920px !important;
    margin-left: auto !important;
    margin-right: auto !important;
  }

  .hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #0f172a;
    margin-bottom: 0.35rem;
    line-height: 1.15;
  }

  .hero-sub {
    font-size: 1.05rem;
    color: #475569;
    margin-bottom: 0.75rem;
    line-height: 1.45;
  }

  .model-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #334155;
    background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
    border: 1px solid #cbd5e1;
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    margin-bottom: 1.5rem;
  }

  .section-heading {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #64748b;
    margin: 0 0 0.5rem 0;
  }

  .metric-placeholder {
    border: 1px dashed #cbd5e1;
    border-radius: 12px;
    background: #f8fafc;
    padding: 1rem 1.1rem;
    min-height: 92px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  .metric-placeholder span {
    font-size: 0.8rem;
    color: #94a3b8;
  }

  .metric-live {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    padding: 1rem 1.1rem;
    min-height: 92px;
  }

  .metric-live .label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 0.35rem;
  }

  .metric-live .value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
  }

  .metric-live .hint {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.25rem;
  }

  .explain-box {
    border-left: 4px solid #3730a3;
    background: #f8fafc;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.15rem;
    margin-top: 1rem;
    color: #334155;
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  }

  div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #172554 100%) !important;
    color: #f8fafc !important;
    border: 1px solid #0f172a !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.55rem 1rem !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.12) !important;
  }

  div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(180deg, #234876 0%, #1e293b 100%) !important;
    border-color: #0f172a !important;
  }

  div[data-testid="stButton"] > button[kind="secondary"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
    color: #334155 !important;
  }

  div[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    background: #fafafa !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
  }

  .chart-wrap {
    margin-top: 0.5rem;
    padding: 0.5rem 0;
  }

  textarea {
    border-radius: 10px !important;
  }
</style>
"""


def _init_session_state() -> None:
    if "resume_field" not in st.session_state:
        st.session_state.resume_field = ""
    if "jd_field" not in st.session_state:
        st.session_state.jd_field = ""
    if "skills_text" not in st.session_state:
        st.session_state.skills_text = "\n".join(DEFAULT_SKILL_LIST)


@st.cache_resource
def _cached_lr_and_vectorizer():
    return load_logistic_regression_and_vectorizer()


def _confidence_display(result: dict) -> tuple[str, str]:
    """Return (main value string, subtitle) for confidence metric."""
    probs = result.get("class_probabilities")
    if not probs:
        return "—", "Model has no probability scores"
    pred = str(result["predicted_label"])
    p = probs.get(pred)
    if p is None:
        top = max(probs, key=probs.get)
        return f"{probs[top]:.0%}", f"Top class: {top}"
    return f"{p:.0%}", f"For predicted class: {pred}"


def _build_why_explanation(
    result: dict,
    top_terms: list[tuple[str, float]] | None,
) -> str:
    pred = html.escape(str(result["predicted_label"]))
    parts = [
        "The <strong>Logistic Regression</strong> model turns your combined résumé + job text into "
        "<strong>TF-IDF</strong> features, then picks the class with the highest score. "
        f"It predicted <strong>{pred}</strong>.",
    ]
    probs = result.get("class_probabilities")
    if probs and str(result["predicted_label"]) in probs:
        pk = str(result["predicted_label"])
        pval = probs[pk]
        parts.append(
            f"Estimated probability for that class is <strong>{pval:.0%}</strong> "
            "(multiclass softmax over the training labels)."
        )
    if top_terms:
        top3 = ", ".join(f"<strong>{html.escape(t[0])}</strong>" for t in top_terms[:3])
        parts.append(
            f"Among the strongest positive contributors for <code>{pred}</code> are: {top3}. "
            "These are interpretable n-grams weighted by learned coefficients."
        )
    parts.append(
        f"The <strong>skill match score ({result['match_score']:.0%})</strong> is separate: it measures "
        "how many skills from your list appear in <em>both</em> texts vs. skills found only in the job text."
    )
    return " ".join(parts)


def _plot_top_terms_chart(terms: list[tuple[str, float]], title: str) -> None:
    if not terms:
        return
    n = min(8, len(terms))
    labels = [t[0][:28] + ("…" if len(t[0]) > 28 else "") for t in terms[:n]]
    values = [t[1] for t in terms[:n]]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    bars = ax.barh(labels[::-1], values[::-1], color="#3730a3", height=0.55, alpha=0.88)
    ax.set_xlabel("Coefficient (positive → supports this class)", fontsize=9, color="#475569")
    ax.set_title(title, fontsize=11, fontweight="600", color="#0f172a", pad=10)
    ax.tick_params(axis="both", labelsize=8, colors="#475569")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#e2e8f0")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_skill_counts_chart(matched: int, missing: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    cats = ["Matched", "Missing (in JD)"]
    vals = [matched, missing]
    colors = ["#059669", "#dc2626"]
    ax.bar(cats, vals, color=colors, width=0.45, alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Count", fontsize=9, color="#475569")
    ax.set_title("Skill list overlap (substring match)", fontsize=11, fontweight="600", color="#0f172a")
    ax.set_ylim(0, max(vals + [1]) * 1.15)
    ax.tick_params(axis="both", labelsize=9, colors="#475569")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#e2e8f0")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.set_page_config(
        page_title="Career Decision Engine",
        page_icon="briefcase",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _init_session_state()
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

    st.markdown('<p class="hero-title">Career Decision Engine</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Interpretable NLP model for resume–job fit and skill gap analysis</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="model-badge">Logistic Regression + TF-IDF</span>',
        unsafe_allow_html=True,
    )

    # --- Input card ---
    st.markdown('<p class="section-heading">Inputs</p>', unsafe_allow_html=True)
    try:
        input_wrap = st.container(border=True)
    except TypeError:
        input_wrap = st.container()

    with input_wrap:
        st.caption("Paste a résumé and a job description, or load a sample pair.")

        b1, b2, _ = st.columns([1, 1, 2])
        with b1:
            if st.button("Load sample SWE pair", use_container_width=True):
                st.session_state.resume_field = SAMPLE_SWE_RESUME
                st.session_state.jd_field = SAMPLE_SWE_JD
                st.session_state.pop("last_result", None)
                st.rerun()
        with b2:
            if st.button("Load sample Data Analyst pair", use_container_width=True):
                st.session_state.resume_field = SAMPLE_DA_RESUME
                st.session_state.jd_field = SAMPLE_DA_JD
                st.session_state.pop("last_result", None)
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            st.text_area(
                "Resume text",
                height=200,
                placeholder="Paste your resume here…",
                key="resume_field",
                label_visibility="visible",
            )
        with c2:
            st.text_area(
                "Job description",
                height=200,
                placeholder="Paste the job description here…",
                key="jd_field",
                label_visibility="visible",
            )

        with st.expander("Skill list (substring overlap)", expanded=False):
            st.caption("One skill or phrase per line. Used only for the skill-gap panel, not for TF-IDF training.")
            st.text_area(
                "Skills",
                height=140,
                key="skills_text",
                label_visibility="collapsed",
            )

        analyze = st.button("Run analysis", type="primary", use_container_width=True)

    resume = st.session_state.resume_field
    jd = st.session_state.jd_field
    skills_text = st.session_state.skills_text

    if analyze:
        skill_list = [s.strip() for s in skills_text.splitlines() if s.strip()]
        if not skill_list:
            st.warning("Add at least one skill in the expander (one per line).")
        else:
            models_dir = Path(__file__).resolve().parent / "outputs" / "models"
            if not (models_dir / "logistic_regression.joblib").exists() or not (
                models_dir / "tfidf_vectorizer.joblib"
            ).exists():
                st.error(
                    "Saved model files were not found under `outputs/models/`. "
                    "Train first: `python main.py train --csv_path data/raw/job_fit_dataset.csv`"
                )
            else:
                try:
                    lr, vectorizer = _cached_lr_and_vectorizer()
                except FileNotFoundError as exc:
                    st.error(f"Could not load model files: {exc}")
                except Exception as exc:
                    st.error(f"Error loading model or vectorizer: {exc}")
                else:
                    with st.spinner("Preprocessing and scoring…"):
                        result = run_single_prediction(
                            resume, jd, lr, vectorizer, skill_list
                        )
                    st.session_state["last_result"] = result

    # --- Results dashboard (placeholders or live) ---
    st.markdown('<p class="section-heading">Results dashboard</p>', unsafe_allow_html=True)

    result = st.session_state.get("last_result")

    m1, m2, m3 = st.columns(3)
    if not result:
        with m1:
            st.markdown(
                '<div class="metric-placeholder"><span>Predicted fit</span></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                '<div class="metric-placeholder"><span>Confidence</span></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                '<div class="metric-placeholder"><span>Skill match score</span></div>',
                unsafe_allow_html=True,
            )
        st.caption("Run an analysis to populate these metrics.")
    else:
        conf_main, conf_sub = _confidence_display(result)
        with m1:
            st.markdown(
                f'<div class="metric-live"><div class="label">Predicted fit</div>'
                f'<div class="value">{html.escape(str(result["predicted_label"]))}</div>'
                f'<div class="hint">Multiclass LR on TF-IDF</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-live"><div class="label">Confidence</div>'
                f'<div class="value">{html.escape(conf_main)}</div>'
                f'<div class="hint">{html.escape(conf_sub)}</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-live"><div class="label">Skill match score</div>'
                f'<div class="value">{result["match_score"]:.0%}</div>'
                f'<div class="hint">JD skills also in résumé</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Matched skills**")
        if result and result["matched_skills"]:
            for s in result["matched_skills"]:
                st.markdown(f"- `{s}`")
        elif result:
            st.caption("No overlap detected with your skill list.")
        else:
            st.caption("—")
    with col_b:
        st.markdown("**Missing skills** *(in job text, not résumé)*")
        if result and result["missing_skills"]:
            for s in result["missing_skills"]:
                st.markdown(f"- `{s}`")
        elif result:
            st.caption("None flagged — nice coverage.")
        else:
            st.caption("—")

    if result and result.get("resume_only_skills"):
        with st.expander("Résumé-only skills (in résumé, not job text)", expanded=False):
            for s in result["resume_only_skills"]:
                st.markdown(f"- `{s}`")

    top_terms: list[tuple[str, float]] | None = None
    if result:
        try:
            lr, vectorizer = _cached_lr_and_vectorizer()
            top_by_class = get_top_terms_per_class(lr, vectorizer, top_n=10)
            top_terms = top_by_class.get(str(result["predicted_label"]))
        except Exception:
            top_terms = None

    if result:
        st.markdown(
            f'<div class="explain-box"><strong>Why this prediction?</strong><br><br>'
            f'{_build_why_explanation(result, top_terms)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-heading" style="margin-top:1.75rem">Charts</p>', unsafe_allow_html=True)
    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        st.caption("Top positive TF-IDF terms (predicted class)")
        if top_terms:
            _plot_top_terms_chart(top_terms, f"Class: {result['predicted_label']}")
        else:
            st.info("Run analysis to show coefficient-backed terms.")
    with chart_c2:
        st.caption("Skill counts (your list)")
        if result:
            _plot_skill_counts_chart(
                len(result["matched_skills"]),
                len(result["missing_skills"]),
            )
        else:
            st.info("Run analysis to compare matched vs. missing skills.")


if __name__ == "__main__":
    main()
