#!/usr/bin/env python3
"""
Streamlit demo UI: Career Decision Engine

This file focuses on UI/UX and presentation. It does NOT change ML logic.
Inference uses saved Logistic Regression + TF-IDF artifacts and calls the
existing helper `run_single_prediction(...)`.
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

SAMPLE_SWE_RESUME = """Software Engineer (Backend) - 4 years building backend services and APIs.
Python (FastAPI, Django), PostgreSQL, Redis, Docker, and AWS (ECS, Lambda).
Experience with CI/CD (GitHub Actions), code review, and mentoring interns.
Computer Science B.S.; contributed to internal observability tooling."""

SAMPLE_SWE_JD = """Senior Backend Engineer - We need strong Python, REST API design,
PostgreSQL, and cloud experience (AWS preferred). Docker/Kubernetes a plus.
You will design scalable services, write tests, and collaborate with product."""

SAMPLE_DA_RESUME = """Data Analyst with SQL, Excel, and dashboard experience.
Built weekly reports for marketing; comfortable with stakeholder presentations.
Familiar with Python basics and data cleaning; eager to grow in analytics."""

SAMPLE_DA_JD = """Data Analyst - Seeking SQL, Python, Pandas, and experience with BI tools.
Machine learning exposure is a plus. You will support forecasting models,
maintain dashboards, and partner with finance on KPI definitions."""


_PAGE_CSS = """
<style>
  /* Typography */
  html, body, [class*="css"]  { font-family: "Times New Roman", Times, serif !important; }
  :root {
    --bg: #F8FAFC; --text: #0F172A; --text2: #475569; --muted: #64748B;
    --card: #FFFFFF; --border: #E2E8F0;
    --blue: #1D4ED8; --blueHover: #1E40AF; --softBlue: #EFF6FF;
    --green: #16A34A; --greenBg: #DCFCE7;
    --amber: #D97706; --amberBg: #FEF3C7;
    --red: #DC2626; --redBg: #FEE2E2;
  }

  body { background: var(--bg) !important; color: var(--text) !important; }
  [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
  /* Remove Streamlit's translucent top header overlay (ghosted bar) */
  [data-testid="stHeader"] { display: none !important; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

  .block-container {
    max-width: 1100px !important;
    padding-left: 24px !important;
    padding-right: 24px !important;
    padding-top: 18px !important;
    padding-bottom: 32px !important;
  }

  /* Header hierarchy */
  .hero-title {
    font-size: 40px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
    margin: 0 0 10px 0;
    line-height: 1.15;
  }
  .hero-subtitle {
    font-size: 18px;
    font-weight: 500;
    color: var(--text2);
    margin: 0 0 12px 0;
    line-height: 1.4;
  }
  .pill {
    display: inline-flex; align-items: center;
    font-size: 14px; font-weight: 600; color: var(--text);
    background: var(--softBlue); border: 1px solid #bfdbfe;
    border-radius: 999px; padding: 6px 12px;
  }
  .header-divider {
    border-top: 1px solid var(--border);
    margin: 18px 0 20px 0;
  }

  .section-title { font-size: 22px; font-weight: 700; color: var(--text); margin: 0 0 10px 0; }
  .helper { font-size: 16px; color: var(--text2); margin: 0 0 14px 0; }

  div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06) !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 20px 20px !important; }

  div[data-testid="stTextArea"] textarea {
    background: var(--bg) !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 16px !important;
    color: var(--text) !important;
    font-size: 16px !important;
    line-height: 1.45 !important;
  }
  div[data-testid="stTextArea"] textarea:focus {
    outline: none !important;
    border: 2px solid #2563EB !important;
    box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.14) !important;
  }
  label { color: var(--text) !important; font-weight: 600 !important; }

  div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: #ffffff !important;
    box-shadow: none !important;
  }

  div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--blue) !important;
    color: #ffffff !important;
    border: 1px solid var(--blue) !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    height: 52px !important;
    box-shadow: 0 8px 18px rgba(29, 78, 216, 0.18) !important;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: var(--blueHover) !important;
    border-color: var(--blueHover) !important;
  }
  div[data-testid="stButton"] > button[kind="secondary"] {
    background: #ffffff !important;
    color: var(--text) !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    height: 48px !important;
    padding: 0 18px !important;
    font-size: 16px !important;
    white-space: nowrap !important;
  }
  div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #94A3B8 !important;
    background: #F8FAFC !important;
  }

  .metric-label { font-size: 14px; font-weight: 600; color: var(--muted); margin-bottom: 10px; }
  .metric-value { font-size: 30px; font-weight: 700; color: var(--text); letter-spacing: -0.02em; line-height: 1.1; }
  .metric-sub { margin-top: 8px; font-size: 13px; color: var(--muted); }

  .chip {
    display: inline-flex; align-items: center;
    font-size: 13px; font-weight: 700;
    border-radius: 999px; padding: 6px 12px; margin-left: 10px;
    border: 1px solid transparent; vertical-align: middle; white-space: nowrap;
  }
  .chip-good { color: var(--green); background: var(--greenBg); border-color: #bbf7d0; }
  .chip-mod { color: var(--amber); background: var(--amberBg); border-color: #fde68a; }
  .chip-poor { color: var(--red); background: var(--redBg); border-color: #fecaca; }

  .explain-panel {
    background: var(--softBlue);
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 16px;
    color: var(--text);
    font-size: 16px;
    line-height: 1.45;
  }

  .skill-chip {
    display: inline-flex; align-items: center;
    border-radius: 999px; padding: 6px 10px;
    font-size: 13px; font-weight: 600;
    margin: 0 8px 8px 0;
    border: 1px solid transparent;
    white-space: nowrap;
  }
  .skill-match { background: var(--greenBg); color: #065f46; border-color: #bbf7d0; }
  .skill-miss-red { background: var(--redBg); color: #991b1b; border-color: #fecaca; }

  .empty { color: var(--muted); font-size: 13px; }
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
    probs = result.get("class_probabilities")
    if not probs:
        return "-", "Model has no probability scores"
    pred = str(result["predicted_label"])
    p = probs.get(pred)
    if p is None:
        top = max(probs, key=probs.get)
        return f"{probs[top]:.0%}", f"Top class: {top}"
    return f"{p:.0%}", f"For predicted class: {pred}"


def _normalize_fit_label(label: str) -> str:
    s = str(label).strip().lower().replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    if "good" in s:
        return "good_fit"
    if "poor" in s or "weak" in s:
        return "poor_fit"
    return "moderate_fit"


def _fit_badge_html(label: str) -> str:
    norm = _normalize_fit_label(label)
    if norm == "good_fit":
        return '<span class="chip chip-good">Good fit</span>'
    if norm == "poor_fit":
        return '<span class="chip chip-poor">Poor fit</span>'
    return '<span class="chip chip-mod">Moderate fit</span>'


def _short_result_sentence(result: dict) -> str:
    matched = list(result.get("matched_skills") or [])
    missing = list(result.get("missing_skills") or [])

    def _join(items: list[str], max_n: int) -> str:
        items = [str(x) for x in items if str(x).strip()]
        if not items:
            return ""
        if len(items) <= max_n:
            if len(items) == 1:
                return items[0]
            if len(items) == 2:
                return f"{items[0]} and {items[1]}"
            return ", ".join(items[:-1]) + f", and {items[-1]}"
        shown = items[:max_n]
        rest = len(items) - max_n
        return ", ".join(shown[:-1]) + f", and {shown[-1]} (+{rest} more)"

    matched_part = _join(matched, 3)
    missing_part = _join(missing, 2)

    if matched_part and missing_part:
        return f"Strong overlap in {matched_part}, but missing {missing_part} lowered the fit score."
    if matched_part and not missing_part:
        return f"Strong overlap in {matched_part}, with no major gaps detected from your skill list."
    if (not matched_part) and missing_part:
        return f"Limited overlap with your skill list; missing {missing_part} appears in the job text."
    return "Skill overlap depends on your skill list; add more relevant skills for a clearer gap view."


def _render_skill_chips(skills: list[str], variant: str) -> None:
    if not skills:
        st.markdown('<div class="empty">None</div>', unsafe_allow_html=True)
        return
    cls = "skill-chip skill-match" if variant == "matched" else "skill-chip skill-miss-red"
    chips = "".join(f'<span class="{cls}">{html.escape(s)}</span>' for s in skills)
    st.markdown(chips, unsafe_allow_html=True)


def _plot_top_terms_chart(terms: list[tuple[str, float]], title: str) -> None:
    if not terms:
        return
    top = terms[:5]
    labels = [t[0][:34] + ("…" if len(t[0]) > 34 else "") for t in top]
    values = [t[1] for t in top]

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    ax.barh(labels[::-1], values[::-1], color="#1D4ED8", alpha=0.92, height=0.62)
    ax.set_title(title, fontsize=12, fontweight="700", color="#0F172A", pad=10)
    ax.set_xlabel("Coefficient (positive → supports this class)", fontsize=10, color="#475569")
    ax.tick_params(axis="both", labelsize=9, colors="#475569")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#E2E8F0")
    ax.grid(axis="x", color="#E2E8F0", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_skill_counts_chart(matched: int, missing: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    cats = ["Matched", "Missing"]
    vals = [matched, missing]
    colors = ["#16A34A", "#DC2626"]
    ax.bar(cats, vals, color=colors, width=0.5, alpha=0.9, edgecolor="#ffffff", linewidth=1.2)
    ax.set_title("Skill count comparison", fontsize=12, fontweight="700", color="#0F172A", pad=10)
    ax.set_ylabel("Count", fontsize=10, color="#475569")
    ax.tick_params(axis="both", labelsize=9, colors="#475569")
    ax.set_ylim(0, max(vals + [1]) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#E2E8F0")
    ax.grid(axis="y", color="#E2E8F0", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)
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

    # Hero header
    st.markdown(
        '<div class="hero-title">Career Decision Engine</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">Interpretable NLP model for resume-job fit and skill gap analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<span class="pill">Logistic Regression + TF-IDF</span>', unsafe_allow_html=True)
    st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)

    # Input card
    with st.container(border=True):
        st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="helper">Paste a resume and job description, or load a sample pair.</div>',
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns(2)
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

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.text_area(
                "Resume text",
                height=220,
                placeholder="Paste your resume here…",
                key="resume_field",
            )
        with c2:
            st.text_area(
                "Job description",
                height=220,
                placeholder="Paste the job description here…",
                key="jd_field",
            )

        with st.expander("Skill list (substring overlap)", expanded=False):
            st.markdown(
                '<div class="helper" style="margin-bottom:8px">'
                "One skill or phrase per line. Used only for the skill-gap panel (not model features)."
                "</div>",
                unsafe_allow_html=True,
            )
            st.text_area("Skills", height=140, key="skills_text", label_visibility="collapsed")

        analyze = st.button("Analyze Fit", type="primary", use_container_width=True)

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
                except Exception as exc:
                    st.error(f"Error loading model or vectorizer: {exc}")
                else:
                    with st.spinner("Preprocessing and scoring…"):
                        result = run_single_prediction(resume, jd, lr, vectorizer, skill_list)
                    st.session_state["last_result"] = result

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Results dashboard
    st.markdown('<div class="section-title">Results Dashboard</div>', unsafe_allow_html=True)
    result = st.session_state.get("last_result")

    m1, m2, m3 = st.columns(3)
    if not result:
        with m1:
            with st.container(border=True):
                st.markdown('<div class="metric-label">Predicted Fit</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-value">-</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-sub">Run analysis to populate</div>', unsafe_allow_html=True)
        with m2:
            with st.container(border=True):
                st.markdown('<div class="metric-label">Confidence</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-value">-</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-sub">Probability (if available)</div>', unsafe_allow_html=True)
        with m3:
            with st.container(border=True):
                st.markdown('<div class="metric-label">Skill Match Score</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-value">-</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-sub">Coverage of JD skills</div>', unsafe_allow_html=True)
    else:
        conf_main, conf_sub = _confidence_display(result)
        with m1:
            with st.container(border=True):
                badge = _fit_badge_html(str(result["predicted_label"]))
                st.markdown('<div class="metric-label">Predicted Fit</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value">{html.escape(str(result["predicted_label"]))}{badge}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="metric-sub">Logistic Regression on TF-IDF</div>', unsafe_allow_html=True)
        with m2:
            with st.container(border=True):
                st.markdown('<div class="metric-label">Confidence</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{html.escape(conf_main)}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-sub">{html.escape(conf_sub)}</div>', unsafe_allow_html=True)
        with m3:
            with st.container(border=True):
                st.markdown('<div class="metric-label">Skill Match Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{result["match_score"]:.0%}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-sub">Fraction of JD skills in resume</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="explain-panel">{html.escape(_short_result_sentence(result))}</div>', unsafe_allow_html=True)

    # Skills analysis
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Skills Analysis</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        with st.container(border=True):
            st.markdown('<div class="metric-label" style="margin-bottom:12px">Matched Skills</div>', unsafe_allow_html=True)
            if result:
                _render_skill_chips(list(result.get("matched_skills") or []), "matched")
            else:
                st.markdown('<div class="empty">Run analysis to show matched skills.</div>', unsafe_allow_html=True)
    with s2:
        with st.container(border=True):
            st.markdown('<div class="metric-label" style="margin-bottom:12px">Missing Skills</div>', unsafe_allow_html=True)
            if result:
                _render_skill_chips(list(result.get("missing_skills") or []), "missing")
            else:
                st.markdown('<div class="empty">Run analysis to show missing skills.</div>', unsafe_allow_html=True)

    # Charts
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Charts</div>', unsafe_allow_html=True)

    top_terms = None
    if result:
        try:
            lr, vectorizer = _cached_lr_and_vectorizer()
            top_by_class = get_top_terms_per_class(lr, vectorizer, top_n=10)
            top_terms = top_by_class.get(str(result["predicted_label"]))
        except Exception:
            top_terms = None

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown('<div class="metric-label" style="margin-bottom:10px">Top Positive Terms</div>', unsafe_allow_html=True)
            if top_terms:
                _plot_top_terms_chart(top_terms, f"Top positive terms - {result['predicted_label']}")
            else:
                st.markdown('<div class="empty">Run analysis to display top terms.</div>', unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown('<div class="metric-label" style="margin-bottom:10px">Skill Count Comparison</div>', unsafe_allow_html=True)
            if result:
                _plot_skill_counts_chart(
                    len(result.get("matched_skills") or []),
                    len(result.get("missing_skills") or []),
                )
            else:
                st.markdown('<div class="empty">Run analysis to compare skill counts.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
