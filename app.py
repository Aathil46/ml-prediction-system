"""ShopperSense AI — Online Shopper Purchase Intention Prediction
Enterprise-grade Streamlit analytics dashboard powered by a pretrained Random Forest Classifier.
"""

import os
import sys
import warnings
from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.validation import (
    validate_uploaded_file,
    validate_input_schema,
    check_data_quality,
    REQUIRED_COLUMNS,
)
from utils.preprocessing import preprocess_features, EXPECTED_MODEL_FEATURES
from utils.prediction import (
    load_rf_model,
    predict_purchase_intent,
    get_model_metadata,
    evaluate_predictions_against_ground_truth,
)
from utils.reporting import (
    compute_summary_metrics,
    generate_csv_report,
    generate_excel_report,
    generate_summary_markdown,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopperSense AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens & global CSS ────────────────────────────────────────────────
PALETTE = {
    "brand":      "#4F46E5",   # indigo-600
    "brand_dark": "#3730A3",   # indigo-700
    "brand_light":"#EEF2FF",   # indigo-50
    "success":    "#10B981",   # emerald-500
    "warning":    "#F59E0B",   # amber-500
    "danger":     "#EF4444",   # red-500
    "neutral":    "#64748B",   # slate-500
    "surface":    "#FFFFFF",
    "bg":         "#F8FAFC",   # slate-50
    "border":     "#E2E8F0",   # slate-200
    "text_primary":   "#0F172A",  # slate-900
    "text_secondary": "#64748B",  # slate-500
}

CSS = f"""
<style>
/* ── Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ══════════════════════════════════════════════════════════
   FORCE LIGHT THEME — override every Streamlit container
   so background is crisp #F8FAFC and cards are #FFFFFF,
   with high-contrast dark text (#0F172A) everywhere.
   ══════════════════════════════════════════════════════════ */
:root, [data-theme="light"], [data-theme="dark"] {{
    --background-color:           #F8FAFC !important;
    --secondary-background-color: #FFFFFF !important;
    --text-color:                 #0F172A !important;
    --font:                       'Inter', system-ui, sans-serif !important;
}}

/* Base app shell, scroll views, headers */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
section.main,
.main,
.block-container {{
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}}

[data-testid="stHeader"] {{
    background-color: #F8FAFC !important;
}}

/* Ensure headings & paragraphs inside markdown are always crisp dark text */
.stApp p, .stApp li,
.stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{
    color: #0F172A !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}}

/* Secondary / caption text */
.stApp .stCaption,
.stApp .stMarkdown small,
[data-testid="stCaptionContainer"] p {{
    color: #475569 !important;
}}

/* Hide Streamlit default header bar so it never overlaps page content */
header[data-testid="stHeader"],
[data-testid="stHeader"] {{
    display: none !important;
}}

/* ── Layout ── */
.block-container {{
    padding: 1.8rem 2.25rem 3rem 2.25rem !important;
    max-width: 1360px;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {{
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    min-width: 275px !important;
    max-width: 295px !important;
}}
[data-testid="stSidebar"] * {{
    color: #0F172A !important;
}}
[data-testid="stSidebar"] .stRadio > div {{
    gap: 0;
}}
[data-testid="stSidebar"] .stRadio label {{
    padding: 0.55rem 0.85rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    color: #334155 !important;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: #F1F5F9 !important;
    color: #4F46E5 !important;
}}
[data-testid="stSidebar"] .stRadio label p {{
    color: inherit !important;
    margin: 0 !important;
}}
/* Selected radio item */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] label {{
    background: #EEF2FF !important;
    color: #4338CA !important;
    font-weight: 600 !important;
}}

/* Hide the radio group title "" label */
[data-testid="stSidebar"] .stRadio > label {{
    display: none;
}}

/* ── Metric widgets ── */
[data-testid="stMetric"] {{
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 0.9rem 1rem !important;
}}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span {{
    color: #475569 !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}}
[data-testid="stMetricValue"] {{
    color: #0F172A !important;
}}
[data-testid="stMetricValue"] * {{
    color: #0F172A !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
}}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"],
[data-testid="stTable"] {{
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}}
[data-testid="stDataFrame"] *,
[data-testid="stTable"] * {{
    color: #0F172A !important;
}}

/* ── Alerts / info boxes from Streamlit ── */
[data-testid="stAlertContainer"],
[data-testid="stInfo"],
[data-testid="stSuccess"],
[data-testid="stWarning"],
[data-testid="stError"] {{
    border-radius: 8px !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}}
[data-testid="stExpander"] summary span {{
    color: #0F172A !important;
    font-weight: 600 !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: #FFFFFF !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 0.5rem;
}}
[data-testid="stFileUploader"] * {{
    color: #0F172A !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] span {{
    color: #475569 !important;
}}

/* ── Selectbox / text input ── */
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label {{
    color: #0F172A !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}}
[data-baseweb="select"] {{
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}}
[data-baseweb="select"] * {{
    color: #0F172A !important;
    background: #FFFFFF !important;
}}
[data-baseweb="input"] input {{
    color: #0F172A !important;
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {{
    color: #475569 !important;
    font-weight: 500 !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: #4F46E5 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #4F46E5 !important;
}}

/* ── Page heading ── */
.page-title {{
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    color: #0F172A !important;
    margin-top: 0 !important;
    margin-bottom: 0.35rem !important;
    line-height: 1.3 !important;
}}
.page-sub {{
    font-size: 0.92rem !important;
    color: #475569 !important;
    margin-bottom: 1.5rem !important;
    font-weight: 400 !important;
    line-height: 1.4 !important;
}}

/* ── KPI cards ── */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}}
.kpi-card {{
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem 1.1rem !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #4F46E5);
    border-radius: 12px 12px 0 0;
}}
.kpi-label {{
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #475569 !important;
    margin-bottom: 0.45rem !important;
}}
.kpi-value {{
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: #0F172A !important;
    line-height: 1 !important;
    margin-bottom: 0.3rem !important;
}}
.kpi-desc {{
    font-size: 0.75rem !important;
    color: #64748B !important;
}}

/* ── Section card & Streamlit native border containers ── */
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.4rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}}
.section-card {{
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 1.35rem 1.5rem !important;
    margin-bottom: 1.1rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}}
.section-header {{
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    margin-bottom: 0.35rem !important;
    line-height: 1.3 !important;
}}
.section-sub {{
    font-size: 0.82rem !important;
    color: #475569 !important;
    margin-bottom: 0.9rem !important;
}}

/* ── Status pills ── */
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.7rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}}
.status-ok   {{ background:#ECFDF5 !important; color:#065F46 !important; border:1px solid #A7F3D0 !important; }}
.status-warn {{ background:#FFFBEB !important; color:#78350F !important; border:1px solid #FDE68A !important; }}
.status-info {{ background:#EEF2FF !important; color:#3730A3 !important; border:1px solid #C7D2FE !important; }}

/* ── Intent Badges ── */
.badge-high   {{ background:#ECFDF5 !important; color:#065F46 !important; border:1px solid #A7F3D0 !important; display:inline-block; padding:0.2rem 0.6rem; border-radius:9999px; font-size:0.75rem; font-weight:600; }}
.badge-medium {{ background:#EEF2FF !important; color:#3730A3 !important; border:1px solid #C7D2FE !important; display:inline-block; padding:0.2rem 0.6rem; border-radius:9999px; font-size:0.75rem; font-weight:600; }}
.badge-low    {{ background:#FFFBEB !important; color:#92400E !important; border:1px solid #FDE68A !important; display:inline-block; padding:0.2rem 0.6rem; border-radius:9999px; font-size:0.75rem; font-weight:600; }}
.badge-bounce {{ background:#FEF2F2 !important; color:#991B1B !important; border:1px solid #FECACA !important; display:inline-block; padding:0.2rem 0.6rem; border-radius:9999px; font-size:0.75rem; font-weight:600; }}

/* ── Callout boxes ── */
.callout {{
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-size: 0.88rem;
    margin: 0.75rem 0;
    border-left: 4px solid;
    line-height: 1.55;
}}
.callout-info    {{ background:#EFF6FF !important; border-color:#3B82F6 !important; color:#1E3A8A !important; }}
.callout-info *  {{ color:#1E3A8A !important; }}
.callout-warn    {{ background:#FFFBEB !important; border-color:#F59E0B !important; color:#78350F !important; }}
.callout-warn *  {{ color:#78350F !important; }}
.callout-success {{ background:#F0FDF4 !important; border-color:#22C55E !important; color:#14532D !important; }}
.callout-success * {{ color:#14532D !important; }}

/* ── Brand block in sidebar ── */
.brand-name {{
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    letter-spacing: -0.01em;
}}
.brand-tag {{
    font-size: 0.72rem !important;
    color: #475569 !important;
    margin-top: 0.1rem;
}}

/* ── Empty state ── */
.empty-state {{
    text-align: center;
    padding: 2.5rem 2rem;
    color: #475569 !important;
}}
.empty-icon  {{ font-size: 2.5rem; margin-bottom: 0.6rem; }}
.empty-title {{
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    margin-bottom: 0.35rem;
}}
.empty-desc  {{ font-size: 0.86rem !important; line-height: 1.55; color: #475569 !important; }}

/* ── Download cards ── */
.dl-card {{
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 1.3rem;
    height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}}
.dl-card-title {{
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    margin-bottom: 0.3rem;
}}
.dl-card-desc {{
    font-size: 0.80rem !important;
    color: #475569 !important;
    margin-bottom: 1rem;
    line-height: 1.5;
}}

/* ── Quick-load / sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {{
    background: #F1F5F9 !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 0.35rem !important;
    white-space: nowrap !important;
}}
[data-testid="stSidebar"] .stButton > button * {{
    color: #0F172A !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: #E2E8F0 !important;
    border-color: #94A3B8 !important;
}}

/* ── Primary action button ── */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {{
    background: #4F46E5 !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}}
.stButton > button[kind="primary"] *,
button[data-testid="baseButton-primary"] * {{
    color: #FFFFFF !important;
}}
.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {{
    background: #3730A3 !important;
}}

/* ── Secondary / Download button ── */
.stButton > button,
.stDownloadButton > button,
button[data-testid="baseButton-secondary"] {{
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
.stButton > button *,
.stDownloadButton > button *,
button[data-testid="baseButton-secondary"] * {{
    color: #0F172A !important;
}}
.stButton > button:hover,
.stDownloadButton > button:hover,
button[data-testid="baseButton-secondary"]:hover {{
    background: #EEF2FF !important;
    border-color: #A5B4FC !important;
    color: #3730A3 !important;
}}
.stButton > button:hover *,
.stDownloadButton > button:hover *,
button[data-testid="baseButton-secondary"]:hover * {{
    color: #3730A3 !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
CHART_COLORS = {
    "purchase":   "#10B981",
    "no_purchase":"#94A3B8",
    "high":       "#10B981",
    "medium":     "#4F46E5",
    "low":        "#F59E0B",
    "bounce":     "#EF4444",
}

PLOTLY_BASE = dict(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#0F172A"),
    margin=dict(t=36, b=30, l=36, r=20),
    xaxis=dict(
        gridcolor="#F1F5F9",
        linecolor="#CBD5E1",
        tickfont=dict(color="#1E293B", size=11),
        title_font=dict(color="#0F172A", size=12),
    ),
    yaxis=dict(
        gridcolor="#F1F5F9",
        linecolor="#CBD5E1",
        tickfont=dict(color="#1E293B", size=11),
        title_font=dict(color="#0F172A", size=12),
    ),
    legend=dict(
        font=dict(color="#0F172A", size=11),
    ),
)

def chart_layout(**kw):
    d = dict(**PLOTLY_BASE)
    d.update(kw)
    return d

def render_plotly_chart(fig, **kwargs):
    """Render Plotly figure with theme=None so Streamlit does not override light colors with dark text."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#0F172A"),
    )
    fig.update_xaxes(tickfont=dict(color="#1E293B"), title_font=dict(color="#0F172A"))
    fig.update_yaxes(tickfont=dict(color="#1E293B"), title_font=dict(color="#0F172A"))
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False}, **kwargs)


def kpi_card(label: str, value: str, desc: str = "", accent: str = "#4F46E5") -> str:
    return f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-desc">{desc}</div>
    </div>"""


def section(title: str, sub: str = "") -> str:
    s = f'<div class="section-card"><div class="section-header">{title}</div>'
    if sub:
        s += f'<div class="section-sub">{sub}</div>'
    return s


def callout(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="callout callout-{kind}">{text}</div>', unsafe_allow_html=True)


def intent_badge(tier: str) -> str:
    cls_map = {
        "High Intent (>=75%)":     ("badge-high",   "High Intent"),
        "Moderate Intent (50-74%)":("badge-medium", "Medium Intent"),
        "Low Intent (25-49%)":     ("badge-low",    "Low Intent"),
        "Bounce Risk (<25%)":      ("badge-bounce", "Bounce Risk"),
    }
    cls, label = cls_map.get(tier, ("badge-medium", tier))
    return f'<span class="badge {cls}">{label}</span>'


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "uploaded_df": None,
    "file_name": None,
    "validation_result": None,
    "quality_result": None,
    "preprocessed_df": None,
    "preprocessing_summary": None,
    "predictions_df": None,
    "metrics": None,
    "evaluation_result": None,
    "active_page": "Dashboard",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource
def get_cached_model():
    path = os.path.join(os.path.dirname(__file__), "model", "random_forest_model.pkl")
    return load_rf_model(path)


def load_demo_data(with_labels: bool = False):
    fname = "sample_customers_with_labels.csv" if with_labels else "sample_customers.csv"
    path = os.path.join(os.path.dirname(__file__), "sample_data", fname)
    df = pd.read_csv(path)
    st.session_state.uploaded_df = df
    st.session_state.file_name = fname
    st.session_state.validation_result = validate_input_schema(df)
    st.session_state.quality_result = check_data_quality(df)
    st.session_state.preprocessed_df = None
    st.session_state.predictions_df = None
    st.session_state.metrics = None
    st.session_state.evaluation_result = None


def run_predictions():
    df = st.session_state.uploaded_df
    X, summary = preprocess_features(df)
    st.session_state.preprocessed_df = X
    st.session_state.preprocessing_summary = summary
    model = get_cached_model()
    results = predict_purchase_intent(model, X, raw_df=df)
    st.session_state.predictions_df = results
    st.session_state.metrics = compute_summary_metrics(results)
    if "Revenue" in df.columns:
        y_pred = results["Predicted_Class"].values
        y_proba = results["Purchase_Probability"].values
        st.session_state.evaluation_result = evaluate_predictions_against_ground_truth(
            df["Revenue"], y_pred, y_proba
        )


def reset_state():
    for k in ["uploaded_df","file_name","validation_result","quality_result",
              "preprocessed_df","preprocessing_summary","predictions_df","metrics","evaluation_result"]:
        st.session_state[k] = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
NAV_PAGES = [
    ("📊", "Dashboard"),
    ("📁", "Upload Dataset"),
    ("🔮", "Purchase Predictions"),
    ("📈", "Analytics"),
    ("⬇️", "Reports & Downloads"),
    ("🤖", "Model Information"),
]

with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    c1, c2 = st.columns([1, 2.8])
    with c1:
        if os.path.exists(logo_path):
            st.image(logo_path, width=44)
    with c2:
        st.markdown("""
        <div style="padding-top:4px">
            <div class="brand-name">ShopperSense AI</div>
            <div class="brand-tag">Purchase Intent Analytics</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.85rem 0;border-color:#E2E8F0'>", unsafe_allow_html=True)

    # Build radio labels with icon + text always visible
    page_labels = [f"{icon}  {name}" for icon, name in NAV_PAGES]
    page_names  = [name for _, name in NAV_PAGES]

    raw_sel = st.radio("", page_labels, key="nav_radio")
    # Map label back to page name
    selected_page = page_names[page_labels.index(raw_sel)]

    st.markdown("<hr style='margin:0.85rem 0;border-color:#E2E8F0'>", unsafe_allow_html=True)

    # ── Session status ─────────────────────────────────────
    if st.session_state.uploaded_df is not None:
        n = len(st.session_state.uploaded_df)
        has_preds = st.session_state.predictions_df is not None
        st.markdown(
            f'<div class="status-pill status-ok">✓ {n:,} rows loaded</div>',
            unsafe_allow_html=True,
        )
        if has_preds:
            m = st.session_state.metrics
            st.markdown(
                f'<div class="status-pill status-info" style="margin-top:0.4rem">'
                f'⚡ {m["predicted_purchases"]:,} / {m["total_sessions"]:,} predicted buyers'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("")
    else:
        st.markdown(
            '<div class="status-pill status-warn">○ No data loaded</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    # ── Quick actions ──────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.06em;color:#94A3B8;margin-bottom:0.5rem">Quick Load</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Sample data", use_container_width=True,
                     help="Load 50 sample sessions (no Revenue label)"):
            load_demo_data(False)
            st.rerun()
    with c2:
        if st.button("With labels", use_container_width=True,
                     help="Load 50 sample sessions with Revenue label for evaluation"):
            load_demo_data(True)
            st.rerun()

    if st.session_state.uploaded_df is not None:
        if st.button("Clear data", use_container_width=True):
            reset_state()
            st.rerun()

    st.markdown("<hr style='margin:0.85rem 0;border-color:#E2E8F0'>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.72rem;color:#94A3B8;line-height:1.6">'
        'Random Forest · 30 features<br>Accuracy 89.6% · AUC 0.926</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if selected_page == "Dashboard":
    st.markdown('<div class="page-title">Purchase Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Predict and analyse online shopper purchase intent using a production Random Forest model.</div>',
        unsafe_allow_html=True,
    )

    has_preds = st.session_state.predictions_df is not None

    # ── KPI row ───────────────────────────────────────────
    if has_preds:
        m = st.session_state.metrics
        rate_color = PALETTE["success"] if m["purchase_intention_rate"] >= 20 else PALETTE["warning"]
        st.markdown(
            '<div class="kpi-row">'
            + kpi_card("Total Sessions Analyzed",
                       f'{m["total_sessions"]:,}',
                       "Sessions processed in this batch",
                       PALETTE["brand"])
            + kpi_card("Predicted Purchases",
                       f'{m["predicted_purchases"]:,}',
                       f'{m["purchase_intention_rate"]}% conversion rate',
                       PALETTE["success"])
            + kpi_card("High-Intent Visitors",
                       f'{m["high_intent_count"]:,}',
                       "Purchase probability ≥ 75%",
                       "#6366F1")
            + kpi_card("Purchase Rate",
                       f'{m["purchase_intention_rate"]}%',
                       f'Avg probability {m["avg_purchase_probability"]}%',
                       rate_color)
            + '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Placeholder KPI cards
        st.markdown(
            '<div class="kpi-row">'
            + kpi_card("Total Sessions Analyzed", "—", "Load a dataset to begin", PALETTE["brand"])
            + kpi_card("Predicted Purchases", "—", "Run predictions first", PALETTE["success"])
            + kpi_card("High-Intent Visitors", "—", "Purchase probability ≥ 75%", "#6366F1")
            + kpi_card("Purchase Rate", "—", "Avg purchase probability", PALETTE["warning"])
            + '</div>',
            unsafe_allow_html=True,
        )

    # ── Charts ────────────────────────────────────────────
    if has_preds:
        m = st.session_state.metrics
        results_df = st.session_state.predictions_df

        col_l, col_r = st.columns([1, 1.35])

        with col_l:
            with st.container(border=True):
                st.markdown('<div class="section-header">Purchase vs No-Purchase</div>', unsafe_allow_html=True)
                fig = go.Figure(go.Pie(
                    labels=["Predicted Purchase", "No Purchase"],
                    values=[m["predicted_purchases"], m["predicted_non_purchases"]],
                    hole=0.6,
                    marker=dict(
                        colors=[CHART_COLORS["purchase"], CHART_COLORS["no_purchase"]],
                        line=dict(color="white", width=2),
                    ),
                    textinfo="percent",
                    textfont=dict(size=13, family="Inter"),
                ))
                fig.add_annotation(
                    text=f"<b>{m['purchase_intention_rate']}%</b><br><span style='font-size:11px'>buyers</span>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=18, family="Inter"),
                )
                fig.update_layout(**chart_layout(
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=12)),
                    margin=dict(t=20, b=35, l=20, r=20),
                    height=280,
                ))
                render_plotly_chart(fig)

        with col_r:
            with st.container(border=True):
                st.markdown('<div class="section-header">Purchase Probability Distribution</div>', unsafe_allow_html=True)
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=results_df[results_df["Predicted_Class"] == 1]["Purchase_Probability"],
                    name="Purchase", nbinsx=20,
                    marker_color=CHART_COLORS["purchase"], opacity=0.85,
                ))
                fig2.add_trace(go.Histogram(
                    x=results_df[results_df["Predicted_Class"] == 0]["Purchase_Probability"],
                    name="No Purchase", nbinsx=20,
                    marker_color=CHART_COLORS["no_purchase"], opacity=0.75,
                ))
                fig2.add_vline(x=0.5, line_dash="dash", line_color=PALETTE["danger"],
                               annotation_text="0.5 threshold", annotation_position="top right",
                               annotation_font_size=11)
                fig2.update_layout(**chart_layout(
                    barmode="overlay", height=280,
                    xaxis_title="Purchase Probability",
                    yaxis_title="Sessions",
                    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                    margin=dict(t=36, b=36, l=36, r=20),
                ))
                render_plotly_chart(fig2)

        # Intent segmentation bar
        with st.container(border=True):
            st.markdown('<div class="section-header">Intent Segmentation</div>', unsafe_allow_html=True)
            tier_df = results_df["Intent_Tier"].value_counts().reset_index()
            tier_df.columns = ["Tier", "Count"]
            tier_order = ["High Intent (>=75%)", "Moderate Intent (50-74%)", "Low Intent (25-49%)", "Bounce Risk (<25%)"]
            tier_df["Tier"] = pd.Categorical(tier_df["Tier"], categories=tier_order, ordered=True)
            tier_df = tier_df.sort_values("Tier")
            fig3 = go.Figure(go.Bar(
                x=tier_df["Tier"],
                y=tier_df["Count"],
                marker_color=[
                    CHART_COLORS["high"], CHART_COLORS["medium"],
                    CHART_COLORS["low"], CHART_COLORS["bounce"],
                ][:len(tier_df)],
                text=tier_df["Count"],
                textposition="outside",
                textfont=dict(size=12, family="Inter"),
            ))
            fig3.update_layout(**chart_layout(height=240, showlegend=False,
                                              xaxis_title="", yaxis_title="Sessions"))
            render_plotly_chart(fig3)

        callout(
            "Predictions are statistical probability estimates based on behavioural browsing patterns — not guaranteed purchases.",
            "warn",
        )

    else:
        # Empty state
        st.markdown("""
        <div class="section-card">
            <div class="empty-state">
                <div class="empty-icon">📂</div>
                <div class="empty-title">No predictions yet</div>
                <div class="empty-desc">
                    Upload a customer session dataset and run predictions to populate this dashboard.<br>
                    You can also use the <strong>Quick Load</strong> buttons in the sidebar to try sample data.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📁 Go to Upload Dataset", use_container_width=True, type="primary"):
                st.session_state["nav_radio"] = "📁  Upload Dataset"
                st.rerun()
        with col_b:
            sample_csv = os.path.join(os.path.dirname(__file__), "sample_data", "sample_customers.csv")
            if os.path.exists(sample_csv):
                with open(sample_csv, "rb") as f:
                    st.download_button(
                        "⬇️ Download Template CSV",
                        data=f.read(),
                        file_name="sample_customers_template.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — UPLOAD DATASET
# ═════════════════════════════════════════════════════════════════════════════
elif selected_page == "Upload Dataset":
    st.markdown('<div class="page-title">Upload Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Upload a CSV or Excel file containing customer browsing session data.</div>',
        unsafe_allow_html=True,
    )

    col_up, col_tmpl = st.columns([1.7, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Drop a file or click to browse",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            is_valid, err_msg, df = validate_uploaded_file(uploaded_file)
            if not is_valid:
                st.error(f"❌ {err_msg}")
            else:
                st.session_state.uploaded_df = df
                st.session_state.file_name = uploaded_file.name
                st.session_state.validation_result = validate_input_schema(df)
                st.session_state.quality_result = check_data_quality(df)
                st.session_state.preprocessed_df = None
                st.session_state.predictions_df = None
                st.session_state.metrics = None
                st.session_state.evaluation_result = None
                st.success(f"✅  **{uploaded_file.name}** ingested — {len(df):,} rows, {len(df.columns)} columns.")

    with col_tmpl:
        st.markdown('<div class="dl-card">', unsafe_allow_html=True)
        st.markdown('<div class="dl-card-title">📋 Input Templates</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="dl-card-desc">Pre-formatted with the exact 17 required features. Remove or ignore the <em>Revenue</em> column — it is used only for evaluation.</div>',
            unsafe_allow_html=True,
        )
        for fname, label, mime in [
            ("sample_customers.csv", "⬇️ CSV template", "text/csv"),
            ("sample_customers.xlsx", "⬇️ Excel template (.xlsx)",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ]:
            fpath = os.path.join(os.path.dirname(__file__), "sample_data", fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    st.download_button(label, data=f.read(),
                                       file_name=fname, mime=mime,
                                       use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Dataset inspection ─────────────────────────────────
    if st.session_state.uploaded_df is not None:
        df  = st.session_state.uploaded_df
        val = st.session_state.validation_result
        q   = st.session_state.quality_result

        st.markdown("---")

        # Validation banner
        if val["is_valid"]:
            callout("✓ Schema valid — all 17 required features present.", "success")
        else:
            for e in val["errors"]:
                st.error(f"❌ {e}")

        if val["has_revenue"]:
            callout(
                "ℹ️ <strong>Revenue</strong> column detected. It will be excluded from model inputs "
                "and used only for optional live evaluation (zero data leakage).",
                "info",
            )

        for w in val.get("warnings", []):
            if "Revenue" not in w:
                callout(f"⚠️ {w}", "warn")

        for rw in q.get("range_warnings", []):
            callout(f"⚠️ {rw}", "warn")

        # Quality metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{q['total_rows']:,}")
        col2.metric("Columns", q["total_cols"])
        col3.metric("Missing cells", q["total_missing_cells"])
        col4.metric("Duplicate rows", f"{q['duplicate_rows']} ({q['duplicate_pct']}%)")

        # Preview
        st.markdown("#### Data Preview")
        st.dataframe(df.head(12), use_container_width=True)

        # Feature checklist
        with st.expander("Required feature schema & verification"):
            rows = []
            for col in REQUIRED_COLUMNS:
                present = col in df.columns
                dtype = "Numeric" if col in [
                    "Administrative", "Administrative_Duration", "Informational",
                    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
                    "OperatingSystems", "Browser", "Region", "TrafficType",
                ] else ("Boolean / Integer" if col == "Weekend" else "Categorical (string)")
                rows.append({"Feature": col, "Type": dtype, "Status": "✅ Present" if present else "❌ Missing"})
            st.table(pd.DataFrame(rows))

        # Generate predictions CTA
        st.markdown("---")
        if val["is_valid"]:
            if st.button("🔮  Generate Predictions", type="primary", use_container_width=False):
                with st.spinner("Running inference pipeline…"):
                    run_predictions()
                st.success(f"✅ Predictions generated for {len(st.session_state.predictions_df):,} sessions.")
                st.info("View results on the **Purchase Predictions** or **Analytics** pages.")
        else:
            st.warning("Fix schema errors above before generating predictions.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PURCHASE PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
elif selected_page == "Purchase Predictions":
    st.markdown('<div class="page-title">Purchase Predictions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Session-level prediction results with purchase probability and intent category.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.predictions_df is None:
        # Allow running predictions from here too
        if st.session_state.uploaded_df is None:
            st.markdown("""
            <div class="section-card">
                <div class="empty-state">
                    <div class="empty-icon">🔮</div>
                    <div class="empty-title">No predictions yet</div>
                    <div class="empty-desc">Upload a dataset first, then run the prediction engine.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            val = st.session_state.validation_result
            if val and val["is_valid"]:
                if st.button("🔮  Run Predictions Now", type="primary"):
                    with st.spinner("Running inference…"):
                        run_predictions()
                    st.rerun()
            else:
                st.warning("Dataset has schema issues — please fix on the Upload Dataset page.")
    else:
        results = st.session_state.predictions_df
        m = st.session_state.metrics

        # ── Summary strip ─────────────────────────────────
        st.markdown(
            '<div class="kpi-row">'
            + kpi_card("Sessions", f'{m["total_sessions"]:,}', "", PALETTE["brand"])
            + kpi_card("Predicted Buyers", f'{m["predicted_purchases"]:,}',
                       f'{m["purchase_intention_rate"]}%', PALETTE["success"])
            + kpi_card("High Intent (≥75%)", f'{m["high_intent_count"]:,}', "", "#6366F1")
            + kpi_card("Avg Buy Probability", f'{m["avg_purchase_probability"]}%', "", PALETTE["warning"])
            + '</div>',
            unsafe_allow_html=True,
        )

        # ── Filters ────────────────────────────────────────
        col_f1, col_f2, col_f3, _ = st.columns([1.4, 1.4, 1.8, 0.5])
        with col_f1:
            intent_filter = st.selectbox(
                "Show",
                ["All sessions", "Purchase predicted", "No purchase predicted", "High intent only (≥ 75%)"],
            )
        with col_f2:
            sort_col = st.selectbox("Sort by probability", ["Highest first", "Lowest first", "Original order"])
        with col_f3:
            search = st.text_input("Search session ID", placeholder="e.g. SESS-0042")

        # Apply filters
        fd = results.copy()
        if intent_filter == "Purchase predicted":
            fd = fd[fd["Predicted_Class"] == 1]
        elif intent_filter == "No purchase predicted":
            fd = fd[fd["Predicted_Class"] == 0]
        elif intent_filter == "High intent only (≥ 75%)":
            fd = fd[fd["Purchase_Probability"] >= 0.75]

        if search.strip():
            fd = fd[fd["Session_ID"].str.lower().str.contains(search.strip().lower())]

        if sort_col == "Highest first":
            fd = fd.sort_values("Purchase_Probability", ascending=False)
        elif sort_col == "Lowest first":
            fd = fd.sort_values("Purchase_Probability", ascending=True)

        st.caption(f"Showing **{len(fd):,}** of **{len(results):,}** sessions")

        # ── Table ──────────────────────────────────────────
        display_cols = [c for c in [
            "Session_ID", "Prediction_Label", "Purchase_Probability_Pct",
            "No_Purchase_Probability", "Intent_Tier",
            "PageValues", "BounceRates", "ExitRates", "ProductRelated_Duration",
            "Month", "VisitorType", "Weekend",
        ] if c in fd.columns]

        st.dataframe(
            fd[display_cols].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "Session_ID":                 st.column_config.TextColumn("Session"),
                "Prediction_Label":           st.column_config.TextColumn("Prediction"),
                "Purchase_Probability_Pct":   st.column_config.TextColumn("Buy Prob."),
                "No_Purchase_Probability":    st.column_config.NumberColumn("No-Buy Prob.", format="%.3f"),
                "Intent_Tier":                st.column_config.TextColumn("Intent Tier"),
                "PageValues":                 st.column_config.NumberColumn("Page Value", format="%.2f"),
                "BounceRates":                st.column_config.NumberColumn("Bounce Rate", format="%.4f"),
                "ExitRates":                  st.column_config.NumberColumn("Exit Rate", format="%.4f"),
                "ProductRelated_Duration":    st.column_config.NumberColumn("Prod. Duration (s)", format="%.0f"),
            },
            height=480,
        )

        callout(
            "Probability thresholds: <strong>High Intent</strong> ≥ 75% · "
            "<strong>Medium Intent</strong> 50–74% · "
            "<strong>Low Intent</strong> 25–49% · "
            "<strong>Bounce Risk</strong> &lt; 25%. "
            "These are statistical estimates, not guaranteed purchase outcomes.",
            "warn",
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif selected_page == "Analytics":
    st.markdown('<div class="page-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Visual breakdown of purchase intent patterns across sessions, visitor types, and time periods.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.predictions_df is None:
        st.markdown("""
        <div class="section-card">
            <div class="empty-state">
                <div class="empty-icon">📈</div>
                <div class="empty-title">No data to visualise</div>
                <div class="empty-desc">Upload a dataset and generate predictions to unlock analytics.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        results = st.session_state.predictions_df
        m = st.session_state.metrics

        # ── Row 1 ──────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown('<div class="section-header">Purchase Intent Distribution</div>', unsafe_allow_html=True)
                fig = px.bar(
                    pd.DataFrame({
                        "Category": ["Purchase", "No Purchase"],
                        "Sessions": [m["predicted_purchases"], m["predicted_non_purchases"]],
                    }),
                    x="Category", y="Sessions",
                    color="Category",
                    color_discrete_map={"Purchase": CHART_COLORS["purchase"], "No Purchase": CHART_COLORS["no_purchase"]},
                    text="Sessions",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(**chart_layout(showlegend=False, height=260,
                                                 xaxis_title="", yaxis_title="Sessions"))
                render_plotly_chart(fig)

        with c2:
            with st.container(border=True):
                st.markdown('<div class="section-header">Purchase Probability Histogram</div>', unsafe_allow_html=True)
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=results["Purchase_Probability"], nbinsx=25,
                    marker_color=PALETTE["brand"], name="All sessions", opacity=0.85,
                ))
                fig2.add_vline(x=0.5, line_dash="dash", line_color=PALETTE["danger"],
                               annotation_text="0.5", annotation_font_size=11)
                fig2.update_layout(**chart_layout(height=260, xaxis_title="Purchase Probability",
                                                  yaxis_title="Sessions", showlegend=False))
                render_plotly_chart(fig2)

        # ── Row 2 ──────────────────────────────────────────
        c3, c4 = st.columns(2)

        with c3:
            if "VisitorType" in results.columns:
                with st.container(border=True):
                    st.markdown('<div class="section-header">Purchase Intent by Visitor Type</div>', unsafe_allow_html=True)
                    vt = results.groupby(["VisitorType", "Prediction_Label"]).size().reset_index(name="Count")
                    fig3 = px.bar(
                        vt, x="VisitorType", y="Count", color="Prediction_Label", barmode="group",
                        color_discrete_map={
                            "Purchase Intention": CHART_COLORS["purchase"],
                            "No Purchase Intention": CHART_COLORS["no_purchase"],
                        },
                    )
                    fig3.update_layout(**chart_layout(height=270, xaxis_title="Visitor Type",
                                                      yaxis_title="Sessions",
                                                      legend=dict(title="", orientation="h", y=1.12, x=0.5, xanchor="center")))
                    render_plotly_chart(fig3)
            else:
                st.info("VisitorType column not present in this dataset.")

        with c4:
            if "Month" in results.columns:
                with st.container(border=True):
                    st.markdown('<div class="section-header">Purchase Rate by Month</div>', unsafe_allow_html=True)
                    month_data = results.groupby("Month").agg(
                        total=("Predicted_Class", "count"),
                        buys=("Predicted_Class", lambda x: (x == 1).sum()),
                    ).reset_index()
                    month_data["Rate_%"] = (month_data["buys"] / month_data["total"] * 100).round(1)
                    month_order = ["Jan","Feb","Mar","Apr","May","Jun","June","Jul","Aug","Sep","Oct","Nov","Dec"]
                    month_data["Month"] = pd.Categorical(
                        month_data["Month"],
                        categories=[mo for mo in month_order if mo in month_data["Month"].values],
                        ordered=True,
                    )
                    month_data = month_data.sort_values("Month")
                    fig4 = px.bar(
                        month_data, x="Month", y="Rate_%",
                        color="Rate_%", color_continuous_scale="Blues",
                        text=month_data["Rate_%"].apply(lambda v: f"{v:.1f}%"),
                    )
                    fig4.update_traces(textposition="outside")
                    fig4.update_layout(**chart_layout(height=270, xaxis_title="Month",
                                                      yaxis_title="Purchase Rate (%)",
                                                      coloraxis_showscale=False, showlegend=False))
                    render_plotly_chart(fig4)
            else:
                st.info("Month column not present in this dataset.")

        # ── Row 3 ──────────────────────────────────────────
        c5, c6 = st.columns(2)

        with c5:
            if "Weekend" in results.columns:
                with st.container(border=True):
                    st.markdown('<div class="section-header">Weekend vs Weekday Purchase Intent</div>', unsafe_allow_html=True)
                    wk = results.copy()
                    wk["Day Type"] = wk["Weekend"].map(
                        lambda v: "Weekend" if str(v) in {"1", "True", "true"} else "Weekday"
                    )
                    wk_data = wk.groupby(["Day Type", "Prediction_Label"]).size().reset_index(name="Count")
                    fig5 = px.bar(
                        wk_data, x="Day Type", y="Count", color="Prediction_Label", barmode="group",
                        color_discrete_map={
                            "Purchase Intention": CHART_COLORS["purchase"],
                            "No Purchase Intention": CHART_COLORS["no_purchase"],
                        },
                    )
                    fig5.update_layout(**chart_layout(height=260, xaxis_title="",
                                                      yaxis_title="Sessions",
                                                      legend=dict(title="", orientation="h", y=1.12, x=0.5, xanchor="center")))
                    render_plotly_chart(fig5)

        with c6:
            if "PageValues" in results.columns and "ExitRates" in results.columns:
                with st.container(border=True):
                    st.markdown('<div class="section-header">Page Value vs Exit Rate</div>', unsafe_allow_html=True)
                    fig6 = px.scatter(
                        results, x="PageValues", y="ExitRates",
                        color="Prediction_Label",
                        color_discrete_map={
                            "Purchase Intention": CHART_COLORS["purchase"],
                            "No Purchase Intention": CHART_COLORS["no_purchase"],
                        },
                        size="Purchase_Probability",
                        size_max=14,
                        opacity=0.7,
                        hover_data=["Session_ID", "Purchase_Probability_Pct"],
                    )
                    fig6.update_layout(**chart_layout(height=260,
                                                      xaxis_title="Page Value",
                                                      yaxis_title="Exit Rate",
                                                      legend=dict(title="", orientation="h", y=1.12, x=0.5, xanchor="center")))
                    render_plotly_chart(fig6)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORTS & DOWNLOADS
# ═════════════════════════════════════════════════════════════════════════════
elif selected_page == "Reports & Downloads":
    st.markdown('<div class="page-title">Reports & Downloads</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Export prediction results and analytics summaries in multiple formats.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.predictions_df is None:
        st.markdown("""
        <div class="section-card">
            <div class="empty-state">
                <div class="empty-icon">⬇️</div>
                <div class="empty-title">No results to export</div>
                <div class="empty-desc">Generate predictions first to enable downloads.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        results = st.session_state.predictions_df
        m = st.session_state.metrics
        meta = get_model_metadata()

        # ── Summary KPIs ──────────────────────────────────
        st.markdown(
            '<div class="kpi-row">'
            + kpi_card("Total Sessions", f'{m["total_sessions"]:,}', "", PALETTE["brand"])
            + kpi_card("Predicted Purchases", f'{m["predicted_purchases"]:,}',
                       f'{m["purchase_intention_rate"]}%', PALETTE["success"])
            + kpi_card("Predicted No-Purchase", f'{m["predicted_non_purchases"]:,}', "", PALETTE["neutral"])
            + kpi_card("High-Intent Visitors", f'{m["high_intent_count"]:,}', "", "#6366F1")
            + '</div>',
            unsafe_allow_html=True,
        )

        # ── Download cards ────────────────────────────────
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown('<div class="dl-card">', unsafe_allow_html=True)
            st.markdown('<div class="dl-card-title">📄 Predictions CSV</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="dl-card-desc">All sessions with predicted class, purchase probability, '
                'no-purchase probability, intent tier, and original input features in a flat CSV file.</div>',
                unsafe_allow_html=True,
            )
            csv_bytes = generate_csv_report(results)
            st.download_button(
                "⬇️ Download CSV",
                data=csv_bytes,
                file_name="ShopperSense_Predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="dl-card">', unsafe_allow_html=True)
            st.markdown('<div class="dl-card-title">📊 Executive Excel Report</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="dl-card-desc">Multi-sheet workbook with: (1) full predictions table, '
                '(2) executive KPI summary, (3) model governance details. Styled with professional formatting.</div>',
                unsafe_allow_html=True,
            )
            excel_bytes = generate_excel_report(results, model_meta=meta)
            st.download_button(
                "⬇️ Download Excel (.xlsx)",
                data=excel_bytes,
                file_name="ShopperSense_Executive_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_c:
            st.markdown('<div class="dl-card">', unsafe_allow_html=True)
            st.markdown('<div class="dl-card-title">📝 Markdown Summary</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="dl-card-desc">Plain-text executive summary with KPI metrics, intent tier breakdown, '
                'and model governance notes. Paste directly into reports, wikis, or emails.</div>',
                unsafe_allow_html=True,
            )
            md_str = generate_summary_markdown(results, model_meta=meta)
            st.download_button(
                "⬇️ Download Markdown (.md)",
                data=md_str.encode("utf-8"),
                file_name="ShopperSense_Summary.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Live evaluation (if ground truth available) ───
        if st.session_state.evaluation_result is not None:
            ev = st.session_state.evaluation_result
            cm = ev["confusion_matrix"]
            st.markdown("---")
            st.markdown("#### Live Evaluation Against Ground-Truth Labels")
            callout(
                "Your uploaded dataset contained a <strong>Revenue</strong> column. "
                "The metrics below compare model predictions against the actual purchase outcomes.",
                "info",
            )
            col_e1, col_e2, col_e3, col_e4, col_e5 = st.columns(5)
            col_e1.metric("Accuracy",   f"{ev['accuracy']*100:.2f}%")
            col_e2.metric("Precision",  f"{ev['precision']:.3f}")
            col_e3.metric("Recall",     f"{ev['recall']:.3f}")
            col_e4.metric("F1-Score",   f"{ev['f1_score']:.3f}")
            col_e5.metric("ROC-AUC",    f"{ev['roc_auc']:.4f}" if isinstance(ev['roc_auc'], float) else "N/A")

        # ── Markdown preview ─────────────────────────────
        with st.expander("Preview Markdown Summary"):
            st.text_area("", value=generate_summary_markdown(results, model_meta=meta), height=260, label_visibility="collapsed")

        # ── Top sessions table ────────────────────────────
        st.markdown("---")
        st.markdown("#### Top 15 Highest-Probability Sessions")
        top15 = results.sort_values("Purchase_Probability", ascending=False).head(15)
        show_cols = [c for c in [
            "Session_ID", "Purchase_Probability_Pct", "Intent_Tier",
            "PageValues", "ProductRelated_Duration", "Month", "VisitorType",
        ] if c in top15.columns]
        st.dataframe(top15[show_cols].reset_index(drop=True), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MODEL INFORMATION
# ═════════════════════════════════════════════════════════════════════════════
elif selected_page == "Model Information":
    st.markdown('<div class="page-title">Model Information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Architecture, provenance, and documented performance benchmarks for the production model.</div>',
        unsafe_allow_html=True,
    )

    meta = get_model_metadata()
    bench = meta["documented_benchmarks"]

    c1, c2 = st.columns([1.1, 1])

    with c1:
        with st.container(border=True):
            st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)
            st.markdown(f"""
            | Parameter | Value |
            |---|---|
            | **Algorithm** | RandomForestClassifier (scikit-learn) |
            | **Estimators** | 100 decision trees |
            | **Class weight** | `balanced` (handles imbalance in purchase conversions) |
            | **Random seed** | `42` (deterministic, reproducible) |
            | **Input features** | 30 aligned numerical + one-hot features |
            | **Target variable** | `Revenue` (0 = no purchase, 1 = purchase) |
            | **Scaler** | Not applied (Random Forest operates on raw values) |
            | **Model file** | `model/random_forest_model.pkl` |
            """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="section-header">Training Dataset</div>', unsafe_allow_html=True)
            st.markdown("""
            | Item | Detail |
            |---|---|
            | **Dataset** | UCI Online Shoppers Intention |
            | **Raw rows** | 12,330 customer sessions |
            | **After cleaning** | 12,205 sessions (duplicates removed) |
            | **Train / test split** | 80% / 20% · stratified by `Revenue` |
            | **Positive rate** | ~15.5% sessions resulted in a purchase |
            """, unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            st.markdown(
                '<div class="section-header">Reported Test-Set Benchmarks</div>'
                '<div class="section-sub">Evaluated on the 20% stratified hold-out set (2,441 sessions) '
                'during original model training. These are <strong>not live metrics</strong> — they document '
                'original model performance on historical data.</div>',
                unsafe_allow_html=True,
            )
            col_b1, col_b2 = st.columns(2)
            col_b1.metric("Accuracy",   bench["overall_accuracy_str"])
            col_b2.metric("ROC-AUC",   str(bench["roc_auc"]))
            col_b3, col_b4, col_b5 = st.columns(3)
            col_b3.metric("Precision", str(bench["precision_class_1"]))
            col_b4.metric("Recall",    str(bench["recall_class_1"]))
            col_b5.metric("F1-Score",  str(bench["f1_score_class_1"]))
            callout(
                "Accuracy and ROC-AUC are for the purchase class (class 1). "
                "The <em>balanced</em> class weight improves recall for the minority purchase class "
                "at the cost of some precision.",
                "info",
            )

    # ── Feature importance chart ───────────────────────────
    with st.container(border=True):
        st.markdown('<div class="section-header">Top 15 Feature Importances</div>', unsafe_allow_html=True)
        try:
            model = get_cached_model()
            fi = (
                pd.Series(model.feature_importances_, index=model.feature_names_in_)
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            fi.columns = ["Feature", "Importance"]
            fi["Importance_%"] = (fi["Importance"] * 100).round(2)
            fig_fi = px.bar(
                fi, x="Importance_%", y="Feature", orientation="h",
                color="Importance_%", color_continuous_scale="Blues",
                text=fi["Importance_%"].apply(lambda v: f"{v:.1f}%"),
            )
            fig_fi.update_traces(textposition="outside")
            fig_fi.update_layout(**chart_layout(
                height=420,
                yaxis=dict(autorange="reversed"),
                xaxis_title="Importance (%)",
                yaxis_title="",
                coloraxis_showscale=False,
                showlegend=False,
            ))
            render_plotly_chart(fig_fi)
        except Exception as ex:
            st.warning(f"Could not load feature importances: {ex}")

    # ── 30-feature schema reference ───────────────────────
    with st.expander("Full 30-feature input schema"):
        schema_rows = []
        for i, feat in enumerate(EXPECTED_MODEL_FEATURES):
            if feat.startswith("Month_"):
                ftype = "One-hot (Month dummy)"
                origin = "get_dummies(Month, drop_first=True, ref=Aug)"
            elif feat.startswith("VisitorType_"):
                ftype = "One-hot (VisitorType dummy)"
                origin = "get_dummies(VisitorType, drop_first=True, ref=New_Visitor)"
            elif feat in ["Total_Pages", "Total_Duration", "Avg_Duration_Per_Page", "Bounce_Exit_Avg"]:
                ftype = "Engineered feature"
                origin = "Feature engineering step"
            else:
                ftype = "Raw input"
                origin = "Original dataset column"
            schema_rows.append({"#": i + 1, "Feature": feat, "Type": ftype, "Origin": origin})
        st.table(pd.DataFrame(schema_rows))

    # ── Note on scaler ────────────────────────────────────
    callout(
        "ℹ️ <strong>Why the scaler is bypassed:</strong> <code>scaler.pkl</code> was fitted and saved "
        "during Logistic Regression training. The Random Forest splits on raw feature thresholds "
        "(e.g. PageValues &gt; 42.5), so scaling would distort those thresholds without improving accuracy. "
        "Applying the scaler to the Random Forest would produce incorrect predictions.",
        "info",
    )

    # ── Live evaluation if available ─────────────────────
    if st.session_state.evaluation_result is not None:
        ev = st.session_state.evaluation_result
        cm = ev["confusion_matrix"]
        st.markdown("---")
        st.markdown("#### Live Evaluation on Uploaded Data")
        callout(
            f"Evaluated on {ev['total_evaluated_samples']:,} sessions from the uploaded dataset "
            f"that included a <strong>Revenue</strong> ground-truth column.",
            "info",
        )
        col_ev1, col_ev2, col_ev3, col_ev4 = st.columns(4)
        col_ev1.metric("Accuracy", f"{ev['accuracy']*100:.2f}%")
        col_ev2.metric("Precision", f"{ev['precision']:.3f}")
        col_ev3.metric("Recall",    f"{ev['recall']:.3f}")
        col_ev4.metric("F1-Score",  f"{ev['f1_score']:.3f}")

        cm_arr = np.array([
            [cm["true_negatives"],  cm["false_positives"]],
            [cm["false_negatives"], cm["true_positives"]],
        ])
        fig_cm = px.imshow(
            cm_arr, text_auto=True,
            x=["Predicted: No Purchase", "Predicted: Purchase"],
            y=["Actual: No Purchase",    "Actual: Purchase"],
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(**chart_layout(height=300, coloraxis_showscale=False))
        render_plotly_chart(fig_cm)
