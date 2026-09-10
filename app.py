"""
app.py
======
PharmaGuard AI — Explainable Pharmacovigilance Signal Detection
& Adverse Event Risk Prioritization

Tabs:
  1. Overview        — Dataset-level monitoring dashboard
  2. Predict Risk    — Per-report fatality prediction + SHAP explanation
  3. Signal Detection— PRR/ROR drug-reaction signal table
  4. Risk Heatmap    — Age × Drug-Count fatal-rate heatmap

Run:  streamlit run app.py
"""

import os
import json
import warnings
import hashlib

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from signal_detection import get_signals
from explainability import (
    build_explainer,
    shap_values_for_input,
    local_bar_figure,
    plain_english_summary,
)
from report_generator import generate_pdf_report

warnings.filterwarnings("ignore")
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.json")
DATA_PATH  = os.path.join(BASE_DIR, "data",   "dataset.csv")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS overrides ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2130; border: 1px solid #2d3250;
        border-radius: 10px; padding: 16px 20px; margin-bottom: 8px;
    }
    .badge-green  { background:#16a34a; color:#fff; padding:4px 14px;
                    border-radius:20px; font-weight:700; font-size:15px; }
    .badge-yellow { background:#d97706; color:#fff; padding:4px 14px;
                    border-radius:20px; font-weight:700; font-size:15px; }
    .badge-red    { background:#dc2626; color:#fff; padding:4px 14px;
                    border-radius:20px; font-weight:700; font-size:15px; }
    .tier-pill    { display:inline-block; padding:3px 12px; border-radius:12px;
                    font-size:13px; font-weight:600; }
    .tier-fatal   { background:#7f1d1d; color:#fca5a5; }
    .tier-life    { background:#7c2d12; color:#fdba74; }
    .tier-serious { background:#713f12; color:#fde68a; }
    .tier-mild    { background:#14532d; color:#86efac; }
</style>
""", unsafe_allow_html=True)


# ── Loaders (cached) ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML model ...")
def load_model_and_meta():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


GDRIVE_FILE_ID = "10RkEYIwq2YqIzIRDTlw3W3tSON-61ppQ"

def _download_gdrive(file_id: str, dest: str):
    """Download a large file from Google Drive, handling the virus-scan warning page."""
    import requests
    session = requests.Session()
    URL = "https://drive.google.com/uc?export=download"
    response = session.get(URL, params={"id": file_id}, stream=True)
    # Extract confirm token if present (large-file warning)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token is None:
        # newer Google Drive uses a query param in the response URL
        for chunk in response.iter_content(chunk_size=8192):
            text = chunk.decode("utf-8", errors="ignore")
            if "confirm=" in text:
                import re
                m = re.search(r'confirm=([0-9A-Za-z_\-]+)', text)
                if m:
                    token = m.group(1)
                break
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

@st.cache_data(show_spinner="Loading dataset ...")
def load_data():
    if not os.path.exists(DATA_PATH):
        with st.spinner("Downloading dataset from Google Drive (this may take a minute) ..."):
            _download_gdrive(GDRIVE_FILE_ID, DATA_PATH)
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Building SHAP explainer ...")
def load_explainer(model_hash: str):
    """Build SHAP explainer once and cache. model_hash ensures invalidation on retrain."""
    pipeline = joblib.load(MODEL_PATH)
    return build_explainer(pipeline, background_data=None)


@st.cache_data(show_spinner="Computing signal detection (PRR/ROR) ...")
def load_signals():
    df = load_data()
    return get_signals(df, min_count=3)


# ── Severity helpers ──────────────────────────────────────────────────────────

SEVERITY_ORDER = ["Fatal", "Life-Threatening", "Serious", "Mild"]

def prob_to_severity(prob_fatal: float,
                     is_hospitalized: bool = False,
                     is_life_threat: bool  = False) -> str:
    """Derive severity tier from predicted fatal probability."""
    if prob_fatal >= 0.50:
        return "Fatal"
    if prob_fatal >= 0.25 or is_life_threat:
        return "Life-Threatening"
    if prob_fatal >= 0.10 or is_hospitalized:
        return "Serious"
    return "Mild"


def prob_to_badge(prob_fatal: float) -> str:
    """Return HTML badge based on risk level."""
    if prob_fatal >= 0.35:
        return '<span class="badge-red">HIGH RISK</span>'
    if prob_fatal >= 0.15:
        return '<span class="badge-yellow">MODERATE RISK</span>'
    return '<span class="badge-green">LOW RISK</span>'


def tier_pill(tier: str) -> str:
    css = {
        "Fatal"           : "tier-fatal",
        "Life-Threatening": "tier-life",
        "Serious"         : "tier-serious",
        "Mild"            : "tier-mild",
    }.get(tier, "tier-mild")
    return f'<span class="tier-pill {css}">{tier}</span>'


# ── Bootstrap ─────────────────────────────────────────────────────────────────

try:
    model, meta = load_model_and_meta()
except FileNotFoundError:
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

# Stable hash for SHAP cache invalidation
_model_hash = hashlib.md5(open(MODEL_PATH, "rb").read(4096)).hexdigest()
explainer, preprocessor = load_explainer(_model_hash)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/64/shield-health.png", width=56)
    st.title("PharmaGuard AI")
    st.caption("Explainable Pharmacovigilance\nSignal Detection & Risk Prioritization")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Overview", "Predict Risk", "Signal Detection", "Risk Heatmap"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(f"**Model:** `{meta['model_name']}`")
    st.markdown(f"**ROC-AUC:** `{meta['metrics']['roc_auc']}`")
    st.markdown(f"**Accuracy:** `{meta['metrics']['accuracy']}`")
    st.caption("Data: FDA FAERS 2015–2026")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("🛡️ PharmaGuard AI — Pharmacovigilance Dashboard")
    st.markdown("**Dataset-level monitoring of FDA Adverse Events 2015–2026**")
    st.divider()

    df = load_data()
    total      = len(df)
    pct_fatal  = df["is_fatal"].mean() * 100
    pct_serious= (df["serious"] == "Yes").mean() * 100
    pct_hosp   = df["is_hospitalized"].mean() * 100

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reports",    f"{total:,}")
    c2.metric("Fatal (%)",        f"{pct_fatal:.1f}%")
    c3.metric("Serious (%)",      f"{pct_serious:.1f}%")
    c4.metric("Hospitalized (%)", f"{pct_hosp:.1f}%")

    st.divider()

    col_l, col_r = st.columns(2)

    # Reports per year
    with col_l:
        st.subheader("Reports by Year")
        yr = df.groupby("year").size().reset_index(name="Reports")
        fig_yr = px.bar(yr, x="year", y="Reports",
                        color_discrete_sequence=["#3b82f6"])
        fig_yr.update_layout(
            height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"), margin=dict(t=10, b=30),
            xaxis=dict(gridcolor="#2d2d2d"), yaxis=dict(gridcolor="#2d2d2d"),
        )
        st.plotly_chart(fig_yr, use_container_width=True)

    # Fatal rate by year
    with col_r:
        st.subheader("Fatal Rate by Year")
        fr = df.groupby("year")["is_fatal"].mean().reset_index()
        fr.columns = ["year", "Fatal Rate"]
        fig_fr = px.line(fr, x="year", y="Fatal Rate", markers=True,
                         color_discrete_sequence=["#ef4444"])
        fig_fr.update_layout(
            height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"), margin=dict(t=10, b=30),
            xaxis=dict(gridcolor="#2d2d2d"), yaxis=dict(gridcolor="#2d2d2d",
            tickformat=".0%"),
        )
        st.plotly_chart(fig_fr, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    # Severity tier distribution
    with col_l2:
        st.subheader("Severity Tier Distribution")
        sev_dist = meta.get("severity_distribution",
                            df.apply(lambda r: (
                                "Fatal" if r["is_fatal"] else
                                "Life-Threatening" if r["is_life_threat"] else
                                "Serious" if (r["is_hospitalized"] or r["is_disabling"] or r["serious"]=="Yes") else
                                "Mild"
                            ), axis=1).value_counts().to_dict())
        sev_df = pd.DataFrame(list(sev_dist.items()), columns=["Tier", "Count"])
        sev_df = sev_df.set_index("Tier").reindex(
            [s for s in SEVERITY_ORDER if s in sev_df["Tier"].values]
        ).reset_index()
        colors = {"Fatal":"#ef4444","Life-Threatening":"#f97316",
                  "Serious":"#eab308","Mild":"#22c55e"}
        fig_sev = px.bar(sev_df, x="Tier", y="Count",
                         color="Tier",
                         color_discrete_map=colors)
        fig_sev.update_layout(
            height=300, showlegend=False,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"), margin=dict(t=10, b=30),
            xaxis=dict(gridcolor="#2d2d2d"), yaxis=dict(gridcolor="#2d2d2d"),
        )
        st.plotly_chart(fig_sev, use_container_width=True)

    # Top drug categories by fatal rate
    with col_r2:
        st.subheader("Fatal Rate by Drug Count Category")
        cat_fatal = (df.groupby("drug_count_category")["is_fatal"]
                     .mean()
                     .reset_index()
                     .rename(columns={"is_fatal": "Fatal Rate"})
                     .sort_values("Fatal Rate", ascending=True))
        fig_dc = px.bar(cat_fatal, x="Fatal Rate", y="drug_count_category",
                        orientation="h", color_discrete_sequence=["#a855f7"])
        fig_dc.update_layout(
            height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"), margin=dict(t=10, b=30, l=10),
            xaxis=dict(gridcolor="#2d2d2d", tickformat=".0%"),
            yaxis=dict(gridcolor="#2d2d2d", title=""),
        )
        st.plotly_chart(fig_dc, use_container_width=True)

    # Model comparison table
    st.divider()
    st.subheader("Model Comparison (5-fold CV ROC-AUC)")
    if "model_comparison" in meta:
        mc = meta["model_comparison"]
        mc_df = pd.DataFrame([
            {
                "Model"         : name,
                "CV ROC-AUC"    : f"{v['cv_roc_auc_mean']:.4f}",
                "+/- Std"       : f"{v['cv_roc_auc_std']:.4f}",
                "Selected"      : "✅" if name == meta["model_name"] else "",
            }
            for name, v in mc.items()
        ])
        st.dataframe(mc_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run train_model.py to populate model comparison data.")

    st.divider()
    st.subheader("Top Countries by Report Volume")
    top_countries = df["country"].value_counts().head(15).reset_index()
    top_countries.columns = ["Country", "Reports"]
    fig_c = px.bar(top_countries, x="Reports", y="Country",
                   orientation="h", color_discrete_sequence=["#06b6d4"])
    fig_c.update_layout(
        height=350, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), margin=dict(t=10, b=30, l=10),
        xaxis=dict(gridcolor="#2d2d2d"), yaxis=dict(gridcolor="#2d2d2d", title=""),
    )
    st.plotly_chart(fig_c, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT RISK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Predict Risk":
    st.title("💊 Predict Adverse Event Fatality Risk")
    st.markdown(
        f"Model: **{meta['model_name']}** | "
        f"ROC-AUC: **{meta['metrics']['roc_auc']}** | "
        f"Accuracy: **{meta['metrics']['accuracy']}**"
    )
    st.divider()

    cat_cats = meta["cat_categories"]

    st.subheader("📋 Enter Report Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Patient Information**")
        patient_age_years = st.number_input("Patient Age (years)",
            min_value=0.0, max_value=120.0, value=50.0, step=1.0,
            help="Enter 0 if unknown (median imputed)")
        patient_weight_kg = st.number_input("Patient Weight (kg)",
            min_value=0.0, max_value=300.0, value=0.0, step=0.5,
            help="Enter 0 if unknown (median imputed)")
        age_opts = sorted([c for c in cat_cats["age_group"] if c != "Unknown"]) + ["Unknown"]
        age_group = st.selectbox("Age Group", age_opts,
            index=age_opts.index("Middle-Aged(41-65)") if "Middle-Aged(41-65)" in age_opts else 0)
        sex_opts = sorted([c for c in cat_cats["patient_sex"] if c != "Unknown"]) + ["Unknown"]
        patient_sex = st.selectbox("Patient Sex", sex_opts)

    with col2:
        st.markdown("**Drug Information**")
        num_drugs = st.number_input("Number of Suspect Drugs",
            min_value=1, max_value=200, value=3, step=1)
        dc_opts = sorted([c for c in cat_cats["drug_count_category"] if c != "Unknown"]) + ["Unknown"]
        drug_count_category = st.selectbox("Drug Count Category", dc_opts,
            index=dc_opts.index("2-3 Drugs") if "2-3 Drugs" in dc_opts else 0)
        st.markdown("**Reaction Information**")
        num_reactions = st.number_input("Number of Reactions",
            min_value=1, max_value=300, value=4, step=1)

    with col3:
        st.markdown("**Report Metadata**")
        serious_opts = sorted([c for c in cat_cats["serious"] if c != "Unknown"]) + ["Unknown"]
        serious = st.selectbox("Serious Report?", serious_opts,
            index=serious_opts.index("Yes") if "Yes" in serious_opts else 0)
        country_top = ["US","CA","JP","DE","FR","GB","IT","BR","EU","AU","Unknown"]
        country_rest= sorted([c for c in cat_cats["country"] if c not in country_top])
        country = st.selectbox("Country", country_top + country_rest)
        year  = st.number_input("Report Year",  min_value=2015, max_value=2030, value=2022, step=1)
        month = st.number_input("Report Month", min_value=1,    max_value=12,   value=6,    step=1)
        report_age_days = st.number_input("Report Age (days)",
            min_value=0, max_value=5000, value=365, step=1)

    st.divider()
    predict_btn = st.button("🔮 Predict Fatality Risk", type="primary", use_container_width=True)

    if predict_btn:
        age_val    = patient_age_years if patient_age_years > 0 else np.nan
        weight_val = patient_weight_kg if patient_weight_kg > 0 else np.nan

        input_data = pd.DataFrame([{
            "num_reactions"      : num_reactions,
            "num_drugs"          : num_drugs,
            "patient_age_years"  : age_val,
            "patient_weight_kg"  : weight_val,
            "report_age_days"    : report_age_days,
            "month"              : month,
            "year"               : year,
            "age_group"          : age_group,
            "patient_sex"        : patient_sex,
            "drug_count_category": drug_count_category,
            "country"            : country,
            "serious"            : serious,
        }])[meta["features_order"]]

        prediction    = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        prob_fatal    = float(probabilities[1])
        prob_not_fatal= float(probabilities[0])

        severity = prob_to_severity(prob_fatal,
                                    is_hospitalized=(serious == "Yes"),
                                    is_life_threat=(prob_fatal >= 0.25))

        # ── Result cards ──────────────────────────────────────────────────────
        st.subheader("🔍 Prediction Result")
        r1, r2, r3, r4 = st.columns(4)

        with r1:
            if prediction:
                st.error("⚠️ **FATAL** predicted")
            else:
                st.success("✅ **NON-FATAL** predicted")

        with r2:
            st.metric("Fatal Probability",     f"{prob_fatal:.1%}")

        with r3:
            st.metric("Non-Fatal Probability", f"{prob_not_fatal:.1%}")

        with r4:
            st.markdown("**Risk Badge**")
            st.markdown(prob_to_badge(prob_fatal), unsafe_allow_html=True)
            st.markdown(f"<br>{tier_pill(severity)}", unsafe_allow_html=True)

        # ── Probability bar ───────────────────────────────────────────────────
        prob_df = pd.DataFrame({
            "Outcome"    : ["Not Fatal", "Fatal"],
            "Probability": [prob_not_fatal, prob_fatal],
            "Color"      : ["#22c55e", "#ef4444"],
        })
        fig_prob = go.Figure(go.Bar(
            x=prob_df["Outcome"], y=prob_df["Probability"],
            marker_color=prob_df["Color"],
            text=[f"{v:.1%}" for v in prob_df["Probability"]],
            textposition="outside",
        ))
        fig_prob.update_layout(
            height=260, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"), yaxis=dict(tickformat=".0%", range=[0, 1.1]),
            margin=dict(t=20, b=20), showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # ── SHAP explanation ──────────────────────────────────────────────────
        shap_vals   = None
        shap_summary_txt = ""
        st.subheader("🧠 SHAP Explanation")
        try:
            shap_vals = shap_values_for_input(
                explainer, preprocessor, input_data, meta["features_order"]
            )
            # Plain-English summary
            shap_summary_txt = plain_english_summary(shap_vals, input_data, top_n=3)
            st.info(f"💬 {shap_summary_txt}")

            # SHAP bar chart
            fig_shap = local_bar_figure(shap_vals, top_n=10,
                                        title="Top 10 Feature Contributions (SHAP Local Explanation)")
            st.plotly_chart(fig_shap, use_container_width=True)

            with st.expander("All SHAP values"):
                sv_df = (shap_vals
                         .reset_index()
                         .rename(columns={"index": "Feature", 0: "SHAP Value"}))
                sv_df["Direction"] = sv_df["SHAP Value"].apply(
                    lambda v: "↑ Fatal risk" if v > 0 else "↓ Fatal risk")
                sv_df["SHAP Value"] = sv_df["SHAP Value"].round(4)
                sv_df = sv_df.reindex(
                    shap_vals.abs().sort_values(ascending=False).index
                ).reset_index(drop=True)
                st.dataframe(sv_df, use_container_width=True)

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        # ── Input summary ─────────────────────────────────────────────────────
        with st.expander("📊 Input Summary"):
            st.dataframe(input_data.T.rename(columns={0: "Value"}),
                         use_container_width=True)

        st.info("⚠️ **Disclaimer**: For research and educational purposes only. "
                "Not a substitute for professional medical judgment.")

        # ── PDF Export ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📄 Export Report")
        st.markdown("Download a complete PDF risk report including prediction, "
                    "SHAP explanation, feature table and model information.")

        try:
            import datetime
            pdf_bytes = generate_pdf_report(
                input_data      = input_data,
                prob_fatal      = prob_fatal,
                prob_not_fatal  = prob_not_fatal,
                prediction      = bool(prediction),
                severity        = severity,
                shap_vals       = shap_vals,
                shap_summary    = shap_summary_txt,
                model_name      = meta["model_name"],
                metrics         = meta["metrics"],
            )
            fname = (f"pharmagard_report_"
                     f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            st.download_button(
                label     = "⬇️ Download PDF Report",
                data      = pdf_bytes,
                file_name = fname,
                mime      = "application/pdf",
                type      = "primary",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Signal Detection":
    st.title("📡 Pharmacovigilance Signal Detection")
    st.markdown(
        "Disproportionality analysis using **PRR** (Proportional Reporting Ratio) "
        "and **ROR** (Reporting Odds Ratio) computed from the full FDA FAERS dataset.\n\n"
        "Signal criteria: count ≥ 3 · PRR ≥ 2 · Chi² ≥ 4 *(Evans et al., 2001)*"
    )
    st.divider()

    with st.spinner("Computing PRR/ROR signals ..."):
        signals = load_signals()

    flagged = signals[signals["is_signal"]].copy()

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Drug-Reaction Pairs Evaluated", f"{len(signals):,}")
    k2.metric("Flagged Signals",               f"{len(flagged):,}")
    k3.metric("Max PRR",                       f"{signals['prr'].max():.1f}")

    st.divider()

    # Filter controls
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        drug_filter = st.text_input("Filter by Drug Name (partial match)",
                                    placeholder="e.g. ADALIMUMAB")
    with fc2:
        reaction_filter = st.text_input("Filter by Reaction (partial match)",
                                        placeholder="e.g. Death")
    with fc3:
        show_all = st.checkbox("Show all pairs (not just signals)", value=False)

    display_df = signals.copy() if show_all else flagged.copy()
    if drug_filter:
        display_df = display_df[
            display_df["drug"].str.upper().str.contains(drug_filter.upper(), na=False)
        ]
    if reaction_filter:
        display_df = display_df[
            display_df["reaction"].str.contains(reaction_filter, case=False, na=False)
        ]

    # Sort controls
    sort_col = st.selectbox("Sort by", ["prr", "ror", "chi2", "count_a"], index=0)
    display_df = display_df.sort_values(sort_col, ascending=False).head(500)

    # Rename for display
    display_df = display_df.rename(columns={
        "drug"    : "Drug",
        "reaction": "Reaction",
        "count_a" : "Reports (n)",
        "prr"     : "PRR",
        "ror"     : "ROR",
        "chi2"    : "Chi²",
        "is_signal": "Signal",
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        height=450,
        column_config={
            "PRR": st.column_config.ProgressColumn(
                "PRR",
                help="Proportional Reporting Ratio",
                min_value=0,
                max_value=float(display_df["PRR"].clip(upper=100).max()) if len(display_df) else 100,
                format="%.2f",
            ),
        },
    )

    st.caption(
        f"Showing {min(len(display_df), 500):,} of "
        f"{len(flagged if not show_all else signals):,} rows. "
        "PRR = Proportional Reporting Ratio. ROR = Reporting Odds Ratio."
    )

    # PRR distribution chart
    st.divider()
    st.subheader("PRR Distribution of Flagged Signals")
    prr_clipped = flagged["prr"].clip(upper=50)
    fig_prr = px.histogram(prr_clipped, x="prr", nbins=60,
                           color_discrete_sequence=["#f59e0b"],
                           labels={"prr": "PRR (capped at 50)"})
    fig_prr.update_layout(
        height=280, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), margin=dict(t=10, b=30),
        xaxis=dict(gridcolor="#2d2d2d"), yaxis=dict(gridcolor="#2d2d2d"),
    )
    st.plotly_chart(fig_prr, use_container_width=True)

    # Top 15 drugs by signal count
    st.subheader("Top 15 Drugs by Number of Flagged Signals")
    top_drugs = (flagged.groupby("drug").size()
                 .reset_index(name="Signals")
                 .sort_values("Signals", ascending=False)
                 .head(15))
    fig_td = px.bar(top_drugs, x="Signals", y="drug", orientation="h",
                    color_discrete_sequence=["#3b82f6"])
    fig_td.update_layout(
        height=380, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), margin=dict(t=10, b=30, l=10),
        xaxis=dict(gridcolor="#2d2d2d"),
        yaxis=dict(gridcolor="#2d2d2d", title=""),
    )
    st.plotly_chart(fig_td, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RISK HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Risk Heatmap":
    st.title("🗺️ Risk Concentration Heatmap")
    st.markdown(
        "Historical fatal rate (%) across **Age Group × Drug Count Category** — "
        "computed from the full FDA FAERS dataset."
    )
    st.divider()

    df = load_data()

    # ── Heatmap data ──────────────────────────────────────────────────────────
    age_order = ["Infant(0-2)", "Child(3-12)", "Teen(13-18)", "Adult(19-40)",
                 "Middle-Aged(41-65)", "Senior(66-80)", "Elderly(81+)", "Unknown"]
    dc_order  = ["Single", "2-3 Drugs", "4-5 Drugs", "Polypharmacy(6+)", "Unknown"]

    heat = (df.groupby(["age_group", "drug_count_category"])["is_fatal"]
            .mean()
            .unstack()
            .reindex(index=[a for a in age_order if a in df["age_group"].unique()],
                     columns=[c for c in dc_order if c in df["drug_count_category"].unique()]))

    fig_heat = go.Figure(go.Heatmap(
        z          = heat.values * 100,
        x          = heat.columns.tolist(),
        y          = heat.index.tolist(),
        colorscale = "RdYlGn_r",
        text       = np.round(heat.values * 100, 1),
        texttemplate="%{text}%",
        hovertemplate="Age: %{y}<br>Drugs: %{x}<br>Fatal Rate: %{z:.1f}%<extra></extra>",
        colorbar   = dict(title="Fatal %", ticksuffix="%"),
    ))
    fig_heat.update_layout(
        title      = "Fatal Rate (%) by Age Group × Drug Count Category",
        height     = 450,
        plot_bgcolor  = "#0e1117",
        paper_bgcolor = "#0e1117",
        font       = dict(color="#fafafa", size=12),
        margin     = dict(t=50, b=60, l=10, r=10),
        xaxis      = dict(title="Drug Count Category"),
        yaxis      = dict(title="Age Group"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Observations ──────────────────────────────────────────────────────────
    st.subheader("Key Observations")
    # Highest-risk cell
    max_idx  = np.unravel_index(np.nanargmax(heat.values), heat.shape)
    max_age  = heat.index[max_idx[0]]
    max_dc   = heat.columns[max_idx[1]]
    max_val  = heat.values[max_idx] * 100

    st.markdown(
        f"- 🔴 **Highest risk cell**: `{max_age}` × `{max_dc}` — "
        f"fatal rate **{max_val:.1f}%**"
    )
    st.markdown(
        "- ⬆️ Fatal rate generally **increases with age group**, peaking in Elderly(81+)"
    )
    st.markdown(
        "- 💊 **Polypharmacy(6+)** shows elevated risk across all age groups"
    )

    st.divider()

    # ── Age group bar chart ───────────────────────────────────────────────────
    st.subheader("Fatal Rate by Age Group")
    age_fatal = (df.groupby("age_group")["is_fatal"]
                 .mean()
                 .reindex([a for a in age_order if a in df["age_group"].unique()])
                 .reset_index()
                 .rename(columns={"is_fatal": "Fatal Rate"}))
    fig_ag = px.bar(age_fatal, x="age_group", y="Fatal Rate",
                    color="Fatal Rate",
                    color_continuous_scale="RdYlGn_r",
                    labels={"age_group": "Age Group"})
    fig_ag.update_layout(
        height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), margin=dict(t=10, b=30),
        xaxis=dict(gridcolor="#2d2d2d"),
        yaxis=dict(gridcolor="#2d2d2d", tickformat=".0%"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_ag, use_container_width=True)

    # ── Sex breakdown ─────────────────────────────────────────────────────────
    st.subheader("Fatal Rate by Patient Sex")
    sex_fatal = (df.groupby("patient_sex")["is_fatal"]
                 .mean()
                 .reset_index()
                 .rename(columns={"is_fatal": "Fatal Rate"})
                 .sort_values("Fatal Rate", ascending=False))
    fig_sex = px.bar(sex_fatal, x="patient_sex", y="Fatal Rate",
                     color_discrete_sequence=["#8b5cf6"],
                     labels={"patient_sex": "Sex"})
    fig_sex.update_layout(
        height=250, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), margin=dict(t=10, b=20),
        xaxis=dict(gridcolor="#2d2d2d"),
        yaxis=dict(gridcolor="#2d2d2d", tickformat=".0%"),
    )
    st.plotly_chart(fig_sex, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "PharmaGuard AI · FDA FAERS 2015-2026 · "
    "scikit-learn + SHAP + Plotly + Streamlit"
)
