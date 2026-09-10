"""
explainability.py
=================
SHAP-based local explanation helpers for the PharmaGuard AI pipeline.

Provides:
  - shap_values_for_input()  : compute SHAP values for a single prediction row
  - local_bar_figure()       : Plotly bar chart of top feature contributions
  - plain_english_summary()  : auto-generated sentence from top SHAP drivers
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap


# ── SHAP explainer factory ────────────────────────────────────────────────────

def build_explainer(pipeline, background_data=None):
    """
    Build a SHAP explainer for the classifier inside a sklearn Pipeline.

    Uses TreeExplainer with NO background data (tree_path_dependent mode)
    for tree-based models — fully in-memory, no temp file I/O, works on
    Windows/OneDrive paths without Errno 22.

    Parameters
    ----------
    pipeline        : Fitted sklearn Pipeline (preprocessor + clf).
    background_data : Ignored. Kept for API compatibility.

    Returns
    -------
    (explainer, preprocessor)
    """
    preprocessor = pipeline.named_steps["pre"]
    clf          = pipeline.named_steps["clf"]
    clf_name     = type(clf).__name__

    try:
        if clf_name in ("RandomForestClassifier",
                        "GradientBoostingClassifier",
                        "XGBClassifier",
                        "DecisionTreeClassifier",
                        "ExtraTreesClassifier"):
            # No background data → tree_path_dependent, zero file I/O
            explainer = shap.TreeExplainer(clf)
        else:
            # Linear models need a small background — use a tiny zero array
            n_features = (
                pipeline.named_steps["pre"]
                .transform(pd.DataFrame(
                    [[0] * len(pipeline.named_steps["pre"]
                               .transformers_[0][2]) +
                     ["Unknown"] * len(pipeline.named_steps["pre"]
                                       .transformers_[1][2])],
                    columns=(pipeline.named_steps["pre"].transformers_[0][2] +
                             pipeline.named_steps["pre"].transformers_[1][2])
                )).shape[1]
            )
            bg_zeros = np.zeros((1, n_features))
            explainer = shap.LinearExplainer(clf, bg_zeros)
    except Exception:
        # Universal fallback — predict_proba wrapper, no file I/O
        def _predict(x):
            return clf.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(_predict, np.zeros((1, 1)))

    return explainer, preprocessor


def shap_values_for_input(explainer,
                           preprocessor,
                           input_row: pd.DataFrame,
                           feature_names: list) -> pd.Series:
    """
    Compute SHAP values for a single prediction row.

    Parameters
    ----------
    explainer     : Fitted SHAP explainer.
    preprocessor  : Fitted ColumnTransformer (from the pipeline).
    input_row     : 1-row DataFrame with raw feature values.
    feature_names : Ordered list of feature names matching the pipeline input.

    Returns
    -------
    pd.Series of SHAP values indexed by feature name.
    """
    X_transformed = preprocessor.transform(input_row)

    raw = explainer.shap_values(X_transformed)

    # shap_values can return list[array] (binary) or 2D array
    if isinstance(raw, list):
        # Binary: index 1 = positive class (Fatal)
        sv = raw[1][0] if raw[1].ndim > 1 else raw[1]
    else:
        sv = raw[0] if raw.ndim > 1 else raw

    sv = np.array(sv).ravel()

    # Pad / truncate to match feature_names length
    n = len(feature_names)
    if len(sv) > n:
        sv = sv[:n]
    elif len(sv) < n:
        sv = np.pad(sv, (0, n - len(sv)))

    return pd.Series(sv, index=feature_names)


# ── Plotly local explanation bar chart ───────────────────────────────────────

def local_bar_figure(shap_series: pd.Series,
                     top_n: int = 10,
                     title: str = "Feature Contributions (SHAP)") -> go.Figure:
    """
    Create a horizontal bar chart showing top-n SHAP contributions for
    a single prediction.

    Positive = pushes toward Fatal, Negative = pushes toward Not Fatal.
    """
    # Sort by absolute magnitude, take top_n
    top = shap_series.reindex(
        shap_series.abs().nlargest(top_n).index
    )
    top = top.sort_values()   # ascending so largest bar is at top in horizontal chart

    colours = ["#ef4444" if v >= 0 else "#22c55e" for v in top.values]
    hover   = [f"{'+ increases' if v >= 0 else '- decreases'} Fatal risk ({v:+.4f})"
               for v in top.values]

    fig = go.Figure(go.Bar(
        x           = top.values,
        y           = top.index.tolist(),
        orientation = "h",
        marker_color= colours,
        hovertext   = hover,
        hoverinfo   = "y+text",
    ))

    fig.update_layout(
        title       = dict(text=title, font=dict(size=14)),
        xaxis_title = "SHAP value (impact on Fatal probability)",
        yaxis_title = "",
        height      = 380,
        margin      = dict(l=10, r=20, t=40, b=40),
        plot_bgcolor= "#0e1117",
        paper_bgcolor="#0e1117",
        font        = dict(color="#fafafa", size=12),
        xaxis       = dict(gridcolor="#2d2d2d", zerolinecolor="#555"),
        yaxis       = dict(gridcolor="#2d2d2d"),
    )
    return fig


# ── Plain-English summary ─────────────────────────────────────────────────────

def plain_english_summary(shap_series: pd.Series,
                           input_row: pd.DataFrame,
                           top_n: int = 3) -> str:
    """
    Generate a plain-English sentence from SHAP values.

    Parameters
    ----------
    shap_series : SHAP values indexed by feature name.
    input_row   : 1-row DataFrame with the actual input values.
    top_n       : Number of top drivers to mention.

    Returns
    -------
    Human-readable string, e.g.:
      "This risk score is driven mainly by Polypharmacy(6+) drugs,
       Elderly(81+) age group, and 12 concurrent reactions."
    """
    top_features = shap_series.abs().nlargest(top_n).index.tolist()
    shap_top     = shap_series[top_features]

    parts = []
    for feat in top_features:
        val       = input_row[feat].iloc[0]
        shap_val  = shap_top[feat]
        direction = "increasing" if shap_val > 0 else "reducing"

        # Safe int/float converters that handle NaN
        def safe_int(v, default="?"):
            try:
                return int(v) if not pd.isna(v) else default
            except Exception:
                return default

        def safe_float(v, fmt=".0f", default="?"):
            try:
                return format(float(v), fmt) if not pd.isna(v) else default
            except Exception:
                return default

        # Human-readable feature descriptions
        descriptions = {
            "num_reactions"      : f"{safe_int(val)} concurrent reaction(s)",
            "num_drugs"          : f"{safe_int(val)} suspect drug(s)",
            "patient_age_years"  : f"patient age {safe_float(val)} years",
            "patient_weight_kg"  : (f"weight {safe_float(val, '.1f')} kg" if not pd.isna(val) else "unknown weight"),
            "report_age_days"    : f"{safe_int(val)}-day report lag",
            "month"              : f"month {safe_int(val)}",
            "year"               : f"year {safe_int(val)}",
            "age_group"          : f"{val} age group",
            "patient_sex"        : f"{val} sex",
            "drug_count_category": f"{val}",
            "country"            : f"{val} country",
            "serious"            : f"{'serious' if val == 'Yes' else 'non-serious'} report",
        }
        label = descriptions.get(feat, f"{feat}={val}")
        parts.append(f"**{label}** ({direction} fatal risk)")

    if not parts:
        return "Insufficient SHAP data to generate an explanation."

    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return f"This risk score is driven mainly by {body}."


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, json, joblib

    BASE  = os.path.dirname(__file__)
    MODEL = os.path.join(BASE, "models", "model.pkl")
    META  = os.path.join(BASE, "models", "model_meta.json")
    DATA  = os.path.join(BASE, "data",   "dataset.csv")

    pipeline = joblib.load(MODEL)
    with open(META) as f:
        meta = json.load(f)

    df       = pd.read_csv(DATA)
    features = meta["features_order"]

    X = df[features].head(500)

    print("Building SHAP explainer (no background data)...")
    explainer, pre = build_explainer(pipeline)

    sample_row = X.iloc[[0]]
    print("Computing SHAP values for one row...")
    sv = shap_values_for_input(explainer, pre, sample_row, features)
    print("SHAP values:")
    print(sv.sort_values(key=abs, ascending=False).head(5))

    summary = plain_english_summary(sv, sample_row)
    print("\nPlain-English Summary:")
    print(summary)
    print("\nSmoke test PASSED.")
