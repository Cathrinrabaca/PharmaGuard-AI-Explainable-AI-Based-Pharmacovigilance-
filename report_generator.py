"""
report_generator.py
===================
PharmaGuard AI — PDF Report Generator

Generates a professional single-patient adverse event risk report as a
PDF byte stream (suitable for st.download_button).

Dependencies: reportlab (pure Python, no system binaries needed)
"""

import io
import datetime
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF


# ── Colour palette ────────────────────────────────────────────────────────────
C_NAVY      = colors.HexColor("#1e3a5f")
C_BLUE      = colors.HexColor("#2563eb")
C_LIGHT     = colors.HexColor("#eff6ff")
C_BORDER    = colors.HexColor("#bfdbfe")
C_RED       = colors.HexColor("#dc2626")
C_RED_LIGHT = colors.HexColor("#fef2f2")
C_ORANGE    = colors.HexColor("#d97706")
C_ORA_LIGHT = colors.HexColor("#fffbeb")
C_GREEN     = colors.HexColor("#16a34a")
C_GRN_LIGHT = colors.HexColor("#f0fdf4")
C_GRAY      = colors.HexColor("#6b7280")
C_DARK      = colors.HexColor("#111827")
C_WHITE     = colors.white
C_ROW_ALT   = colors.HexColor("#f8fafc")


# ── Style helpers ─────────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()

    def _s(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    return {
        "title"    : _s("pg_title",   fontSize=22, textColor=C_NAVY,
                        fontName="Helvetica-Bold", spaceAfter=2),
        "subtitle" : _s("pg_sub",     fontSize=10, textColor=C_GRAY,
                        fontName="Helvetica",      spaceAfter=6),
        "h2"       : _s("pg_h2",      fontSize=13, textColor=C_NAVY,
                        fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4),
        "body"     : _s("pg_body",    fontSize=9,  textColor=C_DARK,
                        fontName="Helvetica",      leading=14),
        "bold"     : _s("pg_bold",    fontSize=9,  textColor=C_DARK,
                        fontName="Helvetica-Bold"),
        "small"    : _s("pg_small",   fontSize=8,  textColor=C_GRAY,
                        fontName="Helvetica"),
        "center"   : _s("pg_center",  fontSize=9,  textColor=C_DARK,
                        fontName="Helvetica",      alignment=TA_CENTER),
        "disclaimer": _s("pg_disc",   fontSize=7.5,textColor=C_GRAY,
                        fontName="Helvetica-Oblique", leading=11),
        "shap_pos" : _s("shap_pos",   fontSize=8.5,textColor=C_RED,
                        fontName="Helvetica-Bold"),
        "shap_neg" : _s("shap_neg",   fontSize=8.5,textColor=C_GREEN,
                        fontName="Helvetica-Bold"),
    }


# ── Risk badge drawing ────────────────────────────────────────────────────────
def _badge_drawing(label: str, bg_hex: str, fg_hex: str,
                   width=110, height=22) -> Drawing:
    d   = Drawing(width, height)
    bg  = colors.HexColor(bg_hex)
    fg  = colors.HexColor(fg_hex)
    d.add(Rect(0, 0, width, height, rx=4, ry=4,
               fillColor=bg, strokeColor=bg))
    d.add(String(width / 2, 5, label,
                 fontName="Helvetica-Bold", fontSize=9,
                 fillColor=fg, textAnchor="middle"))
    return d


def _risk_badge(prob_fatal: float):
    if prob_fatal >= 0.35:
        return _badge_drawing("HIGH RISK",      "#fef2f2", "#dc2626")
    if prob_fatal >= 0.15:
        return _badge_drawing("MODERATE RISK",  "#fffbeb", "#d97706")
    return     _badge_drawing("LOW RISK",       "#f0fdf4", "#16a34a")


def _tier_badge(tier: str):
    palettes = {
        "Fatal"           : ("#fee2e2", "#991b1b"),
        "Life-Threatening": ("#ffedd5", "#9a3412"),
        "Serious"         : ("#fef9c3", "#854d0e"),
        "Mild"            : ("#dcfce7", "#166534"),
    }
    bg, fg = palettes.get(tier, ("#f3f4f6", "#374151"))
    return _badge_drawing(tier, bg, fg, width=130)


# ── SHAP bar drawing ──────────────────────────────────────────────────────────
def _shap_bar_drawing(shap_series: pd.Series, top_n: int = 8,
                      width=440, row_h=18) -> Drawing:
    top = shap_series.reindex(shap_series.abs().nlargest(top_n).index)
    top = top.sort_values()

    bar_area_w = 200
    label_w    = 160
    val_w      = 55
    max_abs    = max(top.abs().max(), 1e-9)
    height     = top_n * row_h + 20

    d = Drawing(width, height)

    # Zero line
    zero_x = label_w + bar_area_w / 2
    d.add(Rect(zero_x, 0, 0.8, height - 10,
               fillColor=colors.HexColor("#9ca3af"),
               strokeColor=colors.HexColor("#9ca3af")))

    for i, (feat, val) in enumerate(top.items()):
        y     = i * row_h + 4
        bw    = abs(val) / max_abs * (bar_area_w / 2 - 4)
        bw    = max(bw, 2)
        color = colors.HexColor("#ef4444") if val >= 0 else colors.HexColor("#22c55e")

        if val >= 0:
            bx = zero_x
        else:
            bx = zero_x - bw

        d.add(Rect(bx, y, bw, row_h - 4,
                   fillColor=color, strokeColor=color))

        # Feature label
        label = str(feat)[:26]
        d.add(String(label_w - 4, y + 3, label,
                     fontName="Helvetica", fontSize=7,
                     fillColor=colors.HexColor("#374151"),
                     textAnchor="end"))

        # Value label
        d.add(String(zero_x + bar_area_w / 2 + 6, y + 3,
                     f"{val:+.4f}",
                     fontName="Helvetica", fontSize=7,
                     fillColor=colors.HexColor("#374151")))

    return d


# ── Main PDF builder ──────────────────────────────────────────────────────────

def generate_pdf_report(
    input_data   : pd.DataFrame,
    prob_fatal   : float,
    prob_not_fatal: float,
    prediction   : bool,
    severity     : str,
    shap_vals    : "pd.Series | None",
    shap_summary : str,
    model_name   : str,
    metrics      : dict,
) -> bytes:
    """
    Build and return a PDF report as bytes.

    Parameters
    ----------
    input_data     : 1-row DataFrame of features used for prediction.
    prob_fatal     : Predicted probability of fatal outcome.
    prob_not_fatal : Predicted probability of non-fatal outcome.
    prediction     : Boolean — True = Fatal predicted.
    severity       : Severity tier string.
    shap_vals      : pd.Series of SHAP values (or None if unavailable).
    shap_summary   : Plain-English SHAP explanation string.
    model_name     : Name of the ML model used.
    metrics        : Dict with accuracy, roc_auc, f1_score.

    Returns
    -------
    bytes — PDF content ready for st.download_button.
    """
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )
    S      = _styles()
    W      = doc.width
    story  = []

    now    = datetime.datetime.now().strftime("%d %b %Y  %H:%M")

    # ── HEADER ────────────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("PharmaGuard AI", S["title"]),
        Paragraph(f"Generated: {now}", ParagraphStyle(
            "right", parent=S["small"], alignment=TA_RIGHT)),
    ]]
    header_tbl = Table(header_data, colWidths=[W * 0.7, W * 0.3])
    header_tbl.setStyle(TableStyle([
        ("VALIGN",      (0,0), (-1,-1), "BOTTOM"),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(header_tbl)
    story.append(Paragraph(
        "Adverse Event Fatality Risk Report  ·  FDA FAERS 2015–2026",
        S["subtitle"]
    ))
    story.append(HRFlowable(width=W, thickness=2, color=C_NAVY, spaceAfter=10))

    # ── PREDICTION RESULT ─────────────────────────────────────────────────────
    story.append(Paragraph("Prediction Result", S["h2"]))

    outcome_txt = "FATAL" if prediction else "NON-FATAL"
    outcome_col = C_RED if prediction else C_GREEN
    outcome_bg  = C_RED_LIGHT if prediction else C_GRN_LIGHT

    result_data = [
        ["Predicted Outcome", "Fatal Probability", "Non-Fatal Probability",
         "Risk Level", "Severity Tier"],
        [
            Paragraph(f"<b>{outcome_txt}</b>",
                      ParagraphStyle("oc", fontSize=11,
                                     textColor=outcome_col,
                                     fontName="Helvetica-Bold",
                                     alignment=TA_CENTER)),
            Paragraph(f"<b>{prob_fatal:.1%}</b>",
                      ParagraphStyle("fp", fontSize=13,
                                     textColor=C_RED if prob_fatal >= 0.35 else C_ORANGE if prob_fatal >= 0.15 else C_GREEN,
                                     fontName="Helvetica-Bold",
                                     alignment=TA_CENTER)),
            Paragraph(f"<b>{prob_not_fatal:.1%}</b>",
                      ParagraphStyle("nfp", fontSize=13,
                                     textColor=C_GREEN,
                                     fontName="Helvetica-Bold",
                                     alignment=TA_CENTER)),
            _risk_badge(prob_fatal),
            _tier_badge(severity),
        ],
    ]

    col_w = W / 5
    result_tbl = Table(result_data, colWidths=[col_w] * 5, rowHeights=[20, 36])
    result_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 8),
        ("ALIGN",        (0, 0), (-1,-1), "CENTER"),
        ("VALIGN",       (0, 0), (-1,-1), "MIDDLE"),
        ("BACKGROUND",   (0, 1), (-1, 1), outcome_bg),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [outcome_bg]),
        ("GRID",         (0, 0), (-1,-1), 0.4, C_BORDER),
        ("TOPPADDING",   (0, 0), (-1,-1), 6),
        ("BOTTOMPADDING",(0, 0), (-1,-1), 6),
    ]))
    story.append(result_tbl)
    story.append(Spacer(1, 10))

    # ── SHAP SUMMARY ──────────────────────────────────────────────────────────
    if shap_summary:
        story.append(Paragraph("AI Explanation (SHAP)", S["h2"]))
        story.append(Paragraph(shap_summary, S["body"]))
        story.append(Spacer(1, 6))

    # ── SHAP BAR CHART ────────────────────────────────────────────────────────
    if shap_vals is not None:
        story.append(Paragraph("Feature Contributions (SHAP Values)", S["h2"]))
        story.append(Paragraph(
            "<font color='#dc2626'>Red bars</font> = increase Fatal risk  &nbsp;&nbsp; "
            "<font color='#16a34a'>Green bars</font> = reduce Fatal risk",
            S["small"]
        ))
        story.append(Spacer(1, 4))

        bar_d = _shap_bar_drawing(shap_vals, top_n=10, width=int(W))
        story.append(bar_d)
        story.append(Spacer(1, 8))

        # SHAP table
        top_sv = shap_vals.reindex(shap_vals.abs().nlargest(10).index)
        sv_data = [["Feature", "SHAP Value", "Direction"]]
        for feat, val in top_sv.sort_values(key=abs, ascending=False).items():
            direction = "↑ Increases Fatal risk" if val >= 0 else "↓ Reduces Fatal risk"
            d_style   = S["shap_pos"] if val >= 0 else S["shap_neg"]
            sv_data.append([
                Paragraph(str(feat), S["body"]),
                Paragraph(f"{val:+.4f}", d_style),
                Paragraph(direction, d_style),
            ])

        sv_tbl = Table(sv_data, colWidths=[W*0.45, W*0.2, W*0.35])
        sv_style = [
            ("BACKGROUND",    (0,0), (-1,0), C_NAVY),
            ("TEXTCOLOR",     (0,0), (-1,0), C_WHITE),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,0), 8),
            ("ALIGN",         (0,0), (-1,-1),"LEFT"),
            ("VALIGN",        (0,0), (-1,-1),"MIDDLE"),
            ("GRID",          (0,0), (-1,-1), 0.4, C_BORDER),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ]
        for i in range(1, len(sv_data)):
            bg = C_ROW_ALT if i % 2 == 0 else C_WHITE
            sv_style.append(("BACKGROUND", (0, i), (-1, i), bg))
        sv_tbl.setStyle(TableStyle(sv_style))
        story.append(KeepTogether(sv_tbl))
        story.append(Spacer(1, 10))

    # ── INPUT FEATURES ────────────────────────────────────────────────────────
    story.append(Paragraph("Input Features Used for Prediction", S["h2"]))

    # Split into two columns
    rows       = list(input_data.T.iterrows())
    mid        = (len(rows) + 1) // 2
    left_rows  = rows[:mid]
    right_rows = rows[mid:]

    def _input_table(r_list):
        data = [["Feature", "Value"]]
        for feat, row in r_list:
            val = row.iloc[0]
            val_str = "Unknown" if (isinstance(val, float) and np.isnan(val)) else str(val)
            data.append([
                Paragraph(str(feat), S["bold"]),
                Paragraph(val_str,   S["body"]),
            ])
        t = Table(data, colWidths=[(W/2 - 0.5*cm) * 0.55,
                                    (W/2 - 0.5*cm) * 0.45])
        ts = [
            ("BACKGROUND",    (0,0), (-1,0), C_BLUE),
            ("TEXTCOLOR",     (0,0), (-1,0), C_WHITE),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,0), 7.5),
            ("GRID",          (0,0), (-1,-1), 0.3, C_BORDER),
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("LEFTPADDING",   (0,0), (-1,-1), 5),
            ("FONTSIZE",      (0,1), (-1,-1), 8),
        ]
        for i in range(1, len(data)):
            bg = C_ROW_ALT if i % 2 == 0 else C_WHITE
            ts.append(("BACKGROUND", (0, i), (-1, i), bg))
        t.setStyle(TableStyle(ts))
        return t

    two_col = Table(
        [[_input_table(left_rows), _input_table(right_rows)]],
        colWidths=[W / 2 - 0.25*cm, W / 2 - 0.25*cm],
        hAlign="LEFT",
    )
    two_col.setStyle(TableStyle([
        ("VALIGN",  (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("COLPADDING",   (0,0), (-1,-1), 4),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 12))

    # ── MODEL INFO ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=C_BORDER, spaceAfter=6))
    story.append(Paragraph("Model Information", S["h2"]))

    model_data = [
        ["Algorithm", "ROC-AUC", "Accuracy", "F1 Score (weighted)"],
        [
            Paragraph(model_name, S["bold"]),
            Paragraph(str(metrics.get("roc_auc", "-")),  S["body"]),
            Paragraph(str(metrics.get("accuracy", "-")), S["body"]),
            Paragraph(str(metrics.get("f1_score", "-")), S["body"]),
        ],
    ]
    model_tbl = Table(model_data, colWidths=[W*0.35, W*0.2, W*0.2, W*0.25])
    model_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), C_NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0), C_WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 8),
        ("BACKGROUND",    (0,1), (-1,1), C_LIGHT),
        ("GRID",          (0,0), (-1,-1), 0.4, C_BORDER),
        ("ALIGN",         (0,0), (-1,-1), "LEFT"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("FONTSIZE",      (0,1), (-1,-1), 8.5),
    ]))
    story.append(model_tbl)
    story.append(Spacer(1, 14))

    # ── FOOTER / DISCLAIMER ───────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=C_BORDER, spaceAfter=6))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by PharmaGuard AI for research and "
        "educational purposes only. The predictions and SHAP explanations are based on "
        "historical FDA FAERS data and must NOT be used as a substitute for professional "
        "pharmacovigilance review, regulatory decision-making, or clinical medical judgment. "
        "Always consult a qualified healthcare or drug safety professional.",
        S["disclaimer"]
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"PharmaGuard AI  ·  FDA FAERS 2015–2026  ·  {now}",
        ParagraphStyle("footer", parent=S["small"], alignment=TA_CENTER)
    ))

    doc.build(story)
    return buf.getvalue()
