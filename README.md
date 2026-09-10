# PharmaGuard AI
## Explainable AI-Based Pharmacovigilance Signal Detection & Adverse Event Risk Prioritization

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green)](https://shap.readthedocs.io/)

---

## What is PharmaGuard AI?

PharmaGuard AI is a **multi-page Streamlit dashboard** for pharmacovigilance intelligence, built on the FDA Adverse Events Reporting System (FAERS) dataset (2015–2026, 528,000 reports).

It combines four capabilities in one interface:

| Capability | What it does |
|---|---|
| **Risk Prediction** | Predicts fatality probability for any adverse event report using a trained RandomForest classifier |
| **Explainable AI (SHAP)** | Shows *why* a prediction was made — per-prediction local SHAP bar chart + plain-English summary |
| **Signal Detection** | Computes PRR/ROR disproportionality scores for every drug–reaction pair; flags pharmacovigilance signals using Evans et al. (2001) criteria |
| **Risk Heatmap** | Visualizes historical fatal rate concentration across Age Group × Drug Count Category |

---

## Project Structure

```
project/
├── app.py                  # PharmaGuard AI 4-page Streamlit dashboard
├── train_model.py          # Model training, comparison, severity tier derivation
├── signal_detection.py     # PRR/ROR computation module
├── explainability.py       # SHAP helpers (explainer, local chart, plain-English summary)
├── requirements.txt        # All Python dependencies
├── README.md               # This file
├── agent_instructions.md   # Design decisions
├── .env.example            # Environment variable template
├── data/
│   └── dataset.csv         # FDA FAERS 2015–2026 dataset
└── models/
    ├── model.pkl            # Trained ML pipeline
    └── model_meta.json      # Feature names, categories, metrics, comparison table
```

---

## Model Comparison

Three algorithms were trained and compared using 5-fold stratified cross-validation on ROC-AUC:

| Model | CV ROC-AUC | ± Std |
|---|---|---|
| LogisticRegression | 0.7499 | ±0.0013 |
| **RandomForest** ✅ | **0.8128** | **±0.0008** |
| GradientBoosting | 0.8118 | ±0.0013 |

**Winner: RandomForest** — highest CV ROC-AUC; selected and saved as `models/model.pkl`.

### Test-set Metrics (RandomForest)

| Metric | Score |
|---|---|
| Accuracy | 0.6758 |
| ROC-AUC | **0.8117** |
| F1 (weighted) | 0.7393 |

> Note: Lower accuracy with high AUC reflects the class imbalance (~10% fatal). The model uses `class_weight="balanced"` to prioritize recall on the fatal class.

---

## Severity Tiers

Derived from the dataset's seriousness flags using a priority hierarchy:

| Tier | Criteria |
|---|---|
| **Fatal** | `is_fatal = True` |
| **Life-Threatening** | `is_life_threat = True` |
| **Serious** | `is_hospitalized = True` OR `is_disabling = True` OR `serious = "Yes"` |
| **Mild** | All other cases |

---

## Signal Detection

PRR (Proportional Reporting Ratio) and ROR (Reporting Odds Ratio) are computed for every `suspect_drug × primary_reaction` pair in the dataset. A pair is flagged as a **signal** if it meets all three Evans criteria:

- **Count (n) ≥ 3**
- **PRR ≥ 2.0**
- **Chi² ≥ 4.0**

Reference: *Evans SJW et al. Use of proportional reporting ratios (PRRs) for signal generation from spontaneous adverse drug reaction reports. Pharmacoepidemiol Drug Saf. 2001.*

---

## Local Setup & Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd project
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (already done — skip if models/ exists)

```bash
python train_model.py
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Streamlit Community Cloud Deployment

1. Push the **entire `project/`** folder to a **public GitHub repository** (include `models/model.pkl` and `models/model_meta.json`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select:
   - **Repository**: your repo
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**.

> **Important**: Commit `models/model.pkl` and `models/model_meta.json` to the repository before deploying. Run `python train_model.py` locally first if these files are missing.

---

## Render Deployment

1. Push to GitHub.
2. Create a **Web Service** on [render.com](https://render.com):
   - **Build command**: `pip install -r requirements.txt && python train_model.py`
   - **Start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Deploy.

---

## Environment Variables

No API keys are required. If you extend with an LLM, copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

---

## Disclaimer

PharmaGuard AI is for **research and educational purposes only**. Predictions and signal scores must not be used as a substitute for professional pharmacovigilance review, regulatory decisions, or medical advice.
