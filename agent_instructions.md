# Agent Instructions — PharmaGuard AI

## Project
Explainable AI-Based Pharmacovigilance Signal Detection & Adverse Event Risk Prioritization

## Dataset
- Source: FDA Adverse Events Reporting System (FAERS) 2015–2026
- File: `data/dataset.csv`
- Records: 528,000 adverse event reports, 30 columns
- Target: `is_fatal` (binary classification)

## ML Task
Binary classification: predict fatal outcome from patient/drug characteristics.

## Architecture Overview

### train_model.py
- Derives severity tier from: is_fatal → is_life_threat → is_hospitalized/is_disabling → serious → Mild
- Compares: LogisticRegression, RandomForest, GradientBoosting (5-fold CV ROC-AUC)
- Best model (RandomForest, AUC=0.8128) saved to models/model.pkl
- model_meta.json stores: feature names, category lists, metrics, model_comparison dict, severity_distribution

### signal_detection.py
- Computes PRR and ROR for all suspect_drug × primary_reaction pairs
- Signal criteria: count≥3, PRR≥2, chi²≥4 (Evans 2001)
- Uses in-process dict cache (_CACHE) to avoid recomputation on Streamlit reruns

### explainability.py
- build_explainer(): uses TreeExplainer for RF/GB, LinearExplainer for LR, KernelExplainer as fallback
- shap_values_for_input(): transforms 1-row input through preprocessor, returns pd.Series of SHAP values
- local_bar_figure(): Plotly horizontal bar, red=increases Fatal risk, green=decreases
- plain_english_summary(): auto-generates sentence from top-N SHAP drivers using actual feature values

### app.py (4 pages via sidebar radio)
1. Overview: KPI metrics, reports-by-year, fatal-rate-by-year, severity distribution, drug-count fatal rate, model comparison table, country volume
2. Predict Risk: existing input form + prediction + risk badge + severity tier + SHAP bar + plain-English summary
3. Signal Detection: PRR/ROR table with filters (drug, reaction, signal-only toggle), sort by PRR/ROR/chi2, PRR histogram, top-drugs chart
4. Risk Heatmap: Age×DrugCount fatal rate heatmap, age group bar, sex bar

## Feature Selection (no leakage)
Numeric: num_reactions, num_drugs, patient_age_years, patient_weight_kg, report_age_days, month, year
Categorical: age_group, patient_sex, drug_count_category, country, serious

Excluded (post-event outcomes / high-cardinality free text):
- is_hospitalized, is_life_threat, is_disabling, patient_recovered
- reactions, primary_reaction, reaction_outcomes

## No API Keys Required
All functionality is purely local ML + SHAP. No LLM or external API used.

## Workflow
1. python train_model.py  → trains model, saves model.pkl + model_meta.json
2. streamlit run app.py   → launches 4-tab dashboard
