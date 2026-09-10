"""
train_model.py
==============
PharmaGuard AI — FDA Adverse Events 2015-2026
Target: is_fatal (binary classification)

Also derives a Severity Tier column and documents model comparison.

Run:  python train_model.py
"""

import os
import warnings
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading dataset ...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ── 2. Derive Severity Tier ───────────────────────────────────────────────────
# Priority order: Fatal > Life-Threatening > Hospitalized/Disabling > Serious > Mild
def assign_severity(row):
    if row["is_fatal"]:
        return "Fatal"
    if row["is_life_threat"]:
        return "Life-Threatening"
    if row["is_hospitalized"] or row["is_disabling"]:
        return "Serious"
    if row["serious"] == "Yes":
        return "Serious"
    return "Mild"

print("Deriving severity tiers ...")
df["severity_tier"] = df.apply(assign_severity, axis=1)
print(f"  Severity distribution:\n{df['severity_tier'].value_counts()}")

# ── 3. Feature selection ──────────────────────────────────────────────────────
TARGET = "is_fatal"

NUM_FEATURES = [
    "num_reactions",
    "num_drugs",
    "patient_age_years",
    "patient_weight_kg",
    "report_age_days",
    "month",
    "year",
]
CAT_FEATURES = [
    "age_group",
    "patient_sex",
    "drug_count_category",
    "country",
    "serious",
]
FEATURES = NUM_FEATURES + CAT_FEATURES

df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(bool)

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"\nTarget distribution:\n{y.value_counts()}")
print(f"Class balance: {y.mean():.3%} positive (fatal)")

# ── 4. Preprocessing pipelines ────────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUM_FEATURES),
    ("cat", categorical_transformer, CAT_FEATURES),
])

# ── 5. Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y,
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 6. Model comparison ───────────────────────────────────────────────────────
candidates = {
    "LogisticRegression": Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            max_iter=500, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )),
    ]),
    "RandomForest": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )),
    ]),
    "GradientBoosting": Pipeline([
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.1, subsample=0.8, random_state=42,
        )),
    ]),
}

print("\n-- Model Comparison (5-fold CV ROC-AUC) --")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    cv_results[name] = {"mean_auc": scores.mean(), "std_auc": scores.std()}
    print(f"  {name:30s}  AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

best_name = max(cv_results, key=lambda k: cv_results[k]["mean_auc"])
print(f"\nBest model: {best_name}  (AUC = {cv_results[best_name]['mean_auc']:.4f})")

# ── 7. Train best model on full training set ──────────────────────────────────
best_pipeline = candidates[best_name]
print(f"\nTraining {best_name} on full training set ...")
best_pipeline.fit(X_train, y_train)

# ── 8. Evaluation ─────────────────────────────────────────────────────────────
y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
f1  = f1_score(y_test, y_pred, average="weighted")

print(f"\n-- Test-set Metrics --")
print(f"  Accuracy  : {acc:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  F1 (wtd)  : {f1:.4f}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Not Fatal", "Fatal"]))

# ── 9. Save model ──────────────────────────────────────────────────────────────
joblib.dump(best_pipeline, MODEL_PATH)
print(f"\nModel saved -> {MODEL_PATH}")

# ── 10. Save metadata ─────────────────────────────────────────────────────────
fitted_pre   = best_pipeline.named_steps["pre"]
cat_pipeline = fitted_pre.named_transformers_["cat"]
encoder      = cat_pipeline.named_steps["encoder"]

cat_categories = {
    feat: list(cats)
    for feat, cats in zip(CAT_FEATURES, encoder.categories_)
}

severity_dist = df["severity_tier"].value_counts().to_dict()

meta = {
    "model_name"     : best_name,
    "target"         : TARGET,
    "num_features"   : NUM_FEATURES,
    "cat_features"   : CAT_FEATURES,
    "features_order" : FEATURES,
    "cat_categories" : cat_categories,
    "metrics": {
        "accuracy": round(acc, 4),
        "roc_auc" : round(auc, 4),
        "f1_score": round(f1, 4),
    },
    "class_names"    : ["Not Fatal", "Fatal"],
    "severity_distribution": severity_dist,
    "model_comparison": {
        name: {
            "cv_roc_auc_mean": round(v["mean_auc"], 4),
            "cv_roc_auc_std" : round(v["std_auc"],  4),
        }
        for name, v in cv_results.items()
    },
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata saved -> {META_PATH}")

# ── 11. Smoke test ─────────────────────────────────────────────────────────────
print("\n-- Smoke test --")
loaded = joblib.load(MODEL_PATH)
sample = X_test.iloc[[0]]
pred   = loaded.predict(sample)[0]
prob   = loaded.predict_proba(sample)[0]
print(f"  Prediction : {'Fatal' if pred else 'Not Fatal'}")
print(f"  Probability: Not Fatal={prob[0]:.3f}, Fatal={prob[1]:.3f}")
print("\nTraining complete.")
