import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    md("""# Paired M0 versus M1 weather ablation

This notebook tests whether fixed-run weather adds useful predictive information beyond the no-weather baseline. P95 is the primary outcome (XGBoost); P90 is the sensitivity outcome (CatBoost). Model families and hyperparameters are carried forward from the prior validation-based selection; this notebook does not retune them.

Only rows with complete fixed-run weather are used, and precisely the same `row_id` values, outcomes, and frozen `04_master_split.csv` assignments are used for M0 and M1. Models are trained on the matched development rows and evaluated on matched `test_seen` and `test_unseen` cohorts. The 0.50 classification threshold is only a common diagnostic point for confusion matrices; it is not a recommended operating threshold. The table also includes an all-negative accuracy reference to show the effect of class imbalance. Weather uplift is primarily assessed with PR-AUC, ROC-AUC, and Brier score. No test-based tuning or calibration is performed.
"""),
    code("""from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
DATA_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "reports" / "tables" / "08_weather_model_comparison"
FIGURE_DIR = ROOT / "reports" / "figures" / "08_weather_model_comparison"
MODEL_DIR = ROOT / "models" / "08_weather_model_comparison"
for folder in [TABLE_DIR, FIGURE_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

P95_M0_PATH = DATA_DIR / "05_model_ready_m0_p95.csv"
P90_M0_PATH = DATA_DIR / "05_model_ready_m0_p90.csv"
P95_M1_PATH = DATA_DIR / "07_model_ready_m1_p95.csv"
P90_M1_PATH = DATA_DIR / "07_model_ready_m1_p90.csv"
SPLIT_PATH = DATA_DIR / "04_master_split.csv"
PAIR_AUDIT_PATH = DATA_DIR / "07_m0_m1_pairing_audit.csv"

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.50
XGBOOST_PARAMETERS = {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.05}
CATBOOST_PARAMETERS = {"iterations": 300, "depth": 6, "learning_rate": 0.05, "auto_class_weights": "Balanced"}
print("Project:", ROOT)
print("M0/M1 are paired on the same complete-weather row IDs; decision threshold:", DECISION_THRESHOLD)
"""),
    md("""## 1. Load data and verify identical observations, targets, and split assignments

If any one-to-one alignment check fails, stop rather than silently compare different test samples. Missing weather rows are excluded only from this paired experiment; they remain in the full-row M1 datasets.
"""),
    code("""def read_csv(path, **kwargs):
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}. Run 07_weather_data_acquisition.ipynb first.")
    return pd.read_csv(path, encoding="utf-8", low_memory=False, **kwargs)

p95_m0 = read_csv(P95_M0_PATH)
p90_m0 = read_csv(P90_M0_PATH)
p95_m1 = read_csv(P95_M1_PATH)
p90_m1 = read_csv(P90_M1_PATH)
split = read_csv(SPLIT_PATH)
pair_audit = read_csv(PAIR_AUDIT_PATH)

if split["row_id"].duplicated().any():
    raise ValueError("Master split has duplicate row_id values")
if pair_audit["row_id"].duplicated().any():
    raise ValueError("Weather pairing audit has duplicate row_id values")

for label, m0, m1 in [("P95", p95_m0, p95_m1), ("P90", p90_m0, p90_m1)]:
    if not m0["row_id"].reset_index(drop=True).equals(m1["row_id"].reset_index(drop=True)):
        raise ValueError(f"{label} M0/M1 row_id ordering differs")
    if not np.array_equal(pd.to_numeric(m0["target"], errors="coerce"), pd.to_numeric(m1["target"], errors="coerce")):
        raise ValueError(f"{label} M0/M1 target labels differ")
    if len(m0) != len(split) or set(m0["row_id"]) != set(split["row_id"]):
        raise ValueError(f"{label} inputs do not exactly cover the frozen master split")
    for column in ["outer_split", "test_cohort"]:
        if not m1[column].reset_index(drop=True).equals(m1.merge(split[["row_id", column]], on="row_id", how="left", suffixes=("", "_master"), validate="one_to_one")[f"{column}_master"].reset_index(drop=True)):
            raise ValueError(f"{label} M1 {column} differs from master split")

common_ids = set(pair_audit.loc[pair_audit["m1_weather_complete"].astype(bool), "row_id"])
if not common_ids:
    raise ValueError("No complete-weather rows available for paired evaluation")
print("Frozen rows:", f"{len(split):,}", "| Paired complete-weather rows:", f"{len(common_ids):,}")
display(pair_audit.loc[pair_audit.row_id.isin(common_ids)].groupby(["outer_split", "test_cohort"], dropna=False).size().rename("paired_rows").reset_index())
"""),
    md("""## 2. Define features and common-row evaluation helpers

The no-weather model receives the original numeric M0 features. The M1 model receives those same features plus the nine daily forecast summaries. Median imputation is learned from the development sample only. The paired population and all split labels are fixed before any fitting.
"""),
    code("""ID_AND_AUDIT_COLUMNS = {"row_id", "household_id", "measurement_date", "target", "outer_split", "seen_in_training", "test_cohort", "m1_weather_complete", "weather_row_status"}

def numeric_features(frame):
    return [c for c in frame.columns if c not in ID_AND_AUDIT_COLUMNS and pd.api.types.is_numeric_dtype(frame[c])]

P95_M0_FEATURES = numeric_features(p95_m0)
P90_M0_FEATURES = numeric_features(p90_m0)
P95_M1_FEATURES = numeric_features(p95_m1)
P90_M1_FEATURES = numeric_features(p90_m1)
if not set(P95_M0_FEATURES).issubset(P95_M1_FEATURES) or not set(P90_M0_FEATURES).issubset(P90_M1_FEATURES):
    raise ValueError("M1 feature set must contain all M0 numeric features")
print("Feature counts — P95 M0/M1:", len(P95_M0_FEATURES), len(P95_M1_FEATURES))
print("Feature counts — P90 M0/M1:", len(P90_M0_FEATURES), len(P90_M1_FEATURES))

def numeric_x(frame, features):
    return frame[features].apply(lambda c: pd.to_numeric(c, errors="coerce").astype("float64"))

def make_model(model_name, target):
    if model_name == "XGBoost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", XGBClassifier(**XGBOOST_PARAMETERS, scale_pos_weight=1.0,
                eval_metric="logloss", tree_method="hist", n_jobs=4, random_state=RANDOM_STATE, verbosity=0)),
        ])
    if model_name == "CatBoost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", CatBoostClassifier(**CATBOOST_PARAMETERS, loss_function="Logloss",
                random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)),
        ])
    raise ValueError(f"Unsupported selected model: {model_name}")

def score_rows(y_true, probability, label, target, cohort):
    y_true = np.asarray(y_true, dtype="int8")
    predicted = (probability >= DECISION_THRESHOLD).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    has_both = np.unique(y_true).size == 2
    return {
        "target": target, "dataset": label, "test_cohort": cohort,
        "rows": len(y_true), "events": int(y_true.sum()), "event_rate": float(y_true.mean()),
        "threshold": DECISION_THRESHOLD, "accuracy": accuracy_score(y_true, predicted),
        "all_negative_accuracy_reference": float((y_true == 0).mean()),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability) if has_both else np.nan,
        "pr_auc": average_precision_score(y_true, probability) if has_both else np.nan,
        "brier_score": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, probability, labels=[0, 1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

metric_rows = []
"""),
    md("""## 3. P95 primary comparison — XGBoost

Both pipelines use the same frozen M0 XGBoost settings and training rows. Confusion matrices are displayed and saved under this target's evaluation output.
"""),
    code("""p95_ids = p95_m1.loc[p95_m1["m1_weather_complete"].astype(bool), "row_id"]
p95_base = p95_m0.loc[p95_m0.row_id.isin(p95_ids)].merge(split[["row_id", "outer_split", "test_cohort"]], on="row_id", how="left", validate="one_to_one")
p95_weather = p95_m1.loc[p95_m1.row_id.isin(p95_ids)].copy()
if not p95_base["row_id"].reset_index(drop=True).equals(p95_weather["row_id"].reset_index(drop=True)):
    raise ValueError("P95 paired sample row order differs")
p95_train = p95_base["outer_split"].eq("development").to_numpy()
p95_test_seen = (p95_base["outer_split"].eq("test") & p95_base["test_cohort"].eq("test_seen")).to_numpy()
p95_test_unseen = (p95_base["outer_split"].eq("test") & p95_base["test_cohort"].eq("test_unseen")).to_numpy()
if not p95_train.any() or not p95_test_seen.any():
    raise ValueError("P95 paired data needs development and test_seen rows")

p95_models = {}
p95_probability = {}
for dataset_name, frame, features in [("M0_no_weather", p95_base, P95_M0_FEATURES), ("M1_with_weather", p95_weather, P95_M1_FEATURES)]:
    model = make_model("XGBoost", "P95")
    y_train = frame.loc[p95_train, "target"].astype("int8")
    model.set_params(model__scale_pos_weight=float((y_train.eq(0).sum() / max(y_train.eq(1).sum(), 1))))
    model.fit(numeric_x(frame.loc[p95_train], features), y_train)
    p95_models[dataset_name] = model
    joblib.dump(model, MODEL_DIR / f"p95_xgboost_{dataset_name}.joblib")
    for cohort, mask in [("test_seen", p95_test_seen), ("test_unseen", p95_test_unseen)]:
        if mask.any():
            probability = model.predict_proba(numeric_x(frame.loc[mask], features))[:, 1]
            p95_probability[(dataset_name, cohort)] = (frame.loc[mask, "target"].astype("int8").to_numpy(), probability)
            metric_rows.append(score_rows(frame.loc[mask, "target"], probability, dataset_name, "P95", cohort))
            print(dataset_name, cohort, "rows:", int(mask.sum()))

p95_metrics = pd.DataFrame([r for r in metric_rows if r["target"] == "P95"])
display(p95_metrics)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, dataset_name in zip(axes, ["M0_no_weather", "M1_with_weather"]):
    y_true, prob = p95_probability[(dataset_name, "test_seen")]
    ConfusionMatrixDisplay.from_predictions(y_true, prob >= DECISION_THRESHOLD, labels=[0, 1], values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"P95 {dataset_name} test_seen (0.50)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "p95_m0_m1_test_seen_confusion_matrices.png", dpi=160, bbox_inches="tight")
display(fig)
plt.close(fig)
"""),
    md("""## 4. P90 sensitivity comparison — CatBoost

The P90 comparison uses the same paired row IDs and split as P95, but its own target-specific labels/features and the previously selected CatBoost model. It remains a sensitivity analysis, not a replacement for P95.
"""),
    code("""p90_ids = p90_m1.loc[p90_m1["m1_weather_complete"].astype(bool), "row_id"]
p90_base = p90_m0.loc[p90_m0.row_id.isin(p90_ids)].merge(split[["row_id", "outer_split", "test_cohort"]], on="row_id", how="left", validate="one_to_one")
p90_weather = p90_m1.loc[p90_m1.row_id.isin(p90_ids)].copy()
if not p90_base["row_id"].reset_index(drop=True).equals(p90_weather["row_id"].reset_index(drop=True)):
    raise ValueError("P90 paired sample row order differs")
p90_train = p90_base["outer_split"].eq("development").to_numpy()
p90_test_seen = (p90_base["outer_split"].eq("test") & p90_base["test_cohort"].eq("test_seen")).to_numpy()
p90_test_unseen = (p90_base["outer_split"].eq("test") & p90_base["test_cohort"].eq("test_unseen")).to_numpy()
if not p90_train.any() or not p90_test_seen.any():
    raise ValueError("P90 paired data needs development and test_seen rows")

p90_models = {}
p90_probability = {}
for dataset_name, frame, features in [("M0_no_weather", p90_base, P90_M0_FEATURES), ("M1_with_weather", p90_weather, P90_M1_FEATURES)]:
    model = make_model("CatBoost", "P90")
    model.fit(numeric_x(frame.loc[p90_train], features), frame.loc[p90_train, "target"].astype("int8"))
    p90_models[dataset_name] = model
    joblib.dump(model, MODEL_DIR / f"p90_catboost_{dataset_name}.joblib")
    for cohort, mask in [("test_seen", p90_test_seen), ("test_unseen", p90_test_unseen)]:
        if mask.any():
            probability = model.predict_proba(numeric_x(frame.loc[mask], features))[:, 1]
            p90_probability[(dataset_name, cohort)] = (frame.loc[mask, "target"].astype("int8").to_numpy(), probability)
            metric_rows.append(score_rows(frame.loc[mask, "target"], probability, dataset_name, "P90", cohort))
            print(dataset_name, cohort, "rows:", int(mask.sum()))

p90_metrics = pd.DataFrame([r for r in metric_rows if r["target"] == "P90"])
display(p90_metrics)
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, dataset_name in zip(axes, ["M0_no_weather", "M1_with_weather"]):
    y_true, prob = p90_probability[(dataset_name, "test_seen")]
    ConfusionMatrixDisplay.from_predictions(y_true, prob >= DECISION_THRESHOLD, labels=[0, 1], values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"P90 {dataset_name} test_seen (0.50)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "p90_m0_m1_test_seen_confusion_matrices.png", dpi=160, bbox_inches="tight")
display(fig)
plt.close(fig)
"""),
    md("""## 5. Save paired metrics and interpret the ablation

Positive PR-AUC/ROC-AUC changes and lower Brier/log-loss support incremental predictive value, but conclusions should consider uncertainty and the paired test sample size. Accuracy alone is not sufficient under class imbalance. If weather coverage omits a substantial portion of the frozen population, state that limitation and do not claim a population-wide weather benefit.
"""),
    code("""metrics = pd.DataFrame(metric_rows)
metrics.to_csv(TABLE_DIR / "08_m0_m1_paired_metrics.csv", index=False, encoding="utf-8")
paired_summary = metrics.pivot_table(index=["target", "test_cohort"], columns="dataset", values=["rows", "events", "accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "brier_score", "log_loss"], aggfunc="first")
display(paired_summary)
paired_summary.to_csv(TABLE_DIR / "08_m0_m1_paired_metric_summary.csv", encoding="utf-8")
print("All comparisons use matched row IDs and split assignments. No test rows were used to tune parameters, thresholds, or calibration.")
"""),
    md("""## Handoff

Interpret P95 test_seen as the primary comparison, with P95 test_unseen as a household-generalization diagnostic. Report P90 separately as sensitivity. If M1 improves only one test cohort or a single metric, describe it as mixed evidence rather than a universal gain. The next stage is interpretation of the final chosen model using SHAP or model-appropriate feature importance, with weather effects distinguished from predictive association and not causality.
"""),
]

for i, cell in enumerate(cells):
    cell["id"] = f"weather-ablation-{i:02d}"

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.14.0"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("notebooks/08_weather_model_comparison.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote notebooks/08_weather_model_comparison.ipynb with", len(cells), "cells")
