import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    md("""# Model interpretation with SHAP

This notebook interprets the paired models trained in `08_weather_model_comparison.ipynb`: P95 XGBoost (primary) and P90 CatBoost (sensitivity), each with M0 and M1 inputs. It uses the same deterministically sampled, complete-weather `test_seen` row IDs across all four model/dataset combinations. It does not alter models, choose features, or tune thresholds.

Tree SHAP values are in each tree model's raw output space (for these binary classifiers, generally a log-odds contribution). They describe how a fitted model's features move predictions relative to a baseline; they are not causal effects or evidence that weather changes household demand. Approximate depot-area coordinates and gridded forecast variables also limit geographic interpretation.
"""),
    code("""from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from IPython.display import display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "08_weather_model_comparison"
TABLE_DIR = ROOT / "reports" / "tables" / "09_shap_analysis"
FIGURE_DIR = ROOT / "reports" / "figures" / "09_shap_analysis"
for folder in [TABLE_DIR, FIGURE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

P95_PATH = DATA_DIR / "07_model_ready_m1_p95.csv"
P90_PATH = DATA_DIR / "07_model_ready_m1_p90.csv"
PAIR_AUDIT_PATH = DATA_DIR / "07_m0_m1_pairing_audit.csv"
RANDOM_STATE = 42
MAX_EXPLANATION_ROWS = 500
TOP_FEATURES = 20
print("SHAP version:", shap.__version__)
print("Deterministic maximum explanation rows:", MAX_EXPLANATION_ROWS)
"""),
    md("""## 1. Load selected models and define the common held-out sample

The sample is selected once from complete-weather `test_seen` rows and reused unchanged across the four explanations. The split and target rows are not used for model fitting in this notebook.
"""),
    code("""def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}. Run notebooks 07 and 08 first.")
    return pd.read_csv(path, encoding="utf-8", low_memory=False)

p95 = read_csv(P95_PATH)
p90 = read_csv(P90_PATH)
pair_audit = read_csv(PAIR_AUDIT_PATH)
if pair_audit["row_id"].duplicated().any():
    raise ValueError("Pairing audit has duplicate row_id")

eligible_ids = set(pair_audit.loc[pair_audit["m1_weather_complete"].astype(bool) & pair_audit["outer_split"].eq("test") & pair_audit["test_cohort"].eq("test_seen"), "row_id"])
if not eligible_ids:
    raise ValueError("No paired complete-weather test_seen rows are available")
common_ids = set(p95.loc[p95.row_id.isin(eligible_ids), "row_id"]) & set(p90.loc[p90.row_id.isin(eligible_ids), "row_id"])
if not common_ids:
    raise ValueError("P95/P90 do not share eligible test_seen row IDs")
sample_ids = sorted(common_ids)
if len(sample_ids) > MAX_EXPLANATION_ROWS:
    sample_ids = pd.Series(sample_ids).sample(n=MAX_EXPLANATION_ROWS, random_state=RANDOM_STATE).sort_values().tolist()
sample_frame = pd.DataFrame({"row_id": sample_ids})
sample_frame.to_csv(TABLE_DIR / "09_shap_sample_row_ids.csv", index=False, encoding="utf-8")
print("Common complete-weather test_seen rows:", len(common_ids), "| rows explained:", len(sample_ids))
display(pd.DataFrame([{"eligible_test_seen_rows": len(common_ids), "explained_rows": len(sample_ids), "random_state": RANDOM_STATE}]))

EXCLUDED = {"row_id", "household_id", "measurement_date", "target", "outer_split", "seen_in_training", "test_cohort", "m1_weather_complete", "weather_row_status"}
WEATHER_INPUT_FEATURES = {
    "temperature_2m_mean_c", "temperature_2m_min_c", "temperature_2m_max_c",
    "relative_humidity_2m_mean_pct", "wind_speed_10m_mean_kmh", "wind_speed_10m_max_kmh",
    "rain_sum_mm", "shortwave_radiation_mean_wm2", "shortwave_radiation_sum_kwh_m2",
}
def features_for(frame):
    return [c for c in frame.columns if c not in EXCLUDED and pd.api.types.is_numeric_dtype(frame[c])]

def numeric_x(frame, features):
    return frame[features].apply(lambda c: pd.to_numeric(c, errors="coerce").astype("float64"))

models = {
    "P95_M0_XGBoost": (p95, "p95_xgboost_M0_no_weather.joblib"),
    "P95_M1_XGBoost": (p95, "p95_xgboost_M1_with_weather.joblib"),
    "P90_M0_CatBoost": (p90, "p90_catboost_M0_no_weather.joblib"),
    "P90_M1_CatBoost": (p90, "p90_catboost_M1_with_weather.joblib"),
}
"""),
    md("""## 2. Compute SHAP values for the four paired models

Each model explains the same household-day IDs. M0 and M1 use their corresponding input feature sets; the fitted imputer and classifier are reused exactly as saved by the comparison notebook. Global importance is mean absolute SHAP value, not a causal effect size.
"""),
    code("""shap_long = []
summary_plot_paths = []
for label, (source_frame, model_file) in models.items():
    model_path = MODEL_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Expected fitted model not found: {model_path}")
    model = joblib.load(model_path)
    is_m1 = "_M1_" in label
    frame = source_frame.loc[source_frame.row_id.isin(sample_ids)].copy().set_index("row_id").loc[sample_ids].reset_index()
    features = features_for(frame)
    if not is_m1:
        features = [feature for feature in features if feature not in WEATHER_INPUT_FEATURES]
    x_raw = numeric_x(frame, features)
    imputer = model.named_steps["imputer"]
    tree_model = model.named_steps["model"]
    x_transformed = imputer.transform(x_raw)
    feature_names = imputer.get_feature_names_out(features)
    explainer = shap.TreeExplainer(tree_model)
    values = explainer.shap_values(x_transformed)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
    if values.shape != x_transformed.shape:
        raise ValueError(f"SHAP shape {values.shape} does not match transformed features {x_transformed.shape} for {label}")

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    shap_long.extend([
        {"model_dataset": label, "feature": str(feature), "mean_abs_shap_raw_output": float(abs_value), "mean_signed_shap_raw_output": float(signed_value), "explained_rows": len(sample_ids)}
        for feature, abs_value, signed_value in zip(feature_names, mean_abs, mean_signed)
    ])

    safe_label = label.lower().replace("_", "-")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(values, x_transformed, feature_names=feature_names, max_display=TOP_FEATURES, show=False)
    plt.title(f"{label}: SHAP summary (raw model output)")
    plt.tight_layout()
    beeswarm_path = FIGURE_DIR / f"{safe_label}_shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    display(plt.gcf())
    plt.close()
    summary_plot_paths.append(beeswarm_path.name)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(values, x_transformed, feature_names=feature_names, plot_type="bar", max_display=TOP_FEATURES, show=False)
    plt.title(f"{label}: mean absolute SHAP")
    plt.tight_layout()
    bar_path = FIGURE_DIR / f"{safe_label}_shap_importance.png"
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    display(plt.gcf())
    plt.close()
    summary_plot_paths.append(bar_path.name)

importance = pd.DataFrame(shap_long).sort_values(["model_dataset", "mean_abs_shap_raw_output"], ascending=[True, False])
importance["rank_within_model"] = importance.groupby("model_dataset")["mean_abs_shap_raw_output"].rank(method="first", ascending=False).astype(int)
importance.to_csv(TABLE_DIR / "09_mean_absolute_shap_importance.csv", index=False, encoding="utf-8")
display(importance.groupby("model_dataset", as_index=False).head(10))
"""),
    md("""## 3. Compare weather feature attribution and summarize

Compare M0 and M1 rankings within each target. Weather predictors appearing in M1's top ranks indicate the fitted model uses them for prediction on this sample; they do not by themselves establish out-of-sample improvement or causal influence. Refer to the paired performance results in Notebook 08 for whether M1 improved.
"""),
    code("""weather_features = [c for c in importance["feature"].unique() if c.startswith(("temperature_", "relative_humidity_", "wind_speed_", "rain_", "shortwave_radiation_"))]
weather_summary = importance.loc[importance.feature.isin(weather_features)].sort_values(["model_dataset", "mean_abs_shap_raw_output"], ascending=[True, False])
weather_summary.to_csv(TABLE_DIR / "09_weather_feature_shap_attribution.csv", index=False, encoding="utf-8")
print("Weather feature attribution rows:", len(weather_summary))
display(weather_summary)
print("Saved plots:", summary_plot_paths)
print("Interpret SHAP magnitudes in raw model-output units; no feature is selected or removed here.")
"""),
    md("""## Handoff

Use the paired model metrics from Notebook 08 for claims about weather value, and use these SHAP outputs to describe how the fitted models use predictors. Report the sample size and test cohort, distinguish M0 versus M1 and P95 versus P90, and avoid causal wording.
"""),
]

for i, cell in enumerate(cells):
    cell["id"] = f"shap-analysis-{i:02d}"

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.14.0"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("notebooks/09_shap_analysis.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote notebooks/09_shap_analysis.ipynb with", len(cells), "cells")
