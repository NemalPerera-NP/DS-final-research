import json
from pathlib import Path


OUTPUT = Path("notebooks/04_model_training_evaluation.ipynb")


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


cells = [
    markdown("""# Model training and testing (M0, no weather)

This notebook uses one date-ordered 80/20 train/test split and evaluates four models in separate stages: Logistic Regression, Random Forest, XGBoost, and CatBoost. P95 is the primary target; P90 is evaluated separately as a sensitivity target. Each model stage fits, predicts the test rows, prints its metrics, and displays its confusion matrix.

The model probability cutoff is fixed at **0.50** throughout this notebook. There is no validation stage, probability calibration, or data-driven threshold selection here. Change the hyperparameters below each model heading, then rerun the notebook. For research reporting, avoid choosing parameters by repeatedly optimizing the same test results; doing that turns the test period into tuning data.
"""),
    markdown("## Stage 0 — Imports, paths, and shared settings"),
    code("""from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)

ROOT = Path.cwd().resolve()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent
DATA_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "reports" / "tables" / "10_model_training_evaluation_simple_80_20"
FIGURE_DIR = ROOT / "reports" / "figures" / "10_model_training_evaluation_simple_80_20"
MODEL_DIR = ROOT / "models" / "10_m0_simple_80_20"
for folder in [TABLE_DIR, FIGURE_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_DATE_FRACTION = 0.80
DECISION_THRESHOLD = 0.50
CSV_ENCODING = "utf-8"
CSV_SEPARATOR = ","

COMMON_FEATURES = [
    "lag1_peak_kw", "lag1_mean_kw", "lag1_energy_kwh", "lag1_coverage_ratio",
    "lag1_was_complete", "peak_mean_7d", "peak_mean_30d", "peak_max_7d",
    "peak_max_30d", "peak_std_7d", "peak_std_30d", "energy_mean_7d",
    "energy_mean_30d", "lag1_peak_over_p95", "lag1_peak_minus_p95",
    "peak_mean_7d_over_30d", "lag1_peak_zscore_30d", "day_of_week_sin",
    "day_of_week_cos", "is_weekend", "month_sin", "month_cos",
]
TARGETS = {
    "P95": {
        "csv": DATA_DIR / "05_model_ready_m0_p95.csv",
        "contract": DATA_DIR / "05_model_ready_m0_p95_dtype_contract.json",
        "features": COMMON_FEATURES + ["p95_high_count_7d", "p95_high_count_30d", "p95_days_since_last_high"],
        "color": "#b2182b",
    },
    "P90": {
        "csv": DATA_DIR / "05_model_ready_m0_p90.csv",
        "contract": DATA_DIR / "05_model_ready_m0_p90_dtype_contract.json",
        "features": COMMON_FEATURES + ["p90_high_count_7d", "p90_high_count_30d", "p90_days_since_last_high"],
        "color": "#2166ac",
    },
}

RESULTS = []
PREDICTIONS = []

print("Project:", ROOT)
print("Decision threshold for predicted class:", DECISION_THRESHOLD)
"""),
    markdown("""## Stage 1 — Load the model datasets and make the 80/20 split

The split is by unique date, with earlier dates for training and later dates for testing. All households from the same date stay together. This is appropriate for forecasting future household-days. It does not guarantee an 80/20 split by row count because household coverage changes over time.
"""),
    code("""def load_csv_with_contract(csv_path, contract_path, columns):
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    header = pd.read_csv(csv_path, nrows=0, encoding=CSV_ENCODING, sep=CSV_SEPARATOR).columns.tolist()
    if header != list(contract["columns"]):
        raise ValueError(f"CSV header and dtype contract do not match: {csv_path.name}")
    dates = [name for name in contract.get("date_columns", []) if name in columns]
    dtypes = {name: contract["columns"][name] for name in columns if name not in dates}
    return pd.read_csv(
        csv_path,
        usecols=columns,
        dtype=dtypes,
        parse_dates=dates,
        true_values=contract.get("reader", {}).get("true_values", ["True"]),
        false_values=contract.get("reader", {}).get("false_values", ["False"]),
        keep_default_na=contract.get("reader", {}).get("keep_default_na", False),
        na_values=contract.get("reader", {}).get("na_values", [""]),
        encoding=contract.get("encoding", CSV_ENCODING),
        sep=contract.get("separator", CSV_SEPARATOR),
        low_memory=False,
    )

def numeric_features(frame, columns):
    # sklearn estimators receive float columns; missing values remain NaN for the model pipeline.
    return frame[columns].apply(lambda column: pd.to_numeric(column, errors="coerce").astype("float64"))

DATASETS = {}
for target_name, settings in TARGETS.items():
    columns = list(dict.fromkeys([
        "row_id", "household_id", "measurement_date", "target", *settings["features"]
    ]))
    DATASETS[target_name] = load_csv_with_contract(settings["csv"], settings["contract"], columns)

date_values = pd.DatetimeIndex(sorted(DATASETS["P95"]["measurement_date"].dropna().unique()))
split_index = int(np.floor(len(date_values) * TRAIN_DATE_FRACTION))
train_dates = date_values[:split_index]
test_dates = date_values[split_index:]

SPLITS = {}
split_rows = []
for target_name, frame in DATASETS.items():
    train = frame.loc[frame.measurement_date.isin(train_dates)].sort_values(["measurement_date", "household_id"]).reset_index(drop=True)
    test = frame.loc[frame.measurement_date.isin(test_dates)].sort_values(["measurement_date", "household_id"]).reset_index(drop=True)
    if train.target.nunique() != 2 or test.target.nunique() != 2:
        raise ValueError(f"Both target classes must exist in train and test for {target_name}")
    SPLITS[target_name] = {"train": train, "test": test}
    split_rows.extend([
        {"target": target_name, "split": "train", "start_date": train.measurement_date.min(), "end_date": train.measurement_date.max(), "dates": train.measurement_date.nunique(), "rows": len(train), "events": int(train.target.sum()), "event_rate": train.target.mean()},
        {"target": target_name, "split": "test", "start_date": test.measurement_date.min(), "end_date": test.measurement_date.max(), "dates": test.measurement_date.nunique(), "rows": len(test), "events": int(test.target.sum()), "event_rate": test.target.mean()},
    ])

assert DATASETS["P95"].row_id.astype(str).reset_index(drop=True).equals(DATASETS["P90"].row_id.astype(str).reset_index(drop=True))
assert SPLITS["P95"]["train"].row_id.astype(str).reset_index(drop=True).equals(SPLITS["P90"]["train"].row_id.astype(str).reset_index(drop=True))
assert SPLITS["P95"]["test"].row_id.astype(str).reset_index(drop=True).equals(SPLITS["P90"]["test"].row_id.astype(str).reset_index(drop=True))
SPLIT_SUMMARY = pd.DataFrame(split_rows)
display(SPLIT_SUMMARY)
"""),
    markdown("""## Shared evaluation display

Every model stage below calls this small helper after fitting. It reports accuracy and the other study metrics, then displays a confusion matrix using the fixed 0.50 probability cutoff.
"""),
    code("""def show_test_results(model_name, target_name, model, test_frame, features):
    x_test = numeric_features(test_frame, features)
    y_test = test_frame["target"].astype("int8").to_numpy()
    probability = model.predict_proba(x_test)[:, 1]
    predicted = (probability >= DECISION_THRESHOLD).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()

    result = {
        "target": target_name,
        "model": model_name,
        "test_rows": len(y_test),
        "events": int(y_test.sum()),
        "threshold": DECISION_THRESHOLD,
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y_test, predicted, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "pr_auc": average_precision_score(y_test, probability),
        "roc_auc": roc_auc_score(y_test, probability),
        "brier": brier_score_loss(y_test, probability),
        "log_loss": log_loss(y_test, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    RESULTS.append(result)

    prediction = test_frame[["row_id", "household_id", "measurement_date", "target"]].copy()
    prediction["target_name"] = target_name
    prediction["model"] = model_name
    prediction["event_probability"] = probability
    prediction["decision_threshold"] = DECISION_THRESHOLD
    prediction["predicted_target"] = predicted
    PREDICTIONS.append(prediction)

    print(f"{target_name} — {model_name} test results")
    display(pd.DataFrame([result]).drop(columns=["tn", "fp", "fn", "tp"]))
    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual normal", "Actual event"],
        columns=["Predicted normal", "Predicted event"],
    )
    display(matrix)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(matrix, annot=True, fmt=",", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{target_name} — {model_name} (cutoff = {DECISION_THRESHOLD:.2f})")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    plt.tight_layout()
    figure_path = FIGURE_DIR / f"{target_name.lower()}_{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.show()

    return result
"""),
    markdown("## Model 1 — Logistic Regression\n\nKey parameters: `C` controls regularization strength; `class_weight` compensates for the relatively rare event class. Median imputation and scaling are fitted on training rows inside each pipeline."),
    code("""from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LOGISTIC_REGRESSION_PARAMETERS = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 2000,
}

logistic_models = {}
for target_name, settings in TARGETS.items():
    train = SPLITS[target_name]["train"]
    test = SPLITS[target_name]["test"]
    features = settings["features"]
    x_train = numeric_features(train, features)
    y_train = train["target"].astype("int8").to_numpy()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(**LOGISTIC_REGRESSION_PARAMETERS, solver="liblinear", random_state=RANDOM_STATE)),
    ])
    print(f"Fitting Logistic Regression for {target_name}...")
    model.fit(x_train, y_train)
    logistic_models[target_name] = model
    joblib.dump(model, MODEL_DIR / f"{target_name.lower()}_logistic_regression.joblib")
    show_test_results("Logistic Regression", target_name, model, test, features)
"""),
    markdown("## Model 2 — Random Forest\n\nKey parameters: `n_estimators` is the number of trees; `max_depth` limits tree depth; `min_samples_leaf` controls the minimum size of a leaf. Missing values are imputed inside the training pipeline."),
    code("""from sklearn.ensemble import RandomForestClassifier

RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
}

random_forest_models = {}
for target_name, settings in TARGETS.items():
    train = SPLITS[target_name]["train"]
    test = SPLITS[target_name]["test"]
    features = settings["features"]
    x_train = numeric_features(train, features)
    y_train = train["target"].astype("int8").to_numpy()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", RandomForestClassifier(**RANDOM_FOREST_PARAMETERS, n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    print(f"Fitting Random Forest for {target_name}...")
    model.fit(x_train, y_train)
    random_forest_models[target_name] = model
    joblib.dump(model, MODEL_DIR / f"{target_name.lower()}_random_forest.joblib")
    show_test_results("Random Forest", target_name, model, test, features)
"""),
    markdown("## Model 3 — XGBoost\n\nKey parameters: `n_estimators`, `max_depth`, and `learning_rate` control boosting length, tree complexity, and update size. `scale_pos_weight` is calculated from training labels only to account for class imbalance."),
    code("""from xgboost import XGBClassifier

XGBOOST_PARAMETERS = {
    "n_estimators": 250,
    "max_depth": 3,
    "learning_rate": 0.05,
}

xgboost_models = {}
for target_name, settings in TARGETS.items():
    train = SPLITS[target_name]["train"]
    test = SPLITS[target_name]["test"]
    features = settings["features"]
    x_train = numeric_features(train, features)
    y_train = train["target"].astype("int8").to_numpy()
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", XGBClassifier(
            **XGBOOST_PARAMETERS,
            scale_pos_weight=negative_count / positive_count,
            eval_metric="logloss", tree_method="hist", n_jobs=4,
            random_state=RANDOM_STATE, verbosity=0,
        )),
    ])
    print(f"Fitting XGBoost for {target_name}...")
    model.fit(x_train, y_train)
    xgboost_models[target_name] = model
    joblib.dump(model, MODEL_DIR / f"{target_name.lower()}_xgboost.joblib")
    show_test_results("XGBoost", target_name, model, test, features)
"""),
    markdown("## Model 4 — CatBoost\n\nKey parameters: `iterations` is the number of boosting rounds; `depth` controls tree complexity; `learning_rate` controls each update. CatBoost can handle numeric missing values, so this stage passes the raw numeric features to the model."),
    code("""from catboost import CatBoostClassifier

CATBOOST_PARAMETERS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "auto_class_weights": "Balanced",
}

catboost_models = {}
for target_name, settings in TARGETS.items():
    train = SPLITS[target_name]["train"]
    test = SPLITS[target_name]["test"]
    features = settings["features"]
    x_train = numeric_features(train, features)
    y_train = train["target"].astype("int8").to_numpy()
    model = CatBoostClassifier(**CATBOOST_PARAMETERS, loss_function="Logloss", random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)
    print(f"Fitting CatBoost for {target_name}...")
    model.fit(x_train, y_train)
    catboost_models[target_name] = model
    joblib.dump(model, MODEL_DIR / f"{target_name.lower()}_catboost.joblib")
    show_test_results("CatBoost", target_name, model, test, features)
"""),
    markdown("## Save evaluation tables and predictions"),
    code("""metrics = pd.DataFrame(RESULTS)
predictions = pd.concat(PREDICTIONS, ignore_index=True)

metrics_path = TABLE_DIR / "model_test_metrics.csv"
predictions_path = DATA_DIR / "06_test_predictions_m0_simple_80_20.csv"

metrics_tmp = metrics_path.with_suffix(".tmp.csv")
metrics.to_csv(metrics_tmp, index=False, encoding=CSV_ENCODING, sep=CSV_SEPARATOR, na_rep="", float_format="%.17g")
metrics_tmp.replace(metrics_path)

predictions_tmp = predictions_path.with_suffix(".tmp.csv")
predictions.to_csv(predictions_tmp, index=False, encoding=CSV_ENCODING, sep=CSV_SEPARATOR, na_rep="", date_format="%Y-%m-%d %H:%M:%S", float_format="%.17g")
predictions_tmp.replace(predictions_path)

print("Metrics saved to:", metrics_path)
display(metrics)
print("Predictions saved to:", predictions_path)
print("Confusion matrix figures saved to:", FIGURE_DIR)
"""),
    markdown("""## How to read the thresholds in this study

There are different kinds of thresholds in the research:

1. **P95/P90 target cutoffs** were created upstream from each household’s past demand history. The label is 1 when that day’s peak exceeds its historical P95 or P90 cutoff. These define what the model learns to predict; they do not convert model probabilities into classes.
2. **Model probability cutoff = 0.50** is the single decision cutoff in this notebook. A predicted event probability of at least 0.50 becomes class 1; a lower probability becomes class 0. The confusion matrices and accuracy use these predicted classes.
3. **Model parameters** such as Logistic Regression `C`, tree `max_depth`, or CatBoost `iterations` control model fitting. They are hyperparameters, not event or probability thresholds.
4. **Preprocessing rules** such as 96 required intervals, 70 near-complete intervals, and 45 historical days govern data eligibility or aggregation. They are separate from both target cutoffs and the 0.50 model decision cutoff.

This notebook does not calibrate probabilities or search for a threshold. Changing `DECISION_THRESHOLD` changes predicted classes, accuracy, and the confusion matrix while leaving the fitted model probabilities unchanged. Accuracy should be considered with recall, precision, F1, balanced accuracy, and PR-AUC because events are relatively uncommon.
"""),
]

for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "imp-smart-meter", "language": "python", "name": "imp-smart-meter"},
        "language_info": {"name": "python", "version": "3.14.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote simplified staged notebook: {OUTPUT}")
