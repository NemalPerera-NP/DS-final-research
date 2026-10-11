# MSc Smart Meter Research Implementation

This project contains the implementation and analysis code for an MSc research project using smart-meter data. The workflow covers data preprocessing, exploratory data analysis, feature engineering, machine-learning model training, model evaluation, and SHAP-based explainability.

## 1. Project Structure

```text
imp_smart_meeter/
├── data/
│   ├── raw/                 # Original datasets
│   ├── interim/             # Intermediate files
│   └── processed/           # Cleaned and model-ready datasets
├── notebooks/               # Jupyter notebooks executed in VS Code
├── src/
│   └── smart_meter/         # Reusable Python source code
├── models/                  # Trained model artifacts
├── reports/
│   ├── figures/             # Generated plots and visualizations
│   └── tables/              # Generated result tables
├── configs/                 # Configuration files
├── tests/                   # Tests for reusable code
├── .githooks/               # Local Git hooks
├── .venv/                   # Local Python virtual environment
├── requirements.txt         # Python dependencies
└── README.md
```

## 2. Software Requirements

- Python 3.14.4 or a compatible supported Python version
- Visual Studio Code
- VS Code Python extension
- VS Code Jupyter extension
- Git

## 3. Environment Setup

### 3.1 Create and Activate the Virtual Environment

From the project root, create the local virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run the following once for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3.2 Install Dependencies

With the virtual environment activated, install the project dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 3.3 XGBoost on macOS

XGBoost on macOS requires the OpenMP runtime. Install it with Homebrew before importing XGBoost or running the model-training notebook:

```bash
brew install libomp
```

After installation, restart the VS Code notebook kernel and rerun the imports. The Homebrew runtime is system-level and is separate from the Python virtual environment.

### 3.4 Register and Select the Jupyter Kernel

The requirements include `ipykernel`. With the environment activated, register it so VS Code can list it as a notebook kernel:

```bash
python -m ipykernel install --user --name imp-smart-meter --display-name "Python (imp-smart-meter)"
```

Open the project folder in VS Code:

```bash
code .
```

Open a notebook from `notebooks/`, click **Select Kernel** in the upper-right corner, and choose **Python (imp-smart-meter)**. If it is not listed, choose **Python Environments** and select the interpreter inside this project's `.venv` (`.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows).

If VS Code does not list the kernel, activate `.venv` in the VS Code terminal and run the kernel registration command above again.

### 3.5 (Optional) Configure Notebook Output Cleanup

To keep notebook outputs from being committed, install and configure `nbstripout`:

```powershell
python -m pip install --upgrade nbstripout
nbstripout --install
nbstripout --attributes .gitattributes
git config core.hooksPath .githooks
```

Verify the configuration:

```powershell
nbstripout --is-installed
git config --get core.hooksPath
git check-attr filter -- notebooks\01_preprocessing.ipynb
```

## 4. Data Organization

Original data should be placed in:

```text
data/raw/
```

Raw data must not be modified directly. Processed datasets should be written to:

```text
data/processed/
```

Generated charts and tables should be saved in the corresponding `reports/` directories.

## 5. Research Workflow

### 5.1 Data Preprocessing

Clean the raw data, parse timestamps, handle missing values, identify duplicates, validate ranges, and produce model-ready data.

### 5.2 Exploratory Data Analysis

Analyze distributions, time patterns, outliers, correlations, missingness, and differences between meters or other relevant groups.

### 5.3 Feature Engineering

Create time-based, lagged, rolling-window, statistical, and domain-specific features. Features must use only information available before the prediction time.

### 5.4 Model Training

Train baseline and candidate machine-learning models using reproducible pipelines. Preprocessing steps should be fitted using training data only.

### 5.5 Model Evaluation

Evaluate the final models on previously unseen data using appropriate regression or classification metrics and visualizations.

### 5.6 SHAP Analysis

Use SHAP to explain global feature importance and individual model predictions. SHAP results describe model behavior and should not be interpreted as proof of causality.

## 6. Notebook Guide

| Notebook | Purpose | Input | Output |
|---|---|---|---|
| `00_environment_and_data_audit.ipynb` | Inspect the environment and raw data | Raw data | Data-quality findings |
| `01_preprocessing.ipynb` | Clean and prepare the data | Raw data | Processed data |
| `02_eda.ipynb` | Explore patterns and distributions | Processed data | Figures and tables |
| `03_feature_engineering.ipynb` | Create model features | Processed data | Feature dataset |
| `04_model_training.ipynb` | Train candidate models | Feature dataset | Trained models |
| `05_model_evaluation.ipynb` | Compare model performance | Trained models and test data | Metrics and plots |
| `06_shap_analysis.ipynb` | Explain the selected model | Final model and test data | SHAP figures and findings |

## 7. Machine Learning Methodology

### 7.1 Data Splitting

For time-dependent smart-meter data, use chronological training, validation, and test periods. Avoid random splitting when it would allow future information to enter the training data.

### 7.2 Baseline Models

Establish simple baseline models before evaluating more complex algorithms.

### 7.3 Candidate Models

Candidate models may include linear models, random forests, gradient boosting, XGBoost, and LightGBM, depending on the research design and data type.

### 7.4 Data Leakage Prevention

- Fit imputers, scalers, encoders, and feature transformations on training data only.
- Generate lagged and rolling features using past observations only.
- Keep the final test set untouched until final evaluation.
- Use pipelines where possible.

## 8. Evaluation Metrics

The selected metrics depend on the prediction task.

For regression, possible metrics include:

- MAE
- RMSE
- R²
- MAPE, where target values are not close to zero

For classification, possible metrics include:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Balanced accuracy
- Confusion matrix

## 9. Reproducibility

### 9.1 Random Seeds

The notebooks use a fixed seed:

```python
SEED = 42
```

### 9.2 Dependency Versions

Installed package versions are recorded in:

```text
requirements.txt
```

Update it after installing packages:

```powershell
python -m pip freeze | Out-File -Encoding utf8 requirements.txt
```

### 9.3 Notebook Output Clearing

Notebook outputs are configured to be removed before commits using `nbstripout` and the project Git hook.

Setup commands:

```powershell
python -m pip install --upgrade nbstripout
nbstripout --install
nbstripout --attributes .gitattributes
git config core.hooksPath .githooks
```

Verify the configuration:

```powershell
nbstripout --is-installed
git config --get core.hooksPath
git check-attr filter -- notebooks\01_preprocessing.ipynb
```

### 9.4 Git Configuration

The `.venv/` directory, notebook checkpoints, private raw data, generated models, and generated reports are excluded through `.gitignore`.

## 10. Running the Analysis

Run the notebooks in this order, completing each notebook before starting the next so its output artifacts are available downstream:

```text
00_environment_and_data_audit.ipynb
01_preprocessing.ipynb
02_eda.ipynb
02b_holiday_calendar.ipynb
03a_target_design_experiments.ipynb
03_household_day_target.ipynb
03_feature_engineering.ipynb
04_split_design.ipynb
05_model_training_tuning.ipynb
06_calibration_evaluation.ipynb
07_weather_data_acquisition.ipynb
08_weather_model_comparison.ipynb
09_shap_analysis.ipynb
```

In particular, `03_household_day_target.ipynb` creates `data/processed/03_daily_with_target.csv`, its dtype contract, and the stage manifest required by `03_feature_engineering.ipynb`.

Before finalizing results, restart the notebook kernel and use `Run All` to confirm that each notebook works from a clean execution state.

## 11. Results

Final model metrics, plots, tables, and SHAP findings will be documented here after the analysis is completed.

## 12. License

This project is intended for MSc research and academic use. Licensing details will be added when finalized.
