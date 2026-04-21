# Run Guide: Multi-Model NDN Attack Detection

This guide shows exactly how to run training and view results:
- with the project dataset in dataset/ndn_traffic.csv
- with any custom dataset (after simple format checks)

## 1) What this project trains

The training script trains and compares 4 models:
1. XGBoost
2. Random Forest
3. Gradient Boosting
4. 1D CNN (PyTorch)

It also generates:
- evaluation metrics
- confusion matrices
- ROC curves
- per-class metrics
- saved model files

Main script:
- train_multimodel.py

## 2) One-time setup

### Step 1: Open terminal in project root

Project root is the folder containing train_multimodel.py.

### Step 2: Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
pip install xgboost seaborn scipy
```

Why the second line: train_multimodel.py imports xgboost, seaborn, and scipy.

## 3) Run with the project dataset (recommended first run)

The script reads this file automatically:
- dataset/ndn_traffic.csv

### Step 1: Run training

```bash
python train_multimodel.py
```

### Step 2: Check saved outputs

Models are saved in:
- model/

Expected files:
- model/xgboost_model.pkl
- model/randomforest_model.pkl
- model/gradientboosting_model.pkl
- model/metrics_summary.pkl

(Neural network checkpoint may be managed separately in your project state.)

Charts are saved in:
- model_analysis/

Expected charts:
- confusion_matrices.png
- roc_curves.png
- model_comparison.png
- per_class_metrics.png
- neural_network_training.png
- error_analysis.png

## 4) Run with any custom dataset

The script can run on any CSV dataset if it follows the required structure.

### 4.1 Required CSV rules

Your CSV must have:
1. A target column named Label
2. Preferably these traffic columns:
   - InInterests
   - OutInterests
   - InSatisfiedInterests
   - OutSatisfiedInterests
   - InTimedOutInterests
   - OutTimedOutInterests
   - InNacks
   - OutNacks
   - InData
   - OutData

Important:
- If some traffic columns are missing, the script auto-fills defaults.
- Label is required and should contain class names (example: Normal, IFA, Slow_IFA, Cache_Pollution, Distributed_IFA, Pulsing_IFA).
- Keep data numeric for traffic columns.
- Remove empty rows when possible.

### 4.2 Use your custom file

Option A (quickest):
1. Copy your dataset to dataset/ndn_traffic.csv
2. Run:

```bash
python train_multimodel.py
```

Option B (keep multiple files):
1. Keep your file as dataset/my_data.csv
2. Copy it to dataset/ndn_traffic.csv before each run

Windows PowerShell:

```powershell
Copy-Item dataset\my_data.csv dataset\ndn_traffic.csv -Force
python train_multimodel.py
```

Linux/macOS:

```bash
cp dataset/my_data.csv dataset/ndn_traffic.csv
python train_multimodel.py
```

### 4.3 If your label column has another name

If your target column is not Label, rename it before run.

Example with pandas (quick one-time converter):

```python
import pandas as pd

df = pd.read_csv('dataset/my_data.csv')
df = df.rename(columns={'target': 'Label'})  # change target to your real column name
df.to_csv('dataset/ndn_traffic.csv', index=False)
```

Then run:

```bash
python train_multimodel.py
```

## 5) How to read results clearly

After training completes, use these files:

1. model/metrics_summary.pkl
- Contains Accuracy, Precision, Recall, F1, Error Rate, Kappa, MCC, ROC-AUC

2. model_analysis/model_comparison.png
- Quick model-vs-model score view

3. model_analysis/confusion_matrices.png
- Class-level correct and incorrect predictions

4. model_analysis/roc_curves.png
- ROC behavior per class and model

5. Project_Report.md
- Full written analysis with tables and figure references

## 6) Quick rerun workflow

Use this each time you want a fresh run:

```bash
python train_multimodel.py
```

Then check:
- model/metrics_summary.pkl
- model_analysis/*.png

## 7) Common issues and fixes

### Issue: ModuleNotFoundError for xgboost or seaborn

Fix:

```bash
pip install xgboost seaborn scipy
```

### Issue: KeyError: 'Label'

Cause: CSV has no Label column.

Fix: Rename your target column to Label and rerun.

### Issue: Very poor metrics on custom data

Checks:
1. Confirm Label values are correct classes.
2. Confirm traffic columns are numeric and not mostly zeros.
3. Confirm row order is time order (important for sliding windows).

## 8) Minimum dataset checklist (before run)

- CSV file exists
- Column Label exists
- At least 200+ rows (more is better)
- Numeric traffic columns
- Time order preserved

When this checklist is satisfied, the same training command works for both project data and custom data.
