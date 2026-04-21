<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Outfit:wght@300;400;500;600&display=swap');

:root {
   --bg: #fbfbf9;
   --paper: #ffffff;
   --text: #1f2937;
   --muted: #4b5563;
   --line: #d1d5db;
   --head: #111827;
}

body {
   font-family: 'Outfit', 'Segoe UI', sans-serif;
   color: var(--text);
   background: var(--bg);
   max-width: 980px;
   margin: 0 auto;
   padding: 28px 30px 40px;
   line-height: 1.62;
}

h1, h2, h3, h4 {
   font-family: 'Fraunces', Georgia, serif;
   color: var(--head);
   line-height: 1.25;
}

h1 {
   text-align: center;
   margin-bottom: 10px;
}

h2 {
   margin-top: 28px;
   border-bottom: 1px solid var(--line);
   padding-bottom: 7px;
}

p, li {
   text-align: left;
}

table {
   width: 100%;
   border-collapse: collapse;
   margin: 12px 0 18px;
}

th, td {
   border: 1px solid var(--line);
   padding: 8px 10px;
   text-align: left;
   vertical-align: top;
}

th {
   background: #f3f4f6;
}

code, pre {
   font-family: 'Consolas', 'Courier New', monospace;
}

pre {
   background: #f8fafc;
   border: 1px solid var(--line);
   border-radius: 8px;
   padding: 12px;
}
</style>

# Google Colab Notebook Guide

This file explains exactly how to run this project in Google Colab using:
- [NDN_Colab_Run.ipynb](NDN_Colab_Run.ipynb)

It is written for simple, direct use.

## What you need before starting

1. A Google account
2. Google Colab access (free)
3. Your project folder named NDN, including these files:
   - [train_multimodel.py](train_multimodel.py)
   - [requirements.txt](requirements.txt)
   - [dataset/ndn_traffic.csv](dataset/ndn_traffic.csv)
   - [NDN_Colab_Run.ipynb](NDN_Colab_Run.ipynb)

## Two ways to open the project in Colab

## Option A: Use Google Drive (recommended)

1. Open Google Drive.
2. Upload your full NDN folder to MyDrive.
3. In Google Drive, open [NDN_Colab_Run.ipynb](NDN_Colab_Run.ipynb) with Google Colab.

Why this is better:
- Files stay saved in your Drive.
- You can rerun later without uploading again.

## Option B: Upload directly to Colab runtime

1. Open Google Colab.
2. Upload [NDN_Colab_Run.ipynb](NDN_Colab_Run.ipynb).
3. Upload your NDN folder into /content.

This works, but files are temporary and can disappear after session reset.

## Step-by-step run process

## Step 1: Open the notebook

Open [NDN_Colab_Run.ipynb](NDN_Colab_Run.ipynb) in Colab.

## Step 2: Run cells in order

Run all cells from top to bottom.

Important:
- Do not skip cells.
- Wait for each cell to finish before running the next one.

## Step 3: Set project path correctly

In Cell 3 (path setup cell), set PROJECT_ROOT to your real folder path.

If you use Google Drive:
- /content/drive/MyDrive/NDN

If you uploaded directly:
- /content/NDN

If this path is wrong, training will not start.

## Step 4: Install packages

The notebook installs packages automatically using:
- [requirements.txt](requirements.txt)

It also installs missing extra packages required by the script.

## Step 5: Validate required files

The notebook checks these files:
- [train_multimodel.py](train_multimodel.py)
- [dataset/ndn_traffic.csv](dataset/ndn_traffic.csv)

If any file is missing, it stops and shows which file is missing.

## Step 6: Start model training

The notebook runs:
- [train_multimodel.py](train_multimodel.py)

This will train the same models used in your project:
1. XGBoost
2. Random Forest
3. Gradient Boosting
4. 1D CNN

## Step 7: View final results

After training, the notebook shows:
1. Model score table from [model/metrics_summary.pkl](model/metrics_summary.pkl)
2. Result images from [model_analysis](model_analysis)

Expected charts:
- confusion_matrices.png
- roc_curves.png
- model_comparison.png
- per_class_metrics.png
- neural_network_training.png
- error_analysis.png

## Where outputs are saved

After a successful run, outputs are written to:

1. Models and metrics:
   - [model](model)

2. Plots and charts:
   - [model_analysis](model_analysis)

If you run from Google Drive, these outputs stay in Drive.

## Run with your own dataset

You can use a different dataset without changing project code.

Do this:
1. Replace [dataset/ndn_traffic.csv](dataset/ndn_traffic.csv) with your file.
2. Keep the same file name: ndn_traffic.csv
3. Run the notebook again from the start.

Your file must include:
- A target column named Label

Preferred traffic columns:
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

## Common problems and simple fixes

## Problem: File not found

Reason:
- PROJECT_ROOT path is wrong, or folder upload is incomplete.

Fix:
1. Check PROJECT_ROOT text in Cell 3.
2. Confirm files exist in that folder.
3. Rerun from Cell 3 onward.

## Problem: Package install error

Reason:
- Network issue or temporary Colab issue.

Fix:
1. Run the install cell again.
2. If it still fails, restart runtime and rerun all.

## Problem: Training stops with missing Label column

Reason:
- Your custom CSV does not have Label.

Fix:
1. Rename your target column to Label.
2. Save file as dataset/ndn_traffic.csv.
3. Run again.

## Problem: Results look weak on custom dataset

Checks:
1. Label values are correct.
2. Traffic columns are numeric.
3. Data rows are in time order.

## Best practice for reliable runs

1. Use Google Drive mode.
2. Keep one clean folder named NDN.
3. Run cells in order every time.
4. Do not rename output folders:
   - [model](model)
   - [model_analysis](model_analysis)

## Quick checklist before clicking Run All

1. [NDN_Colab_Run.ipynb](NDN_Colab_Run.ipynb) opened in Colab
2. PROJECT_ROOT is correct
3. [dataset/ndn_traffic.csv](dataset/ndn_traffic.csv) present
4. [train_multimodel.py](train_multimodel.py) present
5. Internet connected for package installation

When all five checks are true, you can run all cells.
