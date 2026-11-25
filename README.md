# Predicting Missed 6-Minute Attendance Targets: London Fire Brigade

**6COSC017C-n Machine Learning and Data Analytics: Coursework 1 (50%)**
Westminster International University in Tashkent, 2026–2027

| | |
|---|---|
| **Student ID** | 00018680 |
| **Module leader** | Ruslan Isaev |
| **Submission deadline** | 4 December 2026 |
| **Streamlit app** | _TBD (deployed link)_ |

## Overview

The London Fire Brigade (LFB) aims to get the first fire engine to an incident within **6 minutes** of the 999 call. This project predicts, **using only information available when the call is received**, whether the first engine will arrive **late (more than 360 seconds)**.

The model and its explanations help planners see where and when the target is at risk, and so inform decisions on station cover, moving engines to fill gaps, and staffing.

- **Problem type:** supervised binary classification, `late = FirstPumpArriving_AttendanceTime > 360`
- **Deliverables:** report, Jupyter Notebook, multi-page Streamlit app, Git repository

## Dataset

| | |
|---|---|
| **Source** | London Datastore (Greater London Authority) |
| **Tables** | [LFB Incident Records](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records) and [LFB Mobilisation Records](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records) |
| **Join** | `IncidentNumber` (one incident to many mobilisations) |
| **Period used** | January 2024 to October 2026 |
| **Size** | about 398,000 incidents and 595,000 mobilisations |
| **Licence** | [Open Government Licence v2](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/) |

> Raw files (~1 GB) are **not committed** because they are too large for GitHub. A compact cleaned dataset for the app is stored in `data/processed/`.

## Methodology

1. **Load and join:** incident records plus mobilisation records (two files concatenated), aggregated and joined on `IncidentNumber`.
2. **EDA:** summary statistics, missing values, target distribution, late rate by time, borough and property type, a map, and correlation matrices.
3. **Cleaning:**
   - remove unlabelled and implausible attendance times
   - **correct a time-zone mismatch** (incident times are UK local time, mobilisation times are GMT)
   - fill missing categories
4. **Feature engineering, using call-time information only:**
   - time-of-day features, including sine/cosine encoding of the hour
   - home-station workload at the time of the call
   - recent demand (calls in the previous 60 minutes)
   - distance to estimated station locations
5. **Leakage control:** exclude post-incident fields such as the number of pumps, the incident outcome and the number of calls.
6. **Time-based split:** train on 2024–2025, test on 2026; tuning uses `TimeSeriesSplit`.
7. **Models:** Logistic Regression, Random Forest and LightGBM, tuned with `RandomizedSearchCV`, compared against a majority-class baseline.
8. **Evaluation:** ROC-AUC, PR-AUC, recall (TPR), TNR, precision and F1, with confusion matrices, calibration and permutation importance.

## Results

Held-out test set: 117,793 incidents from January to October 2026. Models were trained on 2024–2025.

| Model | ROC-AUC | PR-AUC | Recall | TNR | Precision | F1 |
|-------|---------|--------|--------|-----|-----------|----|
| Baseline (majority) | 0.500 | 0.363 | 0.000 | 1.000 | 0.000 | 0.000 |
| Logistic Regression | 0.771 | 0.656 | 0.731 | 0.675 | 0.561 | 0.635 |
| Random Forest | 0.784 | 0.690 | 0.705 | 0.723 | 0.592 | 0.644 |
| **LightGBM (final)** | **0.791** | **0.697** | 0.721 | 0.715 | 0.590 | **0.649** |

The strongest predictors are **home-station workload at the time of the call** and **distance to the nearest and home stations**, followed by time of day.

## Project structure

```
ml-00018680/
├── app/                      # Multi-page Streamlit app
├── data/
│   ├── raw/                  # Original LFB files (git-ignored, download separately)
│   └── processed/            # Cleaned dataset used by the app
├── models/                   # Final model and evaluation results
├── notebooks/
│   └── MLDA_CW1_00018680.ipynb
├── reports/figures/          # Figures exported by the notebook
├── scripts/build_notebook.py # Source that generates the notebook
├── pyproject.toml / uv.lock  # Dependencies with exact locked versions
├── requirements.txt          # Exported from uv.lock
└── README.md
```

## Getting started

### 1. Install

Option A, with [uv](https://docs.astral.sh/uv/), the recommended way (uses the exact versions in `uv.lock`):

```bash
git clone https://github.com/Sohibjon-0703/ml-00018680.git
cd ml-00018680
uv sync
```

Option B, with pip:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt jupyter
```

### 2. Get the raw data (needed only for the notebook)

Download the following from the London Datastore links above into `data/raw/`:

```
data/raw/incident_records/LFB Incident data from 2024 onwards.xlsx
data/raw/mobilisation_records/LFB Mobilisation data from 2021 - 2024.csv
data/raw/mobilisation_records/LFB Mobilisation data from 2025.csv
```

### 3. Run the notebook

```bash
uv run jupyter notebook notebooks/MLDA_CW1_00018680.ipynb
```

### 4. Run the Streamlit app

```bash
uv run streamlit run app/Home.py
```

## References

- Greater London Authority (2026) *London Fire Brigade Incident Records*. London Datastore. Available at: https://data.london.gov.uk/dataset/london-fire-brigade-incident-records (Accessed: 22 November 2026).
- Greater London Authority (2026) *London Fire Brigade Mobilisation Records*. London Datastore. Available at: https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records (Accessed: 22 November 2026).

## Licence

Code: MIT Licence. Data: contains public sector information licensed under the Open Government Licence v2.0.
