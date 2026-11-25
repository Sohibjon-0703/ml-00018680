import streamlit as st

from utils import TARGET_SECONDS, load_data, load_eval, page_header, sidebar_note

page_header("🚒 Will the first fire engine be late?",
            "Predicting missed 6-minute attendance targets for the London Fire Brigade")
sidebar_note()

df = load_data()
ev = load_eval()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Incidents (cleaned)", f"{len(df):,}")
c2.metric("Missed 6-min target", f"{df['late'].mean():.1%}")
c3.metric("Median attendance", f"{df['FirstPumpArriving_AttendanceTime'].median() / 60:.1f} min")
best = next(r for r in ev["results"] if r["Model"] == ev["final_model"])
c4.metric(f"Final model ({ev['final_model']})", f"ROC-AUC {best['ROC-AUC']:.3f}")

st.markdown(f"""
### The problem
The London Fire Brigade aims to get the **first fire engine to an incident within 6 minutes** of a 999 call.
In the data, about **one in three** incidents misses that target. This app presents a machine-learning model
that estimates, **using only information known when the call is received**, the probability that the first
engine will arrive **late (> {TARGET_SECONDS} s)**.

**Who would use it?** Planners deciding where engines are stationed, when to move engines to cover gaps,
and how to staff busy periods. It is a decision-support and analysis tool. It does not replace the
control-room dispatch system.

### Pages
| Page | What it shows |
|---|---|
| **1 · Data Exploration** | Interactive dashboard: late rate by time, borough, property type, a map of London |
| **2 · Preprocessing** | Cleaning steps, time-zone correction, engineered features, leakage control |
| **3 · Model Training** | Train a model yourself with your own hyperparameters and compare it |
| **4 · Evaluation** | Test-set (2026) comparison of the tuned models: metrics, curves, confusion matrices, importance |
| **5 · Predict** | Enter the details of a call and get the probability of a late arrival |

### Data
* **LFB Incident Records**: one row per incident (call time, property, location, attendance time)
* **LFB Mobilisation Records**: one row per engine sent (station, mobilised/arrived/left times)
* Joined on `IncidentNumber`. Period: **January 2024 – October 2026**. Source: London Datastore (GLA),
  Open Government Licence v2.
""")
