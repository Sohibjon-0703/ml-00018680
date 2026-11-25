import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import load_eval, page_header, sidebar_note

page_header("✅ Model Evaluation", "Tuned models compared on the held-out 2026 test set")
sidebar_note()
ev = load_eval()
res = pd.DataFrame(ev["results"]).set_index("Model")

st.markdown(f"Trained on **{ev['train_rows']:,}** incidents (2024–2025), tested on **{ev['test_rows']:,}** unseen "
            f"incidents (Jan–Oct 2026). Final model: **{ev['final_model']}** (highest PR-AUC; models within "
            "0.005 PR-AUC are treated as tied and ranked by ROC-AUC).")
st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, subset=[c for c in res.columns if c != "Threshold"],
                                                       color="#d4edda"), width="stretch")

with st.expander("What do the metrics mean here?"):
    st.markdown("""
* **ROC-AUC**: probability that a random late incident gets a higher risk score than a random on-time one (0.5 = random).
* **PR-AUC**: precision–recall trade-off for the *late* class. The no-skill baseline equals the late rate in the test set (~0.36).
* **Recall / TPR**: share of late incidents that the model flags. **TNR**: share of on-time incidents correctly not flagged.
* **Precision**: share of flagged incidents that really were late. **F1**: balance of precision and recall.
* **Accuracy** is shown only for completeness: predicting "on time" for everything already scores ~65%.
* Thresholds were chosen to maximise F1 on a validation period (Oct–Dec 2025), never on the test set.
""")

models = [m for m in ev["curves"]]
a, b = st.columns(2)
roc, pr = go.Figure(), go.Figure()
for m in models:
    c = ev["curves"][m]
    roc.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], name=f"{m} ({res.loc[m, 'ROC-AUC']:.3f})"))
    pr.add_trace(go.Scatter(x=c["recall"], y=c["precision"], name=f"{m} ({res.loc[m, 'PR-AUC']:.3f})"))
roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line_dash="dash")
roc.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
pr.update_layout(title="Precision–Recall curves", xaxis_title="Recall", yaxis_title="Precision")
a.plotly_chart(roc, width="stretch"); b.plotly_chart(pr, width="stretch")

st.subheader("Confusion matrices (at each model's chosen threshold)")
cols = st.columns(len(models))
for col, m in zip(cols, models):
    cm = pd.DataFrame(ev["curves"][m]["cm"], index=["actual on time", "actual late"], columns=["pred on time", "pred late"])
    col.plotly_chart(px.imshow(cm, text_auto=",", color_continuous_scale="Blues", title=m).update_layout(
        coloraxis_showscale=False, height=330), width="stretch")

st.subheader(f"What drives the predictions? Permutation importance ({ev['final_model']})")
imp = pd.Series(ev["importance"]).sort_values().reset_index()
imp.columns = ["feature", "drop in ROC-AUC when shuffled"]
st.plotly_chart(px.bar(imp, x="drop in ROC-AUC when shuffled", y="feature", orientation="h", height=520),
                width="stretch")
st.caption("Permutation importance shuffles one feature at a time on test data and measures how much ROC-AUC falls. "
           "A larger drop means the model relies on that feature more.")
