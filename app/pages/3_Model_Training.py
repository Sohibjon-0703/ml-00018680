import time

import pandas as pd
import plotly.express as px
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import load_data, load_eval, load_model, page_header, sidebar_note

page_header("🧠 Model Training", "Train a model yourself on a sample of the data and compare it with the tuned final model")
sidebar_note()
df = load_data()
bundle = load_model()
ev = load_eval()
CAT, NUM = bundle["cat"], bundle["num"]

with st.form("train"):
    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Algorithm", ["LightGBM", "Random Forest", "Logistic Regression"])
    n_train = c2.select_slider("Training rows (sampled from 2024–25)", [10_000, 25_000, 50_000, 100_000], value=25_000)
    threshold = c3.slider("Decision threshold", 0.1, 0.9, 0.4, 0.05)
    feats_num = st.multiselect("Numeric features", NUM, default=NUM)
    feats_cat = st.multiselect("Categorical features", CAT, default=CAT)

    st.markdown("**Hyperparameters**")
    h1, h2, h3 = st.columns(3)
    if algo == "LightGBM":
        params = dict(n_estimators=h1.slider("n_estimators", 50, 800, 300, 50),
                      learning_rate=h2.select_slider("learning_rate", [0.01, 0.02, 0.05, 0.1, 0.2], 0.05),
                      num_leaves=h3.slider("num_leaves", 7, 127, 31))
    elif algo == "Random Forest":
        params = dict(n_estimators=h1.slider("n_estimators", 50, 400, 150, 50),
                      max_depth=h2.slider("max_depth", 3, 30, 15),
                      min_samples_leaf=h3.slider("min_samples_leaf", 1, 100, 20))
    else:
        params = dict(C=h1.select_slider("C (inverse regularisation)", [0.001, 0.01, 0.1, 1.0, 10.0], 1.0),
                      class_weight=h2.selectbox("class_weight", [None, "balanced"]))
    go = st.form_submit_button("🚀 Train and evaluate", type="primary")

if go:
    if not (feats_num or feats_cat):
        st.error("Select at least one feature."); st.stop()
    train = df[df["split"] == "train (2024-25)"].sample(n_train, random_state=0)
    test = df[df["split"] == "test (2026)"]
    prep = ColumnTransformer([t for t in [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), feats_cat),
                                          ("num", StandardScaler(), feats_num)] if t[2]])
    model = {"LightGBM": lambda: LGBMClassifier(verbose=-1, random_state=0, **params),
             "Random Forest": lambda: RandomForestClassifier(n_jobs=-1, random_state=0, **params),
             "Logistic Regression": lambda: LogisticRegression(max_iter=2000, **params)}[algo]()
    pipe = Pipeline([("prep", prep), ("model", model)])
    with st.spinner("Training..."):
        t = time.time(); pipe.fit(train[feats_cat + feats_num], train["late"]); fit_s = time.time() - t
    p = pipe.predict_proba(test[feats_cat + feats_num])[:, 1]
    y = test["late"]; yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    final = next(r for r in ev["results"] if r["Model"] == ev["final_model"])

    st.success(f"Trained {algo} on {n_train:,} rows in {fit_s:.1f}s and evaluated on {len(test):,} incidents from 2026.")
    m = st.columns(5)
    m[0].metric("ROC-AUC", f"{roc_auc_score(y, p):.3f}", f"{roc_auc_score(y, p) - final['ROC-AUC']:+.3f} vs final")
    m[1].metric("PR-AUC", f"{average_precision_score(y, p):.3f}", f"{average_precision_score(y, p) - final['PR-AUC']:+.3f} vs final")
    m[2].metric("Recall (TPR)", f"{tp / (tp + fn):.3f}")
    m[3].metric("Precision", f"{precision_score(y, yhat, zero_division=0):.3f}")
    m[4].metric("F1", f"{f1_score(y, yhat):.3f}")

    a, b = st.columns(2)
    fpr, tpr, _ = roc_curve(y, p)
    roc = pd.DataFrame({"FPR": fpr, "TPR": tpr}).iloc[:: max(1, len(fpr) // 500)]
    a.plotly_chart(px.line(roc, x="FPR", y="TPR", title="ROC curve (test 2026)").add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1, line_dash="dash"), width="stretch")
    cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["actual on time", "actual late"], columns=["pred on time", "pred late"])
    b.plotly_chart(px.imshow(cm, text_auto=",", color_continuous_scale="Blues", title=f"Confusion matrix @ {threshold}"),
                   width="stretch")
    st.caption("Tip: move the threshold to trade recall (finding late incidents) against precision (avoiding false alerts). "
               "Remove `station_busy` or the distance features to see how much they contribute.")
else:
    st.info("Choose an algorithm, features and hyperparameters, then press **Train and evaluate**. "
            "Training uses a sample so it finishes in seconds. The final model in the notebook was tuned on far "
            "more data with cross-validation.")
