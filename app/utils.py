"""Shared loaders and constants for the Streamlit app."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "incidents_clean.parquet"
MODEL = ROOT / "models" / "final_model.joblib"
EVAL = ROOT / "models" / "evaluation.json"

TARGET_SECONDS = 360
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PALETTE = {"late": "#C44E52", "on time": "#4C72B0"}


@st.cache_data(show_spinner="Loading incidents...")
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["call_local"] = pd.to_datetime(df["call_local"])
    df["outcome"] = np.where(df["late"] == 1, "late", "on time")
    df["weekday_name"] = pd.Categorical(df["weekday"].map(dict(enumerate(WEEKDAYS))), WEEKDAYS, ordered=True)
    df["split"] = np.where(df["call_local"] < "2026-01-01", "train (2024-25)", "test (2026)")
    return df


@st.cache_resource(show_spinner="Loading model...")
def load_model() -> dict:
    return joblib.load(MODEL)


@st.cache_data
def load_eval() -> dict:
    return json.loads(EVAL.read_text(encoding="utf-8"))


def page_header(title: str, subtitle: str = ""):
    st.set_page_config(page_title=f"{title} · LFB 6-minute target", page_icon="🚒", layout="wide")
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def sidebar_note():
    st.sidebar.markdown(
        "**Data:** London Fire Brigade Incident & Mobilisation Records, Jan 2024 – Oct 2026, "
        "London Datastore, [Open Government Licence v2]"
        "(https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/).\n\n"
        "MLDA Coursework 1 · Student 00018680")
