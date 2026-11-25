import plotly.express as px
import streamlit as st

from utils import load_data, load_eval, page_header, sidebar_note

page_header("🧹 Preprocessing", "How the raw data was cleaned, corrected and turned into model features")
sidebar_note()
df = load_data()
ev = load_eval()
log = ev.get("prep_log", {})

st.subheader("1 · Cleaning steps")
if log:
    st.dataframe(log["steps"], width="stretch", hide_index=True)
st.markdown("""
| Problem found in EDA | Treatment | Why |
|---|---|---|
| No attendance time (~5%, mostly Special Services) | Rows dropped | No label exists. Imputing the target would invent the answer. |
| Attendance < 30 s | Rows dropped | Physically implausible, so treated as a recording error |
| Postcode / lat-long missing for ~62% (dwellings redacted for privacy) | Use rounded easting/northing | Complete for all rows and still privacy-safe |
| Missing categories (`PropertyCategory`, ward, …) | Filled with `"Unknown"` | Very few rows. Keeps them without guessing a value. |
| 282 property types, many rare | Types with < 300 training rows grouped into `"Other"` | Avoids sparse, overfitted one-hot columns |
""")

st.subheader("2 · Time-zone correction (error correction)")
st.markdown("""
Incident call times are recorded in **UK local time**, while mobilisation times are in **GMT**. Before
correction, engines appeared to be mobilised **one hour before the call** from April to October (British
Summer Time). Call times were converted from *Europe/London* to UTC, so the two tables line up all year.
""")
if log.get("tz_check"):
    st.plotly_chart(px.line(log["tz_check"], x="month", y=["gap_local_min", "gap_utc_min"], markers=True,
                            labels={"value": "median minutes from call to first mobilisation", "variable": ""},
                            title="Median gap between call and first mobilisation, before vs after the fix"),
                    width="stretch")

st.subheader("3 · Target definition")
thr = st.slider("Attendance threshold (seconds)", 180, 720, 360, 30,
                help="The LFB standard is 360 s (6 minutes) for the first engine.")
share = (df["FirstPumpArriving_AttendanceTime"] > thr).mean()
c1, c2 = st.columns([1, 2])
c1.metric(f"Share of incidents later than {thr} s", f"{share:.1%}")
c1.caption("At 360 s the classes are about 1 : 2. That is imbalanced enough that accuracy is misleading, "
           "but there are enough positives that resampling is not needed.")
c2.plotly_chart(px.pie(values=[share, 1 - share], names=["late", "on time"], hole=.5,
                       color_discrete_sequence=["#C44E52", "#4C72B0"]), width="stretch")

st.subheader("4 · Engineered features (known at call time)")
feat = st.selectbox("Feature", ["station_busy", "dist_home_station_km", "dist_nearest_station_km",
                                "ground_calls_60m", "london_calls_60m", "hour"])
desc = {
    "station_busy": "Engines from the incident's home station already out (mobilised, not yet left) when the call arrives, computed from the mobilisation table.",
    "dist_home_station_km": "Distance from the incident to its home station's estimated location (estimated from the fastest 5% of attendances in the training period).",
    "dist_nearest_station_km": "Distance to the nearest estimated station location.",
    "ground_calls_60m": "Incidents in the same station ground in the previous 60 minutes.",
    "london_calls_60m": "Incidents anywhere in London in the previous 60 minutes.",
    "hour": "Hour of the call (local time). The model uses a sine/cosine encoding so that 23:00 is next to 00:00.",
}
st.caption(desc[feat])
clip = df[feat].quantile(0.99)
rate = df.assign(v=df[feat].clip(upper=clip)).groupby(
    (df[feat].clip(upper=clip) // max(clip / 20, 1e-9)) * max(clip / 20, 1e-9))["late"].agg(["mean", "size"]).reset_index()
a, b = st.columns(2)
a.plotly_chart(px.histogram(df, x=feat, color="outcome", barmode="overlay", histnorm="probability density", nbins=40,
                            range_x=(df[feat].min(), clip), color_discrete_map={"late": "#C44E52", "on time": "#4C72B0"},
                            title="Distribution: late vs on time"), width="stretch")
b.plotly_chart(px.line(rate, x=feat, y="mean", markers=True, hover_data=["size"], labels={"mean": "late rate"},
                       title="Late rate by feature value"), width="stretch")

st.subheader("5 · Leakage control: columns deliberately excluded")
st.markdown("""
| Column(s) | Why excluded |
|---|---|
| Number of pumps / stations attending, pump minutes, notional cost, second-pump times | Size of the response, only known **after** the call |
| Station that actually arrived first, turnout/travel time, delay codes | Part of the outcome being predicted |
| Incident group, stop code, special service type | Recorded after attendance (e.g. "false alarm" is confirmed on arrival) |
| Number of calls | Calls keep coming while people wait, so late incidents collect more calls (reverse causation) |
""")

st.subheader("6 · Train / test split")
counts = df.groupby("split")["late"].agg(["size", "mean"]).rename(columns={"size": "incidents", "mean": "late rate"})
st.dataframe(counts.style.format({"incidents": "{:,}", "late rate": "{:.1%}"}), width="stretch")
st.caption("Time-based split: train on 2024–2025, test on 2026, so the model is always evaluated on the future. "
           "Hyperparameters are tuned with TimeSeriesSplit cross-validation, and the decision threshold is chosen "
           "on Oct–Dec 2025.")
