import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import load_data, load_eval, load_model, page_header, sidebar_note

page_header("🔮 Predict", "Enter the details of a 999 call and estimate the risk that the first engine misses 6 minutes")
sidebar_note()
df = load_data()
bundle = load_model()
ev = load_eval()
stations = pd.DataFrame(ev["station_locations"]).set_index("st")
model, thr = bundle["model"], bundle["threshold"]

c1, c2, c3 = st.columns(3)
borough = c1.selectbox("Borough", sorted(df["IncGeo_BoroughName"].unique()), index=None, placeholder="Choose a borough")
grounds = sorted(df.loc[df["IncGeo_BoroughName"] == borough, "IncidentStationGround"].unique()) if borough else []
ground = c2.selectbox("Station ground", grounds, index=0 if grounds else None, help="The fire station area the address falls in")
prop_cat = c3.selectbox("Property category", sorted(df["PropertyCategory"].unique()), index=None, placeholder="Choose")

c4, c5, c6 = st.columns(3)
types = df.loc[df["PropertyCategory"] == prop_cat, "PropertyType_grouped"].value_counts().index.tolist() if prop_cat else []
prop_type = c4.selectbox("Property type", types, index=0 if types else None)
addr = c5.selectbox("Address qualifier", df["AddressQualifier"].value_counts().index.tolist())
date = c6.date_input("Date of call", dt.date(2026, 6, 15))

c7, c8, c9 = st.columns(3)
time_ = c7.time_input("Time of call", dt.time(15, 0))
busy = c8.number_input("Engines from home station already out", 0, 4, 0,
                       help="Engines from this station mobilised and not yet back at the time of the call")
recent = c9.slider("Incidents in this station ground in the last hour", 0, 6, 0)

if not (borough and ground and prop_cat and prop_type):
    st.info("Fill in borough, station ground and property to get a prediction."); st.stop()

# Location: typical position of incidents in the chosen station ground (users rarely know easting/northing)
g = df[df["IncidentStationGround"] == ground]
with st.expander("Location (default: typical location in the station ground)"):
    e = st.slider("Easting (km)", float(g["easting_km"].min()), float(g["easting_km"].max()), float(g["easting_km"].median()), 0.1)
    n = st.slider("Northing (km)", float(g["northing_km"].min()), float(g["northing_km"].max()), float(g["northing_km"].median()), 0.1)

home = ground.strip().lower()
dists = np.sqrt((stations["easting_km"] - e) ** 2 + (stations["northing_km"] - n) ** 2)
hour = time_.hour
row = pd.DataFrame([{
    "IncGeo_BoroughName": borough, "IncidentStationGround": ground, "PropertyCategory": prop_cat,
    "PropertyType_grouped": prop_type, "AddressQualifier": addr,
    "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
    "weekday": date.weekday(), "month": date.month, "is_weekend": int(date.weekday() >= 5),
    "is_night": int(hour in [23, 0, 1, 2, 3, 4, 5, 6]), "easting_km": e, "northing_km": n,
    "station_busy": busy, "ground_calls_60m": recent,
    "london_calls_60m": int(df["london_calls_60m"].median()),
    "dist_home_station_km": float(dists.get(home, dists.min())), "dist_nearest_station_km": float(dists.min()),
    "home_is_nearest": int(dists.idxmin() == home)}])[bundle["features"]]

p = float(model.predict_proba(row)[:, 1][0])
a, b = st.columns([1, 1])
gauge = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                               title={"text": "Probability first engine is late (> 6 min)"},
                               gauge={"axis": {"range": [0, 100]}, "threshold": {"line": {"color": "black", "width": 3}, "value": thr * 100},
                                      "steps": [{"range": [0, thr * 100], "color": "#d4edda"}, {"range": [thr * 100, 100], "color": "#f8d7da"}],
                                      "bar": {"color": "#333"}}))
a.plotly_chart(gauge, width="stretch")
with b:
    if p >= thr:
        st.error(f"**High risk of missing the 6-minute target** (above the model threshold of {thr:.0%}).")
    else:
        st.success(f"**Likely to be reached within 6 minutes** (below the model threshold of {thr:.0%}).")
    base = df.loc[df["IncidentStationGround"] == ground, "late"].mean()
    st.metric("Historical late rate in this station ground", f"{base:.1%}")
    st.write(f"Distance to home station (estimated): **{row['dist_home_station_km'].iloc[0]:.1f} km** · "
             f"nearest station: **{dists.idxmin().title()}** ({dists.min():.1f} km)")
    st.caption(f"Model: {ev['final_model']}. London-wide demand is set to its median. This is a planning aid "
               "built on historical data, not an operational dispatch tool.")

st.subheader("What if the home station's engines were busy?")
what_if = pd.concat([row.assign(station_busy=k) for k in range(4)])
probs = model.predict_proba(what_if)[:, 1]
st.bar_chart(pd.Series(probs, index=[f"{k} engines out" for k in range(4)], name="P(late)"))
