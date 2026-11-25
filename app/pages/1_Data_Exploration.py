import plotly.express as px
import streamlit as st

from utils import TARGET_SECONDS, WEEKDAYS, load_data, page_header, sidebar_note

page_header("📊 Data Exploration", "Interactive dashboard: filter the incidents and see where and when the target is missed")
df = load_data()

# ---- Filters ----
st.sidebar.header("Filters")
years = st.sidebar.multiselect("Year", sorted(df["CalYear"].unique()), default=sorted(df["CalYear"].unique()))
boroughs = st.sidebar.multiselect("Borough", sorted(df["IncGeo_BoroughName"].unique()), placeholder="All boroughs")
props = st.sidebar.multiselect("Property category", sorted(df["PropertyCategory"].unique()), placeholder="All")
groups = st.sidebar.multiselect("Incident group", sorted(df["IncidentGroup"].unique()), placeholder="All")
hours = st.sidebar.slider("Hour of call", 0, 23, (0, 23))
sidebar_note()

f = df[df["CalYear"].isin(years) & df["hour"].between(*hours)]
if boroughs: f = f[f["IncGeo_BoroughName"].isin(boroughs)]
if props: f = f[f["PropertyCategory"].isin(props)]
if groups: f = f[f["IncidentGroup"].isin(groups)]
if f.empty:
    st.warning("No incidents match the filters."); st.stop()

# ---- KPIs ----
k = st.columns(4)
k[0].metric("Incidents", f"{len(f):,}")
k[1].metric("Late rate (> 6 min)", f"{f['late'].mean():.1%}", f"{(f['late'].mean() - df['late'].mean()) * 100:+.1f} pp vs all",
            delta_color="inverse")
k[2].metric("Median attendance", f"{f['FirstPumpArriving_AttendanceTime'].median():.0f} s")
k[3].metric("90th percentile", f"{f['FirstPumpArriving_AttendanceTime'].quantile(.9):.0f} s")

tab1, tab2, tab3, tab4 = st.tabs(["⏱ When", "📍 Where", "🏠 What", "📈 Distributions & correlation"])

with tab1:
    a, b = st.columns(2)
    by_h = f.groupby("hour", observed=True)["late"].agg(["mean", "size"]).reset_index()
    a.plotly_chart(px.line(by_h, x="hour", y="mean", markers=True, labels={"mean": "late rate"},
                           title="Late rate by hour of call"), width="stretch")
    by_w = f.groupby("weekday_name", observed=True)["late"].mean().reindex(WEEKDAYS).reset_index()
    b.plotly_chart(px.bar(by_w, x="weekday_name", y="late", labels={"late": "late rate", "weekday_name": ""},
                          title="Late rate by weekday"), width="stretch")
    heat = f.pivot_table(index="weekday_name", columns="hour", values="late", aggfunc="mean", observed=True).reindex(WEEKDAYS)
    st.plotly_chart(px.imshow(heat, color_continuous_scale="RdYlGn_r", aspect="auto",
                              labels={"color": "late rate"}, title="Late rate: weekday × hour"), width="stretch")

with tab2:
    a, b = st.columns([3, 2])
    grid = f.assign(x=(f["easting_km"] // 1), y=(f["northing_km"] // 1)) \
            .groupby(["x", "y"])["late"].agg(["mean", "size"]).reset_index()
    grid = grid[grid["size"] >= 15]
    fig = px.scatter(grid, x="x", y="y", color="mean", color_continuous_scale="RdYlGn_r", range_color=(0, 0.8),
                     labels={"mean": "late rate", "x": "Easting (km)", "y": "Northing (km)"},
                     title="Late rate on a 1 km grid (cells with ≥ 15 incidents)", hover_data={"size": True})
    fig.update_traces(marker=dict(size=6, symbol="square")); fig.update_yaxes(scaleanchor="x")
    a.plotly_chart(fig, width="stretch")
    by_b = f.groupby("IncGeo_BoroughName", observed=True)["late"].mean().sort_values().reset_index()
    b.plotly_chart(px.bar(by_b, y="IncGeo_BoroughName", x="late", orientation="h", height=650,
                          labels={"late": "late rate", "IncGeo_BoroughName": ""}, title="Late rate by borough"),
                   width="stretch")

with tab3:
    a, b = st.columns(2)
    by_p = f.groupby("PropertyCategory", observed=True)["late"].agg(["mean", "size"]).reset_index().sort_values("mean")
    a.plotly_chart(px.bar(by_p, x="mean", y="PropertyCategory", orientation="h", hover_data=["size"],
                          labels={"mean": "late rate", "PropertyCategory": ""}, title="Late rate by property category"),
                   width="stretch")
    by_g = f.groupby("IncidentGroup", observed=True)["late"].agg(["mean", "size"]).reset_index()
    b.plotly_chart(px.bar(by_g, x="IncidentGroup", y="mean", hover_data=["size"], labels={"mean": "late rate"},
                          title="Late rate by incident group (recorded after attendance)"), width="stretch")
    top = f["PropertyType"].value_counts().head(15).index
    st.plotly_chart(px.box(f[f["PropertyType"].isin(top)], y="PropertyType", x="FirstPumpArriving_AttendanceTime",
                           points=False, title="Attendance time, 15 most common property types",
                           labels={"FirstPumpArriving_AttendanceTime": "attendance (s)", "PropertyType": ""}, height=550)
                    .add_vline(x=TARGET_SECONDS, line_dash="dash", line_color="red"), width="stretch")

with tab4:
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(f, x="FirstPumpArriving_AttendanceTime", nbins=80, color="outcome",
                                color_discrete_map={"late": "#C44E52", "on time": "#4C72B0"},
                                title="Attendance time distribution", labels={"FirstPumpArriving_AttendanceTime": "seconds"})
                   .add_vline(x=TARGET_SECONDS, line_dash="dash"), width="stretch")
    num = ["FirstPumpArriving_AttendanceTime", "late", "station_busy", "ground_calls_60m", "london_calls_60m",
           "dist_home_station_km", "dist_nearest_station_km", "home_is_nearest", "hour_sin", "hour_cos", "is_night"]
    b.plotly_chart(px.imshow(f[num].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Correlation matrix"), width="stretch")
    st.subheader("Summary statistics")
    st.dataframe(f[num].describe().T.assign(variance=f[num].var()).round(2), width="stretch")
