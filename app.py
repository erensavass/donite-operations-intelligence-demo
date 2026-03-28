import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Donite Operations Intelligence Demo", layout="wide")

# I wanted this to feel closer to a real thermoforming / CNC discussion than a generic factory KPI page.
# The data below is synthetic on purpose: it keeps the project shareable while still letting me model
# throughput, quality drift, trim bottlenecks and maintenance-style signals.

# ---------- Synthetic data generation ----------
@st.cache_data
def generate_data(seed: int = 42, days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    processes = ["Thermoforming Cell 1", "Thermoforming Cell 2", "5-Axis CNC Trim"]
    sectors = [
        "Aircraft Interior Panel",
        "Medical Equipment Cover",
        "Agri-tech Housing",
        "Transport Interior Trim",
    ]
    shifts = ["Day", "Evening", "Night"]

    rows = []
    for date in dates:
        for process in processes:
            for shift in shifts:
                product = rng.choice(sectors, p=[0.36, 0.18, 0.22, 0.24])

                if "Thermoforming" in process:
                    base_output = 510 if process.endswith("1") else 470
                    setpoint_temp = 182 if product == "Aircraft Interior Panel" else 176
                    cycle_sec = rng.normal(81, 7)
                    trim_delay = rng.normal(0.03, 0.015)
                    vibration = rng.normal(1.8, 0.25)
                    tool_wear = rng.normal(36, 12)
                    sheet_util = rng.normal(83, 3.5)
                    energy_kwh = rng.normal(1180, 90)
                else:
                    base_output = 560
                    setpoint_temp = 28
                    cycle_sec = rng.normal(64, 6)
                    trim_delay = rng.normal(0.12, 0.05)
                    vibration = rng.normal(3.0, 0.45)
                    tool_wear = rng.normal(48, 15)
                    sheet_util = rng.normal(91, 2.2)
                    energy_kwh = rng.normal(940, 80)

                shift_factor = {"Day": 1.02, "Evening": 1.00, "Night": 0.95}[shift]
                weekday_factor = 1.02 if date.dayofweek < 5 else 0.93
                seasonal = 1 + 0.03 * math.sin(date.dayofyear / 9)

                output = base_output * shift_factor * weekday_factor * seasonal + rng.normal(0, 28)
                scrap = rng.normal(2.4, 0.7)
                rework = rng.normal(4.8, 1.2)

                # I added a few rough drift windows so the dashboard has realistic-looking problems to explain.
                if "Thermoforming" in process:
                    temp = rng.normal(setpoint_temp, 6)
                    pressure = rng.normal(5.3, 0.5)
                    heating_sec = rng.normal(102, 11)
                    if date.dayofyear in range(38, 48) and process == "Thermoforming Cell 2":
                        temp += rng.normal(10, 3)
                        scrap += rng.normal(2.2, 0.4)
                        output -= rng.normal(30, 8)
                    if date.dayofyear in range(88, 96):
                        sheet_util -= rng.normal(4.0, 1.0)
                        scrap += rng.normal(1.5, 0.5)
                else:
                    temp = rng.normal(31, 4)
                    pressure = rng.normal(4.2, 0.4)
                    heating_sec = rng.normal(0.0, 0.2)
                    if date.dayofyear in range(70, 86):
                        vibration += rng.normal(1.0, 0.2)
                        tool_wear += rng.normal(18, 3)
                        trim_delay += rng.normal(0.11, 0.03)
                        rework += rng.normal(2.6, 0.7)
                        output -= rng.normal(45, 10)

                if product == "Aircraft Interior Panel":
                    scrap += 0.35
                    rework += 0.50
                    energy_kwh += 60
                if shift == "Night":
                    rework += 0.35
                    output -= 9

                first_pass_yield = max(78, min(99.5, 100 - scrap - rework / 2))
                health_penalty = (
                    max(0, abs(temp - setpoint_temp) * 0.45)
                    + max(0, vibration - (2.2 if "Thermoforming" in process else 3.0)) * 5
                    + max(0, trim_delay - (0.05 if "Thermoforming" in process else 0.12)) * 120
                    + max(0, tool_wear - 60) * 0.18
                    + max(0, 84 - sheet_util) * 0.8
                )
                line_health = max(40, min(99, 94 - health_penalty + rng.normal(0, 2)))

                failure_risk = (
                    0.02
                    + max(0, abs(temp - setpoint_temp) - 5) * 0.008
                    + max(0, vibration - (2.2 if "Thermoforming" in process else 3.2)) * 0.06
                    + max(0, tool_wear - 55) * 0.008
                    + max(0, trim_delay - 0.15) * 0.65
                )
                failure_next_7d = int(rng.random() < min(0.92, failure_risk))

                rows.append(
                    {
                        "date": date,
                        "process": process,
                        "product_family": product,
                        "shift": shift,
                        "units_produced": round(max(290, output)),
                        "scrap_rate": round(max(0.6, scrap), 2),
                        "rework_rate": round(max(0.8, rework), 2),
                        "first_pass_yield": round(first_pass_yield, 2),
                        "sheet_utilization": round(min(97, max(72, sheet_util)), 2),
                        "energy_kwh": round(max(620, energy_kwh), 1),
                        "energy_per_100_units": round(energy_kwh / max(1, output) * 100, 1),
                        "temp_c": round(temp, 2),
                        "setpoint_temp_c": setpoint_temp,
                        "pressure_bar": round(max(0.1, pressure), 2),
                        "heating_seconds": round(max(0.0, heating_sec), 1),
                        "cycle_time_sec": round(max(42, cycle_sec), 1),
                        "vibration_mm_s": round(max(0.1, vibration), 2),
                        "tool_wear_index": round(max(5, tool_wear), 1),
                        "trim_delay_rate": round(max(0.0, trim_delay), 3),
                        "line_health_score": round(line_health, 1),
                        "failure_next_7d": failure_next_7d,
                    }
                )

    df = pd.DataFrame(rows)
    df["anomaly_flag"] = (
        (df["scrap_rate"] > 4.3)
        | (df["first_pass_yield"] < 92.5)
        | (df["trim_delay_rate"] > 0.19)
        | (df["vibration_mm_s"] > 3.8)
        | (df["line_health_score"] < 75)
    ).astype(int)
    return df


def build_failure_model(df: pd.DataFrame):
    # I kept the classification model interpretable enough for an interview conversation.
    features = [
        "process",
        "product_family",
        "shift",
        "temp_c",
        "setpoint_temp_c",
        "pressure_bar",
        "cycle_time_sec",
        "vibration_mm_s",
        "tool_wear_index",
        "trim_delay_rate",
        "sheet_utilization",
        "energy_per_100_units",
        "scrap_rate",
        "rework_rate",
        "line_health_score",
    ]
    target = "failure_next_7d"

    X = df[features]
    y = df[target]
    cat = ["process", "product_family", "shift"]
    num = [c for c in features if c not in cat]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ]
    )
    model = Pipeline(
        [
            ("prep", pre),
            ("clf", RandomForestClassifier(n_estimators=220, random_state=42, min_samples_leaf=3)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    perm = permutation_importance(model, X_test, y_test, n_repeats=6, random_state=42, n_jobs=1)
    fi = pd.DataFrame({"feature": features, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
    return model, auc, fi


def build_scrap_model(df: pd.DataFrame):
    # This is a simple scrap model for the what-if simulator, not a claim of production accuracy.
    features = [
        "temp_c",
        "setpoint_temp_c",
        "vibration_mm_s",
        "tool_wear_index",
        "trim_delay_rate",
        "sheet_utilization",
        "energy_per_100_units",
        "cycle_time_sec",
    ]
    reg = LinearRegression()
    reg.fit(df[features], df["scrap_rate"])
    return reg, features


def make_recommendations(row: pd.Series) -> list[str]:
    # The recommendations are deliberately operational and readable rather than overly academic.
    recs = []
    temp_delta = row["temp_c"] - row["setpoint_temp_c"]
    if abs(temp_delta) > 6:
        direction = "down" if temp_delta > 0 else "up"
        recs.append(f"Bring forming temperature {direction} toward setpoint by ~{abs(temp_delta):.1f}°C to stabilize scrap and first-pass yield.")
    if row["tool_wear_index"] > 60:
        recs.append("Schedule tool inspection / replacement soon; wear is high and is likely contributing to rework and short-horizon failure risk.")
    if row["vibration_mm_s"] > 3.3:
        recs.append("Check spindle / router vibration and fixture stability; this looks like a trimming-quality risk.")
    if row["trim_delay_rate"] > 0.16:
        recs.append("Investigate CNC queueing and setup losses; trim delay is likely the main bottleneck on throughput.")
    if row["sheet_utilization"] < 82:
        recs.append("Review nesting/tool layout to improve sheet utilization and reduce material waste.")
    if row["scrap_rate"] > 4.0 and row["product_family"] == "Aircraft Interior Panel":
        recs.append("Aircraft interior work is quality-sensitive; increase first-off inspection frequency on this family until drift settles.")
    if not recs:
        recs.append("Process looks broadly stable; continue monitoring and keep preventive maintenance on schedule.")
    return recs[:4]


# ---------- UI ----------
df = generate_data()
model, auc, fi = build_failure_model(df)
scrap_model, scrap_features = build_scrap_model(df)

st.sidebar.header("Filter controls")
processes = ["All"] + sorted(df["process"].unique().tolist())
products = ["All"] + sorted(df["product_family"].unique().tolist())
shifts = ["All"] + sorted(df["shift"].unique().tolist())

selected_process = st.sidebar.selectbox("Process", processes)
selected_product = st.sidebar.selectbox("Product family", products)
selected_shift = st.sidebar.selectbox("Shift", shifts)
date_min, date_max = df["date"].min().date(), df["date"].max().date()
selected_dates = st.sidebar.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

filtered = df.copy()
if selected_process != "All":
    filtered = filtered[filtered["process"] == selected_process]
if selected_product != "All":
    filtered = filtered[filtered["product_family"] == selected_product]
if selected_shift != "All":
    filtered = filtered[filtered["shift"] == selected_shift]
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
    filtered = filtered[(filtered["date"] >= start_date) & (filtered["date"] <= end_date)]

st.title("Donite Operations Intelligence Demo")
st.caption(
    "Personal portfolio demo built around a thermoforming / CNC trimming scenario. Uses synthetic data to show how I would approach process analytics, risk signals and improvement ideas in an industrial setting."
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Avg Output", f"{filtered['units_produced'].mean():.0f}")
col2.metric("Avg Scrap Rate", f"{filtered['scrap_rate'].mean():.2f}%")
col3.metric("First-Pass Yield", f"{filtered['first_pass_yield'].mean():.2f}%")
col4.metric("Avg Health Score", f"{filtered['line_health_score'].mean():.1f}/100")
col5.metric("Anomaly Days", int(filtered["anomaly_flag"].sum()))

st.markdown("---")

# Risk board
latest = filtered.sort_values("date").groupby(["process", "shift"], as_index=False).tail(1).copy()
feature_cols = [
    "process","product_family","shift","temp_c","setpoint_temp_c","pressure_bar","cycle_time_sec",
    "vibration_mm_s","tool_wear_index","trim_delay_rate","sheet_utilization","energy_per_100_units",
    "scrap_rate","rework_rate","line_health_score"
]
latest["failure_risk_score"] = model.predict_proba(latest[feature_cols])[:, 1] * 100
latest = latest.sort_values("failure_risk_score", ascending=False)

st.subheader("Predictive maintenance watchlist")
st.caption(f"Held-out synthetic demo AUC: {auc:.2f}. I use this here as a discussion tool, not as a production claim.")
st.dataframe(
    latest[["date", "process", "shift", "product_family", "line_health_score", "scrap_rate", "trim_delay_rate", "tool_wear_index", "failure_risk_score"]]
    .rename(columns={
        "date": "Date",
        "process": "Process",
        "shift": "Shift",
        "product_family": "Product family",
        "line_health_score": "Health",
        "scrap_rate": "Scrap %",
        "trim_delay_rate": "Trim delay",
        "tool_wear_index": "Tool wear",
        "failure_risk_score": "Failure risk next 7d %",
    }),
    use_container_width=True,
    hide_index=True,
)

# Operational trends
st.subheader("Operational trends")
trend_choice = st.selectbox(
    "Trend metric",
    ["units_produced", "scrap_rate", "first_pass_yield", "energy_per_100_units", "line_health_score"],
    index=0,
)
trend = filtered.groupby("date", as_index=False)[trend_choice].mean()
fig = plt.figure(figsize=(10, 4.5))
plt.plot(trend["date"], trend[trend_choice])
plt.xlabel("Date")
plt.ylabel(trend_choice)
plt.title(f"{trend_choice} over time")
st.pyplot(fig, clear_figure=True)

# Root cause analysis
left, right = st.columns(2)
with left:
    st.subheader("Root-cause signals")
    fig2 = plt.figure(figsize=(7, 4.5))
    top_fi = fi.head(8).sort_values("importance")
    plt.barh(top_fi["feature"], top_fi["importance"])
    plt.xlabel("Importance")
    plt.title("Top drivers of short-horizon failure risk")
    st.pyplot(fig2, clear_figure=True)

with right:
    st.subheader("Correlation with scrap rate")
    numeric_cols = [
        "scrap_rate", "temp_c", "vibration_mm_s", "tool_wear_index", "trim_delay_rate",
        "sheet_utilization", "energy_per_100_units", "cycle_time_sec", "line_health_score"
    ]
    corr = filtered[numeric_cols].corr(numeric_only=True)["scrap_rate"].drop("scrap_rate").sort_values()
    fig3 = plt.figure(figsize=(7, 4.5))
    plt.barh(corr.index, corr.values)
    plt.xlabel("Correlation")
    plt.title("Variables most associated with scrap")
    st.pyplot(fig3, clear_figure=True)

# Recommendation engine
st.subheader("Action recommendations")
case_row = latest.iloc[0]
st.write(
    f"Highest current risk in filtered view: **{case_row['process']} / {case_row['shift']} / {case_row['product_family']}** on **{case_row['date'].date()}**."
)
for rec in make_recommendations(case_row):
    st.write(f"- {rec}")

# What-if simulator
st.subheader("What-if simulator")
st.caption("This section lets me test whether a small operational change could plausibly improve scrap in one recent operating state.")
base_option = latest[["date", "process", "shift", "product_family"]].astype(str).agg(" | ".join, axis=1).tolist()
base_pick = st.selectbox("Choose a recent operating state", base_option)
base_idx = base_option.index(base_pick)
base_row = latest.iloc[base_idx].copy()

sim_cols = st.columns(4)
new_temp = sim_cols[0].slider("Temperature offset (°C)", -15, 15, 0)
new_wear = sim_cols[1].slider("Tool wear reduction", 0, 30, 0)
new_vibration = sim_cols[2].slider("Vibration reduction", 0.0, 1.8, 0.0, 0.1)
new_delay = sim_cols[3].slider("Trim delay reduction", 0.0, float(min(0.18, base_row['trim_delay_rate'])), 0.0, 0.01)

base_scrap = float(scrap_model.predict(pd.DataFrame([base_row[scrap_features]]))[0])
sim_row = base_row.copy()
sim_row["temp_c"] = base_row["temp_c"] + new_temp
sim_row["tool_wear_index"] = max(0, base_row["tool_wear_index"] - new_wear)
sim_row["vibration_mm_s"] = max(0, base_row["vibration_mm_s"] - new_vibration)
sim_row["trim_delay_rate"] = max(0, base_row["trim_delay_rate"] - new_delay)
sim_scrap = float(scrap_model.predict(pd.DataFrame([sim_row[scrap_features]]))[0])

r1, r2, r3 = st.columns(3)
r1.metric("Predicted scrap before", f"{base_scrap:.2f}%")
r2.metric("Predicted scrap after", f"{sim_scrap:.2f}%", delta=f"{sim_scrap - base_scrap:.2f}%")
r3.metric("Estimated annualized improvement signal", f"{max(0, (base_scrap - sim_scrap) * 12):.1f} pts")

# Executive summary
st.subheader("Executive summary")
summary = filtered.groupby("process").agg(
    avg_output=("units_produced", "mean"),
    avg_scrap=("scrap_rate", "mean"),
    avg_fpy=("first_pass_yield", "mean"),
    avg_health=("line_health_score", "mean"),
    anomaly_days=("anomaly_flag", "sum"),
).reset_index().sort_values("avg_health")
st.dataframe(summary, use_container_width=True, hide_index=True)

worst_process = summary.iloc[0]["process"]
worst_issue = "trim bottleneck and wear" if "CNC" in worst_process else "forming drift and material utilization"
st.info(
    f"Current management readout: **{worst_process}** is the weakest area in the selected slice, with the main issue pattern suggesting **{worst_issue}**."
)

with st.expander("About this demo"):
    st.write(
        "This is not based on Donite's private data. I built it as a synthetic portfolio project to mirror an industrial thermoforming and CNC trimming environment, including quality, throughput, maintenance and process-improvement analytics."
    )
