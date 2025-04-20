import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# === Page Setup ===
st.set_page_config(page_title="YRFI Dashboard", layout="wide")
st.title("🔥 YRFI Prediction Dashboard")

# === Load Data ===
preds = pd.read_csv("data/yrfi_predictions_pregame_with_odds.csv")
live = pd.read_csv("data/yrfi_model_input_live_with_era.csv")

# Normalize column names
preds["Game Date"] = pd.to_datetime(preds["Game Date"])
live["Game Date"] = pd.to_datetime(live["date"])
live.rename(columns={"date": "Game Date"}, inplace=True)

# === Merge Predictions with Actual Outcomes ===
merged = preds.merge(
    live[["Game Date", "Away Team", "Home Team", "Away 1st_x", "Home 1st_x"]],
    on=["Game Date", "Away Team", "Home Team"],
    how="left"
)

# === Calculate actual YRFI outcome ===
merged["YRFI"] = ((merged["Away 1st_x"].fillna(0) > 0) | (merged["Home 1st_x"].fillna(0) > 0)).astype(int)

# === Fireball Confidence Tiers ===
def to_fireballs(p):
    if p >= 0.80: return "🔥🔥🔥🔥🔥"
    elif p >= 0.60: return "🔥🔥🔥🔥"
    elif p >= 0.40: return "🔥🔥🔥"
    elif p >= 0.20: return "🔥🔥"
    else: return "🔥"

merged["YRFI🔥"] = merged["YRFI_Prob"].apply(to_fireballs)
merged["NRFI🔥"] = merged["NRFI_Prob"].apply(lambda x: to_fireballs(1 - x))

# === Determine prediction correctness ===
def outcome_check(row):
    if pd.isna(row["YRFI"]):
        return ""
    correct = (
        (row["YRFI_Prob"] >= 0.5 and row["YRFI"] == 1) or
        (row["YRFI_Prob"] < 0.5 and row["YRFI"] == 0)
    )
    return "✅" if correct else "❌"

merged["Correct"] = merged.apply(outcome_check, axis=1)

# === Calendar Selector ===
today = datetime.today().date()
tomorrow = today + timedelta(days=1)

available_dates = sorted(merged["Game Date"].dt.date.unique())
if today not in available_dates:
    available_dates.append(today)
if tomorrow not in available_dates:
    available_dates.append(tomorrow)

available_dates = sorted(set(available_dates))

default_date = tomorrow if tomorrow in available_dates else today if today in available_dates else available_dates[-1]

selected_date = st.date_input(
    "📅 Select Game Date",
    value=default_date,
    min_value=min(available_dates),
    max_value=max(available_dates)
)

# === Filter by selected date ===
filtered = merged[merged["Game Date"].dt.date == selected_date]

# === Show Predictions Table ===
if not filtered.empty:
    st.subheader(f"📋 Games for {selected_date.strftime('%Y-%m-%d')}")
    display_cols = ["Away Team", "Home Team", "YRFI_Prob", "YRFI🔥", "NRFI_Prob", "NRFI🔥", "YRFI", "Correct"]
    st.dataframe(filtered[display_cols].sort_values("YRFI_Prob", ascending=False), use_container_width=True)
else:
    st.warning("No predictions available for this date.")

# === Accuracy Summary ===
today_total = filtered.shape[0]
today_correct = (filtered["Correct"] == "✅").sum()
today_wrong = (filtered["Correct"] == "❌").sum()

cumulative = merged[merged["Game Date"].dt.date <= selected_date]
cumulative_total = cumulative.shape[0]
cumulative_correct = (cumulative["Correct"] == "✅").sum()
cumulative_wrong = (cumulative["Correct"] == "❌").sum()

st.markdown("---")
st.subheader("📊 Prediction Accuracy Summary")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📅 Daily Accuracy")
    st.metric("Correct", today_correct)
    st.metric("Incorrect", today_wrong)
    st.metric("Total Predictions", today_total)
with col2:
    st.markdown("#### 🔁 Cumulative Accuracy")
    st.metric("Correct", cumulative_correct)
    st.metric("Incorrect", cumulative_wrong)
    st.metric("Total Predictions", cumulative_total)

# === Fireball Tier Summary
st.markdown("---")
st.subheader("🔥 Fireball Tier Performance")

tiers = ["🔥", "🔥🔥", "🔥🔥🔥", "🔥🔥🔥🔥", "🔥🔥🔥🔥🔥"]
fire_df = cumulative[cumulative["YRFI🔥"].isin(tiers)]

tier_stats = (
    fire_df.groupby("YRFI🔥")["Correct"]
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={"✅": "Correct", "❌": "Incorrect"})
)

tier_stats["Total"] = tier_stats["Correct"] + tier_stats["Incorrect"]
tier_stats["Accuracy %"] = (tier_stats["Correct"] / tier_stats["Total"].replace(0, 1) * 100).round(1)
tier_stats = tier_stats.reindex(tiers).fillna(0).astype(int)

st.dataframe(tier_stats, use_container_width=True)

# === Compact Summary View
st.markdown("---")
st.subheader("🔥 Fireball Accuracy Summary (Compact View)")

def summarize_fireballs(df):
    result = {}
    for tier in tiers:
        subset = df[df["YRFI🔥"] == tier]
        correct = (subset["Correct"] == "✅").sum()
        incorrect = (subset["Correct"] == "❌").sum()
        total = correct + incorrect
        result[tier] = {"Correct": correct, "Incorrect": incorrect, "Total": total}
    return result

daily_stats = summarize_fireballs(filtered)
rolling_stats = summarize_fireballs(cumulative)

summary_df = pd.DataFrame([
    {
        "Tier": tier,
        "Daily Correct": daily_stats[tier]["Correct"],
        "Daily Incorrect": daily_stats[tier]["Incorrect"],
        "Daily Total": daily_stats[tier]["Total"],
        "Rolling Correct": rolling_stats[tier]["Correct"],
        "Rolling Incorrect": rolling_stats[tier]["Incorrect"],
        "Rolling Total": rolling_stats[tier]["Total"],
    }
    for tier in tiers
])

st.dataframe(summary_df.set_index("Tier"), use_container_width=True)
