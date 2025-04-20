import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# === Page Setup ===
st.set_page_config(page_title="YRFI Dashboard", layout="wide")
st.title("🔥 YRFI Prediction Dashboard")

# === Safe Deduplication ===
def dedupe_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            if i == 0:
                continue
            cols[idx] = f"{cols[idx]}.{i}"
    df.columns = cols
    return df

# === Load and Prepare Data ===
try:
    live = pd.read_csv("data/yrfi_model_input_live_with_era.csv")
    live = dedupe_columns(live)
except Exception as e:
    st.error(f"💥 Failed to load data file: {e}")
    st.stop()

# === Check Required Columns ===
required_cols = ["date", "Away 1st_x", "Home 1st_x", "YRFI"]
missing_cols = [col for col in required_cols if col not in live.columns]

if missing_cols:
    st.error(f"❌ Missing required column(s): {', '.join(missing_cols)}")
    st.stop()

# === Convert Date and Set Predictions ===
live["Game Date"] = pd.to_datetime(live["date"]).dt.date
live.rename(columns={"YRFI": "YRFI_pred"}, inplace=True)

# === Compute actual YRFI result from scores ===
live["YRFI_actual"] = ((live["Away 1st_x"].fillna(0) > 0) | (live["Home 1st_x"].fillna(0) > 0)).astype(int)

# === Debug Info ===
st.markdown("### 🛠 Data Debug Info")
debug_info = {
    "Total Rows": len(live),
    "Columns Present": list(live.columns),
    "Sample Predictions": live["YRFI_pred"].head(5).tolist(),
}
st.code(str(debug_info), language="python")

# === Confidence Fireball 🔥 Tiers ===
def to_fireballs(p):
    if p >= 0.80: return "🔥🔥🔥🔥🔥"
    elif p >= 0.60: return "🔥🔥🔥🔥"
    elif p >= 0.40: return "🔥🔥🔥"
    elif p >= 0.20: return "🔥🔥"
    else: return "🔥"

live["YRFI🔥"] = live["YRFI_pred"].apply(to_fireballs)
live["NRFI🔥"] = (1 - live["YRFI_pred"]).apply(to_fireballs)

# === Correct/Incorrect Outcome ===
def outcome_check(row):
    if pd.isna(row["YRFI_actual"]): return ""
    return "✅" if (row["YRFI_pred"] >= 0.5 and row["YRFI_actual"] == 1) or \
                  (row["YRFI_pred"] < 0.5 and row["YRFI_actual"] == 0) else "❌"

live["Correct"] = live.apply(outcome_check, axis=1)

# === Calendar Picker ===
available_dates = sorted(live["Game Date"].unique())
today = datetime.today().date()
tomorrow = today + timedelta(days=1)
if today not in available_dates: available_dates.append(today)
if tomorrow not in available_dates: available_dates.append(tomorrow)
available_dates = sorted(set(available_dates))
default_date = tomorrow if tomorrow in available_dates else today

selected_date = st.date_input("📅 Select Game Date", value=default_date,
                              min_value=min(available_dates), max_value=max(available_dates))

# === Filter for Selected Date ===
filtered = live[live["Game Date"] == selected_date]

# === Main Table ===
if not filtered.empty:
    st.subheader(f"📋 Games for {selected_date.strftime('%Y-%m-%d')}")
    display_cols = ["away_team", "home_team", "YRFI_pred", "YRFI🔥", "NRFI🔥", "YRFI_actual", "Correct"]
    st.dataframe(filtered[display_cols].sort_values("YRFI_pred", ascending=False), use_container_width=True)
else:
    st.warning("No predictions available for this date.")

# === Accuracy Summary ===
today_total = filtered.shape[0]
today_correct = (filtered["Correct"] == "✅").sum()
today_wrong = (filtered["Correct"] == "❌").sum()
cumulative = live[live["Game Date"] <= selected_date]
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
    st.metric("Total", today_total)

with col2:
    st.markdown("#### 🔁 Cumulative Accuracy")
    st.metric("Correct", cumulative_correct)
    st.metric("Incorrect", cumulative_wrong)
    st.metric("Total", cumulative_total)

# === Fireball Tier Performance
st.markdown("---")
st.subheader("🔥 Fireball Tier Performance")

tiers = ["🔥", "🔥🔥", "🔥🔥🔥", "🔥🔥🔥🔥", "🔥🔥🔥🔥🔥"]
fire_df = cumulative[cumulative["YRFI🔥"].isin(tiers)]
tier_stats = (
    fire_df.groupby("YRFI🔥")["Correct"]
    .value_counts().unstack(fill_value=0)
    .rename(columns={"✅": "Correct", "❌": "Incorrect"})
)
tier_stats["Total"] = tier_stats.sum(axis=1)
tier_stats["Accuracy %"] = (tier_stats["Correct"] / tier_stats["Total"].replace(0, 1) * 100).round(1)
tier_stats = tier_stats.reindex(tiers).fillna(0).astype(int)

st.dataframe(tier_stats, use_container_width=True)

# === Compact Summary
st.markdown("---")
st.subheader("🔥 Fireball Accuracy Summary (Compact View)")

def summarize_fireballs(df):
    summary = {}
    for tier in tiers:
        subset = df[df["YRFI🔥"] == tier]
        correct = (subset["Correct"] == "✅").sum()
        incorrect = (subset["Correct"] == "❌").sum()
        total = correct + incorrect
        summary[tier] = {"Correct": correct, "Incorrect": incorrect, "Total": total}
    return summary

daily_summary = summarize_fireballs(filtered)
rolling_summary = summarize_fireballs(cumulative)

summary_rows = []
for tier in tiers:
    summary_rows.append({
        "Tier": tier,
        "Daily Correct": daily_summary[tier]["Correct"],
        "Daily Incorrect": daily_summary[tier]["Incorrect"],
        "Daily Total": daily_summary[tier]["Total"],
        "Rolling Correct": rolling_summary[tier]["Correct"],
        "Rolling Incorrect": rolling_summary[tier]["Incorrect"],
        "Rolling Total": rolling_summary[tier]["Total"],
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df.set_index("Tier"), use_container_width=True)
