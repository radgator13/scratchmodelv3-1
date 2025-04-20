# yrfi_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/model_vs_inferred_yrfi_market.csv", parse_dates=["date"])
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.title("⚾ Filter Games")
min_edge, max_edge = st.sidebar.slider("Predicted Edge Range", -0.5, 0.5, (-0.1, 0.1), 0.01)
min_prob = st.sidebar.slider("Minimum YRFI Probability", 0.0, 1.0, 0.5, 0.01)
teams = sorted(set(df["home_team"]) | set(df["away_team"]))
selected_teams = st.sidebar.multiselect("Filter by Team", teams, default=teams)

# --- Filter Data ---
filtered = df[
    (df["predicted_edge"].between(min_edge, max_edge)) &
    (df["YRFI_Prob"] >= min_prob) &
    (df["home_team"].isin(selected_teams) | df["away_team"].isin(selected_teams))
]

# --- Main Display ---
st.title("🔥 YRFI Prediction Dashboard")
st.caption("Model vs. Fireball-Inferred Market Odds")

st.markdown(f"**{len(filtered)} games shown** | Sorted by `predicted_edge`")
filtered = filtered.sort_values(by="predicted_edge", ascending=False)

st.dataframe(
    filtered[[
        "date", "away_team", "home_team", "YRFI_Prob", "yrfi_odds",
        "implied_prob", "predicted_edge"
    ]].round(3),
    use_container_width=True
)

# --- Download Option ---
st.download_button(
    label="📥 Download CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_yrfi_predictions.csv",
    mime="text/csv"
)

# --- Tips ---
with st.expander("ℹ️ About this Dashboard"):
    st.markdown("""
    - **YRFI_Prob**: model-predicted probability of a run in the 1st inning
    - **yrfi_odds**: estimated market odds (based on 🔥 count)
    - **implied_prob**: implied probability from the odds
    - **predicted_edge**: model_prob - implied_prob
    """)
