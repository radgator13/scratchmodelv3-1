import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# === Load today's matchups (from Rotowire)
matchups = pd.read_csv(DATA_DIR / "today_matchups.csv", parse_dates=["Game Date"])

# === Load your training feature data (with ERA, hands, etc.)
reference = pd.read_csv(DATA_DIR / "yrfi_model_input_with_era_and_team_rates.csv", parse_dates=["date"])

# === Get latest known values per team/hand
latest = (
    reference.sort_values("date")
    .groupby(["away_team", "home_team"])
    .tail(1)
    .reset_index(drop=True)
)

# === Merge team/pitcher features into today's matchups
merged = matchups.merge(
    latest.drop(columns=["date", "yrfi"]),  # drop target + date
    left_on=["Away Team", "Home Team"],
    right_on=["away_team", "home_team"],
    how="left"
)

# === Feature engineering
merged["day_of_week"] = merged["Game Date"].dt.dayofweek
if "away_hand" in merged.columns and "home_hand" in merged.columns:
    merged["same_hand"] = (merged["away_hand"] == merged["home_hand"]).astype(int)
else:
    merged["same_hand"] = 0

# === Drop incomplete games
required_cols = [
    "home_era", "away_era", "home_team_avg_1st", "away_team_avg_1st",
    "away_team", "home_team", "away_hand", "home_hand"
]
merged = merged.dropna(subset=required_cols)

# === Rename for consistency
merged.rename(columns={"Game Date": "date"}, inplace=True)

# === Save it
output_path = DATA_DIR / "yrfi_model_input_live.csv"
merged.to_csv(output_path, index=False)
print(f"✅ Saved live model input to: {output_path}")
