import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# === Load matchups (from get_scores output)
box = pd.read_csv(DATA_DIR / "mlb_boxscores_cleaned.csv", parse_dates=["Game Date"])

# === Normalize team names and clean whitespace
box["Away Team"] = box["Away Team"].str.strip().str.title()
box["Home Team"] = box["Home Team"].str.strip().str.title()

# === Rename for consistency
box.rename(columns={"Game Date": "date"}, inplace=True)

# === Load enriched data from your main input source
ref = pd.read_csv(DATA_DIR / "yrfi_model_input_with_era.csv", parse_dates=["date"])

# === Match on team names and date, extract latest stats
latest = (
    ref.sort_values("date")
    .groupby(["away_team", "home_team"])
    .tail(1)
    .drop(columns=["yrfi", "date"])
    .reset_index(drop=True)
)

# === Merge enriched stats into today's box scores
merged = box.merge(
    latest,
    left_on=["Away Team", "Home Team"],
    right_on=["away_team", "home_team"],
    how="left"
)

# === Add additional columns
merged["day_of_week"] = merged["date"].dt.dayofweek
merged["same_hand"] = (merged["away_hand"] == merged["home_hand"]).astype(int)

# === Drop incomplete rows
required = ["away_hand", "home_hand", "away_era", "home_era"]
merged = merged.dropna(subset=required)

# === Save for modeling
output = DATA_DIR / "yrfi_model_input_live_with_era.csv"
merged.to_csv(output, index=False)
print(f"✅ Final live input saved to: {output}")
