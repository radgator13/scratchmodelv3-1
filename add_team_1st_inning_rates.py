# add_team_1st_inning_rates.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# Load boxscores
box = pd.read_csv(DATA_DIR / "mlb_boxscores_cleaned.csv", parse_dates=["Game Date"])
box = box.rename(columns={
    "Game Date": "date",
    "Home Team": "home_team",
    "Away Team": "away_team",
    "Home 1st": "home_1st",
    "Away 1st": "away_1st"
})

# Calculate team scoring rates as HOME
home_rates = (
    box.groupby("home_team")["home_1st"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "home_team_avg_1st", "count": "home_games"})
    .reset_index()
)

# Calculate team scoring rates as AWAY
away_rates = (
    box.groupby("away_team")["away_1st"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "away_team_avg_1st", "count": "away_games"})
    .reset_index()
)

# Load model input
games = pd.read_csv(DATA_DIR / "yrfi_model_input_with_era.csv", parse_dates=["date"])

# Merge in the scoring rates
games = games.merge(home_rates, left_on="home_team", right_on="home_team", how="left")
games = games.merge(away_rates, left_on="away_team", right_on="away_team", how="left")

# Final output
out_path = DATA_DIR / "yrfi_model_input_with_era_and_team_rates.csv"
games.to_csv(out_path, index=False)
print(f"✅ Team 1st inning rates added. Saved to: {out_path.resolve()}")

# Preview
print(games[[
    "home_team", "home_team_avg_1st", "away_team", "away_team_avg_1st"
]].head())
