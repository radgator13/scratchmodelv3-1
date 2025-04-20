import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
main_file = DATA_DIR / "yrfi_model_input_with_era_and_team_rates.csv"
live_file = DATA_DIR / "yrfi_model_input_live_with_era.csv"

# === Load both files
main_df = pd.read_csv(main_file, parse_dates=["date"])
live_df = pd.read_csv(live_file, parse_dates=["date"])

# Ensure datetime type
main_df["date"] = pd.to_datetime(main_df["date"])
live_df["date"] = pd.to_datetime(live_df["date"])

# === Identify unique matchup keys (date + teams)
match_keys = ["date", "away_team", "home_team"]

# Perform left merge to find new matchups
merged = live_df.merge(main_df[match_keys], on=match_keys, how="left", indicator=True)
new_rows = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

# === Append new rows if found
if new_rows.empty:
    print("ℹ️ No new matchups to append. Everything is current.")
else:
    combined = pd.concat([main_df, new_rows], ignore_index=True)
    combined.to_csv(main_file, index=False)
    print(f"✅ Appended {len(new_rows)} new matchups to {main_file.name}")
    print("📋 Sample new rows:")
    print(new_rows[match_keys + ['home_era', 'away_era']].head())
