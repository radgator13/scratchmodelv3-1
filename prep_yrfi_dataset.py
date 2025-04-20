# prep_yrfi_dataset.py
import pandas as pd
from pathlib import Path
import re

DATA_DIR = Path("data")
input_path = DATA_DIR / "boxscores_with_starters.csv"
output_path = DATA_DIR / "yrfi_model_input.csv"

# Load data
df = pd.read_csv(input_path, parse_dates=["date"])

# Drop rows with missing or placeholder starters
df = df.dropna(subset=["home_starter", "away_starter"])
df = df[~df["home_starter"].str.contains("POSTPONED|---", na=False)]
df = df[~df["away_starter"].str.contains("POSTPONED|---", na=False)]

# Extract handedness using regex
def extract_hand(s):
    match = re.search(r"\((L|R)\)", s)
    return match.group(1) if match else None

df["away_hand"] = df["away_starter"].apply(extract_hand)
df["home_hand"] = df["home_starter"].apply(extract_hand)

# Drop rows where handedness wasn't found (just to keep it clean)
df = df.dropna(subset=["away_hand", "home_hand"])

# Create YRFI label
df["yrfi"] = ((df["Away 1st"] > 0) | (df["Home 1st"] > 0)).astype(int)

# Remove any duplicate games
df = df.drop_duplicates(subset=["date", "home_team", "away_team"])

# Final output
model_df = df[[
    "date", "away_team", "home_team",
    "away_starter", "away_hand",
    "home_starter", "home_hand",
    "Away 1st", "Home 1st", "yrfi"
]]

# Save output
model_df.to_csv(output_path, index=False)
print(f"✅ YRFI model input saved to: {output_path.resolve()}")
print("🧪 Preview:")
print(model_df.head())
