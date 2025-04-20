import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# Load your fireball-enhanced file
df = pd.read_csv(DATA_DIR / "yrfi_predictions_pregame_with_odds.csv")
df["date"] = pd.to_datetime(df["Game Date"], errors="coerce")

# Clean team names
df["away_team"] = df["Away Team"].str.strip().str.title()
df["home_team"] = df["Home Team"].str.strip().str.title()

# Convert odds to implied probability
def odds_to_implied_prob(o):
    try:
        o = float(o)
        if o < 0:
            return abs(o) / (abs(o) + 100)
        else:
            return 100 / (o + 100)
    except:
        return None

df["implied_prob"] = df["yrfi_odds"].apply(odds_to_implied_prob)

# Calculate model edge
df["predicted_edge"] = (df["YRFI_Prob"] - df["implied_prob"]).round(3)

# Save output
output_path = DATA_DIR / "model_vs_inferred_yrfi_market.csv"
df.to_csv(output_path, index=False)

print(f"✅ Model vs inferred market saved to: {output_path.resolve()}")
print(df[[
    "date", "away_team", "home_team", "YRFI_Prob", "yrfi_odds", "implied_prob", "predicted_edge"
]].head())
