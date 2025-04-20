# compare_model_to_predictions_file.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# Load your predictions + odds file
df = pd.read_csv(DATA_DIR / "yrfi_predictions_pregame.csv")
df["date"] = pd.to_datetime(df["Game Date"], errors="coerce")
df["away_team"] = df["Away Team"].str.strip().str.title()
df["home_team"] = df["Home Team"].str.strip().str.title()

# ✅ Ensure 'yrfi_odds' column exists
if "yrfi_odds" not in df.columns:
    raise ValueError("❌ 'yrfi_odds' column is missing from yrfi_predictions_pregame.csv!")

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
out_path = DATA_DIR / "yrfi_model_edge_vs_market.csv"
df.to_csv(out_path, index=False)

print(f"✅ Comparison complete — saved to: {out_path.resolve()}")
print(df[[
    "date", "away_team", "home_team", "YRFI_Prob", "yrfi_odds", "implied_prob", "predicted_edge"
]].round(3).head())
