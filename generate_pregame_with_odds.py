import pandas as pd
from pathlib import Path

# Load original file
DATA_DIR = Path("data")
df = pd.read_csv(DATA_DIR / "yrfi_predictions_pregame.csv")

# 🔥 Map fireball count to estimated odds
fireball_map = {
    1: 120,
    2: 105,
    3: -110,
    4: -125,
    5: -140
}

# Count number of fireballs in the YRFI🔥 column
df["fire_count"] = df["YRFI🔥"].astype(str).str.count("🔥")

# Estimate odds based on fireball intensity
df["yrfi_odds"] = df["fire_count"].map(fireball_map)

# Drop temp column
df.drop(columns=["fire_count"], inplace=True)

# Save updated file
output_path = DATA_DIR / "yrfi_predictions_pregame_with_odds.csv"
df.to_csv(output_path, index=False)

print(f"✅ File saved with estimated odds to: {output_path.resolve()}")
