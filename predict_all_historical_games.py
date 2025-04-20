# predict_all_historical_games.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Paths
DATA_DIR = Path("data")
MODEL_DIR = Path("model")

# Date range
today = pd.Timestamp("2025-04-19")
start_date = pd.Timestamp("2025-03-27")

# Load model and encoder
model = joblib.load(MODEL_DIR / "yrfi_xgb_model.pkl")
encoder = joblib.load(MODEL_DIR / "yrfi_encoder.pkl")

# Load enriched data with ERA and team 1st inning rates
df = pd.read_csv(DATA_DIR / "yrfi_model_input_with_era_and_team_rates.csv", parse_dates=["date"])

# Filter to historical range (up to today)
df = df[(df["date"] >= start_date) & (df["date"] <= today)].copy()

# Add modeling features
df["day_of_week"] = df["date"].dt.dayofweek
df["same_hand"] = (df["away_hand"] == df["home_hand"]).astype(int)

# Drop rows with missing model features
categorical_cols = ["away_team", "home_team", "away_hand", "home_hand"]
numeric_cols = [
    "home_era", "away_era", "home_team_avg_1st", "away_team_avg_1st",
    "day_of_week", "same_hand"
]
df = df.dropna(subset=categorical_cols + numeric_cols)

# Preprocess
X_cat = encoder.transform(df[categorical_cols])
X_num = df[numeric_cols].values
X = np.hstack([X_cat, X_num])

# Predict
df["yrfi_probability"] = model.predict_proba(X)[:, 1]
df["yrfi_predicted"] = (df["yrfi_probability"] >= 0.5).astype(int)

# Save results
output_path = DATA_DIR / "yrfi_backtest_results_through_apr19.csv"
df.to_csv(output_path, index=False)

print(f"✅ Backtest saved to: {output_path.resolve()}")
print(df[[
    "date", "away_team", "home_team", "away_starter", "home_starter",
    "yrfi", "yrfi_predicted", "yrfi_probability"
]].round(3).head())
