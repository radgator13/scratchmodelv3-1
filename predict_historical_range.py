# predict_historical_range.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Settings
DATA_DIR = Path("data")
MODEL_DIR = Path("model")

start_date = pd.Timestamp("2025-03-27")
end_date = pd.Timestamp("2025-04-20")

# Load model + encoder
model = joblib.load(MODEL_DIR / "yrfi_xgb_model.pkl")
encoder = joblib.load(MODEL_DIR / "yrfi_encoder.pkl")

# Load full dataset
df = pd.read_csv(DATA_DIR / "yrfi_model_input_with_era_and_team_rates.csv", parse_dates=["date"])

# Filter to date range
df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

# Add features
df["day_of_week"] = df["date"].dt.dayofweek
df["same_hand"] = (df["away_hand"] == df["home_hand"]).astype(int)

# Drop any rows with missing model features
features_needed = [
    "home_era", "away_era", "home_team_avg_1st", "away_team_avg_1st",
    "day_of_week", "same_hand"
]
categorical_cols = ["away_team", "home_team", "away_hand", "home_hand"]
df = df.dropna(subset=categorical_cols + features_needed)

# Encode
X_cat = encoder.transform(df[categorical_cols])
X_num = df[features_needed].values
X = np.hstack([X_cat, X_num])

# Predict
df["yrfi_probability"] = model.predict_proba(X)[:, 1]
df["yrfi_predicted"] = (df["yrfi_probability"] >= 0.5).astype(int)

# Output
output_path = DATA_DIR / "backtest_yrfi_predictions_mar27_to_apr20.csv"
df.to_csv(output_path, index=False)

print(f"✅ Backtest predictions saved to: {output_path.resolve()}")
print(df[[
    "date", "away_team", "home_team", "away_starter", "home_starter",
    "yrfi", "yrfi_probability", "yrfi_predicted"
]].round(3).head())
