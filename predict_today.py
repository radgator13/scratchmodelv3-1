# predict_today.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

DATA_DIR = Path("data")
MODEL_DIR = Path("model")

# Load model and encoder
model = joblib.load(MODEL_DIR / "yrfi_xgb_model.pkl")
encoder = joblib.load(MODEL_DIR / "yrfi_encoder.pkl")

# Load today's matchups
df = pd.read_csv(DATA_DIR / "today_matchups.csv", parse_dates=["Game Date"])


# Create features
df["day_of_week"] = df["date"].dt.dayofweek
df["same_hand"] = (df["away_hand"] == df["home_hand"]).astype(int)

# Clean starter names
df["away_starter_clean"] = df["away_starter"].str.strip()
df["home_starter_clean"] = df["home_starter"].str.strip()

# Columns
categorical_cols = ["away_team", "home_team", "away_hand", "home_hand"]
numeric_cols = [
    "home_era", "away_era", "home_team_avg_1st", "away_team_avg_1st",
    "day_of_week", "same_hand"
]

# Drop incomplete rows
df = df.dropna(subset=categorical_cols + numeric_cols)

# Preprocess
X_cat = encoder.transform(df[categorical_cols])
X_num = df[numeric_cols].values
X = np.hstack([X_cat, X_num])

# Predict
df["yrfi_probability"] = model.predict_proba(X)[:, 1]
df["yrfi_predicted"] = (df["yrfi_probability"] >= 0.5).astype(int)

# Output
output_path = DATA_DIR / "today_yrfi_predictions.csv"
df.to_csv(output_path, index=False)

print(f"✅ Predictions saved to: {output_path.resolve()}")
print(df[[
    "date", "away_team", "home_team", "away_starter", "home_starter",
    "yrfi_probability", "yrfi_predicted"
]].round(3).head())
