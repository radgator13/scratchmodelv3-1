import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

# === Setup paths ===
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "yrfi_xgb_model.pkl"
ENCODER_PATH = MODEL_DIR / "yrfi_encoder.pkl"

# === Load data ===
df = pd.read_csv(DATA_DIR / "today_matchups.csv", parse_dates=["Game Date"])
df["day_of_week"] = df["Game Date"].dt.dayofweek

# === Optional: Build same_hand flag
if "away_hand" in df.columns and "home_hand" in df.columns:
    df["same_hand"] = (df["away_hand"] == df["home_hand"]).astype(int)
else:
    df["same_hand"] = 0

# === Define feature sets
categorical_cols = ["away_team", "home_team", "away_hand", "home_hand"]
numeric_cols = [
    "home_era", "away_era", "home_team_avg_1st", "away_team_avg_1st",
    "day_of_week", "same_hand"
]

# Drop any rows missing required columns
required_cols = numeric_cols + categorical_cols
df = df.dropna(subset=required_cols)

# === One-hot encode categoricals
encoder = joblib.load(ENCODER_PATH)
X_cat = encoder.transform(df[categorical_cols])
X_num = df[numeric_cols].values
X = np.hstack([X_cat, X_num])

# === Load model
model = joblib.load(MODEL_PATH)

# === Predict
df["YRFI_Prob"] = model.predict_proba(X)[:, 1]
df["NRFI_Prob"] = 1 - df["YRFI_Prob"]

# === Fireballs
def to_fireballs(p):
    if p >= 0.80: return "🔥🔥🔥🔥🔥"
    elif p >= 0.60: return "🔥🔥🔥🔥"
    elif p >= 0.40: return "🔥🔥🔥"
    elif p >= 0.20: return "🔥🔥"
    else: return "🔥"

df["YRFI🔥"] = df["YRFI_Prob"].apply(to_fireballs)
df["NRFI🔥"] = df["NRFI_Prob"].apply(to_fireballs)

# === Save output
output_path = DATA_DIR / "yrfi_predictions_pregame_with_odds.csv"
df.to_csv(output_path, index=False)
print(f"✅ Saved predictions to {output_path}")
