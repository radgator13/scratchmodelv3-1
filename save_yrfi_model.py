# save_yrfi_model.py
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np

DATA_DIR = Path("data")
df = pd.read_csv(DATA_DIR / "yrfi_model_input_with_era_and_team_rates.csv", parse_dates=["date"])

# Prep data (same as training script)
df["day_of_week"] = df["date"].dt.dayofweek
df["same_hand"] = (df["away_hand"] == df["home_hand"]).astype(int)

feature_cols = [
    "home_era", "away_era", "home_team_avg_1st", "away_team_avg_1st",
    "day_of_week", "same_hand"
]
df = df.dropna(subset=feature_cols)

categorical_cols = ["away_team", "home_team", "away_hand", "home_hand"]
numeric_cols = feature_cols

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(df[categorical_cols])
X_num = df[numeric_cols].values
X = np.hstack([X_cat, X_num])
y = df["yrfi"]

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model again (just to capture the latest)
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Save model + encoder
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODEL_DIR / "yrfi_xgb_model.pkl")
joblib.dump(encoder, MODEL_DIR / "yrfi_encoder.pkl")

print("✅ Model and encoder saved to /model/")
