# train_yrfi_xgb_with_era_and_team.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
df = pd.read_csv(DATA_DIR / "yrfi_model_input_with_era_and_team_rates.csv", parse_dates=["date"])

# --- Feature Engineering ---
df["day_of_week"] = df["date"].dt.dayofweek
df["same_hand"] = (df["away_hand"] == df["home_hand"]).astype(int)

# Drop rows with missing data
feature_cols = [
    "home_era", "away_era", "home_team_avg_1st", "away_team_avg_1st",
    "day_of_week", "same_hand"
]
df = df.dropna(subset=feature_cols)

# Set up features
categorical_cols = ["away_team", "home_team", "away_hand", "home_hand"]
numeric_cols = feature_cols

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(df[categorical_cols])
X_num = df[numeric_cols].values
X = np.hstack([X_cat, X_num])
y = df["yrfi"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train XGBoost
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

# Predict + evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("✅ Classification Report:")
print(classification_report(y_test, y_pred))
print("📈 ROC AUC Score:", round(roc_auc_score(y_test, y_proba), 4))

# Feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=15, height=0.5)
plt.title("Top 15 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()
