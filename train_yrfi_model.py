# train_yrfi_model.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

DATA_DIR = Path("data")
df = pd.read_csv(DATA_DIR / "yrfi_model_input.csv", parse_dates=["date"])

# Select features and target
categorical_cols = ["away_team", "home_team", "away_hand", "home_hand"]
X_cat = df[categorical_cols]
y = df["yrfi"]

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X_cat)

# Train/test split (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

print("📈 ROC AUC Score:", round(roc_auc_score(y_test, y_proba), 4))
