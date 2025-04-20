# summarize_backtest_results.py
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

DATA_DIR = Path("data")
df = pd.read_csv(DATA_DIR / "yrfi_backtest_results_through_apr19.csv")

# Round probabilities to 3 decimals for display
df["yrfi_probability"] = df["yrfi_probability"].round(3)

# Basic counts
total_games = len(df)
yrfi_rate = df["yrfi"].mean()
accuracy = accuracy_score(df["yrfi"], df["yrfi_predicted"])
precision = precision_score(df["yrfi"], df["yrfi_predicted"], zero_division=0)
recall = recall_score(df["yrfi"], df["yrfi_predicted"], zero_division=0)
roc_auc = roc_auc_score(df["yrfi"], df["yrfi_probability"])

# Summary table
summary = pd.DataFrame({
    "Total Games": [total_games],
    "Actual YRFI Rate": [round(yrfi_rate, 3)],
    "Model Accuracy": [round(accuracy, 3)],
    "YRFI Precision": [round(precision, 3)],
    "YRFI Recall": [round(recall, 3)],
    "ROC AUC": [round(roc_auc, 3)]
})

# Save to CSV
summary_path = DATA_DIR / "yrfi_backtest_summary_through_apr19.csv"
summary.to_csv(summary_path, index=False)
print(f"✅ Summary saved to: {summary_path.resolve()}")
print(summary)
