import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
odds = pd.read_csv(DATA_DIR / "yrfi_predictions_pregame.csv")
print("📋 Predictions Pregame columns:")
print(odds.columns.tolist())
