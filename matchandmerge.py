# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

# Set data folder relative to script location
DATA_DIR = Path("data")

# Load boxscores
boxscores = pd.read_csv(DATA_DIR / "mlb_boxscores_cleaned.csv", parse_dates=["Game Date"])

# Load and reshape starters file (wide → long)
starters_raw = pd.read_csv(DATA_DIR / "rotowire-projstarters.csv")
starters_long = starters_raw.melt(id_vars=["Unnamed: 0"], var_name="Date", value_name="starter_name")
starters_long = starters_long.rename(columns={"Unnamed: 0": "Team"})

# Parse date strings into actual datetime
starters_long["Date"] = pd.to_datetime(starters_long["Date"] + " 2025", format="%a %m/%d %Y")

# Team abbreviation → full name mapping
TEAM_NAME_MAP = {
    'ARI': 'Arizona Diamondbacks', 'ATL': 'Atlanta Braves', 'BAL': 'Baltimore Orioles',
    'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs', 'CWS': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians', 'COL': 'Colorado Rockies',
    'DET': 'Detroit Tigers', 'HOU': 'Houston Astros', 'KC': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers', 'MIA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
    'NYY': 'New York Yankees', 'OAK': 'Athletics', 'PHI': 'Philadelphia Phillies',
    'PIT': 'Pittsburgh Pirates', 'SD': 'San Diego Padres', 'SEA': 'Seattle Mariners',
    'SF': 'San Francisco Giants', 'STL': 'St. Louis Cardinals', 'TB': 'Tampa Bay Rays',
    'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays', 'WSH': 'Washington Nationals'
}

# Apply team name mapping and drop junk rows
starters_long["team"] = starters_long["Team"].map(TEAM_NAME_MAP)
starters_long = starters_long.dropna(subset=["team"])

# Normalize columns for merging
boxscores = boxscores.rename(columns={
    "Game Date": "date", "Home Team": "home_team", "Away Team": "away_team"
})
starters_long = starters_long.rename(columns={"Date": "date"})

# Merge home starters
home = starters_long.rename(columns={"team": "home_team", "starter_name": "home_starter"})
merged = pd.merge(boxscores, home[["date", "home_team", "home_starter"]], on=["date", "home_team"], how="left")

# Merge away starters
away = starters_long.rename(columns={"team": "away_team", "starter_name": "away_starter"})
merged = pd.merge(merged, away[["date", "away_team", "away_starter"]], on=["date", "away_team"], how="left")

# Save merged output in /data folder
output_path = DATA_DIR / "boxscores_with_starters.csv"
merged.to_csv(output_path, index=False)

# Final info
print("✅ Merged file saved to:", output_path.resolve())
print("🔢 Total rows:", merged.shape[0])
print("❌ Rows missing a starter:",
      (merged['home_starter'].isna() | merged['away_starter'].isna()).sum())
print("🧪 Preview:")
print(merged[["date", "away_team", "away_starter", "home_team", "home_starter"]].head())
