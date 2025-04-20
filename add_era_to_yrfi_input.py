# extract_era_from_rotowire_blocked.py
import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path("data")
games = pd.read_csv(DATA_DIR / "yrfi_model_input.csv", parse_dates=["date"])
raw = pd.read_csv(DATA_DIR / "rotowire-projstarters.csv")

# Date columns from rotowire header
date_cols = raw.columns[1:]

# Create container for cleaned rows
records = []

TEAM_MAP = {
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

# Iterate through rows in groups of 3: [starter, game, stats]
for i in range(0, len(raw), 3):
    try:
        starter_row = raw.iloc[i]
        stats_row = raw.iloc[i+2]
    except IndexError:
        continue

    team_abbr = starter_row['Unnamed: 0']
    if team_abbr in ['game', 'stats']:
        continue

    full_team = TEAM_MAP.get(team_abbr)
    if not full_team:
        continue

    for col in date_cols:
        try:
            game_date = pd.to_datetime(col + " 2025", format="%a %m/%d %Y", errors="coerce")
            starter = starter_row[col]
            stats = stats_row[col]
        except Exception:
            continue

        if pd.isna(starter) or pd.isna(stats):
            continue

        # Clean starter name
        starter_clean = starter.split(" (")[0].strip()

        # Extract ERA
        match = re.search(r"(\d+\.\d+)\s*ERA", stats)
        era = float(match.group(1)) if match else None

        records.append({
            "date": game_date,
            "team": full_team,
            "starter_clean": starter_clean,
            "era": era
        })

# Create long DataFrame
era_df = pd.DataFrame(records)

# Merge into game data for both home and away
for side in ["home", "away"]:
    games[f"{side}_starter_clean"] = games[f"{side}_starter"].str.extract(r"^(.*?)\s\(")[0]

    games = pd.merge(
        games,
        era_df.rename(columns={
            "team": f"{side}_team",
            "starter_clean": f"{side}_starter_clean",
            "era": f"{side}_era"
        }),
        on=["date", f"{side}_team", f"{side}_starter_clean"],
        how="left"
    )

    print(f"🔍 {side}_era nulls after merge: {games[f'{side}_era'].isna().sum()}")

# Save updated data
out_path = DATA_DIR / "yrfi_model_input_with_era.csv"
games.to_csv(out_path, index=False)
print(f"\n✅ ERA values extracted and saved to: {out_path.resolve()}")
