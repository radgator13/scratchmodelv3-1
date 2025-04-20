import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path("data")

# Load model input for today/tomorrow
games = pd.read_csv(DATA_DIR / "yrfi_model_input_live.csv", parse_dates=["date"])
raw = pd.read_csv(DATA_DIR / "rotowire-projstarters.csv")

# Setup
date_cols = raw.columns[1:]
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

# Extract ERA records from rotowire
for i in range(0, len(raw), 3):
    try:
        starter_row = raw.iloc[i]
        stats_row = raw.iloc[i + 2]
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

        starter_clean = starter.split(" (")[0].strip()
        match = re.search(r"(\d+\.\d+)\s*ERA", stats)
        era = float(match.group(1)) if match else None

        records.append({
            "date": game_date,
            "team": full_team,
            "starter_clean": starter_clean,
            "era": era
        })

era_df = pd.DataFrame(records)

# Add starter_clean columns to games
for side in ["home", "away"]:
    games[f"{side}_starter_clean"] = games[f"{side}_starter"].str.extract(r"^(.*?)\s\(")[0]

    # Prepare merge columns
    games = pd.merge(
        games,
        era_df.rename(columns={
            "team": f"{side}_team",
            "starter_clean": f"{side}_starter_clean",
            "era": f"{side}_era"
        }),
        how="left",
        on=["date", f"{side}_team", f"{side}_starter_clean"]
    )

    # Check for successful merge
    if f"{side}_era" in games.columns:
        print(f"✅ {side}_era merged. Nulls: {games[f'{side}_era'].isna().sum()}")
    else:
        print(f"❌ Failed to merge {side}_era!")

# Save output
out_path = DATA_DIR / "yrfi_model_input_live_with_era.csv"
games.to_csv(out_path, index=False)
print(f"\n✅ ERA-enriched live input saved to: {out_path.resolve()}")
