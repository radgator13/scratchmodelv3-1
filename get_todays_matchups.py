import pandas as pd
from datetime import datetime, timedelta

# === Config ===
input_file = "data/rotowire-projstarters.csv"
output_file = "data/today_matchups.csv"

# Get formatted date labels like "Sun 4/21" and "Mon 4/22"
def format_date_label(dt):
    return f"{dt.strftime('%a')} {dt.month}/{dt.day}"

today = datetime.today()
tomorrow = today + timedelta(days=1)

dates_to_pull = {
    format_date_label(today): today.strftime("%Y-%m-%d"),
    format_date_label(tomorrow): tomorrow.strftime("%Y-%m-%d")
}

# === Load and transform Rotowire data ===
df = pd.read_csv(input_file, header=0)
df_t = df.set_index(df.columns[0]).T.reset_index()
df_t.rename(columns={"index": "Date"}, inplace=True)

# === Extract matchups for today + tomorrow ===
matchups = []

for label, iso_date in dates_to_pull.items():
    daily_games = df_t[df_t["Date"].str.strip() == label]
    if daily_games.empty:
        print(f"⚠️ No games found for {label} in Rotowire file.")
        continue

    for team, game in daily_games.iloc[0].items():
        if team == "Date" or pd.isna(game):
            continue
        if " vs " in game:
            home = team
            away = game.split(" vs ")[1].split()[0]
        elif " @ " in game:
            away = team
            home = game.split(" @ ")[1].split()[0]
        else:
            continue  # skip OFF DAY or POSTPONED
        matchups.append({
            "Game Date": iso_date,
            "Away Team": away.upper(),
            "Home Team": home.upper()
        })

# === Save output ===
if matchups:
    pd.DataFrame(matchups).to_csv(output_file, index=False)
    print(f"✅ Saved {len(matchups)} matchups for today + tomorrow to {output_file}")
else:
    print("❌ No matchups generated. Check rotowire-projstarters.csv format.")
