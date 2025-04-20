﻿import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import re
import time
import os

def get_game_ids(date_obj):
    date_str = date_obj.strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date_str}"
    r = requests.get(url)
    events = r.json().get("events", [])
    return [{"gameId": e["id"], "date": date_obj.strftime("%Y-%m-%d")} for e in events]

def extract_boxscore(game_id, game_date):
    url = f"https://www.espn.com/mlb/boxscore/_/gameId/{game_id}"
    print(f"🌐 Scraping HTML: {url}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.content, "html.parser")

    team_names = soup.select("h2.ScoreCell__TeamName")
    if len(team_names) < 2:
        print("⚠️ Team names not found.")
        return None

    away_team = team_names[0].text.strip()
    home_team = team_names[1].text.strip()

    records = soup.select("div.Gamestrip__Record")
    away_record = records[0].text.strip().split(',')[0] if len(records) > 0 else ""
    home_record = records[1].text.strip().split(',')[0] if len(records) > 1 else ""

    scores = soup.select("div.Gamestrip__Score")
    away_runs = scores[0].get_text(strip=True) if len(scores) > 0 else ""
    home_runs = scores[1].get_text(strip=True) if len(scores) > 1 else ""

    away_1st = home_1st = 0

    try:
        linescore_table = soup.find("table", class_="Table Table--align-center")
        if not linescore_table:
            print("❌ Linescore table not found.")
            return {
                "Game Date": game_date,
                "Away Team": away_team,
                "Away Record": away_record,
                "Away Score": None,
                "Home Team": home_team,
                "Home Record": home_record,
                "Home Score": None,
                "Away 1st": 0,
                "Home 1st": 0
            }

        header_row = linescore_table.find("thead").find("tr")
        header_cells = header_row.find_all("th")
        headers = [cell.text.strip() for cell in header_cells]

        try:
            first_inning_index = headers.index("1")
        except ValueError:
            print("❌ '1' not found in headers:", headers)
            first_inning_index = None

        if first_inning_index is not None:
            rows = linescore_table.find("tbody").find_all("tr")
            if len(rows) >= 2:
                away_cells = rows[0].find_all("td")
                home_cells = rows[1].find_all("td")

                if first_inning_index < len(away_cells) and first_inning_index < len(home_cells):
                    away_1st_str = away_cells[first_inning_index].text.strip()
                    home_1st_str = home_cells[first_inning_index].text.strip()

                    away_1st = int(away_1st_str) if away_1st_str.isdigit() else 0
                    home_1st = int(home_1st_str) if home_1st_str.isdigit() else 0
                else:
                    print(f"⚠️ Index {first_inning_index} out of bounds for this row.")
            else:
                print("⚠️ Not enough team rows in linescore.")
    except Exception as e:
        print(f"⚠️ Error parsing inning data: {e}")

    print(f"✅ Parsed: {away_team} {away_1st} | {home_team} {home_1st}")

    return {
        "Game Date": game_date,
        "Away Team": away_team,
        "Away Record": away_record,
        "Away Score": re.sub(r"\D", "", away_runs),
        "Home Team": home_team,
        "Home Record": home_record,
        "Home Score": re.sub(r"\D", "", home_runs),
        "Away 1st": away_1st,
        "Home 1st": home_1st
    }

def scrape_range(start_date, end_date, output_file="data/mlb_boxscores_cleaned.csv"):
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        print(f"📄 Found existing file with {len(existing_df)} rows.")
    else:
        existing_df = pd.DataFrame()
        print("🆕 No previous file found. Starting fresh.")

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    new_rows = []

    while current <= end:
        games = get_game_ids(current)
        for game in games:
            try:
                row = extract_boxscore(game["gameId"], game["date"])
                if row:
                    new_rows.append(row)
            except Exception as e:
                print(f"❌ Error parsing {game['gameId']}: {e}")
            time.sleep(0.75)
        current += timedelta(days=1)

    if new_rows:
        new_df = pd.DataFrame(new_rows)

        if not existing_df.empty:
            merge_keys = ["Game Date", "Away Team", "Home Team"]
            existing_df = existing_df[
                ~existing_df[merge_keys].apply(tuple, axis=1).isin(
                    new_df[merge_keys].apply(tuple, axis=1)
                )
            ]

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.sort_values(by=["Game Date", "Home Team"], inplace=True)

        if 'Away 1st' in combined.columns and 'Home 1st' in combined.columns:
            combined['YRFI'] = ((combined['Away 1st'] + combined['Home 1st']) > 0).astype(int)
            print("✅ YRFI column created.")
        else:
            print("⚠️ Could not create YRFI column — missing 1st inning data.")

        combined.to_csv(output_file, index=False)
        print(f"\n✅ Updated and saved to {output_file} ({len(combined)} total rows)")
    else:
        print("ℹ️ No new games found to append.")

if __name__ == "__main__":
    today = datetime.today()
    start_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")  # now includes tomorrow
    print(f"🚀 Scraping boxscores for: {start_date} to {end_date}")
    scrape_range(start_date, end_date)
