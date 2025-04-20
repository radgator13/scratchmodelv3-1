import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import argparse
import sys
import numpy as np


# === Config ===
API_KEY = "591b5b68a9802e9b588155794300ed47"
SPORT_KEY = "baseball_mlb"
MARKETS = "h2h,spreads,totals"
REGION = "us"
BOOKMAKER_PRIORITY = ["mybookieag", "fanduel", "draftkings", "betmgm"]

# === Paths ===
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ODDS_CSV = os.path.join(DATA_DIR, "mlb_odds_mybookie.csv")
BOXSCORE_CSV = os.path.join(DATA_DIR, "mlb_boxscores_cleaned.csv")
MERGED_CSV = os.path.join(DATA_DIR, "mlb_model_and_odds.csv")

# === Scrape Range ===
START_DATE = datetime.strptime("2025-03-27", "%Y-%m-%d")
END_DATE = datetime.today() + timedelta(days=1)

# === Normalize Merge Keys ===
def normalize_merge_keys(df):
    df["Game Date"] = pd.to_datetime(df["Game Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Home Team"] = df["Home Team"].astype(str).str.strip().str.title()
    df["Away Team"] = df["Away Team"].astype(str).str.strip().str.title()
    return df

# === Fetch Odds Function ===
def fetch_odds_for_day(date_obj):
    is_future = date_obj.date() > datetime.today().date()

    if is_future:
        url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
        params = {
            "apiKey": API_KEY,
            "markets": MARKETS,
            "regions": REGION,
            "oddsFormat": "decimal"
        }
    else:
        snapshot_time = date_obj.replace(hour=16).isoformat() + "Z"
        url = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT_KEY}/odds"
        params = {
            "apiKey": API_KEY,
            "markets": MARKETS,
            "regions": REGION,
            "oddsFormat": "decimal",
            "date": snapshot_time
        }

    print(f"📅 Fetching odds for {date_obj.strftime('%Y-%m-%d')}...")

    try:
        res = requests.get(url, params=params)
        if res.status_code != 200:
            print(f"⚠️ API Error {res.status_code}: {res.text}")
            return []

        snapshot = res.json()
        if not is_future:
            snapshot = snapshot.get("data", [])

        rows = []
        for game in snapshot:
            home = game["home_team"]
            away = game["away_team"]
            game_date = game["commence_time"][:10]

            for bk in BOOKMAKER_PRIORITY:
                book = next((b for b in game.get("bookmakers", []) if b["key"] == bk), None)
                if not book:
                    continue

                row = {
                    "Game Date": game_date,
                    "Home Team": home,
                    "Away Team": away,
                    "Bookmaker Used": book["title"]
                }

                for market in book.get("markets", []):
                    if market["key"] == "h2h":
                        for o in market["outcomes"]:
                            if o["name"] == home:
                                row["ML Home"] = o["price"]
                            elif o["name"] == away:
                                row["ML Away"] = o["price"]
                    elif market["key"] == "spreads":
                        for o in market["outcomes"]:
                            if o["name"] == home:
                                row["Spread Home"] = o["point"]
                                row["Spread Home Odds"] = o["price"]
                            elif o["name"] == away:
                                row["Spread Away"] = o["point"]
                                row["Spread Away Odds"] = o["price"]
                    elif market["key"] == "totals":
                        for o in market["outcomes"]:
                            if "Over" in o["name"]:
                                row["Total"] = o["point"]
                                row["Over Odds"] = o["price"]
                            elif "Under" in o["name"]:
                                row["Under Odds"] = o["price"]

                rows.append(row)
                break
        return rows
    except Exception as e:
        print(f"❌ Error on {date_obj.strftime('%Y-%m-%d')}: {e}")
        return []

# === Scrape Range ===
def scrape_range(start_date, end_date, update_existing=False):
    columns = [
        "Game Date", "Home Team", "Away Team", "Bookmaker Used",
        "ML Home", "ML Away", "Spread Home", "Spread Home Odds",
        "Spread Away", "Spread Away Odds", "Total", "Over Odds", "Under Odds"
    ]

    if os.path.exists(ODDS_CSV):
        existing_df = pd.read_csv(ODDS_CSV)
    else:
        print("🆕 No odds file found. Starting new.")
        existing_df = pd.DataFrame(columns=columns)

    all_rows = []
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        if not update_existing and date_str in existing_df["Game Date"].astype(str).unique():
            print(f"⏭ Skipping {date_str} (already exists)")
        else:
            rows = fetch_odds_for_day(current)
            if rows:
                all_rows.extend(rows)
            else:
                print(f"⚠️ No odds found for {date_str}")
            time.sleep(1.25)
        current += timedelta(days=1)

    if all_rows:
        new_df = pd.DataFrame(all_rows)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["Game Date", "Home Team", "Away Team"], keep="last", inplace=True)
        combined.to_csv(ODDS_CSV, index=False)
        print(f"\n✅ Updated odds saved to {ODDS_CSV} ({len(combined)} total rows)")
    else:
        print("❌ No new odds scraped.")

# === Merge Scores ===
def merge_with_model_results():
    print("🔗 Merging odds and scores...")

    if not os.path.exists(ODDS_CSV):
        print("⚠️ Missing odds file.")
        sys.exit(1)

    try:
        odds = normalize_merge_keys(pd.read_csv(ODDS_CSV))
    except Exception as e:
        print(f"❌ Failed to load odds: {e}")
        sys.exit(1)

    if os.path.exists(BOXSCORE_CSV):
        try:
            scores = normalize_merge_keys(pd.read_csv(BOXSCORE_CSV))
            scores = scores.drop_duplicates(subset=["Game Date", "Home Team", "Away Team"], keep="last")

            # Use a merge key to ensure proper join
            odds["merge_key"] = odds["Game Date"] + "|" + odds["Home Team"] + "|" + odds["Away Team"]
            scores["merge_key"] = scores["Game Date"] + "|" + scores["Home Team"] + "|" + scores["Away Team"]

            # Drop keys to avoid overwrite
            scores_clean = scores.drop(columns=["Game Date", "Home Team", "Away Team"])

            # ⚠️ DO NOT replace 0.0 with NaN here

            merged = pd.merge(odds, scores_clean, on="merge_key", how="left")
            merged[["Game Date", "Home Team", "Away Team"]] = merged["merge_key"].str.split("|", expand=True)
            merged.drop(columns=["merge_key"], inplace=True)

        except Exception as e:
            print(f"❌ Failed to merge with scores: {e}")
            sys.exit(1)
    else:
        merged = odds

    merged.to_csv(MERGED_CSV, index=False)
    print(f"✅ Merged dataset saved to {MERGED_CSV} ({len(merged)} rows)")

    # 🧪 Show some samples for debugging
    merged_check = merged[merged["Game Date"] == "2025-04-18"]
    print("\n🧪 Sample April 18 merged data:")
    print(merged_check[["Game Date", "Home Team", "Away Team", "Home Score", "Away Score"]].head())




# === Run ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-existing", action="store_true", help="Update odds for existing dates")
    args = parser.parse_args()

    print("🚀 Starting odds scrape from 2025-03-27 to today + 1...")
    scrape_range(START_DATE, END_DATE, update_existing=args.update_existing)
    merge_with_model_results()
