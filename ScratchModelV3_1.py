import os
from datetime import datetime

print("🚀 Starting ScratchModelV3.1 Pipeline...")

# === STEP 1: Generate matchups from Rotowire for today + tomorrow
print("📅 Step 1: Building matchups for today + tomorrow...")
os.system("python get_todays_matchups.py")

# === STEP 2: Build input features with odds
print("📊 Step 2: Generating features with market odds...")
os.system("python generate_pregame_with_odds.py")

# === STEP 3: Run predictions for all available matchups
print("🧠 Step 3: Running predictions...")
os.system("python predict_today.py")

# === STEP 4: Force Git to always commit (touch .last_push.txt)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
touch_file = "data/.last_push.txt"
with open(touch_file, "w") as f:
    f.write(f"Last push: {timestamp}\n")

# === STEP 5: Git commit + push
print("📤 Step 4: Committing and pushing updates to GitHub...")
os.system("git add .")  # Adds everything (data, .txt, .py, .csv, etc.)
os.system(f'git commit -m "🤖 Auto push from pipeline @ {timestamp}"')
os.system("git push origin main")

# === STEP 6: Launch dashboard (local only)
print("📈 Step 5: Launching Streamlit dashboard...")
os.system("start streamlit run yrfi_dashboard.py")  # For Windows

print("✅ ScratchModelV3.1 pipeline complete.")
