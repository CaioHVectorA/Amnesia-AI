#!/usr/bin/env python3
"""
Ladder Replay & Performance Auditor for Amnesia-AI
Analyzes saved replay JSONs from ladder_rl directory and displays live win rates and trends.
"""

import os
import glob
import json
import time
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPLAYS_DIR = "data/replays/ladder_rl"

def analyze_replays():
    files = sorted(glob.glob(os.path.join(REPLAYS_DIR, "*.json")), key=os.path.getmtime)
    total = len(files)
    if total == 0:
        print("No replays found in", REPLAYS_DIR)
        return

    wins = 0
    losses = 0
    active = 0
    turns_list = []
    recent_results = []

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            
            bot = data.get("bot_username", "").strip().lower()
            winner = data.get("winner", "").strip().lower()
            turns = data.get("turns", 0)
            if turns > 0:
                turns_list.append(turns)

            if not winner:
                active += 1
                continue

            # Check if bot won
            if bot in winner or winner in bot:
                wins += 1
                recent_results.append(1)
            else:
                losses += 1
                recent_results.append(0)
        except Exception:
            pass

    decided = wins + losses
    overall_wr = (wins / decided * 100) if decided > 0 else 0.0
    avg_turns = sum(turns_list) / len(turns_list) if turns_list else 0.0

    print("=" * 65)
    print("🏆 AMNESIA-AI LADDER RL PERFORMANCE REPORT")
    print("=" * 65)
    print(f"📁 Total Saved Replay Files : {total}")
    print(f"⚔️ Decided Battles           : {decided}")
    print(f"✅ Victories                 : {wins}")
    print(f"❌ Defeats                   : {losses}")
    print(f"📈 Overall Win Rate          : {overall_wr:.1f}%")
    print(f"⏱️ Avg Turns / Game         : {avg_turns:.1f}")

    if len(recent_results) >= 10:
        last_10 = recent_results[-10:]
        wr_10 = sum(last_10) / len(last_10) * 100
        print(f"🔥 Last 10 Games Win Rate   : {wr_10:.1f}%")

    if len(recent_results) >= 20:
        last_20 = recent_results[-20:]
        wr_20 = sum(last_20) / len(last_20) * 100
        print(f"🚀 Last 20 Games Win Rate   : {wr_20:.1f}%")
    print("=" * 65)

if __name__ == "__main__":
    analyze_replays()
