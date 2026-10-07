"""
Replay Dataset Analytics & Inspection
=====================================
Analyzes the scraped Showdown replays in SQLite using pure standard library.
Computes:
- Elo / rating cluster distributions
- Average turns per bracket
- Forfeit rates
- Top winners and high-weight replay rankings
"""

import argparse
import os
import sqlite3
import sys

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def analyze(db_path: str = "data/replays/replays_metadata.sqlite"):
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*), AVG(winner_rating), AVG(turn_count), AVG(is_forfeit) FROM replays")
    total, avg_elo, avg_turns, avg_forfeit = cursor.fetchone()

    if not total:
        print("No replays found in database.")
        conn.close()
        return

    print("=" * 75)
    print("  AMNESIA-AI REPLAY DATASET SUMMARY")
    print("=" * 75)
    print(f"Total Replays:          {total:,}")
    print(f"Average Winner Elo:     {avg_elo:.1f}" if avg_elo else "Average Winner Elo:     N/A")
    print(f"Average Turn Count:     {avg_turns:.1f} turns")
    print(f"Overall Forfeit Rate:   {(avg_forfeit * 100):.1f}%\n")

    print("-" * 75)
    print(f"{'Rating Bracket':<18} | {'Count':<6} | {'Avg Elo':<8} | {'Avg Turns':<10} | {'Forfeit %':<10} | {'Avg Weight'}")
    print("-" * 75)

    cursor.execute("""
        SELECT rating_bracket, COUNT(*), AVG(winner_rating), AVG(turn_count), AVG(is_forfeit), AVG(weight)
        FROM replays
        GROUP BY rating_bracket
    """)
    rows = cursor.fetchall()
    tier_order = {"elite_1800plus": 0, "high_1650_1799": 1, "mid_1500_1649": 2, "low_1300_1499": 3, "sub1300": 4, "unrated": 5}
    rows.sort(key=lambda r: tier_order.get(r[0], 99))

    for bracket, count, b_elo, b_turns, b_forfeit, b_weight in rows:
        elo_str = f"{b_elo:.1f}" if b_elo else "N/A"
        forfeit_str = f"{(b_forfeit * 100):.1f}%"
        print(f"{bracket:<18} | {count:<6} | {elo_str:<8} | {b_turns:<10.1f} | {forfeit_str:<10} | {b_weight:.4f}")

    print("\n" + "-" * 75)
    print(" TOP 5 HIGHEST QUALITY REPLAYS (Highest BC Sample Weight)")
    print("-" * 75)
    cursor.execute("""
        SELECT id, rating_bracket, winner_name, winner_rating, turn_count, weight
        FROM replays
        ORDER BY weight DESC, winner_rating DESC
        LIMIT 5
    """)
    top_replays = cursor.fetchall()
    for r_id, bracket, winner, w_elo, turns, weight in top_replays:
        print(f"* ID: {r_id:<28} | Bracket: {bracket:<15} | Winner: {str(winner):<15} (Elo: {str(w_elo):<4}) | Turns: {turns:<2} | Weight: {weight:.4f}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze scraped replay dataset")
    parser.add_argument("--db", type=str, default="data/replays/replays_metadata.sqlite", help="Path to SQLite db")
    args = parser.parse_args()
    analyze(args.db)
