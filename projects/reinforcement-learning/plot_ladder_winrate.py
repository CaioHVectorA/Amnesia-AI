#!/usr/bin/env python3
"""
Ladder Win Rate & PPO Progression Visualizer
============================================
Parses all recorded Showdown Blitz ladder battles and plots:
1. Cumulative Win Rate progression over time / games.
2. Rolling Win Rate (window = 20 games).
3. PPO Update checkpoints and turns accumulated.
"""

import os
import glob
import json
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt

def generate_ladder_plot(output_path: str, artifact_path: str = None):
    files = glob.glob("data/replays/ladder_rl/*.json")
    battles = []
    
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            log = d.get("log", "")
            bot = d.get("bot_username", "").strip().lower()
            winner = d.get("winner", "").strip().lower()
            if not winner:
                continue
            won = 1 if (bot in winner or winner in bot) else 0
            
            t_match = re.search(r"\|t:\|(\d+)", log)
            ts = int(t_match.group(1)) if t_match else int(os.path.getmtime(f))
            turns = d.get("turns", 0)
            battles.append({"ts": ts, "won": won, "turns": turns, "file": f})
        except Exception:
            pass

    battles.sort(key=lambda x: x["ts"])
    total_games = len(battles)
    if total_games == 0:
        print("[-] No valid battles found.")
        return

    game_numbers = np.arange(1, total_games + 1)
    wins = np.array([b["won"] for b in battles])
    turns = np.array([b["turns"] for b in battles])
    cum_turns = np.cumsum(turns)
    
    # Cumulative Win Rate
    cum_wins = np.cumsum(wins)
    cum_wr = (cum_wins / game_numbers) * 100.0
    
    # Rolling Win Rate (window = 20)
    window = 20
    rolling_wr = []
    for i in range(total_games):
        start_idx = max(0, i - window + 1)
        sub = wins[start_idx : i + 1]
        rolling_wr.append(np.mean(sub) * 100.0)
    rolling_wr = np.array(rolling_wr)

    # Estimate PPO Update checkpoints (~ every 250 turns)
    update_interval_turns = 250
    estimated_updates = cum_turns // update_interval_turns
    
    # Matplotlib styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), dpi=300, sharex=True, gridspec_kw={'height_ratios': [2.2, 1.0]})

    # Top Plot: Win Rates
    ax1.plot(game_numbers, rolling_wr, color="#e63946", linewidth=2.4, label=f"Rolling Win Rate ({window}-game window)")
    ax1.plot(game_numbers, cum_wr, color="#1d3557", linewidth=2.0, linestyle="--", label="Cumulative Win Rate")
    
    # Target / Reference lines
    ax1.axhline(y=21.5, color="#457b9d", linestyle=":", alpha=0.8, label=f"Final Overall WR ({cum_wr[-1]:.1f}%)")
    ax1.axhline(y=50.0, color="#2a9d8f", linestyle="-.", alpha=0.6, label="50% Target Benchmark")

    # Annotate peak rolling performance
    peak_idx = np.argmax(rolling_wr[15:]) + 15
    ax1.annotate(f"Peak: {rolling_wr[peak_idx]:.1f}%",
                 xy=(game_numbers[peak_idx], rolling_wr[peak_idx]),
                 xytext=(game_numbers[peak_idx] - 15, rolling_wr[peak_idx] + 8),
                 arrowprops=dict(facecolor="#e63946", shrink=0.08, width=1.5, headwidth=6),
                 fontweight="bold", color="#e63946", fontsize=10)

    ax1.set_title("Amnesia-AI: Online Ladder RL Win Rate & PPO Optimization Progress\n(Gen 9 Random Battle Blitz - 214 Ranked Battles)", fontsize=14, weight="bold", pad=15)
    ax1.set_ylabel("Win Rate (%)", fontsize=12, weight="bold")
    ax1.set_ylim(0, 60)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Bottom Plot: Cumulative Turns and PPO Updates
    ax2.plot(game_numbers, cum_turns, color="#2b2d42", linewidth=2.0, label="Cumulative Transitions / Turns")
    ax2.set_ylabel("Total Turns", fontsize=11, weight="bold")
    ax2.set_xlabel("Game Number (Chronological Order)", fontsize=12, weight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Secondary Y axis for PPO updates
    ax2_twin = ax2.twinx()
    ax2_twin.plot(game_numbers, estimated_updates, color="#8338ec", linewidth=2.0, linestyle="-.", label="PPO GPU Updates (~250 turns/batch)")
    ax2_twin.set_ylabel("PPO Updates", color="#8338ec", fontsize=11, weight="bold")
    ax2_twin.tick_params(axis="y", labelcolor="#8338ec")

    # Combined legend for bottom plot
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"[+] Saved ladder progression plot to: {output_path}")

    if artifact_path:
        os.makedirs(os.path.dirname(os.path.abspath(artifact_path)), exist_ok=True)
        plt.savefig(artifact_path, bbox_inches="tight", dpi=300)
        print(f"[+] Saved artifact copy to: {artifact_path}")
    plt.close()

if __name__ == "__main__":
    out_file = "projects/reinforcement-learning/ladder_winrate_progression.png"
    artifact_file = "/home/usuario/.gemini/antigravity/brain/388067c6-5104-418e-add4-880610759fc0/ladder_winrate_progression.png"
    generate_ladder_plot(out_file, artifact_file)
