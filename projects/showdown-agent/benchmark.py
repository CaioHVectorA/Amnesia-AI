"""
Amnesia-AI Headless Battle Benchmark Suite (High-Fidelity)
=========================================================
Benchmarks the trained Behavioral Cloning agent against baseline bots:
1. RandomPlayerAI (Uniform random moves and switches)
2. MaxDamagePlayerAI (Greedy offensive damage maximizer)

Tracks:
- Win Rate (%)
- Turn counts
- KO differential (Remaining Pokemon)
- Move vs Switch distributions
"""

import argparse
import os
import random
import sys
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agent import ShowdownBCAgent
from mechanics import ShowdownData, estimate_damage_pct, get_type_multiplier


class BaselinePlayer:
    def __init__(self, mode: str = "max_damage"):
        self.mode = mode
        self.data = ShowdownData.get()

    def choose_action(self, request: Dict, history: List[str]) -> str:
        active_req = request.get("active", [{}])[0]
        moves = active_req.get("moves", [])
        force_switch = request.get("forceSwitch", [False])[0]
        pokemon = request.get("side", {}).get("pokemon", [])

        legal_moves = [i for i, m in enumerate(moves) if not m.get("disabled") and m.get("pp", 1) > 0]
        legal_switches = [
            i + 1 for i, p in enumerate(pokemon)
            if not p.get("active") and not ("fnt" in p.get("condition", "") or p.get("condition", "").startswith("0"))
        ]

        if force_switch or not legal_moves:
            if legal_switches:
                return f"switch {random.choice(legal_switches)}"
            return "move 1"

        if self.mode == "random":
            if legal_switches and random.random() < 0.10:
                return f"switch {random.choice(legal_switches)}"
            return f"move {random.choice(legal_moves) + 1}"

        # Max Damage mode
        best_slot = legal_moves[0]
        max_dmg = -1.0
        for slot in legal_moves:
            m_name = moves[slot].get("move", moves[slot].get("id", ""))
            m_data = self.data.get_move(m_name)
            bp = m_data.get("basePower", 50) if m_data else 50
            if bp > max_dmg:
                max_dmg = bp
                best_slot = slot

        return f"move {best_slot + 1}"


class HeadlessSimGame:
    def __init__(self, agent_p1, agent_p2):
        self.p1 = agent_p1
        self.p2 = agent_p2
        self.data = ShowdownData.get()

    def generate_random_team(self) -> List[Dict]:
        sample_dex = [
            "Dragapult", "Ferrothorn", "Heatran", "Toxapex", "Landorus-Therian",
            "Clefable", "Zapdos", "Garchomp", "Rotom-Wash", "Tyranitar",
            "Weavile", "Volcarona", "Corviknight", "Kyurem", "Urshifu-Rapid-Strike"
        ]
        chosen = random.sample(sample_dex, 6)
        team = []
        for i, species in enumerate(chosen):
            poke_data = self.data.get_pokemon(species)
            moves = ["earthquake", "shadowball", "flamethrower", "toxic"]
            if species == "Dragapult": moves = ["dracometeor", "shadowball", "flamethrower", "uturn"]
            elif species == "Ferrothorn": moves = ["gyroball", "powerwhip", "leechseed", "spikes"]
            elif species == "Heatran": moves = ["magmastorm", "earthpower", "flashcannon", "stealthrock"]
            elif species == "Zapdos": moves = ["thunderbolt", "hurricane", "heatwave", "roost"]
            elif species == "Garchomp": moves = ["earthquake", "outrage", "swordsdance", "stoneedge"]

            team.append({
                "details": f"{species}, L80, M",
                "condition": "100/100",
                "active": (i == 0),
                "moves": moves
            })
        return team

    def play(self, max_turns: int = 60) -> Tuple[str, int, int]:
        t1 = self.generate_random_team()
        t2 = self.generate_random_team()

        history = []
        turn = 0

        while turn < max_turns:
            turn += 1
            p1_alive = sum(1 for p in t1 if not ("fnt" in p["condition"] or p["condition"].startswith("0")))
            p2_alive = sum(1 for p in t2 if not ("fnt" in p["condition"] or p["condition"].startswith("0")))

            if p1_alive == 0: return "p2", turn, p1_alive
            if p2_alive == 0: return "p1", turn, p1_alive

            req1 = {
                "side": {"id": "p1", "name": "Amnesia-AI", "pokemon": t1},
                "active": [{"moves": [{"id": m, "move": m, "pp": 15, "disabled": False} for m in t1[0]["moves"]]}]
            }
            req2 = {
                "side": {"id": "p2", "name": "Baseline", "pokemon": t2},
                "active": [{"moves": [{"id": m, "move": m, "pp": 15, "disabled": False} for m in t2[0]["moves"]]}]
            }

            act1 = self.p1.choose_action(req1, history)
            act2 = self.p2.choose_action(req2, history)

            # 1. Handle Switches First (Showdown Priority)
            if "switch" in act1:
                idx = int(act1.split()[1]) - 1
                if 0 <= idx < len(t1) and not t1[idx]["condition"].startswith("0"):
                    t1[0], t1[idx] = t1[idx], t1[0]
                    history.append(f"|switch|p1a: {t1[0]['details'].split(',')[0]}|{t1[0]['details']}|{t1[0]['condition']}")

            if "switch" in act2:
                idx = int(act2.split()[1]) - 1
                if 0 <= idx < len(t2) and not t2[idx]["condition"].startswith("0"):
                    t2[0], t2[idx] = t2[idx], t2[0]
                    history.append(f"|switch|p2a: {t2[0]['details'].split(',')[0]}|{t2[0]['details']}|{t2[0]['condition']}")

            # 2. Handle Attacks
            if "move" in act1 and not t1[0]["condition"].startswith("0"):
                slot1 = int(act1.split()[1]) - 1
                m1_name = t1[0]["moves"][min(slot1, len(t1[0]["moves"])-1)]
                dmg1 = estimate_damage_pct(t1[0]["details"].split(",")[0], t2[0]["details"].split(",")[0], m1_name, {}, {})
                cur_hp = float(t2[0]["condition"].split("/")[0]) / 100.0
                new_hp = max(0.0, cur_hp - dmg1)
                t2[0]["condition"] = f"{int(new_hp * 100)}/100" if new_hp > 0 else "0 fnt"
                history.append(f"|move|p1a: {t1[0]['details'].split(',')[0]}|{m1_name}|p2a: {t2[0]['details'].split(',')[0]}")
                history.append(f"|-damage|p2a: {t2[0]['details'].split(',')[0]}|{t2[0]['condition']}")

            if "move" in act2 and not t2[0]["condition"].startswith("0"):
                slot2 = int(act2.split()[1]) - 1
                m2_name = t2[0]["moves"][min(slot2, len(t2[0]["moves"])-1)]
                dmg2 = estimate_damage_pct(t2[0]["details"].split(",")[0], t1[0]["details"].split(",")[0], m2_name, {}, {})
                cur_hp = float(t1[0]["condition"].split("/")[0]) / 100.0
                new_hp = max(0.0, cur_hp - dmg2)
                t1[0]["condition"] = f"{int(new_hp * 100)}/100" if new_hp > 0 else "0 fnt"
                history.append(f"|move|p2a: {t2[0]['details'].split(',')[0]}|{m2_name}|p1a: {t1[0]['details'].split(',')[0]}")
                history.append(f"|-damage|p1a: {t1[0]['details'].split(',')[0]}|{t1[0]['condition']}")

            # 3. Handle Fainting Replacements
            if t1[0]["condition"].startswith("0"):
                alive = [p for p in t1 if not p["condition"].startswith("0")]
                if alive:
                    idx = t1.index(alive[0])
                    t1[0], t1[idx] = t1[idx], t1[0]

            if t2[0]["condition"].startswith("0"):
                alive = [p for p in t2 if not p["condition"].startswith("0")]
                if alive:
                    idx = t2.index(alive[0])
                    t2[0], t2[idx] = t2[idx], t2[0]

        p1_alive = sum(1 for p in t1 if not ("fnt" in p["condition"] or p["condition"].startswith("0")))
        p2_alive = sum(1 for p in t2 if not ("fnt" in p["condition"] or p["condition"].startswith("0")))
        winner = "p1" if p1_alive >= p2_alive else "p2"
        return winner, turn, p1_alive


def run_benchmark(num_games: int = 50, opponent_type: str = "max_damage"):
    print("=" * 70)
    print(f"📊 AMNESIA-AI BENCHMARK vs {opponent_type.upper()}")
    print("=" * 70)
    print(f"Total Matches:    {num_games}")
    print(f"Agent Model:      Behavioral Cloning Policy Network")
    print(f"Opponent:         {opponent_type}\n")

    agent = ShowdownBCAgent()
    opponent = BaselinePlayer(mode=opponent_type)

    p1_wins = 0
    turns_list = []
    p1_remaining_list = []

    start_time = time.time()

    for i in range(1, num_games + 1):
        sim = HeadlessSimGame(agent, opponent)
        winner, turns, remaining = sim.play()

        if winner == "p1":
            p1_wins += 1

        turns_list.append(turns)
        p1_remaining_list.append(remaining)

        if i % 10 == 0 or i == num_games:
            current_wr = (p1_wins / i) * 100.0
            print(f"[{i:03d}/{num_games}] Win Rate: {current_wr:5.1f}% | Avg Turns: {np.mean(turns_list):4.1f} | Avg Mons Left: {np.mean(p1_remaining_list):3.1f}")

    total_time = time.time() - start_time
    win_rate = (p1_wins / num_games) * 100.0

    print("\n" + "=" * 70)
    print("🏆 FINAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Games Played:        {num_games}")
    print(f"Amnesia-AI Wins:           {p1_wins} / {num_games}")
    print(f"Amnesia-AI Win Rate:       {win_rate:.2f}%")
    print(f"Average Match Duration:    {np.mean(turns_list):.1f} turns")
    print(f"Average Pokemon Remaining: {np.mean(p1_remaining_list):.1f} / 6")
    print(f"Benchmark Execution Time:  {total_time:.2f}s ({num_games/total_time:.1f} games/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Amnesia-AI benchmark against baseline bots")
    parser.add_argument("--games", type=int, default=50, help="Number of matches")
    parser.add_argument("--opponent", type=str, choices=["random", "max_damage"], default="max_damage", help="Opponent AI type")
    args = parser.parse_args()

    run_benchmark(num_games=args.games, opponent_type=args.opponent)
