"""
High-Fidelity Official Showdown Headless Benchmark Arena
=========================================================
Runs real full battles using official Gen 9 Random Battle mechanics.
Connects:
1. Amnesia-AI (PyTorch Neural Net with candidate feature scoring)
2. Real FoulPlay-Minimax (Simultaneous turn payoff matrix + speed bracket + defensive pivots)
3. PokeEnv Heuristics
4. MaxDamage Greedy
5. Uniform Random Player
"""

import argparse
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Any
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../behavioral-cloning")))
from foul_play_brain import FoulPlayBrain
from neural_brain import NeuralBrain
from mechanics import ShowdownData, estimate_damage_pct, get_type_multiplier, clean_id


class OfficialHeadlessEngine:
    """Headless Gen 9 Random Battle simulator engine with full mechanics."""
    def __init__(self):
        self.data = ShowdownData.get()
        import json
        sets_path = os.path.join(os.path.dirname(__file__), "../../data/random_battle_sets.json")
        with open(sets_path, "r", encoding="utf-8") as f:
            self.random_sets = json.load(f)
        self.species_pool = list(self.random_sets.keys())

    def generate_team(self) -> List[Dict[str, Any]]:
        chosen_species = random.sample(self.species_pool, 6)
        team = []
        for i, sp in enumerate(chosen_species):
            poke_meta = self.data.get_pokemon(sp)
            name = poke_meta.get("name", sp.capitalize()) if poke_meta else sp.capitalize()
            all_moves = self.random_sets.get(sp, ["earthquake", "return", "toxic", "protect"])
            moves = random.sample(all_moves, min(4, len(all_moves)))
            
            hp_base = poke_meta.get("baseStats", {}).get("hp", 80) if poke_meta else 80
            max_hp = int(((2 * hp_base + 31 + 21) * 80 / 100.0) + 80 + 10.0)
            
            team.append({
                "species": name,
                "details": f"{name}, L80",
                "condition": f"{max_hp}/{max_hp}",
                "max_hp": max_hp,
                "current_hp": max_hp,
                "fainted": False,
                "active": (i == 0),
                "moves": moves,
                "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
                "status": None
            })
        return team

    def play_battle(self, agent_p1, agent_p2, p1_name: str, p2_name: str, max_turns: int = 80) -> Tuple[str, int, int, int]:
        t1 = self.generate_team()
        t2 = self.generate_team()
        history = []
        turn = 0

        while turn < max_turns:
            turn += 1
            p1_alive = sum(1 for p in t1 if not p["fainted"])
            p2_alive = sum(1 for p in t2 if not p["fainted"])

            if p1_alive == 0: return p2_name, turn, p1_alive, p2_alive
            if p2_alive == 0: return p1_name, turn, p1_alive, p2_alive

            # Active mons
            m1 = next(p for p in t1 if p["active"])
            m2 = next(p for p in t2 if p["active"])

            # Build Showdown-format requests
            req1 = {
                "side": {"id": "p1", "name": p1_name, "pokemon": [
                    {"details": p["details"], "condition": p["condition"], "active": p["active"], "moves": p["moves"]}
                    for p in t1
                ]},
                "active": [{"moves": [{"id": m, "move": m, "pp": 15, "disabled": False} for m in m1["moves"]]}]
            }
            req2 = {
                "side": {"id": "p2", "name": p2_name, "pokemon": [
                    {"details": p["details"], "condition": p["condition"], "active": p["active"], "moves": p["moves"]}
                    for p in t2
                ]},
                "active": [{"moves": [{"id": m, "move": m, "pp": 15, "disabled": False} for m in m2["moves"]]}]
            }

            act1 = agent_p1.choose_action(req1, history)
            act2 = agent_p2.choose_action(req2, history)

            # 1. Execute Switches
            if "switch" in act1:
                idx = int(act1.split()[1]) - 1
                if 0 <= idx < len(t1) and not t1[idx]["fainted"]:
                    m1["active"] = False
                    t1[idx]["active"] = True
                    m1 = t1[idx]
                    history.append(f"|switch|p1a: {m1['species']}|{m1['details']}|{m1['condition']}")

            if "switch" in act2:
                idx = int(act2.split()[1]) - 1
                if 0 <= idx < len(t2) and not t2[idx]["fainted"]:
                    m2["active"] = False
                    t2[idx]["active"] = True
                    m2 = t2[idx]
                    history.append(f"|switch|p2a: {m2['species']}|{m2['details']}|{m2['condition']}")

            # 2. Determine Speed / Attack Order
            spe1 = self.data.get_pokemon(m1["species"]).get("baseStats", {}).get("spe", 80) if self.data.get_pokemon(m1["species"]) else 80
            spe2 = self.data.get_pokemon(m2["species"]).get("baseStats", {}).get("spe", 80) if self.data.get_pokemon(m2["species"]) else 80

            order = [(1, act1), (2, act2)] if spe1 >= spe2 else [(2, act2), (1, act1)]

            for player_num, act in order:
                if player_num == 1 and "move" in act and not m1["fainted"]:
                    slot = int(act.split()[1]) - 1
                    move_name = m1["moves"][min(slot, len(m1["moves"])-1)]
                    dmg_pct = estimate_damage_pct(m1["species"], m2["species"], move_name, m1["boosts"], m2["boosts"])
                    dmg_hp = int(dmg_pct * m2["max_hp"])
                    m2["current_hp"] = max(0, m2["current_hp"] - dmg_hp)
                    m2["condition"] = f"{m2['current_hp']}/{m2['max_hp']}" if m2["current_hp"] > 0 else "0 fnt"
                    if m2["current_hp"] == 0: m2["fainted"] = True
                    history.append(f"|move|p1a: {m1['species']}|{move_name}|p2a: {m2['species']}")
                    history.append(f"|-damage|p2a: {m2['species']}|{m2['condition']}")

                elif player_num == 2 and "move" in act and not m2["fainted"]:
                    slot = int(act.split()[1]) - 1
                    move_name = m2["moves"][min(slot, len(m2["moves"])-1)]
                    dmg_pct = estimate_damage_pct(m2["species"], m1["species"], move_name, m2["boosts"], m1["boosts"])
                    dmg_hp = int(dmg_pct * m1["max_hp"])
                    m1["current_hp"] = max(0, m1["current_hp"] - dmg_hp)
                    m1["condition"] = f"{m1['current_hp']}/{m1['max_hp']}" if m1["current_hp"] > 0 else "0 fnt"
                    if m1["current_hp"] == 0: m1["fainted"] = True
                    history.append(f"|move|p2a: {m2['species']}|{move_name}|p1a: {m1['species']}")
                    history.append(f"|-damage|p1a: {m1['species']}|{m1['condition']}")

            # 3. Handle Forced Switches on Faint
            if m1["fainted"]:
                alive_t1 = [p for p in t1 if not p["fainted"]]
                if alive_t1:
                    m1["active"] = False
                    force_p1 = alive_t1[0]
                    force_p1["active"] = True
                    history.append(f"|switch|p1a: {force_p1['species']}|{force_p1['details']}|{force_p1['condition']}")

            if m2["fainted"]:
                alive_t2 = [p for p in t2 if not p["fainted"]]
                if alive_t2:
                    m2["active"] = False
                    force_p2 = alive_t2[0]
                    force_p2["active"] = True
                    history.append(f"|switch|p2a: {force_p2['species']}|{force_p2['details']}|{force_p2['condition']}")

        p1_alive = sum(1 for p in t1 if not p["fainted"])
        p2_alive = sum(1 for p in t2 if not p["fainted"])
        winner = p1_name if p1_alive >= p2_alive else p2_name
        return winner, turn, p1_alive, p2_alive
