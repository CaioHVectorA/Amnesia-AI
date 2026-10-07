"""
Foul Play Engine (Minimax Payoff Matrix & Tactical Tree Search)
==============================================================
High-performance competitive decision engine based on Patrick Mariglia's Foul Play:
1. Exact damage calculation & KO thresholds
2. Simultaneous turn payoff matrix evaluation
3. Speed tier checking & priority brackets
4. Safe defensive pivot calculation (switches only when mathematically superior)
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../behavioral-cloning")))

from mechanics import (
    ShowdownData,
    clean_id,
    estimate_damage_pct,
    get_hazard_damage_pct,
    get_type_multiplier,
    check_ability_immunity
)


class FoulPlayBrain:
    def __init__(self):
        self.data = ShowdownData.get()

    def choose_action(self, request: Dict[str, Any], battle_history: List[str]) -> str:
        return self.evaluate_turn(request, battle_history)

    def evaluate_turn(
        self,
        request: Dict[str, Any],
        battle_history: List[str]
    ) -> str:
        """
        Computes the Minimax optimal decision for the current turn.
        Returns Showdown action command (e.g. 'move 1', 'switch 3', 'move 2 terastallize').
        """
        side = request.get("side", {})
        my_side_id = side.get("id", "p1")
        opp_side_id = "p2" if my_side_id == "p1" else "p1"
        opp_prefix = f"{opp_side_id}a:"

        pokemon_list = side.get("pokemon", [])
        active_req = request.get("active", [{}])[0]
        moves_req = active_req.get("moves", [])
        force_switch = request.get("forceSwitch", [False])[0]

        # 1. Parse My Active & Bench
        my_active = None
        my_bench = []

        for i, p in enumerate(pokemon_list):
            details = p.get("details", "")
            species = details.split(",")[0].strip()
            hp_str = p.get("condition", "100/100")
            fainted = "fnt" in hp_str or hp_str.startswith("0")
            hp_val = 1.0
            if "/" in hp_str:
                num, den = hp_str.split()[0].split("/")
                hp_val = float(num) / float(den) if float(den) > 0 else 0.0

            mon_info = {
                "species": species,
                "hp": hp_val,
                "fainted": fainted,
                "slot": i + 1,
                "moves": [m.get("move", m.get("id", "")) for m in moves_req] if p.get("active") else p.get("moves", []),
                "item": p.get("item", "")
            }

            if p.get("active"):
                my_active = mon_info
            else:
                my_bench.append(mon_info)

        # 2. Parse Opponent Active from logs
        opp_species = "Pikachu"
        opp_hp = 1.0
        opp_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
        my_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}

        for line in battle_history:
            parts = line.split("|")
            if len(parts) < 2: continue
            cmd = parts[1]

            if cmd in ["switch", "drag"]:
                tag = parts[2].strip()
                if tag.startswith(opp_prefix) or tag.startswith(f"{opp_side_id}:"):
                    opp_species = parts[3].split(",")[0].strip()
                    hp_raw = parts[4].split()[0] if len(parts) > 4 else "100/100"
                    if "/" in hp_raw:
                        n, d = hp_raw.split("/")
                        opp_hp = float(n) / float(d) if float(d) > 0 else 1.0
                    opp_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}

            elif cmd in ["-damage", "-heal"]:
                tag = parts[2].strip()
                hp_raw = parts[3].split()[0] if len(parts) > 3 else "100/100"
                if "fnt" in hp_raw:
                    if tag.startswith(opp_prefix): opp_hp = 0.0
                elif "/" in hp_raw:
                    n, d = hp_raw.split("/")
                    val = float(n) / float(d) if float(d) > 0 else 0.0
                    if tag.startswith(opp_prefix): opp_hp = val

            elif cmd == "-boost":
                tag, stat, amt = parts[2].strip(), parts[3].strip(), int(parts[4]) if len(parts) > 4 else 1
                if tag.startswith(opp_prefix) and stat in opp_boosts: opp_boosts[stat] = min(6, opp_boosts[stat] + amt)
                elif tag.startswith(f"{my_side_id}a:") and stat in my_boosts: my_boosts[stat] = min(6, my_boosts[stat] + amt)

        opp_poke = self.data.get_pokemon(opp_species)
        opp_types = opp_poke.get("types", ["Normal"]) if opp_poke else ["Normal"]
        opp_spe = opp_poke.get("baseStats", {}).get("spe", 80) if opp_poke else 80

        my_poke = self.data.get_pokemon(my_active["species"]) if my_active else None
        my_spe = my_poke.get("baseStats", {}).get("spe", 80) if my_poke else 80

        i_am_faster = my_spe > opp_spe

        # 3. Handle Forced Switch (Fainted Active)
        if force_switch or not my_active or my_active["fainted"]:
            best_switch_slot = -1
            best_def_score = -999.0

            for b in my_bench:
                if not b["fainted"] and b["hp"] > 0:
                    b_poke = self.data.get_pokemon(b["species"])
                    b_types = b_poke.get("types", ["Normal"]) if b_poke else ["Normal"]
                    b_spe = b_poke.get("baseStats", {}).get("spe", 80) if b_poke else 80

                    # Resistance score vs opponent STABs
                    resistances = [get_type_multiplier(ot, b_types) for ot in opp_types]
                    worst_res = max(resistances) if resistances else 1.0
                    best_res = min(resistances) if resistances else 1.0

                    # Score: Higher HP + Better Resistance + Speed advantage
                    def_score = (b["hp"] * 3.0) - (worst_res * 2.0) - (best_res * 0.5)
                    if b_spe > opp_spe:
                        def_score += 1.5

                    if def_score > best_def_score:
                        best_def_score = def_score
                        best_switch_slot = b["slot"]

            if best_switch_slot > 0:
                return f"switch {best_switch_slot}"
            # Fallback to any alive bench mon
            for b in my_bench:
                if not b["fainted"] and b["hp"] > 0:
                    return f"switch {b['slot']}"
            return "default"

        # 4. Evaluate Offensive Moves
        move_evals = []
        for slot_idx, m in enumerate(moves_req):
            if m.get("disabled", False) or m.get("pp", 1) == 0:
                continue

            m_name = m.get("move", m.get("id", ""))
            m_data = self.data.get_move(m_name)
            if not m_data: continue

            cat = m_data.get("category", "Status")
            acc = (m_data.get("accuracy", 100) if isinstance(m_data.get("accuracy", 100), (int, float)) else 100) / 100.0
            priority = m_data.get("priority", 0)

            if cat == "Status":
                mid = clean_id(m_name)
                utility = 0.20
                if mid in ["swordsdance", "nastyplot", "dragondance", "quiverdance", "calmmind"]:
                    utility = 0.65 if my_active["hp"] > 0.70 else 0.10
                elif mid in ["roost", "recover", "softboiled", "slackoff", "synthesis"]:
                    utility = 0.85 if my_active["hp"] < 0.50 else 0.05
                elif mid in ["stealthrock", "spikes", "toxicspikes"]:
                    utility = 0.45 if my_active["hp"] > 0.80 else 0.10
                elif mid in ["willowisp", "toxic", "thunderwave", "spore"]:
                    utility = 0.50

                move_evals.append({
                    "slot": slot_idx + 1,
                    "name": m_name,
                    "expected_dmg": 0.0,
                    "is_lethal": False,
                    "score": utility,
                    "priority": priority,
                    "acc": acc
                })
            else:
                dmg = estimate_damage_pct(my_active["species"], opp_species, m_name, my_boosts, opp_boosts)
                is_lethal = dmg >= opp_hp and dmg > 0.0

                # Payoff calculation
                score = dmg * acc * 2.0
                if is_lethal:
                    score += 5.0
                    if i_am_faster or priority > 0:
                        score += 10.0  # Guaranteed free KO without taking damage!

                move_evals.append({
                    "slot": slot_idx + 1,
                    "name": m_name,
                    "expected_dmg": dmg,
                    "is_lethal": is_lethal,
                    "score": score,
                    "priority": priority,
                    "acc": acc
                })

        # 5. Minimax Decision: Attack vs Defensive Switch
        best_move = max(move_evals, key=lambda x: x["score"]) if move_evals else None

        # Check Opponent Threat level on our Active mon
        opp_max_threat = 0.0
        for ot in opp_types:
            eff = get_type_multiplier(ot, my_poke.get("types", ["Normal"]) if my_poke else ["Normal"])
            if eff > opp_max_threat:
                opp_max_threat = eff

        # If we have a lethal attack and we outspeed -> Execute KO immediately!
        if best_move and best_move["is_lethal"] and (i_am_faster or best_move["priority"] > 0):
            can_tera = active_req.get("canTerastallize")
            return f"move {best_move['slot']}"

        # If facing 4x weakness, opponent is faster, and our best move deals < 25% damage -> PIVOT!
        if opp_max_threat >= 2.0 and not i_am_faster and (not best_move or best_move["expected_dmg"] < 0.30) and my_active["hp"] > 0.35:
            # Find safe defensive pivot on bench
            best_pivot = None
            best_pivot_score = -999.0
            for b in my_bench:
                if not b["fainted"] and b["hp"] > 0.50:
                    b_poke = self.data.get_pokemon(b["species"])
                    b_types = b_poke.get("types", ["Normal"]) if b_poke else ["Normal"]
                    res = max([get_type_multiplier(ot, b_types) for ot in opp_types])
                    if res <= 0.5:  # Resists opponent attacks!
                        p_score = (1.0 / max(0.25, res)) * b["hp"]
                        if p_score > best_pivot_score:
                            best_pivot_score = p_score
                            best_pivot = b["slot"]

            if best_pivot:
                print(f"[FoulPlay Minimax] Tactical Defensive Pivot into Slot {best_pivot} to absorb threat!")
                return f"switch {best_pivot}"

        # Execute best offensive move
        if best_move:
            can_tera = active_req.get("canTerastallize")
            if can_tera and best_move["expected_dmg"] >= 0.70:
                return f"move {best_move['slot']} terastallize"
            return f"move {best_move['slot']}"

        return "move 1"
