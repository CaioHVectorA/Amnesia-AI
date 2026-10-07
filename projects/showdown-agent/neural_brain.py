"""
Slot-Agnostic Neural Behavioral Cloning Brain & Dynamic Decision Engine
========================================================================
1. Candidate-Centric Action Scoring: Evaluates actions purely by intrinsic mechanics
   (Base Power, STAB, Type Effectiveness, KO-Threshold, Priority, Speed, Status Utility).
2. Slot Deserialization: Maps chosen high-value action to physical Showdown slot dynamically.
3. Anti-Loop Mechanism: Tracks previous turns and forbids repetitive 0-damage moves.
4. Opponent Moveset Belief: Queries data/random_battle_sets.json to evaluate threats.
"""

import json
import os
import sys
import torch
import numpy as np
from typing import Any, Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../behavioral-cloning")))

from model import ContextualActionScoringNet
from mechanics import (
    ShowdownData,
    clean_id,
    estimate_damage_pct,
    get_type_multiplier,
    TYPE_TO_IDX,
    TYPES
)

STATE_DIM = 76
ACTION_DIM = 16
NUM_ACTION_SLOTS = 9


def encode_types(types: List[str]) -> np.ndarray:
    vec = np.zeros(len(TYPES), dtype=np.float32)
    for t in types:
        idx = TYPE_TO_IDX.get(t.lower())
        if idx is not None:
            vec[idx] = 1.0
    return vec


class NeuralBrain:
    def __init__(self, checkpoint_path: str = "projects/reinforcement-learning/weights/ppo_model.pt"):
        self.data = ShowdownData.get()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Random Battle Sets for Opponent Belief
        sets_path = os.path.join(os.path.dirname(__file__), "../../data/random_battle_sets.json")
        self.random_sets = {}
        if os.path.exists(sets_path):
            with open(sets_path, "r", encoding="utf-8") as f:
                self.random_sets = json.load(f)

        # Action History for Anti-Looping
        self.action_history: List[str] = []
        self.last_dmg_dealt = 1.0

        # Load Neural Model
        self.model = ContextualActionScoringNet(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=128,
            num_actions=NUM_ACTION_SLOTS
        ).to(self.device)

        target_ckpt = checkpoint_path
        if not os.path.exists(target_ckpt):
            target_ckpt = "projects/behavioral-cloning/weights/bc_model.pt"

        if os.path.exists(target_ckpt):
            ckpt = torch.load(target_ckpt, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith("actor_scorer."):
                    cleaned[k.replace("actor_scorer.", "scorer.")] = v
                elif not k.startswith("critic_head."):
                    cleaned[k] = v
            self.model.load_state_dict(cleaned, strict=False)
            print(f"[+] Loaded Neural Model ({self.device}): {target_ckpt}")

        self.model.eval()

    def reset_battle_state(self):
        self.action_history.clear()
        self.last_dmg_dealt = 1.0

    def choose_action(self, request: Dict[str, Any], battle_history: List[str]) -> str:
        side = request.get("side", {})
        my_side_id = side.get("id", "p1")
        opp_side_id = "p2" if my_side_id == "p1" else "p1"
        opp_prefix = f"{opp_side_id}a:"

        pokemon_list = side.get("pokemon", [])
        active_req = request.get("active", [{}])[0] if request.get("active") else {}
        moves_req = active_req.get("moves", [])
        force_switch = request.get("forceSwitch", [False])[0]

        # 1. Parse My Team
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

        # 2. Parse Opponent Active & Hazards
        opp_species = "Pikachu"
        opp_hp = 1.0
        opp_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "acc": 0}
        my_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "acc": 0}
        weather = None
        terrain = None
        opp_has_status = False

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
                    opp_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "acc": 0}
                    opp_has_status = any(st in parts[4] for st in ["brn", "psn", "tox", "par", "slp", "frz"])
            elif cmd in ["-damage", "-heal"]:
                tag = parts[2].strip()
                hp_raw = parts[3].split()[0] if len(parts) > 3 else "100/100"
                if "fnt" in hp_raw:
                    if tag.startswith(opp_prefix): opp_hp = 0.0
                elif "/" in hp_raw:
                    n, d = hp_raw.split("/")
                    val = float(n) / float(d) if float(d) > 0 else 0.0
                    if tag.startswith(opp_prefix):
                        opp_hp = val
                        opp_has_status = any(st in parts[3] for st in ["brn", "psn", "tox", "par", "slp", "frz"])
            elif cmd == "-boost":
                tag, stat, amt = parts[2].strip(), parts[3].strip(), int(parts[4]) if len(parts) > 4 else 1
                if tag.startswith(opp_prefix) and stat in opp_boosts: opp_boosts[stat] = min(6, opp_boosts[stat] + amt)
                elif tag.startswith(f"{my_side_id}a:") and stat in my_boosts: my_boosts[stat] = min(6, my_boosts[stat] + amt)
            elif cmd == "-status":
                if parts[2].startswith(opp_prefix): opp_has_status = True
            elif cmd == "-curestatus":
                if parts[2].startswith(opp_prefix): opp_has_status = False

        # Speed Tier Analysis
        opp_poke = self.data.get_pokemon(opp_species)
        opp_types = opp_poke.get("types", ["Normal"]) if opp_poke else ["Normal"]
        opp_spe = opp_poke.get("baseStats", {}).get("spe", 80) if opp_poke else 80

        my_poke = self.data.get_pokemon(my_active["species"]) if my_active else None
        my_types = my_poke.get("types", ["Normal"]) if my_poke else ["Normal"]
        my_spe = my_poke.get("baseStats", {}).get("spe", 80) if my_poke else 80

        i_am_faster = my_spe > opp_spe

        # 3. Handle Forced Switch (Smart Defensive Matchup)
        if force_switch or not my_active or my_active["fainted"]:
            best_switch_slot = -1
            best_score = -999.0

            for b in my_bench:
                if not b["fainted"] and b["hp"] > 0:
                    b_poke = self.data.get_pokemon(b["species"])
                    b_types = b_poke.get("types", ["Normal"]) if b_poke else ["Normal"]
                    b_spe = b_poke.get("baseStats", {}).get("spe", 80) if b_poke else 80

                    # Matchup Resistances
                    resistances = [get_type_multiplier(ot, b_types) for ot in opp_types]
                    worst_res = max(resistances) if resistances else 1.0
                    best_res = min(resistances) if resistances else 1.0

                    score = (b["hp"] * 4.0) - (worst_res * 2.5) - (best_res * 0.5)
                    if b_spe > opp_spe: score += 2.0

                    if score > best_score:
                        best_score = score
                        best_switch_slot = b["slot"]

            if best_switch_slot > 0:
                self.action_history.append(f"switch {best_switch_slot}")
                return f"switch {best_switch_slot}"
            for b in my_bench:
                if not b["fainted"] and b["hp"] > 0:
                    return f"switch {b['slot']}"
            return "default"

        # 4. Construct Action Candidate Matrix
        a_mat = np.zeros((NUM_ACTION_SLOTS, ACTION_DIM), dtype=np.float32)
        mask = np.zeros(NUM_ACTION_SLOTS, dtype=np.float32)
        status_moves = ["willowisp", "toxic", "thunderwave", "spore", "hypnosis", "yawn", "sing", "glare"]

        # Calculate all move damages
        move_damages = []
        for i, m in enumerate(moves_req[:4]):
            if not m.get("disabled", False) and m.get("pp", 1) > 0:
                m_name = m.get("move", m.get("id", ""))
                dmg = estimate_damage_pct(my_active["species"], opp_species, m_name, my_boosts, opp_boosts, weather=weather)
                move_damages.append((i, m_name, dmg))

        max_dmg = max([d[2] for d in move_damages], default=0.01)

        # Fill Action Matrix
        for i, m_name, dmg in move_damages:
            mid = clean_id(m_name)
            m_data = self.data.get_move(m_name)
            if not m_data: continue

            # Redundant Status Mask
            if mid in status_moves and opp_has_status:
                continue

            # Anti-Loop Mask: If last 2 moves were this move and dmg was 0, mask out
            recent_count = self.action_history[-2:].count(f"move {i+1}")
            if recent_count >= 2 and dmg <= 0.01:
                continue

            mask[i] = 1.0
            a_mat[i, 0] = 1.0  # is_move
            a_mat[i, 2] = min(2.0, dmg)
            a_mat[i, 3] = 1.0 if (dmg >= opp_hp and dmg > 0) else 0.0

            cat = m_data.get("category", "Status")
            if cat == "Physical": a_mat[i, 4] = 1.0
            elif cat == "Special": a_mat[i, 5] = 1.0
            else: a_mat[i, 6] = 1.0

            m_type = m_data.get("type", "Normal")
            a_mat[i, 7] = get_type_multiplier(m_type, opp_types) / 4.0
            a_mat[i, 8] = 1.5 if m_type in my_types else 1.0
            acc = m_data.get("accuracy", 100)
            a_mat[i, 9] = (acc if isinstance(acc, (int, float)) else 100) / 100.0
            a_mat[i, 10] = (m_data.get("priority", 0) + 6.0) / 12.0
            a_mat[i, 13] = dmg / max(max_dmg, 0.01)
            a_mat[i, 14] = 1.0 if (dmg == max_dmg and dmg > 0.05) else 0.0

        # Bench Switches
        for i, b in enumerate(my_bench[:5]):
            slot = 4 + i
            if not b["fainted"] and b["hp"] > 0:
                mask[slot] = 1.0
                a_mat[slot, 1] = 1.0
                a_mat[slot, 2] = b["hp"]
                b_poke = self.data.get_pokemon(b["species"])
                b_types = b_poke.get("types", ["Normal"]) if b_poke else ["Normal"]
                res = [get_type_multiplier(ot, b_types) for ot in opp_types]
                a_mat[slot, 7] = (min(res) if res else 1.0) / 4.0

        if mask.sum() == 0:
            return "move 1"

        # 5. Deterministic Clean KO Rule (Always execute lethal outspeeding move!)
        for i, m_name, dmg in move_damages:
            if mask[i] > 0 and dmg >= opp_hp and (i_am_faster or a_mat[i, 10] > 0.5):
                action = f"move {i + 1}"
                self.action_history.append(action)
                return action

        # 6. Neural Network Inference
        s_vec = np.zeros(STATE_DIM, dtype=np.float32)
        s_vec[0] = my_active["hp"] if my_active else 0.0
        s_vec[1] = opp_hp
        s_vec[2:8] = [my_boosts.get(k, 0) / 6.0 for k in ["atk", "def", "spa", "spd", "spe", "acc"]]
        s_vec[8:14] = [opp_boosts.get(k, 0) / 6.0 for k in ["atk", "def", "spa", "spd", "spe", "acc"]]
        my_alive = (1 if my_active and not my_active["fainted"] else 0) + sum(1 for b in my_bench if not b["fainted"])
        s_vec[14] = my_alive / 6.0
        s_vec[15] = 0.8
        s_vec[34:52] = encode_types(my_types)
        s_vec[52:70] = encode_types(opp_types)

        s_tensor = torch.tensor(s_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_tensor = torch.tensor(a_mat, dtype=torch.float32, device=self.device).unsqueeze(0)
        m_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(s_tensor, a_tensor, m_tensor)
            # Heuristic penalty for weak non-status moves (<40% of max damage)
            for slot_idx in range(4):
                if mask[slot_idx] > 0 and a_mat[slot_idx, 13] < 0.40 and a_mat[slot_idx, 6] == 0.0:
                    logits[0, slot_idx] -= 4.0

            best_idx = int(torch.argmax(logits, dim=-1).item())

        if best_idx < 4:
            can_tera = active_req.get("canTerastallize")
            move_slot = best_idx + 1
            action = f"move {move_slot} terastallize" if (can_tera and a_mat[best_idx, 2] >= 0.70) else f"move {move_slot}"
        else:
            bench_idx = best_idx - 4
            action = f"switch {my_bench[bench_idx]['slot']}" if bench_idx < len(my_bench) else "default"

        self.action_history.append(action)
        return action
