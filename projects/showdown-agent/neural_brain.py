"""
Slot-Agnostic Neural Behavioral Cloning Brain & Dynamic Decision Engine
========================================================================
1. Candidate-Centric Action Scoring: Evaluates actions purely by intrinsic mechanics
   (Base Power, STAB, Type Effectiveness, KO-Threshold, Priority, Speed, Status Utility).
2. Semantic Move Gating: Forbids redundant or useless status moves
   (e.g. Turn 1 Defog without hazards, Recover at full HP, Will-O-Wisp on Fire types).
3. Anti-Loop & Switch-Gating Engine: Forbids repetitive back-and-forth switches.
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
    evaluate_status_move_utility,
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
    def __init__(self, checkpoint_path: str = "projects/reinforcement-learning/weights/real_sim_gen8ou_ppo.pt"):
        self.data = ShowdownData.get()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_history: List[str] = []

        self.model = ContextualActionScoringNet(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=128,
            num_actions=NUM_ACTION_SLOTS,
            dropout=0.0
        ).to(self.device)

        target_ckpt = checkpoint_path
        if not os.path.exists(target_ckpt):
            target_ckpt = "projects/reinforcement-learning/weights/ppo_model.pt"
        if not os.path.exists(target_ckpt):
            target_ckpt = "projects/behavioral-cloning/weights/bc_model.pt"

        if os.path.exists(target_ckpt):
            try:
                ckpt = torch.load(target_ckpt, map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                cleaned = {}
                for k, v in state_dict.items():
                    if k.startswith("actor."):
                        cleaned[k.replace("actor.", "")] = v
                    elif not k.startswith("critic."):
                        cleaned[k] = v
                self.model.load_state_dict(cleaned, strict=False)
                print(f"[+] Loaded Neural Model ({self.device}): {target_ckpt}")
            except Exception as e:
                print(f"[-] Checkpoint load warning: {e}")

        self.model.eval()

    def reset_battle_state(self):
        self.action_history.clear()

    def choose_action(self, request: Dict[str, Any], battle_history: List[str]) -> str:
        side = request.get("side", {})
        my_side_id = side.get("id", "p1")
        opp_side_id = "p2" if my_side_id == "p1" else "p1"
        my_prefix = f"{my_side_id}:"
        opp_prefix = f"{opp_side_id}a:"

        pokemon_list = side.get("pokemon", [])
        active_req = request.get("active", [{}])[0] if request.get("active") else {}
        moves_req = active_req.get("moves", [])
        force_switch = bool(request.get("forceSwitch", [False])[0])

        # 1. Parse active and bench Pokemon
        my_active = None
        my_bench = []

        for i, p in enumerate(pokemon_list):
            is_active = p.get("active", False)
            cond = p.get("condition", "100/100")
            fainted = "fnt" in cond or cond.startswith("0")

            if "/" in cond and not fainted:
                cur_hp, max_hp = cond.split(" ")[0].split("/")
                hp_pct = float(cur_hp) / float(max_hp)
            else:
                hp_pct = 0.0 if fainted else 1.0

            details = p.get("details", "")
            species = clean_id(details.split(",")[0])

            poke_info = {
                "slot": i + 1,
                "species": species,
                "hp": hp_pct,
                "fainted": fainted,
                "active": is_active,
                "item": p.get("item", ""),
                "ability": p.get("ability", ""),
                "moves": p.get("moves", [])
            }

            if is_active:
                my_active = poke_info
            else:
                my_bench.append(poke_info)

        # 2. Parse Opponent & Field State (Hazards, Screens, Status) from Battle History
        opp_species = "pikachu"
        opp_hp = 1.0
        opp_boosts: Dict[str, int] = {}
        my_boosts: Dict[str, int] = {}
        weather = "none"
        opp_has_status = False

        my_side_hazards = {"stealthrock": False, "spikes": False, "toxicspikes": False, "stickyweb": False}
        opp_side_hazards = {"stealthrock": False, "spikes": False, "toxicspikes": False, "stickyweb": False}
        opp_screens = False

        for line in battle_history:
            parts = line.split("|")
            if len(parts) < 2: continue
            cmd = parts[1]

            if cmd in ("switch", "drag") and parts[2].startswith(opp_prefix):
                opp_species = clean_id(parts[2].split(" ")[-1].replace(opp_prefix, ""))
                cond = parts[4] if len(parts) > 4 else "100/100"
                if "fnt" in cond: opp_hp = 0.0
                elif "/" in cond:
                    c, m = cond.split(" ")[0].split("/")
                    opp_hp = float(c) / float(m)

            elif cmd == "-sidestart":
                target_side = parts[2]
                hazard_type = clean_id(parts[3])
                if target_side.startswith(my_side_id):
                    for h in my_side_hazards:
                        if h in hazard_type: my_side_hazards[h] = True
                else:
                    for h in opp_side_hazards:
                        if h in hazard_type: opp_side_hazards[h] = True
                    if any(s in hazard_type for s in ("reflect", "lightscreen", "auroraveil")):
                        opp_screens = True

            elif cmd == "-sideend":
                target_side = parts[2]
                hazard_type = clean_id(parts[3])
                if target_side.startswith(my_side_id):
                    for h in my_side_hazards:
                        if h in hazard_type: my_side_hazards[h] = False
                else:
                    for h in opp_side_hazards:
                        if h in hazard_type: opp_side_hazards[h] = False

            elif cmd == "-status" and parts[2].startswith(opp_prefix):
                opp_has_status = True
            elif cmd == "-curestatus" and parts[2].startswith(opp_prefix):
                opp_has_status = False

        # Speed Tier & Types
        opp_poke = self.data.get_pokemon(opp_species)
        opp_types = opp_poke.get("types", ["Normal"]) if opp_poke else ["Normal"]
        opp_spe = opp_poke.get("baseStats", {}).get("spe", 80) if opp_poke else 80

        my_poke = self.data.get_pokemon(my_active["species"]) if my_active else None
        my_types = my_poke.get("types", ["Normal"]) if my_poke else ["Normal"]
        my_spe = my_poke.get("baseStats", {}).get("spe", 80) if my_poke else 80
        i_am_faster = my_spe >= opp_spe
        my_hp_pct = my_active["hp"] if my_active else 1.0

        # 3. Handle Forced Switch (Faint / U-turn)
        if force_switch or not my_active or my_active["fainted"]:
            best_switch_slot = -1
            best_score = -999.0

            for b in my_bench:
                if not b["fainted"] and b["hp"] > 0:
                    b_poke = self.data.get_pokemon(b["species"])
                    b_types = b_poke.get("types", ["Normal"]) if b_poke else ["Normal"]
                    b_spe = b_poke.get("baseStats", {}).get("spe", 80) if b_poke else 80

                    resistances = [get_type_multiplier(ot, b_types) for ot in opp_types]
                    worst_res = max(resistances) if resistances else 1.0
                    score = (b["hp"] * 4.0) - (worst_res * 2.5)
                    if b_spe >= opp_spe: score += 1.5

                    if score > best_score:
                        best_score = score
                        best_switch_slot = b["slot"]

            if best_switch_slot > 0:
                action = f"switch {best_switch_slot}"
                self.action_history.append(action)
                return action
            return "default"

        # 4. Action Candidates Construction
        a_mat = np.zeros((NUM_ACTION_SLOTS, ACTION_DIM), dtype=np.float32)
        mask = np.zeros(NUM_ACTION_SLOTS, dtype=np.float32)

        move_damages = []
        for i, m in enumerate(moves_req[:4]):
            if not m.get("disabled", False) and m.get("pp", 1) > 0:
                m_name = m.get("move", m.get("id", ""))
                dmg = estimate_damage_pct(my_active["species"], opp_species, m_name, my_boosts, opp_boosts, weather=weather)
                move_damages.append((i, m_name, dmg))

        max_dmg = max([d[2] for d in move_damages], default=0.0)

        for i, m_name, dmg in move_damages:
            m_data = self.data.get_move(m_name)
            if not m_data: continue

            cat = m_data.get("category", "Status")
            
            # GATING STATUS MOVES BY SEMANTIC UTILITY
            if cat == "Status":
                util = evaluate_status_move_utility(
                    move_name=m_name,
                    my_hp_pct=my_hp_pct,
                    opp_hp_pct=opp_hp,
                    opp_species=opp_species,
                    opp_has_status=opp_has_status,
                    my_side_hazards=my_side_hazards,
                    opp_side_hazards=opp_side_hazards,
                    opp_screens=opp_screens
                )
                if util <= 0.0:
                    # STRICTLY MASK OUT INVALID / USELESS STATUS MOVES!
                    continue
                a_mat[i, 2] = util * 0.35 # Status utility value
                a_mat[i, 6] = 1.0
            else:
                a_mat[i, 2] = min(2.0, dmg)
                if cat == "Physical": a_mat[i, 4] = 1.0
                elif cat == "Special": a_mat[i, 5] = 1.0

            mask[i] = 1.0
            a_mat[i, 0] = 1.0
            a_mat[i, 3] = 1.0 if (dmg >= opp_hp and dmg > 0) else 0.0

            m_type = m_data.get("type", "Normal")
            a_mat[i, 7] = get_type_multiplier(m_type, opp_types) / 4.0
            a_mat[i, 8] = 1.5 if m_type in my_types else 1.0
            acc = m_data.get("accuracy", 100)
            a_mat[i, 9] = (acc if isinstance(acc, (int, float)) else 100) / 100.0
            a_mat[i, 10] = (m_data.get("priority", 0) + 6.0) / 12.0
            a_mat[i, 13] = dmg / max(max_dmg, 0.01) if max_dmg > 0 else 0.0

        # 5. Guaranteed Clean KO Execution
        for i, m_name, dmg in move_damages:
            if mask[i] > 0 and dmg >= opp_hp and (i_am_faster or a_mat[i, 10] > 0.5):
                action = f"move {i + 1}"
                self.action_history.append(action)
                return action

        # 6. ANTI-SWITCH LOOP & STRICT SWITCH GATING:
        last_action_was_switch = bool(self.action_history and "switch" in self.action_history[-1])
        allow_voluntary_switch = (not last_action_was_switch) and (max_dmg < 0.20 or my_hp_pct < 0.25)

        if allow_voluntary_switch:
            for i, b in enumerate(my_bench[:5]):
                slot = 4 + i
                if not b["fainted"] and b["hp"] > 0:
                    b_poke = self.data.get_pokemon(b["species"])
                    b_types = b_poke.get("types", ["Normal"]) if b_poke else ["Normal"]
                    res = [get_type_multiplier(ot, b_types) for ot in opp_types]
                    worst_res = max(res) if res else 1.0
                    if worst_res <= 1.0 and b["hp"] >= 0.50:
                        mask[slot] = 1.0
                        a_mat[slot, 1] = 1.0
                        a_mat[slot, 2] = b["hp"]
                        a_mat[slot, 7] = worst_res / 4.0

        if mask.sum() == 0 or (mask[:4].sum() > 0 and not allow_voluntary_switch):
            mask[4:] = 0.0

        # 7. Neural Model Forward Pass
        s_vec = np.zeros(STATE_DIM, dtype=np.float32)
        s_vec[0] = my_hp_pct
        s_vec[1] = opp_hp
        s_vec[14] = (1 + sum(1 for b in my_bench if not b["fainted"])) / 6.0
        s_vec[34:52] = encode_types(my_types)
        s_vec[52:70] = encode_types(opp_types)

        s_tensor = torch.tensor(s_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_tensor = torch.tensor(a_mat, dtype=torch.float32, device=self.device).unsqueeze(0)
        m_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(s_tensor, a_tensor, m_tensor)
            # High reward for maximum damage moves
            for slot_idx in range(4):
                if mask[slot_idx] > 0:
                    logits[0, slot_idx] += (a_mat[slot_idx, 2] * 2.5) # Strong damage preference

            best_idx = int(torch.argmax(logits, dim=-1).item())

        if best_idx < 4:
            action = f"move {best_idx + 1}"
        else:
            bench_idx = best_idx - 4
            action = f"switch {my_bench[bench_idx]['slot']}" if bench_idx < len(my_bench) else "move 1"

        self.action_history.append(action)
        return action
