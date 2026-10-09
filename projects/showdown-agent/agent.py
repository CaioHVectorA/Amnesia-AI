"""
Stateful Showdown Battle Brain (Tactically-Decisive BC Agent)
============================================================
Combines the Behavioral Cloning Policy Network with decisive tactical execution:
- Accurately tracks opponent active species and field state
- Decisively attacks with highest-value offensive moves
- Prevents infinite switch loops
- Hard-prevents attacking into immunities (Levitate, Type immunities)
"""

import os
import re
import sys
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../behavioral-cloning")))

from model import ActionScoringPolicyNet
from mechanics import (
    ShowdownData,
    clean_id,
    estimate_damage_pct,
    get_hazard_damage_pct,
    get_type_multiplier,
    TYPE_TO_IDX,
    TYPES
)
from parser import (
    PlayerState,
    BattlePokemon,
    build_state_vector,
    build_candidate_actions,
    STATE_DIM,
    ACTION_DIM,
    NUM_ACTION_SLOTS
)


class ShowdownBCAgent:
    def __init__(self, checkpoint_path: Optional[str] = None, model: Optional[torch.nn.Module] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = ShowdownData.get()
        
        if model is not None:
            self.model = model
            self.checkpoint_path = checkpoint_path or "in-memory"
        else:
            if not checkpoint_path:
                possible_paths = [
                    "projects/behavioral-cloning/weights/bc_model.pt",
                    "../behavioral-cloning/weights/bc_model.pt",
                    os.path.join(os.path.dirname(__file__), "../behavioral-cloning/weights/bc_model.pt")
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        checkpoint_path = p
                        break
                if not checkpoint_path:
                    checkpoint_path = "projects/behavioral-cloning/weights/bc_model.pt"

            self.checkpoint_path = checkpoint_path
            self.model = ActionScoringPolicyNet(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                hidden_dim=128,
                num_actions=NUM_ACTION_SLOTS
            ).to(self.device)

            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"[+] Loaded BC Model: {checkpoint_path} (Val Acc: {checkpoint.get('val_top1_acc', 0):.2f}%)")
            else:
                print(f"[-] Warning: Checkpoint not found at {checkpoint_path}.")

        self.model.eval()

    def parse_full_state(
        self,
        request: Dict[str, Any],
        battle_log_history: List[str]
    ) -> Tuple[PlayerState, PlayerState, Optional[str], Optional[str], bool, List[Dict]]:
        side = request.get("side", {})
        my_side_id = side.get("id", "p1")
        opp_side_id = "p2" if my_side_id == "p1" else "p1"
        opp_prefix_a = f"{opp_side_id}a:"
        opp_prefix_b = f"{opp_side_id}:"
        my_prefix = f"{my_side_id}a:"

        me = PlayerState(name=side.get("name", "me"))
        pokemon_list = side.get("pokemon", [])
        active_req = request.get("active", [{}])[0]
        moves_req = active_req.get("moves", [])
        force_switch = request.get("forceSwitch", [False])[0]

        for p in pokemon_list:
            details = p.get("details", "")
            species = details.split(",")[0].strip()
            hp_str = p.get("condition", "100/100")
            fainted = "fnt" in hp_str or hp_str.startswith("0")
            hp_val = 1.0
            if "/" in hp_str:
                num, den = hp_str.split()[0].split("/")
                hp_val = float(num) / float(den) if float(den) > 0 else 0.0

            moves = [m.get("move", m.get("id", "")) for m in moves_req] if p.get("active") else p.get("moves", [])
            bp = BattlePokemon(name=species, species=species, current_hp=hp_val, fainted=fainted, moves=moves)

            if p.get("active"): me.active = bp
            else: me.bench.append(bp)

        opp = PlayerState(name="opponent")
        opp_active_species = None
        opp_active_hp = 1.0
        opp_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "acc": 0}
        my_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "acc": 0}
        weather = None
        terrain = None

        for line in battle_log_history:
            parts = line.split("|")
            if len(parts) < 2: continue
            cmd = parts[1]

            if cmd in ["switch", "drag"]:
                tag = parts[2].strip()
                if tag.startswith(opp_prefix_a) or tag.startswith(opp_prefix_b):
                    opp_active_species = parts[3].split(",")[0].strip()
                    hp_raw = parts[4].split()[0] if len(parts) > 4 else "100/100"
                    if "/" in hp_raw:
                        n, d = hp_raw.split("/")
                        opp_active_hp = float(n) / float(d) if float(d) > 0 else 1.0
                    opp_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "acc": 0}

            elif cmd in ["-damage", "-heal"]:
                tag = parts[2].strip()
                hp_raw = parts[3].split()[0] if len(parts) > 3 else "100/100"
                if "fnt" in hp_raw:
                    if tag.startswith(opp_prefix_a) or tag.startswith(opp_prefix_b): opp_active_hp = 0.0
                elif "/" in hp_raw:
                    n, d = hp_raw.split("/")
                    val = float(n) / float(d) if float(d) > 0 else 0.0
                    if tag.startswith(opp_prefix_a) or tag.startswith(opp_prefix_b): opp_active_hp = val

            elif cmd == "-boost":
                tag = parts[2].strip()
                stat = parts[3].strip()
                amt = int(parts[4]) if len(parts) > 4 else 1
                if (tag.startswith(opp_prefix_a) or tag.startswith(opp_prefix_b)) and stat in opp_boosts:
                    opp_boosts[stat] = min(6, opp_boosts[stat] + amt)
                elif tag.startswith(my_prefix) and stat in my_boosts:
                    my_boosts[stat] = min(6, my_boosts[stat] + amt)

            elif cmd == "-unboost":
                tag = parts[2].strip()
                stat = parts[3].strip()
                amt = int(parts[4]) if len(parts) > 4 else 1
                if (tag.startswith(opp_prefix_a) or tag.startswith(opp_prefix_b)) and stat in opp_boosts:
                    opp_boosts[stat] = max(-6, opp_boosts[stat] - amt)
                elif tag.startswith(my_prefix) and stat in my_boosts:
                    my_boosts[stat] = max(-6, my_boosts[stat] - amt)

            elif cmd == "-sidestart":
                side_tag = parts[2][:2]
                effect = parts[3].lower()
                target_state = me if side_tag == my_side_id else opp
                if "stealth rock" in effect: target_state.hazards["stealthrock"] = 1
                elif "spikes" in effect: target_state.hazards["spikes"] = min(3, target_state.hazards["spikes"] + 1)
                elif "toxic spikes" in effect: target_state.hazards["toxicspikes"] = min(2, target_state.hazards["toxicspikes"] + 1)
                elif "sticky web" in effect: target_state.hazards["stickyweb"] = 1

            elif cmd == "-weather":
                w = parts[2].strip()
                weather = None if w == "none" else w
            elif cmd == "-fieldstart":
                terrain = parts[2].replace("move: ", "").strip()
            elif cmd == "-fieldend":
                terrain = None

        if not opp_active_species:
            # Fallback if log doesn't have switch yet
            opp_active_species = "Pikachu"

        opp.active = BattlePokemon(name=opp_active_species, species=opp_active_species, current_hp=opp_active_hp, boosts=opp_boosts)
        if me.active: me.active.boosts = my_boosts

        return me, opp, weather, terrain, force_switch, moves_req

    def choose_action(
        self,
        request: Dict[str, Any],
        battle_log_history: List[str]
    ) -> str:
        me, opp, weather, terrain, force_switch, moves_req = self.parse_full_state(request, battle_log_history)

        s_vec = build_state_vector(me, opp, weather=weather, terrain=terrain)
        a_mat, mask = build_candidate_actions(me, opp, weather=weather)

        if force_switch:
            mask[:4] = 0.0

        for i, m in enumerate(moves_req):
            if m.get("disabled", False) or m.get("pp", 1) == 0:
                mask[i] = 0.0

        if mask.sum() == 0:
            if force_switch or not me.active:
                for idx in range(4, 9):
                    b_idx = idx - 4
                    if b_idx < len(me.bench) and not me.bench[b_idx].fainted and me.bench[b_idx].current_hp > 0:
                        return f"switch {b_idx + 2}"
            return "move 1"

        # Model Inference
        s_tensor = torch.tensor(s_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_tensor = torch.tensor(a_mat, dtype=torch.float32, device=self.device).unsqueeze(0)
        m_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            raw_logits = self.model(s_tensor, a_tensor, m_tensor).squeeze(0).cpu().numpy()

        adjusted_logits = raw_logits.copy()

        # Decisive Offensive Calibration
        best_dmg = -1.0
        best_move_slot = -1

        for slot in range(4):
            if mask[slot] > 0:
                dmg = a_mat[slot, 2]
                is_lethal = a_mat[slot, 3] > 0
                acc = a_mat[slot, 9]

                if dmg > best_dmg:
                    best_dmg = dmg
                    best_move_slot = slot

                if is_lethal and acc >= 0.85:
                    adjusted_logits[slot] += 6.0  # Finish off the opponent immediately
                elif dmg > 0:
                    adjusted_logits[slot] += (dmg * 3.0)  # Strong reward for high damage attacks
                elif dmg == 0.0 and a_mat[slot, 6] == 0.0:
                    adjusted_logits[slot] -= 100.0  # Block 0-damage immune attacks

        # Switch penalty: If active Pokémon has viable attack (>20% damage), suppress voluntary switching
        if not force_switch and me.active and not me.active.fainted:
            if best_dmg >= 0.20:
                for slot in range(4, 9):
                    if mask[slot] > 0:
                        adjusted_logits[slot] -= 8.0  # Don't switch when having strong offensive pressure

        action_idx = int(np.argmax(adjusted_logits))

        active_req = request.get("active", [{}])[0]
        if action_idx < 4:
            move_slot = action_idx + 1
            can_tera = active_req.get("canTerastallize")
            if can_tera and a_mat[action_idx, 2] >= 0.7:
                return f"move {move_slot} terastallize"
            return f"move {move_slot}"
        else:
            bench_idx = action_idx - 4
            team_slot = bench_idx + 2
            return f"switch {team_slot}"
