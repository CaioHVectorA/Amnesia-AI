"""
Replay Parser & Dataset Extractor for Behavioral Cloning (Enhanced)
===================================================================
Reconstructs battle states with both Move and Voluntary Switch decisions,
with normalized semantic action features.
"""

import gzip
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from mechanics import (
    ShowdownData,
    clean_id,
    estimate_damage_pct,
    get_hazard_damage_pct,
    get_type_multiplier,
    TYPE_TO_IDX,
    TYPES
)

STATE_DIM = 76
ACTION_DIM = 16
NUM_ACTION_SLOTS = 9  # 4 moves + 5 switches


@dataclass
class BattlePokemon:
    name: str
    species: str
    level: int = 80
    current_hp: float = 1.0
    fainted: bool = False
    moves: List[str] = field(default_factory=list)
    boosts: Dict[str, int] = field(default_factory=lambda: {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "acc": 0})
    item: Optional[str] = None
    ability: Optional[str] = None
    status: Optional[str] = None


@dataclass
class PlayerState:
    name: str
    active: Optional[BattlePokemon] = None
    bench: List[BattlePokemon] = field(default_factory=list)
    hazards: Dict[str, int] = field(default_factory=lambda: {"stealthrock": 0, "spikes": 0, "toxicspikes": 0, "stickyweb": 0})
    screens: Dict[str, int] = field(default_factory=lambda: {"reflect": 0, "lightscreen": 0, "auroraveil": 0})


def encode_types(types: List[str]) -> np.ndarray:
    vec = np.zeros(len(TYPES), dtype=np.float32)
    for t in types:
        idx = TYPE_TO_IDX.get(t.lower())
        if idx is not None:
            vec[idx] = 1.0
    return vec


def build_state_vector(me: PlayerState, opp: PlayerState, weather: Optional[str], terrain: Optional[str]) -> np.ndarray:
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    data = ShowdownData.get()

    vec[0] = me.active.current_hp if me.active else 0.0
    vec[1] = opp.active.current_hp if opp.active else 0.0

    if me.active:
        b = me.active.boosts
        vec[2:8] = [b.get("atk", 0)/6.0, b.get("def", 0)/6.0, b.get("spa", 0)/6.0, b.get("spd", 0)/6.0, b.get("spe", 0)/6.0, b.get("acc", 0)/6.0]
    if opp.active:
        ob = opp.active.boosts
        vec[8:14] = [ob.get("atk", 0)/6.0, ob.get("def", 0)/6.0, ob.get("spa", 0)/6.0, ob.get("spd", 0)/6.0, ob.get("spe", 0)/6.0, ob.get("acc", 0)/6.0]

    my_alive = (1 if (me.active and not me.active.fainted) else 0) + sum(1 for p in me.bench if not p.fainted)
    opp_alive = (1 if (opp.active and not opp.active.fainted) else 0) + sum(1 for p in opp.bench if not p.fainted)
    vec[14] = my_alive / 6.0
    vec[15] = opp_alive / 6.0

    vec[16] = float(me.hazards.get("stealthrock", 0) > 0)
    vec[17] = me.hazards.get("spikes", 0) / 3.0
    vec[18] = me.hazards.get("toxicspikes", 0) / 2.0
    vec[19] = float(me.hazards.get("stickyweb", 0) > 0)

    vec[20] = float(opp.hazards.get("stealthrock", 0) > 0)
    vec[21] = opp.hazards.get("spikes", 0) / 3.0
    vec[22] = opp.hazards.get("toxicspikes", 0) / 2.0
    vec[23] = float(opp.hazards.get("stickyweb", 0) > 0)

    weathers = ["RainDance", "SunnyDay", "Sandstorm", "Snow", None]
    w_idx = weathers.index(weather) if weather in weathers else 4
    vec[24 + w_idx] = 1.0

    terrains = ["ElectricTerrain", "GrassyTerrain", "PsychicTerrain", "MistyTerrain", None]
    t_idx = terrains.index(terrain) if terrain in terrains else 4
    vec[29 + t_idx] = 1.0

    my_types = []
    if me.active:
        poke = data.get_pokemon(me.active.name)
        my_types = poke.get("types", []) if poke else []
    vec[34:52] = encode_types(my_types)

    opp_types = []
    if opp.active:
        poke = data.get_pokemon(opp.active.name)
        opp_types = poke.get("types", []) if poke else []
    vec[52:70] = encode_types(opp_types)

    bench_hps = [p.current_hp for p in me.bench if not p.fainted]
    vec[70] = float(np.mean(bench_hps)) if bench_hps else 0.0
    vec[71] = len(bench_hps) / 5.0

    return vec


def build_candidate_actions(
    me: PlayerState,
    opp: PlayerState,
    weather: Optional[str]
) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((NUM_ACTION_SLOTS, ACTION_DIM), dtype=np.float32)
    mask = np.zeros(NUM_ACTION_SLOTS, dtype=np.float32)

    data = ShowdownData.get()

    # 1. Moves (Slots 0..3)
    if me.active and not me.active.fainted:
        for slot_idx in range(4):
            if slot_idx < len(me.active.moves):
                m_name = me.active.moves[slot_idx]
                m_data = data.get_move(m_name)
                if m_data:
                    mask[slot_idx] = 1.0
                    matrix[slot_idx, 0] = 1.0  # is_move
                    matrix[slot_idx, 1] = 0.0  # is_switch

                    opp_name = opp.active.name if opp.active else "Pikachu"
                    dmg = estimate_damage_pct(
                        me.active.name,
                        opp_name,
                        m_name,
                        me.active.boosts,
                        opp.active.boosts if opp.active else {},
                        weather=weather
                    )
                    matrix[slot_idx, 2] = min(2.0, dmg)
                    matrix[slot_idx, 3] = 1.0 if (opp.active and dmg >= opp.active.current_hp) else 0.0

                    cat = m_data.get("category", "Status")
                    if cat == "Physical": matrix[slot_idx, 4] = 1.0
                    elif cat == "Special": matrix[slot_idx, 5] = 1.0
                    else: matrix[slot_idx, 6] = 1.0

                    m_type = m_data.get("type", "Normal")
                    opp_poke = data.get_pokemon(opp_name) if opp.active else None
                    opp_types = opp_poke.get("types", []) if opp_poke else []
                    matrix[slot_idx, 7] = get_type_multiplier(m_type, opp_types) / 4.0

                    my_poke = data.get_pokemon(me.active.name)
                    my_types = my_poke.get("types", []) if my_poke else []
                    matrix[slot_idx, 8] = 1.0 if m_type in my_types else 0.67

                    acc = m_data.get("accuracy", 100)
                    matrix[slot_idx, 9] = (acc if isinstance(acc, (int, float)) else 100) / 100.0
                    matrix[slot_idx, 10] = (m_data.get("priority", 0) + 6.0) / 12.0

                    mid = clean_id(m_name)
                    is_pivot = mid in ["uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception"]
                    is_recovery = mid in ["roost", "recover", "synthesis", "moonlight", "softboiled", "slackoff", "wish"]
                    matrix[slot_idx, 11] = float(is_pivot)
                    matrix[slot_idx, 12] = float(is_recovery)

    # 2. Switches (Slots 4..8)
    for i, bench_mon in enumerate(me.bench[:5]):
        switch_slot = 4 + i
        if not bench_mon.fainted and bench_mon.current_hp > 0:
            mask[switch_slot] = 1.0
            matrix[switch_slot, 0] = 0.0
            matrix[switch_slot, 1] = 1.0
            matrix[switch_slot, 2] = bench_mon.current_hp

            h_dmg = get_hazard_damage_pct(bench_mon.name, me.hazards, bench_mon.item)
            matrix[switch_slot, 3] = 1.0 - h_dmg

            opp_poke = data.get_pokemon(opp.active.name) if opp.active else None
            opp_types = opp_poke.get("types", []) if opp_poke else ["Normal"]
            bench_poke = data.get_pokemon(bench_mon.name)
            bench_types = bench_poke.get("types", []) if bench_poke else ["Normal"]

            resistances = [get_type_multiplier(ot, bench_types) for ot in opp_types]
            matrix[switch_slot, 7] = (min(resistances) if resistances else 1.0) / 4.0

            bench_spe = bench_poke.get("baseStats", {}).get("spe", 80) if bench_poke else 80
            opp_spe = opp_poke.get("baseStats", {}).get("spe", 80) if opp_poke else 80
            matrix[switch_slot, 10] = 1.0 if bench_spe > opp_spe else 0.0

    return matrix, mask


def parse_replay_file(file_path: str) -> List[Tuple[np.ndarray, np.ndarray, int, float, np.ndarray]]:
    try:
        if file_path.endswith(".gz"):
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception:
        return []

    log = data.get("log", "")
    weight = data.get("weight", 1.0)
    winner_side = data.get("winner_side")

    if not log or weight <= 0:
        return []

    p1 = PlayerState(name=data.get("p1", "p1"))
    p2 = PlayerState(name=data.get("p2", "p2"))

    weather = None
    terrain = None
    is_new_turn = False

    samples: List[Tuple[np.ndarray, np.ndarray, int, float, np.ndarray]] = []

    lines = log.split("\n")
    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue

        cmd = parts[1]

        if cmd == "turn":
            is_new_turn = True

        # Switch event
        elif cmd in ["switch", "drag"]:
            player_tag = parts[2][:2]
            species = parts[3].split(",")[0].strip()
            hp_str = parts[4].split()[0] if len(parts) > 4 else "100/100"
            hp_val = 1.0
            if "/" in hp_str:
                num, den = hp_str.split("/")
                hp_val = float(num) / float(den)

            target_player = p1 if player_tag == "p1" else p2
            opp_player = p2 if player_tag == "p1" else p1

            # Voluntary Switch on new turn (before fainting)
            if is_new_turn and target_player.active and not target_player.active.fainted and winner_side == player_tag:
                # Find which bench slot this was
                for b_idx, b_mon in enumerate(target_player.bench[:5]):
                    if b_mon.name == species:
                        switch_slot = 4 + b_idx
                        s_vec = build_state_vector(target_player, opp_player, weather, terrain)
                        a_mat, mask = build_candidate_actions(target_player, opp_player, weather)
                        if mask[switch_slot] > 0:
                            samples.append((s_vec, a_mat, switch_slot, weight * 0.8, mask))
                        break

            new_mon = BattlePokemon(name=species, species=species, current_hp=hp_val)
            if target_player.active:
                target_player.bench.append(target_player.active)
            target_player.active = new_mon
            is_new_turn = False

        # Move used
        elif cmd == "move":
            player_tag = parts[2][:2]
            move_name = parts[3].strip()

            target_player = p1 if player_tag == "p1" else p2
            opp_player = p2 if player_tag == "p1" else p1

            if target_player.active:
                if move_name not in target_player.active.moves:
                    target_player.active.moves.append(move_name)

                if winner_side == player_tag:
                    slot_idx = target_player.active.moves.index(move_name)
                    if slot_idx < 4:
                        s_vec = build_state_vector(target_player, opp_player, weather, terrain)
                        a_mat, mask = build_candidate_actions(target_player, opp_player, weather)
                        if mask[slot_idx] > 0:
                            samples.append((s_vec, a_mat, slot_idx, weight, mask))
            is_new_turn = False

        # Damage
        elif cmd == "-damage":
            player_tag = parts[2][:2]
            hp_str = parts[3].split()[0] if len(parts) > 3 else "0/100"
            if "/" in hp_str:
                num, den = hp_str.split("/")
                val = float(num) / float(den)
                target = p1 if player_tag == "p1" else p2
                if target.active: target.active.current_hp = val
            elif "0 fnt" in parts[3]:
                target = p1 if player_tag == "p1" else p2
                if target.active:
                    target.active.current_hp = 0.0
                    target.active.fainted = True

        # Weather / Terrain
        elif cmd == "-weather":
            w = parts[2].strip()
            weather = None if w == "none" else w
        elif cmd == "-fieldstart":
            terrain = parts[2].replace("move: ", "").strip()
        elif cmd == "-fieldend":
            terrain = None

        # Hazards
        elif cmd == "-sidestart":
            player_tag = parts[2][:2]
            effect = parts[3].lower()
            target = p1 if player_tag == "p1" else p2
            if "stealth rock" in effect: target.hazards["stealthrock"] = 1
            elif "spikes" in effect: target.hazards["spikes"] = min(3, target.hazards["spikes"] + 1)
            elif "toxic spikes" in effect: target.hazards["toxicspikes"] = min(2, target.hazards["toxicspikes"] + 1)
            elif "sticky web" in effect: target.hazards["stickyweb"] = 1

    return samples
