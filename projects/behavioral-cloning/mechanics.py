"""
Pokemon Showdown Mechanics & Semantic Feature Extractor (Full Immunities)
========================================================================
High-speed lookup for type effectiveness, ability immunities (Levitate, Flash Fire,
Volt Absorb, Water Absorb, Sap Sipper, etc.), and accurate Gen 9 damage calculations.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

TYPE_CHART: Dict[str, Dict[str, float]] = {
    "Normal":   {"Rock": 0.5, "Ghost": 0.0, "Steel": 0.5},
    "Fire":     {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 2.0, "Bug": 2.0, "Rock": 0.5, "Dragon": 0.5, "Steel": 2.0},
    "Water":    {"Fire": 2.0, "Water": 0.5, "Grass": 0.5, "Ground": 2.0, "Rock": 2.0, "Dragon": 0.5},
    "Electric": {"Water": 2.0, "Electric": 0.5, "Grass": 0.5, "Ground": 0.0, "Flying": 2.0, "Dragon": 0.5},
    "Grass":    {"Fire": 0.5, "Water": 2.0, "Grass": 0.5, "Poison": 0.5, "Ground": 2.0, "Flying": 0.5, "Bug": 0.5, "Rock": 2.0, "Dragon": 0.5, "Steel": 0.5},
    "Ice":      {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 0.5, "Ground": 2.0, "Flying": 2.0, "Dragon": 2.0, "Steel": 0.5},
    "Fighting": {"Normal": 2.0, "Ice": 2.0, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Rock": 2.0, "Ghost": 0.0, "Dark": 2.0, "Steel": 2.0, "Fairy": 0.5},
    "Poison":   {"Grass": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5, "Steel": 0.0, "Fairy": 2.0},
    "Ground":   {"Fire": 2.0, "Electric": 2.0, "Grass": 0.5, "Poison": 2.0, "Flying": 0.0, "Bug": 0.5, "Rock": 2.0, "Steel": 2.0},
    "Flying":   {"Electric": 0.5, "Grass": 2.0, "Fighting": 2.0, "Bug": 2.0, "Rock": 0.5, "Steel": 0.5},
    "Psychic":  {"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5, "Dark": 0.0, "Steel": 0.5},
    "Bug":      {"Fire": 0.5, "Grass": 2.0, "Fighting": 0.5, "Poison": 0.5, "Flying": 0.5, "Psychic": 2.0, "Ghost": 0.5, "Dark": 2.0, "Steel": 0.5, "Fairy": 0.5},
    "Rock":     {"Fire": 2.0, "Ice": 2.0, "Fighting": 0.5, "Ground": 0.5, "Flying": 2.0, "Bug": 2.0, "Steel": 0.5},
    "Ghost":    {"Normal": 0.0, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5, "Steel": 0.5},
    "Dragon":   {"Dragon": 2.0, "Steel": 0.5, "Fairy": 0.0},
    "Dark":     {"Fighting": 0.5, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5, "Fairy": 0.5},
    "Steel":    {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2.0, "Rock": 2.0, "Steel": 0.5, "Fairy": 2.0},
    "Fairy":    {"Fire": 0.5, "Fighting": 2.0, "Poison": 0.5, "Dragon": 2.0, "Dark": 2.0, "Steel": 0.5}
}

TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
    "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"
]
TYPE_TO_IDX = {t.lower(): i for i, t in enumerate(TYPES)}

# Common ability immunities
ABILITY_IMMUNITIES: Dict[str, str] = {
    "levitate": "Ground",
    "flashfire": "Fire",
    "waterabsorb": "Water",
    "stormdrain": "Water",
    "dryskin": "Water",
    "voltabsorb": "Electric",
    "motordrive": "Electric",
    "lightningrod": "Electric",
    "sapsipper": "Grass",
    "eartheater": "Ground",
    "purifyingsalt": "Ghost"
}


def clean_id(name: str) -> str:
    """Normalize string into Showdown ID format."""
    return "".join(c.lower() for c in name if c.isalnum())


class ShowdownData:
    """Loads and caches dex.json and moves.json."""
    _instance = None

    def __init__(self, data_dir: str = "data"):
        dex_path = os.path.join(data_dir, "dex.json")
        moves_path = os.path.join(data_dir, "moves.json")

        self.dex: Dict[str, Any] = {}
        self.moves: Dict[str, Any] = {}

        if os.path.exists(dex_path):
            with open(dex_path, "r", encoding="utf-8") as f:
                self.dex = json.load(f)
        if os.path.exists(moves_path):
            with open(moves_path, "r", encoding="utf-8") as f:
                self.moves = json.load(f)

    @classmethod
    def get(cls, data_dir: str = "data") -> "ShowdownData":
        if cls._instance is None:
            cls._instance = cls(data_dir)
        return cls._instance

    def get_pokemon(self, name: str) -> Optional[Dict[str, Any]]:
        pid = clean_id(name)
        return self.dex.get(pid)

    def get_move(self, name: str) -> Optional[Dict[str, Any]]:
        mid = clean_id(name)
        return self.moves.get(mid)


def get_type_multiplier(attack_type: str, defender_types: List[str]) -> float:
    """Calculates type effectiveness multiplier against 1 or 2 defender types."""
    atk = attack_type.capitalize()
    eff = 1.0
    for d in defender_types:
        def_cap = d.capitalize()
        if atk in TYPE_CHART and def_cap in TYPE_CHART[atk]:
            eff *= TYPE_CHART[atk][def_cap]
    return eff


def check_ability_immunity(move_type: str, defender_abilities: Dict[str, str]) -> bool:
    """Checks if defender has an innate ability immunity (e.g. Levitate vs Ground)."""
    for _, ab_name in defender_abilities.items():
        ab_id = clean_id(ab_name)
        if ab_id in ABILITY_IMMUNITIES and ABILITY_IMMUNITIES[ab_id] == move_type:
            return True
    return False


def estimate_damage_pct(
    attacker_name: str,
    defender_name: str,
    move_name: str,
    attacker_boosts: Dict[str, int],
    defender_boosts: Dict[str, int],
    weather: Optional[str] = None
) -> float:
    """Computes an approximate expected damage % factoring types, abilities, and boosts."""
    data = ShowdownData.get()
    p_atk = data.get_pokemon(attacker_name)
    p_def = data.get_pokemon(defender_name)
    m = data.get_move(move_name)

    if not m or not p_atk or not p_def:
        return 0.0

    category = m.get("category", "Status")
    if category == "Status":
        return 0.0

    move_type = m.get("type", "Normal")
    atk_types = p_atk.get("types", [])
    def_types = p_def.get("types", [])
    def_abilities = p_def.get("abilities", {})

    # Check Ability Immunity
    if check_ability_immunity(move_type, def_abilities):
        return 0.0

    # Type effectiveness
    type_eff = get_type_multiplier(move_type, def_types)
    if type_eff == 0.0:
        return 0.0

    bp = m.get("basePower", 0)
    if bp == 0:
        bp = 80

    # STAB (Same Type Attack Bonus)
    stab = 1.5 if move_type in atk_types else 1.0

    level = 80
    atk_stats = p_atk.get("baseStats", {"atk": 80, "spa": 80})
    def_stats = p_def.get("baseStats", {"hp": 80, "def": 80, "spd": 80})

    if category == "Physical":
        a = atk_stats.get("atk", 80)
        d = def_stats.get("def", 80)
        a_boost = attacker_boosts.get("atk", 0)
        d_boost = defender_boosts.get("def", 0)
    else:
        a = atk_stats.get("spa", 80)
        d = def_stats.get("spd", 80)
        a_boost = attacker_boosts.get("spa", 0)
        d_boost = defender_boosts.get("spd", 0)

    def get_boost_mult(stage: int) -> float:
        if stage >= 0: return (2.0 + stage) / 2.0
        return 2.0 / (2.0 - stage)

    a_effective = a * get_boost_mult(a_boost)
    d_effective = d * get_boost_mult(d_boost)

    weather_mult = 1.0
    if weather == "RainDance":
        if move_type == "Water": weather_mult = 1.5
        elif move_type == "Fire": weather_mult = 0.5
    elif weather == "SunnyDay":
        if move_type == "Fire": weather_mult = 1.5
        elif move_type == "Water": weather_mult = 0.5

    base_damage = (((2.0 * level / 5.0 + 2.0) * bp * (a_effective / max(1.0, d_effective))) / 50.0) + 2.0
    total_damage = base_damage * stab * type_eff * weather_mult

    hp_base = def_stats.get("hp", 80)
    approx_max_hp = ((2 * hp_base + 31 + 21) * level / 100.0) + level + 10.0

    return min(2.5, total_damage / max(1.0, approx_max_hp))


def get_hazard_damage_pct(defender_name: str, hazards: Dict[str, int], item: Optional[str] = None) -> float:
    if item and clean_id(item) == "heavydutyboots":
        return 0.0

    data = ShowdownData.get()
    p_def = data.get_pokemon(defender_name)
    def_types = p_def.get("types", []) if p_def else ["Normal"]
    def_abilities = p_def.get("abilities", {}) if p_def else {}

    damage = 0.0
    if hazards.get("stealthrock", 0) > 0:
        eff = get_type_multiplier("Rock", def_types)
        damage += 0.125 * eff

    # Spikes (Grounded only - check Flying and Levitate)
    is_grounded = ("Flying" not in def_types) and not check_ability_immunity("Ground", def_abilities)
    if is_grounded:
        spikes_count = hazards.get("spikes", 0)
        if spikes_count == 1: damage += 0.125
        elif spikes_count == 2: damage += 0.166
        elif spikes_count >= 3: damage += 0.25

    return min(1.0, damage)
