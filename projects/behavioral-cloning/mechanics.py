"""
Pokemon Showdown Mechanics & Semantic Feature Extractor (Full Immunities & Status Utility)
========================================================================================
High-speed lookup for type effectiveness, ability immunities, accurate damage calculations,
and semantic move utility evaluation (Defog, Stealth Rock, Recovery, Status, Setup).
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
    _instance = None

    def __init__(self, data_dir: str = "data"):
        self.dex: Dict[str, Any] = {}
        self.moves: Dict[str, Any] = {}
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
        dex_path = os.path.join(base_dir, "dex.json")
        moves_path = os.path.join(base_dir, "moves.json")

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
    atk = attack_type.capitalize()
    eff = 1.0
    for d in defender_types:
        def_cap = d.capitalize()
        if atk in TYPE_CHART and def_cap in TYPE_CHART[atk]:
            eff *= TYPE_CHART[atk][def_cap]
    return eff


def check_ability_immunity(move_type: str, defender_abilities: Dict[str, str]) -> bool:
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

    if check_ability_immunity(move_type, def_abilities):
        return 0.0

    type_mult = get_type_multiplier(move_type, def_types)
    if type_mult == 0.0:
        return 0.0

    stab = 1.5 if move_type in atk_types else 1.0
    bp = m.get("basePower", 50)
    if bp == 0:
        bp = 60

    stats_atk = p_atk.get("baseStats", {})
    stats_def = p_def.get("baseStats", {})

    if category == "Physical":
        a_stat = stats_atk.get("atk", 100)
        d_stat = stats_def.get("def", 100)
        a_boost = attacker_boosts.get("atk", 0)
        d_boost = defender_boosts.get("def", 0)
    else:
        a_stat = stats_atk.get("spa", 100)
        d_stat = stats_def.get("spd", 100)
        a_boost = attacker_boosts.get("spa", 0)
        d_boost = defender_boosts.get("spd", 0)

    def boost_mult(b: int) -> float:
        return (2.0 + b) / 2.0 if b >= 0 else 2.0 / (2.0 - b)

    a_stat *= boost_mult(a_boost)
    d_stat *= boost_mult(d_boost)

    weather_mult = 1.0
    if weather:
        w_clean = clean_id(weather)
        if "rain" in w_clean:
            if move_type == "Water": weather_mult = 1.5
            elif move_type == "Fire": weather_mult = 0.5
        elif "sun" in w_clean:
            if move_type == "Fire": weather_mult = 1.5
            elif move_type == "Water": weather_mult = 0.5

    level = 80
    base_dmg = (((2 * level / 5 + 2) * bp * (a_stat / max(d_stat, 1.0))) / 50 + 2)
    final_dmg = base_dmg * stab * type_mult * weather_mult
    def_hp = stats_def.get("hp", 100) * 2 + 140

    return min(2.0, (final_dmg / def_hp) * 0.92)


def evaluate_status_move_utility(
    move_name: str,
    my_hp_pct: float,
    opp_hp_pct: float,
    opp_species: str,
    opp_has_status: bool,
    my_side_hazards: Dict[str, bool],
    opp_side_hazards: Dict[str, bool],
    opp_screens: bool = False
) -> float:
    """
    Evaluates semantic utility for status moves.
    Returns:
    0.0 -> Move is completely useless in current state (MUST BE MASKED OUT!)
    > 0.0 -> Valid utility score.
    """
    mid = clean_id(move_name)
    data = ShowdownData.get()
    opp_poke = data.get_pokemon(opp_species) or {}
    opp_types = opp_poke.get("types", [])

    # 1. Defog / Hazard Clearing
    if mid in ("defog", "rapidspin", "courtchange", "tidyup", "mortalspin"):
        has_our_hazards = any(my_side_hazards.values())
        if not has_our_hazards and not opp_screens:
            return 0.0  # Useless! No hazards or screens to clear
        return 2.5 if has_our_hazards else 1.0

    # 2. Setting Entry Hazards
    if mid in ("stealthrock", "stickyweb"):
        if opp_side_hazards.get(mid, False):
            return 0.0  # Already on field!
        return 2.0
    if mid in ("spikes", "toxicspikes"):
        # Maximum layers reached check
        if opp_side_hazards.get(f"{mid}_max", False):
            return 0.0
        return 1.8

    # 3. Status Inducers
    if mid in ("thunderwave", "glare"):
        if opp_has_status or "Electric" in opp_types or "Ground" in opp_types:
            return 0.0
        return 1.8
    if mid in ("willowisp",):
        if opp_has_status or "Fire" in opp_types:
            return 0.0
        return 2.2
    if mid in ("toxic", "poisonpowder"):
        if opp_has_status or "Poison" in opp_types or "Steel" in opp_types:
            return 0.0
        return 2.0
    if mid in ("spore", "hypnosis", "sleeppowder", "yawn", "sing"):
        if opp_has_status or ("Grass" in opp_types and mid in ("spore", "sleeppowder")):
            return 0.0
        return 2.5

    # 4. Recovery Moves
    if mid in ("recover", "roost", "softboiled", "slackoff", "milkdrink", "synthesis", "moonlight", "morningsun", "wish", "shoreup"):
        if my_hp_pct >= 0.75:
            return 0.0  # Already healthy, do not waste turn
        if my_hp_pct < 0.45:
            return 3.5  # Critical healing
        return 1.5

    # 5. Setup Boosts
    if mid in ("swordsdance", "nastyplot", "dragondance", "calmmind", "quiverdance", "bulkup", "agility"):
        if my_hp_pct < 0.35:
            return 0.0  # Too low HP to setup safely
        return 1.8

    # 6. Taunt
    if mid == "taunt":
        return 1.5

    return 1.0


def get_hazard_damage_pct(pokemon_name: str, side_hazards: Dict[str, int], item: Optional[str] = None) -> float:
    """Estimates hazard damage percentage taken when switching in."""
    if item and clean_id(item) == "heavydutyboots":
        return 0.0

    data = ShowdownData.get()
    poke = data.get_pokemon(pokemon_name)
    types = poke.get("types", []) if poke else []
    is_airborne = "Flying" in types or (poke and poke.get("abilities", {}).get("0") == "Levitate")

    total_dmg = 0.0

    # Stealth Rock (Rock damage, modified by type effectiveness)
    if side_hazards.get("stealthrock", 0) > 0:
        rock_mult = get_type_multiplier("Rock", types)
        total_dmg += 0.125 * rock_mult

    # Spikes (Grounded only)
    spikes_layers = side_hazards.get("spikes", 0)
    if spikes_layers > 0 and not is_airborne:
        spikes_dmg = [0.0, 0.125, 0.1667, 0.25][min(3, spikes_layers)]
        total_dmg += spikes_dmg

    return min(1.0, total_dmg)

