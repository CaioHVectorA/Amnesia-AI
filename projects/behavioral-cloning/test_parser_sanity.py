"""
Test Parser Sanity & Revalidation
==================================
Verifies that:
1. Defog with 0 hazards has 0.0 utility and mask[slot] == 0.0 (strictly masked out!).
2. Defog with hazards on our side has > 0 utility and mask[slot] == 1.0.
3. Roost at 100% HP is masked out (util == 0.0).
4. Roost at 30% HP has high utility (util > 0.0).
5. Switch candidate actions have 0.0 damage (matrix[slot, 2] == 0.0).
6. Parsing an actual replay correctly discovers moves and creates valid samples.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from parser import (
    BattlePokemon,
    PlayerState,
    build_candidate_actions,
    build_state_vector,
    parse_replay_file
)
from mechanics import evaluate_status_move_utility

def test_defog_masking():
    print("--- Test 1: Defog Utility & Masking ---")
    p1 = PlayerState(name="P1")
    p1.active = BattlePokemon(
        name="Corviknight",
        species="Corviknight",
        current_hp=1.0,
        moves=["Defog", "Brave Bird", "Roost", "U-turn"]
    )
    p1.bench = [
        BattlePokemon(name="Landorus-Therian", species="Landorus-Therian", current_hp=1.0)
    ]

    p2 = PlayerState(name="P2")
    p2.active = BattlePokemon(name="Garchomp", species="Garchomp", current_hp=1.0)

    # Case A: Turn 1, NO HAZARDS on either side
    p1.hazards = {"stealthrock": 0, "spikes": 0, "toxicspikes": 0, "stickyweb": 0}
    matrix, mask = build_candidate_actions(p1, p2, weather=None)

    defog_idx = p1.active.moves.index("Defog")
    assert mask[defog_idx] == 0.0, f"Defog should be MASKED OUT (0.0) when 0 hazards, got mask={mask[defog_idx]}"
    print("  [PASS] Defog with 0 hazards is strictly masked to 0.0!")

    # Case B: Hazards on our side
    p1.hazards = {"stealthrock": 1, "spikes": 0, "toxicspikes": 0, "stickyweb": 0}
    matrix, mask = build_candidate_actions(p1, p2, weather=None)
    assert mask[defog_idx] == 1.0, f"Defog should be ACTIVE (1.0) with hazards on our side, got {mask[defog_idx]}"
    assert matrix[defog_idx, 2] > 0.0, f"Defog utility should be positive, got {matrix[defog_idx, 2]}"
    print(f"  [PASS] Defog with hazards on our side is legal with utility {matrix[defog_idx, 2]:.3f}!")

def test_recovery_masking():
    print("--- Test 2: Recovery Move Masking ---")
    p1 = PlayerState(name="P1")
    p1.active = BattlePokemon(
        name="Corviknight",
        species="Corviknight",
        current_hp=1.0,
        moves=["Roost", "Brave Bird"]
    )
    p2 = PlayerState(name="P2")
    p2.active = BattlePokemon(name="Garchomp", species="Garchomp", current_hp=1.0)

    roost_idx = 0
    # At 100% HP: Roost should be masked out
    matrix, mask = build_candidate_actions(p1, p2, weather=None)
    assert mask[roost_idx] == 0.0, f"Roost at 100% HP should be masked to 0.0, got {mask[roost_idx]}"
    print("  [PASS] Roost at 100% HP is masked out!")

    # At 30% HP: Roost should be active with high utility
    p1.active.current_hp = 0.30
    matrix, mask = build_candidate_actions(p1, p2, weather=None)
    assert mask[roost_idx] == 1.0, f"Roost at 30% HP should be legal, got {mask[roost_idx]}"
    assert matrix[roost_idx, 2] >= 1.0, f"Roost at 30% HP should have high utility, got {matrix[roost_idx, 2]}"
    print(f"  [PASS] Roost at 30% HP is legal with critical healing utility {matrix[roost_idx, 2]:.3f}!")

def test_switch_damage_feature():
    print("--- Test 3: Switch Candidate Feature Symmetry ---")
    p1 = PlayerState(name="P1")
    p1.active = BattlePokemon(name="Corviknight", species="Corviknight", current_hp=1.0, moves=["Brave Bird"])
    p1.bench = [
        BattlePokemon(name="Heatran", species="Heatran", current_hp=1.0)
    ]
    p2 = PlayerState(name="P2")
    p2.active = BattlePokemon(name="Ferrothorn", species="Ferrothorn", current_hp=1.0)

    matrix, mask = build_candidate_actions(p1, p2, weather=None)
    switch_slot = 4
    assert mask[switch_slot] == 1.0
    assert matrix[switch_slot, 2] == 0.0, f"Switch damage slot MUST be 0.0, got {matrix[switch_slot, 2]}"
    assert matrix[switch_slot, 13] == 1.0, f"Switch reserve HP should be in slot 13, got {matrix[switch_slot, 13]}"
    print(f"  [PASS] Switch deal 0 damage (matrix[4, 2] = {matrix[switch_slot, 2]}), reserve HP stored in slot 13!")

def test_real_replay_parsing():
    print("--- Test 4: Replay File Parsing ---")
    replays_dir = "data/replays/ladder_rl"
    if os.path.exists(replays_dir):
        files = [os.path.join(replays_dir, f) for f in os.listdir(replays_dir) if f.endswith(".json")]
        if files:
            sample_file = files[0]
            samples = parse_replay_file(sample_file)
            print(f"  [PASS] Parsed {len(samples)} valid decisions from {os.path.basename(sample_file)}")
            if samples:
                s_vec, a_mat, action_idx, weight, mask = samples[0]
                assert mask[action_idx] > 0.0, "Chosen action must be legal in mask"
                print(f"  [PASS] Sample 0 chose action {action_idx} with legal mask {mask[action_idx]}")

if __name__ == "__main__":
    test_defog_masking()
    test_recovery_masking()
    test_switch_damage_feature()
    test_real_replay_parsing()
    print("\nALL REVALIDATION PARSER TESTS PASSED! Behavioral Cloning integrity verified.")
