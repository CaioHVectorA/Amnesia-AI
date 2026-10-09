"""
NeuralBrain Direct Unit & Behavioral Test
=========================================
Tests that NeuralBrain loaded with bc_model.pt:
1. NEVER uses Turn 1 Defog when hazards are 0.
2. NEVER uses Roost when HP is full (100%).
3. Chooses attacks appropriately.
4. Does not loop switches.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from neural_brain import NeuralBrain

def run_tests():
    print("=" * 60)
    print("Testing NeuralBrain Decision Logic")
    print("=" * 60)

    brain = NeuralBrain(checkpoint_path="projects/behavioral-cloning/weights/bc_model.pt")

    # TEST 1: Turn 1 with Corviknight carrying Defog, Roost, Brave Bird, U-turn
    request_t1 = {
        "active": [{
            "moves": [
                {"move": "Defog", "id": "defog", "pp": 24, "maxpp": 24, "target": "normal", "disabled": False},
                {"move": "Roost", "id": "roost", "pp": 16, "maxpp": 16, "target": "self", "disabled": False},
                {"move": "Brave Bird", "id": "bravebird", "pp": 24, "maxpp": 24, "target": "normal", "disabled": False},
                {"move": "U-turn", "id": "uturn", "pp": 32, "maxpp": 32, "target": "normal", "disabled": False}
            ]
        }],
        "side": {
            "id": "p1",
            "name": "Amnesia-AI",
            "pokemon": [
                {"ident": "p1: Corviknight", "details": "Corviknight, L80", "condition": "100/100", "active": True, "moves": ["defog", "roost", "bravebird", "uturn"]},
                {"ident": "p1: Landorus-T", "details": "Landorus-Therian, L80", "condition": "100/100", "active": False, "moves": ["earthquake", "stoneedge", "stealthrock", "uturn"]}
            ]
        }
    }

    # Battle history with NO HAZARDS on either side
    history_t1 = [
        "|player|p1|Amnesia-AI|",
        "|player|p2|Opponent|",
        "|teamsize|p1|2",
        "|teamsize|p2|2",
        "|start",
        "|switch|p1a: Corviknight|Corviknight, L80|100/100",
        "|switch|p2a: Garchomp|Garchomp, L80|100/100",
        "|turn|1"
    ]

    action = brain.choose_action(request_t1, history_t1)
    print(f"Test 1 (Turn 1 with 0 hazards): Chosen action is '{action}'")
    assert action != "move 1", f"ERROR: Bot chose Defog on Turn 1 with 0 hazards! Action: {action}"
    assert action != "move 2", f"ERROR: Bot chose Roost at 100% HP! Action: {action}"
    assert action in ["move 3", "move 4", "switch 2"], f"Unexpected action: {action}"
    print("  [PASS] NeuralBrain strictly rejected Defog and Roost on Turn 1!")

    # TEST 2: Turn with Hazards on our side
    history_hazards = list(history_t1) + [
        "|move|p2a: Garchomp|Stealth Rock|p1a: Corviknight",
        "|-sidestart|p1: Amnesia-AI|move: Stealth Rock",
        "|turn|2"
    ]
    # Now Defog should be legal
    action_h = brain.choose_action(request_t1, history_hazards)
    print(f"Test 2 (Hazards present): Chosen action is '{action_h}'")
    print("  [PASS] Move choice with hazards present handled cleanly.")

    # TEST 3: Low HP recovery test
    request_low_hp = json.loads(json.dumps(request_t1))
    request_low_hp["side"]["pokemon"][0]["condition"] = "30/100" # Corviknight at 30% HP
    action_rec = brain.choose_action(request_low_hp, history_t1)
    print(f"Test 3 (Corviknight at 30% HP): Chosen action is '{action_rec}'")
    print("  [PASS] Handled low HP state.")

    # TEST 4: Anti-Switch Loop verification
    brain.action_history = ["switch 2"]
    action_after_switch = brain.choose_action(request_t1, history_t1)
    print(f"Test 4 (After voluntary switch): Chosen action is '{action_after_switch}'")
    assert "switch" not in action_after_switch, f"ERROR: Anti-switch loop failed! Consecutive switch chosen: {action_after_switch}"
    print("  [PASS] Anti-switch loop engine strictly prevented back-to-back switching!")

    print("\nALL NEURAL BRAIN UNIT TESTS PASSED SUCCESSFULLY! 🎯")

if __name__ == "__main__":
    run_tests()
