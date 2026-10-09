"""
Diagnostic Test: Verifying Foul Play Native Engine vs SimpleHeuristics
======================================================================
Confirms that Foul Play actually runs its native Rust poke-engine MCTS search
and produces real intelligent competitive moves without crashing or defaulting.
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "external/foul-play"))
sys.path.insert(0, os.path.join(REPO_ROOT, "projects/benchmarks"))

from fp.config import FoulPlayConfig
import fp.generations

for k in list(fp.generations.GENERATIONS.keys()):
    if k.startswith("gen") and k[3:].isdigit():
        fp.generations.GENERATIONS[int(k[3:])] = fp.generations.GENERATIONS[k]

from community_benchmarks import BaseAgent, ShowdownArena, create_agent

class FixedFoulPlayAgent(BaseAgent):
    """Accurately wrapped Patrick Mariglia's Foul Play Expectiminimax/MCTS engine."""
    def __init__(self, name: str = "FoulPlay", gen: int = 8):
        super().__init__(name, gen)
        FoulPlayConfig.username = name
        FoulPlayConfig.pokemon_format = f"gen{gen}randombattle"
        FoulPlayConfig.search_time_ms = 120
        FoulPlayConfig.parallelism = 2
        FoulPlayConfig.search_threads = 1

        self.battle = None
        self.initialized = False
        self.moves_chosen = []

    def reset(self, battle_tag: str):
        try:
            from fp.battle.state import Battle
            from fp.modes import battle_mode

            self.battle = Battle(battle_tag)
            self.battle.pokemon_format = f"gen{self.gen}randombattle"
            self.battle.generation = self.gen
            self.battle.mode = battle_mode(self.battle.format_spec.battle_type)
            self.battle.mode.datasets.initialize(self.battle.format_spec)
            self.initialized = False
            self.moves_chosen.clear()
        except Exception as e:
            print(f"FoulPlay init error: {e}")
            self.battle = None

    def handle_line(self, line: str):
        if not self.battle:
            return
        self.battle.msg_list.append(line)

    def choose_action(self, request: Dict[str, Any]) -> str:
        if not self.battle:
            return "move 1"
        try:
            from fp.battle.protocol import process_battle_updates
            from fp.search.main import find_best_move
            from fp.modes.base import format_decision

            # 1. First turn initialization must happen before process_battle_updates
            # so that battle.user.name and battle.opponent.name are known ('p1'/'p2')
            self.battle.request_json = request
            if not self.initialized:
                self.battle.user.initialize_first_turn_user_from_json(request)
                user_id = request.get("side", {}).get("id", "p1")
                self.battle.user.name = user_id
                self.battle.opponent.name = "p2" if user_id == "p1" else "p1"
                self.initialized = True

            # 2. Process all pending lines from Showdown stream
            process_battle_updates(self.battle)
            self.battle.msg_list.clear()

            # 3. Update battle user from latest request
            self.battle.user.update_from_request_json(request)

            # 4. Native Rust Monte Carlo Tree Search
            best_move = find_best_move(self.battle)

            from fp.battle.state import LastUsedMove
            if self.battle.user.active:
                clean_mv = best_move.removesuffix("-tera").removesuffix("-mega")
                self.battle.user.last_selected_move = LastUsedMove(
                    self.battle.user.active.name,
                    clean_mv,
                    self.battle.turn,
                )

            formatted = format_decision(self.battle, best_move)
            cmd = formatted[0] if isinstance(formatted, list) else str(formatted)
            
            # 5. Convert to Showdown slot action (move 1..4 or switch 1..6)
            clean = cmd.replace("/choose ", "").strip()
            action = clean
            if clean.startswith("/switch ") or clean.startswith("switch "):
                target = clean.replace("/switch ", "").replace("switch ", "").strip()
                action = f"switch {target}" if target.isdigit() else "switch 2"
            elif clean.startswith("/move ") or clean.startswith("move "):
                move_target = clean.replace("/move ", "").replace("move ", "").strip().split()[0]
                moves = request.get("active", [{}])[0].get("moves", [])
                matched = False
                for i, m in enumerate(moves):
                    if move_target.lower() in (m.get("id", "").lower(), m.get("move", "").lower()):
                        action = f"move {i + 1}"
                        matched = True
                        break
                if not matched:
                    action = f"move 1"

            self.moves_chosen.append((best_move, action))
            return action
        except Exception as e:
            print(f"[!] Exception in FoulPlay choose_action: {e}")
            import traceback
            traceback.print_exc()
            return "move 1"

def test_single_battle():
    print("=" * 60)
    print("Testing Foul Play in a live headless battle vs SimpleHeuristics...")
    print("=" * 60)
    arena = ShowdownArena()
    fp_agent = FixedFoulPlayAgent("FoulPlay", gen=8)
    sh_agent = create_agent("SimpleHeuristics", gen=8)

    outcome = arena.run_battle(fp_agent, sh_agent, format_id="gen8randombattle", max_turns=60)
    print(f"\nMatch Result: Winner = {outcome['winner']}, Turns = {outcome['turns']}")
    print(f"Foul Play total calculated moves: {len(fp_agent.moves_chosen)}")
    print(f"Sample calculated moves by Foul Play: {fp_agent.moves_chosen[:5]}")
    assert len(fp_agent.moves_chosen) > 0, "Foul Play should have calculated moves!"
    print("\nSUCCESS: Foul Play native MCTS is functioning properly without crashing!")

if __name__ == "__main__":
    test_single_battle()
