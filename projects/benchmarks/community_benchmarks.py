"""
Community Benchmark Suite for Amnesia-AI
========================================
Runs headless, authentic Showdown battles against the 5 real community bots:
1. Patrick Mariglia's Foul Play (Expectiminimax / MCTS via Rust poke-engine)
2. Metamon KaizoPlus (UT-Austin RPL Expert Heuristic)
3. Metamon EmeraldKaizo (UT-Austin RPL Competitive AI)
4. poke-env SimpleHeuristicsPlayer (Haris Sahovic)
5. poke-env MaxBasePowerPlayer (Haris Sahovic)
6. Metamon WinsOnlyRNN (Reference Behavioral Cloning policy)

Driven by the official Pokemon Showdown simulate-battle engine via stdin/stdout.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark")

# Setup module search paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "external/metamon"))
sys.path.insert(0, os.path.join(REPO_ROOT, "external/foul-play"))
sys.path.insert(0, os.path.join(REPO_ROOT, "projects/behavioral-cloning"))
sys.path.insert(0, os.path.join(REPO_ROOT, "projects/showdown-agent"))

# Fix Gen 8 in Metamon type chart
try:
    from metamon.baselines.base import GEN_DATA
    if 9 in GEN_DATA and 8 not in GEN_DATA:
        GEN_DATA[8] = GEN_DATA[9]
except Exception:
    pass

from poke_env.environment.battle import Battle as PokeEnvBattle
from poke_env.player import SimpleHeuristicsPlayer, MaxBasePowerPlayer
from metamon.baselines.heuristic.kaizoplus import KaizoPlus
from metamon.baselines.heuristic.kaizo import EmeraldKaizo


class BaseAgent(ABC):
    def __init__(self, name: str, gen: int = 8):
        self.name = name
        self.gen = gen

    @abstractmethod
    def reset(self, battle_tag: str):
        pass

    @abstractmethod
    def handle_line(self, line: str):
        pass

    @abstractmethod
    def choose_action(self, request: Dict[str, Any]) -> str:
        """Returns standard showdown action string, e.g. 'move 1' or 'switch 2'"""
        pass


class PokeEnvAgentAdapter(BaseAgent):
    """Wraps any poke-env or metamon Player heuristic into the Showdown stream."""
    def __init__(self, name: str, player_cls, gen: int = 8):
        super().__init__(name, gen)
        self.player = player_cls(start_listening=False)
        self.battle: Optional[PokeEnvBattle] = None

    def reset(self, battle_tag: str):
        self.battle = PokeEnvBattle(battle_tag, self.name, logging.getLogger(self.name), gen=self.gen)

    def handle_line(self, line: str):
        if not self.battle:
            return
        parts = line.split("|")
        if len(parts) >= 2:
            try:
                self.battle.parse_message(parts)
            except Exception:
                pass

    def choose_action(self, request: Dict[str, Any]) -> str:
        if not self.battle:
            return "move 1"
        self.battle.parse_request(request)
        try:
            order = self.player.choose_move(self.battle)
            msg = order.message if hasattr(order, "message") else str(order)
            msg = msg.replace("/choose ", "").strip()
            if msg and msg != "default":
                return msg
        except Exception as e:
            logger.debug(f"[{self.name}] Exception in choose_move: {e}")

        # Guaranteed legal fallback
        side_pokemon = request.get("side", {}).get("pokemon", [])
        is_force = bool(request.get("forceSwitch", [False])[0])
        if is_force:
            for i, bp in enumerate(side_pokemon):
                cond = bp.get("condition", "")
                if "fnt" not in cond and not cond.startswith("0") and not bp.get("active", False):
                    return f"switch {i + 1}"
        moves = request.get("active", [{}])[0].get("moves", [])
        for i, m in enumerate(moves):
            if not m.get("disabled", False) and m.get("pp", 1) > 0:
                return f"move {i + 1}"
        return "move 1"


class FoulPlayAgent(BaseAgent):
    """Wraps Patrick Mariglia's Foul Play Expectiminimax/MCTS engine (Native Rust poke-engine)."""
    def __init__(self, name: str = "FoulPlay", gen: int = 8):
        super().__init__(name, gen)
        from fp.config import FoulPlayConfig
        import fp.generations

        # Map integer generations to string keys in fp.generations.GENERATIONS
        for k in list(fp.generations.GENERATIONS.keys()):
            if isinstance(k, str) and k.startswith("gen") and k[3:].isdigit():
                fp.generations.GENERATIONS[int(k[3:])] = fp.generations.GENERATIONS[k]

        FoulPlayConfig.username = name
        FoulPlayConfig.pokemon_format = f"gen{gen}randombattle"
        FoulPlayConfig.search_time_ms = 120
        FoulPlayConfig.parallelism = 2
        FoulPlayConfig.search_threads = 1

        self.battle = None
        self.initialized = False

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
        except Exception as e:
            logger.warning(f"FoulPlay init warning: {e}")
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
            from fp.battle.state import LastUsedMove

            # 1. First turn initialization must happen before process_battle_updates
            # so that battle.user.name and battle.opponent.name are known ('p1'/'p2')
            self.battle.request_json = request
            self.battle.rqid = request.get("rqid", 1)
            self.battle.force_switch = bool(request.get("forceSwitch", [False])[0])
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

            if best_move == "No Move" or (self.battle.force_switch and not best_move.startswith("switch ")):
                for p in self.battle.user.reserve:
                    if p.hp > 0:
                        best_move = f"switch {p.name}"
                        break

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
                    if move_target.lower() in (m.get("id", "").lower(), m.get("move", "").lower()) and not m.get("disabled", False) and m.get("pp", 1) > 0:
                        action = f"move {i + 1}"
                        matched = True
                        break
                if not matched:
                    # Pick first non-disabled move with PP
                    for i, m in enumerate(moves):
                        if not m.get("disabled", False) and m.get("pp", 1) > 0:
                            action = f"move {i + 1}"
                            break
                    else:
                        action = "move 1"

            # 6. Strict legality checks (ForceSwitch & Switch validity)
            side_pokemon = request.get("side", {}).get("pokemon", [])
            is_force_switch = bool(request.get("forceSwitch", [False])[0])

            if is_force_switch and not action.startswith("switch "):
                for i, bp in enumerate(side_pokemon):
                    b_cond = bp.get("condition", "")
                    if "fnt" not in b_cond and not b_cond.startswith("0") and not bp.get("active", False):
                        action = f"switch {i + 1}"
                        break

            if action.startswith("switch "):
                slot_str = action.split()[1]
                slot_num = int(slot_str) if slot_str.isdigit() else 2
                if 1 <= slot_num <= len(side_pokemon):
                    p = side_pokemon[slot_num - 1]
                    cond = p.get("condition", "")
                    if "fnt" in cond or cond.startswith("0") or p.get("active", False):
                        for i, bp in enumerate(side_pokemon):
                            b_cond = bp.get("condition", "")
                            if "fnt" not in b_cond and not b_cond.startswith("0") and not bp.get("active", False):
                                action = f"switch {i + 1}"
                                break

            if not action.startswith("move ") and not action.startswith("switch "):
                moves = request.get("active", [{}])[0].get("moves", [])
                for i, m in enumerate(moves):
                    if not m.get("disabled", False) and m.get("pp", 1) > 0:
                        action = f"move {i + 1}"
                        break
                else:
                    action = "move 1"

            return action
        except Exception as e:
            logger.error(f"[FoulPlay] Error in search: {e}")
            side_pokemon = request.get("side", {}).get("pokemon", [])
            is_force_switch = bool(request.get("forceSwitch", [False])[0])
            if is_force_switch:
                for i, bp in enumerate(side_pokemon):
                    b_cond = bp.get("condition", "")
                    if "fnt" not in b_cond and not b_cond.startswith("0") and not bp.get("active", False):
                        return f"switch {i + 1}"
            moves = request.get("active", [{}])[0].get("moves", [])
            for i, m in enumerate(moves):
                if not m.get("disabled", False) and m.get("pp", 1) > 0:
                    return f"move {i + 1}"
            return "move 1"


class AmnesiaNeuralAgent(BaseAgent):
    """Wraps Amnesia-AI's Neural Brain policy."""
    def __init__(self, name: str = "Amnesia-AI", gen: int = 8, checkpoint_path: Optional[str] = None):
        super().__init__(name, gen)
        from neural_brain import NeuralBrain
        self.brain = NeuralBrain(checkpoint_path=checkpoint_path or "projects/behavioral-cloning/weights/bc_model.pt")
        self.history: List[str] = []

    def reset(self, battle_tag: str):
        self.history.clear()
        self.brain.reset_battle_state()

    def handle_line(self, line: str):
        self.history.append(line)

    def choose_action(self, request: Dict[str, Any]) -> str:
        return self.brain.choose_action(request, self.history)


def create_agent(agent_name: str, gen: int = 8, checkpoint_path: Optional[str] = None) -> BaseAgent:
    lower = agent_name.lower()
    if "foul" in lower or "play" in lower:
        return FoulPlayAgent("FoulPlay", gen=gen)
    elif "kaizoplus" in lower:
        return PokeEnvAgentAdapter("KaizoPlus", KaizoPlus, gen=gen)
    elif "kaizo" in lower:
        return PokeEnvAgentAdapter("EmeraldKaizo", EmeraldKaizo, gen=gen)
    elif "heuristic" in lower or "simple" in lower:
        return PokeEnvAgentAdapter("SimpleHeuristics", SimpleHeuristicsPlayer, gen=gen)
    elif "maxbp" in lower or "basepower" in lower:
        return PokeEnvAgentAdapter("MaxBasePower", MaxBasePowerPlayer, gen=gen)
    elif "amnesia" in lower:
        return AmnesiaNeuralAgent("Amnesia-AI", gen=gen, checkpoint_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


class ShowdownArena:
    """Orchestrates 1-on-1 battles via Showdown's headless simulate-battle CLI."""
    def __init__(self, showdown_path: str = "external/pokemon-showdown/pokemon-showdown"):
        self.showdown_path = os.path.abspath(os.path.join(REPO_ROOT, showdown_path))
        self.bun_bin = "/home/usuario/.bun/bin/bun"

    def run_battle(
        self,
        agent_p1: BaseAgent,
        agent_p2: BaseAgent,
        format_id: str = "gen8randombattle",
        p1_team: Optional[str] = None,
        p2_team: Optional[str] = None,
        max_turns: int = 100
    ) -> Dict[str, Any]:
        battle_tag = f"battle-{format_id}-{int(time.time()*1000)%1000000}"
        agent_p1.reset(battle_tag)
        agent_p2.reset(battle_tag)

        cmd = [self.bun_bin, self.showdown_path, "simulate-battle", "--skip-build"]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Start message
        start_payload = {"formatid": format_id}
        proc.stdin.write(f">start {json.dumps(start_payload)}\n")

        p1_spec = {"name": agent_p1.name}
        if p1_team: p1_spec["team"] = p1_team
        proc.stdin.write(f">player p1 {json.dumps(p1_spec)}\n")

        p2_spec = {"name": agent_p2.name}
        if p2_team: p2_spec["team"] = p2_team
        proc.stdin.write(f">player p2 {json.dumps(p2_spec)}\n")
        proc.stdin.flush()

        winner = None
        turn_count = 0
        switches_p1 = 0
        moves_p1 = 0
        switches_p2 = 0
        moves_p2 = 0

        current_target = None
        current_request = None

        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line_str = line.strip()

            if line_str.startswith("|win|"):
                winner = line_str.replace("|win|", "").strip()
                break
            elif line_str == "|tie" or line_str.startswith("|tie|"):
                winner = "tie"
                break
            elif line_str.startswith("|turn|"):
                try:
                    turn_count = int(line_str.split("|")[2])
                except Exception:
                    turn_count += 1
                if turn_count > max_turns:
                    winner = "tie (timeout)"
                    break

            if line_str.startswith("|error|[Invalid choice]"):
                logger.warning(f"Showdown invalid choice: {line_str}. Sending recovery actions.")
                try:
                    proc.stdin.write(">p1 move 1\n>p2 move 1\n")
                    proc.stdin.flush()
                except Exception:
                    pass

            # Forward lines to agents
            agent_p1.handle_line(line_str)
            agent_p2.handle_line(line_str)

            # Detect side update for requests
            if line_str == "sideupdate":
                target_side = proc.stdout.readline().strip()  # 'p1' or 'p2'
                req_line = proc.stdout.readline().strip()
                if req_line.startswith("|request|"):
                    req_json_str = req_line[len("|request|"):].strip()
                    if req_json_str:
                        try:
                            req_data = json.loads(req_json_str)
                            if not req_data.get("wait", False):
                                if target_side == "p1":
                                    act = agent_p1.choose_action(req_data)
                                    if "switch" in act: switches_p1 += 1
                                    else: moves_p1 += 1
                                    proc.stdin.write(f">p1 {act}\n")
                                    proc.stdin.flush()
                                elif target_side == "p2":
                                    act = agent_p2.choose_action(req_data)
                                    if "switch" in act: switches_p2 += 1
                                    else: moves_p2 += 1
                                    proc.stdin.write(f">p2 {act}\n")
                                    proc.stdin.flush()
                        except Exception as e:
                            logger.debug(f"Request parse error: {e}")

        try:
            proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            pass

        return {
            "winner": winner,
            "turns": turn_count,
            "p1_stats": {"name": agent_p1.name, "moves": moves_p1, "switches": switches_p1},
            "p2_stats": {"name": agent_p2.name, "moves": moves_p2, "switches": switches_p2}
        }


def run_tournament(
    hero_agent_name: str,
    opponent_names: List[str],
    n_battles: int = 10,
    format_id: str = "gen8randombattle",
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    arena = ShowdownArena()
    results = {}

    print("=" * 70)
    print(f"⚔️  COMMUNITY BENCHMARK ARENA: {hero_agent_name} vs REAL COMMUNITY BOTS")
    print(f"Format: {format_id} | Battles per Opponent: {n_battles}")
    print("=" * 70)

    hero = create_agent(hero_agent_name, checkpoint_path=checkpoint_path)

    for opp_name in opponent_names:
        print(f"\n▶ Battling against {opp_name} ({n_battles} matches)...", flush=True)
        opp = create_agent(opp_name)
        wins = 0
        losses = 0
        ties = 0
        total_turns = 0
        total_hero_switches = 0
        total_hero_moves = 0

        for b_idx in range(1, n_battles + 1):
            outcome = arena.run_battle(hero, opp, format_id=format_id)
            w = outcome["winner"]
            total_turns += outcome["turns"]
            total_hero_switches += outcome["p1_stats"]["switches"]
            total_hero_moves += outcome["p1_stats"]["moves"]

            if w == hero.name:
                wins += 1
                status = "WIN "
            elif w == "tie" or "timeout" in str(w):
                ties += 1
                status = "TIE "
            else:
                losses += 1
                status = "LOSS"

            print(f"  Match {b_idx:02d}/{n_battles:02d} | Result: {status} | Turns: {outcome['turns']:02d} | Hero Switches: {outcome['p1_stats']['switches']}", flush=True)

        win_rate = (wins / n_battles) * 100.0
        avg_turns = total_turns / n_battles
        avg_switches = total_hero_switches / n_battles
        avg_moves = total_hero_moves / n_battles

        results[opp_name] = {
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "avg_turns": avg_turns,
            "avg_hero_switches": avg_switches,
            "avg_hero_moves": avg_moves
        }

        print(f"  --> Final vs {opp_name}: {win_rate:.1f}% Win Rate ({wins}W / {losses}L / {ties}T)", flush=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Headless Community Benchmark Battles")
    parser.add_argument("--hero", type=str, default="Amnesia-AI", help="Hero agent to evaluate")
    parser.add_argument("--opponents", nargs="+", default=["SimpleHeuristics", "MaxBasePower", "KaizoPlus", "EmeraldKaizo", "FoulPlay"])
    parser.add_argument("--battles", type=int, default=5, help="Number of battles per opponent")
    parser.add_argument("--format", type=str, default="gen8randombattle", help="Battle format")
    parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint")
    parser.add_argument("--output", type=str, default="projects/benchmarks/benchmark_results.json")

    args = parser.parse_args()
    res = run_tournament(
        args.hero,
        args.opponents,
        n_battles=args.battles,
        format_id=args.format,
        checkpoint_path=args.checkpoint
    )

    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[+] Benchmark results saved to: {args.output}")
