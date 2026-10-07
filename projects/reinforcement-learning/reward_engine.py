"""
Potential-Based Reward Shaping (PBRS) Engine for Pokemon RL
============================================================
Calculates smooth, dense, non-exploitable rewards from state transitions:
1. Terminal Reward: +1.0 (Win) / -1.0 (Loss)
2. HP Differential: +0.40 * (dHP_opp) - 0.40 * (dHP_me)
3. KO Reward: +0.50 (Opp Fainted) / -0.50 (Me Fainted)
4. Opponent Boost Hazard Penalty: -0.20 per dangerous stat stage
5. Sleep Clause Penalty: -0.40 for illegal/wasted sleep attempts
"""

from typing import Dict, Any, Optional


class RewardEngine:
    def __init__(
        self,
        w_dmg: float = 0.40,
        w_dmg_taken: float = 0.40,
        w_ko: float = 0.50,
        w_faint: float = 0.50,
        w_boost_threat: float = 0.20,
        w_sleep_penalty: float = 0.40,
        w_win: float = 1.0,
        gamma: float = 0.99
    ):
        self.w_dmg = w_dmg
        self.w_dmg_taken = w_dmg_taken
        self.w_ko = w_ko
        self.w_faint = w_faint
        self.w_boost_threat = w_boost_threat
        self.w_sleep_penalty = w_sleep_penalty
        self.w_win = w_win
        self.gamma = gamma

    def calculate_turn_reward(
        self,
        prev_state: Dict[str, Any],
        curr_state: Dict[str, Any],
        action_taken: int,
        is_terminal: bool = False,
        won: bool = False
    ) -> float:
        if is_terminal:
            return self.w_win if won else -self.w_win

        reward = 0.0

        # 1. HP Differentials
        prev_my_hp = prev_state.get("my_hp", 1.0)
        curr_my_hp = curr_state.get("my_hp", 1.0)
        prev_opp_hp = prev_state.get("opp_hp", 1.0)
        curr_opp_hp = curr_state.get("opp_hp", 1.0)

        delta_opp_hp = max(0.0, prev_opp_hp - curr_opp_hp)
        delta_my_hp = max(0.0, prev_my_hp - curr_my_hp)

        reward += self.w_dmg * delta_opp_hp
        reward -= self.w_dmg_taken * delta_my_hp

        # 2. Knockouts (KOs)
        prev_opp_alive = prev_state.get("opp_alive", 6)
        curr_opp_alive = curr_state.get("opp_alive", 6)
        prev_my_alive = prev_state.get("my_alive", 6)
        curr_my_alive = curr_state.get("my_alive", 6)

        if curr_opp_alive < prev_opp_alive:
            reward += self.w_ko * (prev_opp_alive - curr_opp_alive)

        if curr_my_alive < prev_my_alive:
            reward -= self.w_faint * (prev_my_alive - curr_my_alive)

        # 3. Opponent Dangerous Boost Penalty
        prev_opp_threat = sum(prev_state.get("opp_boosts", {}).values())
        curr_opp_threat = sum(curr_state.get("opp_boosts", {}).values())
        delta_threat = curr_opp_threat - prev_opp_threat
        if delta_threat > 0:
            reward -= self.w_boost_threat * delta_threat

        # 4. Sleep Clause Protection Penalty
        attempted_sleep = curr_state.get("attempted_invalid_sleep", False)
        if attempted_sleep:
            reward -= self.w_sleep_penalty

        return float(reward)
