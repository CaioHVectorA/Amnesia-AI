"""
Dynamic Potential-Based Matrix Reward Engine (State-Adaptive Advantage & Anti-Loop)
=====================================================================================
Calculates context-sensitive, dynamic rewards scaling with game phase, speed control,
type advantage, clean KOs, and penalizes repetitive/ineffective loops.

Formulation:
R(s, a, s') = Gamma * Phi(s') - Phi(s) + R_aux(s, a, s')
where Phi(s) is the Dynamic Potential Matrix evaluated across:
1. Material Advantage (Alive count differential modulated by phase)
2. HP Ratio & Clean KO thresholds
3. Speed Tier Control (faster mon gets proactive tempo weight)
4. Defensive Type Positioning (resistances on active matchup)
5. Strict Anti-Loop & Redundancy Penalties
"""

import numpy as np
from typing import Dict, Any, List, Optional


class DynamicMatrixRewardEngine:
    def __init__(
        self,
        gamma: float = 0.99,
        w_terminal: float = 2.0,
        w_clean_ko: float = 0.8,
        w_faint: float = 0.8,
        w_loop_penalty: float = 1.0,
        w_status_waste_penalty: float = 0.6
    ):
        self.gamma = gamma
        self.w_terminal = w_terminal
        self.w_clean_ko = w_clean_ko
        self.w_faint = w_faint
        self.w_loop_penalty = w_loop_penalty
        self.w_status_waste_penalty = w_status_waste_penalty

    def calculate_potential(self, state: Dict[str, Any]) -> float:
        """
        Computes the dense scalar potential Phi(s) for potential-based shaping.
        Guarantees policy invariance while guiding exploration smoothly.
        """
        my_alive = state.get("my_alive", 6)
        opp_alive = state.get("opp_alive", 6)
        my_hp = state.get("my_hp", 1.0)
        opp_hp = state.get("opp_hp", 1.0)
        
        # Game Phase Weight (Early game = 1.0, Late game = 2.0 for tighter precision)
        total_alive = my_alive + opp_alive
        phase_weight = 1.0 + (12.0 - total_alive) / 12.0

        # 1. Material Differential
        material_pot = (my_alive - opp_alive) * 0.75 * phase_weight

        # 2. HP Differential
        hp_pot = (my_hp - opp_hp) * 0.50

        # 3. Tempo / Speed Advantage
        speed_pot = 0.20 if state.get("i_am_faster", False) else -0.10

        # 4. Defensive Type Resistance
        type_res = state.get("type_resistance_factor", 1.0)
        defensive_pot = 0.25 * (1.0 - min(2.0, type_res))

        return float(material_pot + hp_pot + speed_pot + defensive_pot)

    def calculate_turn_reward(
        self,
        prev_state: Dict[str, Any],
        curr_state: Dict[str, Any],
        action_name: str,
        dmg_dealt: float,
        dmg_taken: float,
        repeated_action_count: int = 0,
        is_terminal: bool = False,
        won: bool = False
    ) -> float:
        """Calculates dense transition reward with dynamic modulation."""
        if is_terminal:
            return self.w_terminal if won else -self.w_terminal

        # 1. Potential-Based Base Reward (Phi(s') - Phi(s))
        phi_prev = self.calculate_potential(prev_state)
        phi_curr = self.calculate_potential(curr_state)
        pbrs_reward = (self.gamma * phi_curr) - phi_prev

        aux_reward = 0.0

        # 2. Clean Knockout Bonus (KO without taking lethal counter-damage)
        prev_opp_alive = prev_state.get("opp_alive", 6)
        curr_opp_alive = curr_state.get("opp_alive", 6)
        if curr_opp_alive < prev_opp_alive:
            aux_reward += self.w_clean_ko
            if dmg_taken < 0.10:
                aux_reward += 0.30  # Extra reward for clean outspeed KO

        # 3. Faint Penalty
        prev_my_alive = prev_state.get("my_alive", 6)
        curr_my_alive = curr_state.get("my_alive", 6)
        if curr_my_alive < prev_my_alive:
            aux_reward -= self.w_faint

        # 4. Anti-Loop & Ineffective Action Penalty
        if repeated_action_count >= 2 and dmg_dealt <= 0.01:
            # Penalize doing the exact same ineffective move repeatedly
            aux_reward -= self.w_loop_penalty * (repeated_action_count - 1)

        # 5. Redundant Status Attempt Penalty
        if curr_state.get("attempted_invalid_status", False):
            aux_reward -= self.w_status_waste_penalty

        return float(pbrs_reward + aux_reward)
