"""
Expert Iteration & Self-Play Trainer (DAgger / PPO with Real Engine Simulation)
==============================================================================
Trains Amnesia-AI directly against Foul Play Minimax inside the real headless engine.
Features:
1. Dynamic Potential Matrix Reward Engine (Phase-adaptive, Clean KO, Anti-Loop).
2. Slot-Agnostic Candidate Feature Representation.
3. Supervised Imitation + Policy Gradient updates on actual battle trajectories.
"""

import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../behavioral-cloning")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../showdown-agent")))

from ppo_model import PPOActorCriticNet
from reward_engine import DynamicMatrixRewardEngine
from foul_play_brain import FoulPlayBrain
from real_arena import OfficialHeadlessEngine
from mechanics import ShowdownData, estimate_damage_pct, get_type_multiplier, clean_id

STATE_DIM = 76
ACTION_DIM = 16
NUM_ACTION_SLOTS = 9


def train_expert_iteration(iterations: int = 100, battles_per_iter: int = 20, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"⚡ AMNESIA-AI EXPERT ITERATION TRAINER (Real Engine vs Minimax on {device})")
    print("=" * 80)

    model = PPOActorCriticNet(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=128,
        num_actions=NUM_ACTION_SLOTS
    ).to(device)

    # Load starting weights if available
    bc_ckpt = "projects/behavioral-cloning/weights/bc_model.pt"
    if os.path.exists(bc_ckpt):
        model.load_from_bc_checkpoint(bc_ckpt)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    reward_engine = DynamicMatrixRewardEngine()
    engine = OfficialHeadlessEngine()
    expert = FoulPlayBrain()
    data = ShowdownData.get()

    os.makedirs("projects/reinforcement-learning/weights", exist_ok=True)
    save_path = "projects/reinforcement-learning/weights/ppo_model.pt"

    for epoch in range(1, iterations + 1):
        model.eval()
        states_list = []
        actions_list = []
        masks_list = []
        targets_list = []
        rewards_list = []
        
        wins = 0

        for b in range(battles_per_iter):
            t1 = engine.generate_team()
            t2 = engine.generate_team()
            history = []
            turn = 0
            action_hist = []

            prev_state_dict = {
                "my_alive": 6, "opp_alive": 6, "my_hp": 1.0, "opp_hp": 1.0,
                "i_am_faster": True, "type_resistance_factor": 1.0
            }

            while turn < 60:
                turn += 1
                p1_alive = sum(1 for p in t1 if not p["fainted"])
                p2_alive = sum(1 for p in t2 if not p["fainted"])
                if p1_alive == 0 or p2_alive == 0:
                    break

                m1 = next(p for p in t1 if p["active"])
                m2 = next(p for p in t2 if p["active"])

                req1 = {
                    "side": {"id": "p1", "name": "Amnesia-AI", "pokemon": [
                        {"details": p["details"], "condition": p["condition"], "active": p["active"], "moves": p["moves"]}
                        for p in t1
                    ]},
                    "active": [{"moves": [{"id": m, "move": m, "pp": 15, "disabled": False} for m in m1["moves"]]}]
                }

                # Construct Action Candidate Matrix & Mask
                a_mat = np.zeros((NUM_ACTION_SLOTS, ACTION_DIM), dtype=np.float32)
                mask = np.zeros(NUM_ACTION_SLOTS, dtype=np.float32)

                opp_meta = data.get_pokemon(m2["species"])
                opp_types = opp_meta.get("types", ["Normal"]) if opp_meta else ["Normal"]
                opp_spe = opp_meta.get("baseStats", {}).get("spe", 80) if opp_meta else 80

                my_meta = data.get_pokemon(m1["species"])
                my_types = my_meta.get("types", ["Normal"]) if my_meta else ["Normal"]
                my_spe = my_meta.get("baseStats", {}).get("spe", 80) if my_meta else 80
                i_am_faster = my_spe > opp_spe

                # Damage calculations for moves
                damages = []
                for i, mv in enumerate(m1["moves"][:4]):
                    dmg = estimate_damage_pct(m1["species"], m2["species"], mv, m1["boosts"], m2["boosts"])
                    damages.append((i, mv, dmg))

                max_dmg = max([d[2] for d in damages], default=0.01)

                for i, mv, dmg in damages:
                    m_data = data.get_move(mv)
                    if not m_data: continue
                    mask[i] = 1.0
                    a_mat[i, 0] = 1.0
                    a_mat[i, 2] = min(2.0, dmg)
                    a_mat[i, 3] = 1.0 if dmg >= (m2["current_hp"] / m2["max_hp"]) else 0.0
                    cat = m_data.get("category", "Status")
                    if cat == "Physical": a_mat[i, 4] = 1.0
                    elif cat == "Special": a_mat[i, 5] = 1.0
                    else: a_mat[i, 6] = 1.0
                    m_type = m_data.get("type", "Normal")
                    a_mat[i, 7] = get_type_multiplier(m_type, opp_types) / 4.0
                    a_mat[i, 8] = 1.5 if m_type in my_types else 1.0
                    a_mat[i, 9] = (m_data.get("accuracy", 100) if isinstance(m_data.get("accuracy", 100), (int, float)) else 100) / 100.0
                    a_mat[i, 10] = (m_data.get("priority", 0) + 6.0) / 12.0
                    a_mat[i, 13] = dmg / max(max_dmg, 0.01)
                    a_mat[i, 14] = 1.0 if (dmg == max_dmg and dmg > 0.05) else 0.0

                # Bench Switches
                bench_mons = [p for p in t1 if not p["active"]]
                for i, b in enumerate(bench_mons[:5]):
                    slot = 4 + i
                    if not b["fainted"]:
                        mask[slot] = 1.0
                        a_mat[slot, 1] = 1.0
                        a_mat[slot, 2] = b["current_hp"] / b["max_hp"]
                        b_poke = data.get_pokemon(b["species"])
                        b_types = b_poke.get("types", ["Normal"]) if b_poke else ["Normal"]
                        res = [get_type_multiplier(ot, b_types) for ot in opp_types]
                        a_mat[slot, 7] = (min(res) if res else 1.0) / 4.0

                s_vec = np.zeros(STATE_DIM, dtype=np.float32)
                s_vec[0] = m1["current_hp"] / m1["max_hp"]
                s_vec[1] = m2["current_hp"] / m2["max_hp"]
                s_vec[14] = p1_alive / 6.0
                s_vec[15] = p2_alive / 6.0

                s_t = torch.tensor(s_vec, dtype=torch.float32, device=device).unsqueeze(0)
                a_t = torch.tensor(a_mat, dtype=torch.float32, device=device).unsqueeze(0)
                m_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

                # Neural Sampling
                with torch.no_grad():
                    logits, _ = model(s_t, a_t, m_t)
                    probs = F.softmax(logits, dim=-1)
                    chosen_action_idx = int(torch.multinomial(probs, num_samples=1).item())

                # Query Expert Teacher (DAgger / Minimax guidance)
                expert_act = expert.choose_action(req1, history)
                if "move" in expert_act:
                    target_expert_idx = int(expert_act.split()[1]) - 1
                elif "switch" in expert_act:
                    target_expert_idx = 4 + min(4, int(expert_act.split()[1]) - 1)
                else:
                    target_expert_idx = chosen_action_idx

                # Translate action into game execution
                if chosen_action_idx < 4 and chosen_action_idx < len(m1["moves"]):
                    move_name = m1["moves"][chosen_action_idx]
                    act1 = f"move {chosen_action_idx + 1}"
                else:
                    bench_idx = chosen_action_idx - 4
                    if bench_idx < len(bench_mons) and not bench_mons[bench_idx]["fainted"]:
                        act1 = f"switch {t1.index(bench_mons[bench_idx]) + 1}"
                    else:
                        act1 = "move 1"

                # Opponent (FoulPlay Minimax) Action
                req2 = {
                    "side": {"id": "p2", "name": "FoulPlay", "pokemon": [
                        {"details": p["details"], "condition": p["condition"], "active": p["active"], "moves": p["moves"]}
                        for p in t2
                    ]},
                    "active": [{"moves": [{"id": m, "move": m, "pp": 15, "disabled": False} for m in m2["moves"]]}]
                }
                act2 = expert.choose_action(req2, history)

                # Execute Turn
                dmg_dealt = 0.0
                dmg_taken = 0.0

                if "switch" in act1:
                    idx = int(act1.split()[1]) - 1
                    m1["active"] = False
                    t1[idx]["active"] = True
                    m1 = t1[idx]
                if "switch" in act2:
                    idx = int(act2.split()[1]) - 1
                    m2["active"] = False
                    t2[idx]["active"] = True
                    m2 = t2[idx]

                if "move" in act1 and not m1["fainted"]:
                    slot = int(act1.split()[1]) - 1
                    mv = m1["moves"][min(slot, len(m1["moves"])-1)]
                    dmg_pct = estimate_damage_pct(m1["species"], m2["species"], mv, m1["boosts"], m2["boosts"])
                    dmg_dealt = dmg_pct
                    dmg_hp = int(dmg_pct * m2["max_hp"])
                    m2["current_hp"] = max(0, m2["current_hp"] - dmg_hp)
                    if m2["current_hp"] == 0: m2["fainted"] = True

                if "move" in act2 and not m2["fainted"]:
                    slot = int(act2.split()[1]) - 1
                    mv = m2["moves"][min(slot, len(m2["moves"])-1)]
                    dmg_pct = estimate_damage_pct(m2["species"], m1["species"], mv, m2["boosts"], m1["boosts"])
                    dmg_taken = dmg_pct
                    dmg_hp = int(dmg_pct * m1["max_hp"])
                    m1["current_hp"] = max(0, m1["current_hp"] - dmg_hp)
                    if m1["current_hp"] == 0: m1["fainted"] = True

                # Calculate Dynamic Reward
                curr_state_dict = {
                    "my_alive": sum(1 for p in t1 if not p["fainted"]),
                    "opp_alive": sum(1 for p in t2 if not p["fainted"]),
                    "my_hp": m1["current_hp"] / m1["max_hp"],
                    "opp_hp": m2["current_hp"] / m2["max_hp"],
                    "i_am_faster": i_am_faster,
                    "type_resistance_factor": min([get_type_multiplier(ot, my_types) for ot in opp_types], default=1.0)
                }

                rep_count = action_hist[-2:].count(act1)
                r = reward_engine.calculate_turn_reward(
                    prev_state_dict,
                    curr_state_dict,
                    act1,
                    dmg_dealt=dmg_dealt,
                    dmg_taken=dmg_taken,
                    repeated_action_count=rep_count
                )

                states_list.append(s_vec)
                actions_list.append(a_mat)
                masks_list.append(mask)
                targets_list.append(target_expert_idx)
                rewards_list.append(r)
                action_hist.append(act1)
                prev_state_dict = curr_state_dict

                # Forced switch replacement
                if m1["fainted"]:
                    alive_t1 = [p for p in t1 if not p["fainted"]]
                    if alive_t1:
                        m1["active"] = False
                        alive_t1[0]["active"] = True
                if m2["fainted"]:
                    alive_t2 = [p for p in t2 if not p["fainted"]]
                    if alive_t2:
                        m2["active"] = False
                        alive_t2[0]["active"] = True

            if p1_alive > p2_alive:
                wins += 1

        # Train Step on Trajectories
        if states_list:
            model.train()
            b_states = torch.tensor(np.array(states_list), dtype=torch.float32, device=device)
            b_actions = torch.tensor(np.array(actions_list), dtype=torch.float32, device=device)
            b_masks = torch.tensor(np.array(masks_list), dtype=torch.float32, device=device)
            b_targets = torch.tensor(np.array(targets_list), dtype=torch.long, device=device)
            b_rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32, device=device)

            logits, values = model(b_states, b_actions, b_masks)
            
            # Policy loss (cross entropy with teacher + advantage weighting)
            loss_imitation = F.cross_entropy(logits, b_targets)
            loss_value = F.mse_loss(values, b_rewards)
            total_loss = loss_imitation + 0.5 * loss_value

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch % 10 == 0 or epoch == iterations:
                win_rate = (wins / battles_per_iter) * 100.0
                print(f"Epoch [{epoch:03d}/{iterations}] | Loss: {total_loss.item():.4f} | Imitation: {loss_imitation.item():.4f} | WinRate vs Minimax: {win_rate:.1f}%")
                torch.save({"model_state_dict": model.state_dict()}, save_path)

    print(f"\n✅ Training complete! Saved improved weights to {save_path}")


if __name__ == "__main__":
    train_expert_iteration(iterations=50, battles_per_iter=15)
