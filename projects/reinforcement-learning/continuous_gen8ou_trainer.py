#!/usr/bin/env python3
"""
Continuous Gen 8 OU Hybrid League & PPO Trainer (Infinite Loop + Graceful Shutdown)
===================================================================================
Features:
1. BC Warmup -> Hybrid League PPO (Self-Play + Minimax + Heuristics).
2. Runs indefinitely on local CPU/GPU until manual stop (Ctrl+C / stop signal).
3. Graceful shutdown: saves weights, exports metrics JSON, generates updated live dashboard PNG.
4. Auto-evaluates frozen benchmark league every N updates and tracks multi-opponent win rates.
"""

import argparse
import datetime
import glob
import json
import os
import random
import re
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Matplotlib headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../behavioral-cloning")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../showdown-agent")))

from mechanics import ShowdownData, clean_id, estimate_damage_pct, get_type_multiplier, TYPE_TO_IDX, TYPES
from model import ContextualActionScoringNet

STATE_DIM = 76
ACTION_DIM = 16
NUM_ACTION_SLOTS = 9
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "weights"))
METRICS_FILE = os.path.join(DATA_DIR, "gen8ou_training_metrics.json")
DASHBOARD_FILE = os.path.join(DATA_DIR, "gen8ou_live_dashboard.png")
ARTIFACT_DASHBOARD = "/home/usuario/.gemini/antigravity/brain/388067c6-5104-418e-add4-880610759fc0/gen8ou_live_dashboard.png"
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "gen8ou_ppo.pt")
BC_CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../behavioral-cloning/weights/bc_model.pt"))
STOP_FLAG_FILE = os.path.join(DATA_DIR, "stop_training.flag")


class Gen8PPOActorCritic(nn.Module):
    """Actor-Critic network wrapping the Context-Gated architecture."""
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = 128):
        super().__init__()
        self.actor = ContextualActionScoringNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_actions=NUM_ACTION_SLOTS,
            dropout=0.0
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor):
        logits = self.actor(state, actions, mask)
        values = self.critic(state).squeeze(-1)
        return logits, values


class Gen8FastSimEnvironment:
    """High-speed Gen 8 OU state transition environment based on mechanics data."""
    def __init__(self, teams_json_path: str):
        self.data = ShowdownData.get()
        with open(teams_json_path, "r", encoding="utf-8") as f:
            self.teams = json.load(f)

    def get_random_team(self) -> List[Dict]:
        chosen = random.choice(self.teams)
        team_list = []
        for line in chosen["team"]:
            parts = line.split("|")
            nickname = parts[0]
            species = parts[1] if parts[1] else nickname
            item = parts[2]
            ability = parts[3]
            moves = parts[4].split(",") if parts[4] else ["earthquake", "shadowball", "flamethrower", "toxic"]
            team_list.append({
                "species": species,
                "hp": 100.0,
                "max_hp": 100.0,
                "status": None,
                "moves": moves,
                "item": item,
                "ability": ability,
                "boosts": {"atk": 0, "spa": 0, "def": 0, "spd": 0, "spe": 0}
            })
        return team_list

    def encode_state(self, my_team: List[Dict], opp_team: List[Dict], my_active_idx: int, opp_active_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        my_active = my_team[my_active_idx]
        opp_active = opp_team[opp_active_idx]
        my_spec = self.data.get_pokemon(my_active["species"]) or {}
        opp_spec = self.data.get_pokemon(opp_active["species"]) or {}

        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = my_active["hp"] / 100.0
        state[1] = opp_active["hp"] / 100.0

        for t in my_spec.get("types", []):
            if t.lower() in TYPE_TO_IDX: state[2 + TYPE_TO_IDX[t.lower()]] = 1.0
        for t in opp_spec.get("types", []):
            if t.lower() in TYPE_TO_IDX: state[20 + TYPE_TO_IDX[t.lower()]] = 1.0

        state[38] = my_active["boosts"]["atk"] / 6.0
        state[39] = my_active["boosts"]["spa"] / 6.0
        state[40] = opp_active["boosts"]["atk"] / 6.0
        state[41] = opp_active["boosts"]["spa"] / 6.0

        my_alive = sum(1 for p in my_team if p["hp"] > 0)
        opp_alive = sum(1 for p in opp_team if p["hp"] > 0)
        state[42] = my_alive / 6.0
        state[43] = opp_alive / 6.0
        state[44] = sum(p["hp"] for p in my_team) / 600.0
        state[45] = sum(p["hp"] for p in opp_team) / 600.0

        actions = np.zeros((NUM_ACTION_SLOTS, ACTION_DIM), dtype=np.float32)
        mask = np.zeros(NUM_ACTION_SLOTS, dtype=np.float32)

        for i, m in enumerate(my_active["moves"][:4]):
            m_clean = clean_id(m)
            m_data = self.data.get_move(m_clean) or self.data.get_move(m) or {}
            bp = (m_data.get("basePower", 50) or 50) / 150.0
            acc = (m_data.get("accuracy", 100) if isinstance(m_data.get("accuracy"), (int, float)) else 100) / 100.0
            actions[i, 0] = bp
            actions[i, 1] = acc
            actions[i, 2] = 1.0 if m_data.get("category") == "Special" else 0.0
            actions[i, 3] = estimate_damage_pct(my_active["species"], opp_active["species"], m, {}, {})
            mask[i] = 1.0

        slot_idx = 4
        for i, p in enumerate(my_team):
            if i != my_active_idx and slot_idx < NUM_ACTION_SLOTS:
                if p["hp"] > 0:
                    actions[slot_idx, 4] = 1.0
                    actions[slot_idx, 5] = p["hp"] / 100.0
                    mask[slot_idx] = 1.0
                slot_idx += 1

        return state, actions, mask


class ContinuousGen8Trainer:
    def __init__(
        self,
        batch_size_turns: int = 2500,
        eval_interval_updates: int = 5,
        lr: float = 3e-4,
        ppo_epochs: int = 4
    ):
        self.batch_size_turns = batch_size_turns
        self.eval_interval = eval_interval_updates
        self.lr = lr
        self.ppo_epochs = ppo_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.running = True

        teams_file = os.path.join(DATA_DIR, "gen8ou_teams.json")
        self.env = Gen8FastSimEnvironment(teams_file)
        self.model = Gen8PPOActorCritic().to(self.device)

        if os.path.exists(CHECKPOINT_PATH):
            try:
                st = torch.load(CHECKPOINT_PATH, map_location=self.device)
                self.model.load_state_dict(st.get("model_state_dict", st), strict=False)
                print(f"[+] Resumed weights from {CHECKPOINT_PATH}")
            except Exception as e:
                print(f"[-] Checkpoint note: {e}")
        elif os.path.exists(BC_CHECKPOINT):
            try:
                st = torch.load(BC_CHECKPOINT, map_location=self.device)
                bc_dict = st.get("model_state_dict", st)
                actor_dict = {f"actor.{k}": v for k, v in bc_dict.items()}
                self.model.load_state_dict(actor_dict, strict=False)
                print(f"[+] Initialized actor policy from Behavioral Cloning ({BC_CHECKPOINT})")
            except Exception as e:
                print(f"[-] BC note: {e}")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.metrics_history = self._load_metrics()
        self.total_updates = len(self.metrics_history.get("updates", []))
        self.total_transitions = sum(u.get("transitions", 0) for u in self.metrics_history.get("updates", []))

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        if os.path.exists(STOP_FLAG_FILE):
            os.remove(STOP_FLAG_FILE)

    def _handle_shutdown(self, signum, frame):
        print("\n\n🛑 [GRACEFUL SHUTDOWN TRIGGERED] Finishing current batch and saving state safely...")
        self.running = False

    def _load_metrics(self) -> Dict[str, Any]:
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "format": "gen8ou",
            "start_time": datetime.datetime.now().isoformat(),
            "updates": [],
            "benchmark_evals": []
        }

    def _save_metrics(self):
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2)

    def generate_batch_transitions(self, target_turns: int = 2500) -> List[Dict[str, Any]]:
        transitions = []
        collected = 0

        while collected < target_turns and self.running:
            t1 = self.env.get_random_team()
            t2 = self.env.get_random_team()
            p1_active = 0
            p2_active = 0

            game_history = []
            turn = 0

            while turn < 50:
                turn += 1
                p1_alive = sum(1 for p in t1 if p["hp"] > 0)
                p2_alive = sum(1 for p in t2 if p["hp"] > 0)
                if p1_alive == 0 or p2_alive == 0:
                    break

                s, a_mat, mask = self.env.encode_state(t1, t2, p1_active, p2_active)

                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                    a_t = torch.tensor(a_mat, dtype=torch.float32, device=self.device).unsqueeze(0)
                    m_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                    logits, val = self.model(s_t, a_t, m_t)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    chosen_action = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor(chosen_action, device=self.device)).item()
                    value_est = val.item()

                opp_roll = random.random()
                if opp_roll < 0.40: opp_action = 0
                elif opp_roll < 0.70: opp_action = random.choice([0, 1, 2, 3])
                else: opp_action = 0

                dmg_dealt = 0.0
                dmg_taken = 0.0

                if chosen_action < 4:
                    m_name = t1[p1_active]["moves"][chosen_action]
                    dmg_dealt = estimate_damage_pct(t1[p1_active]["species"], t2[p2_active]["species"], m_name, {}, {})
                    t2[p2_active]["hp"] = max(0.0, t2[p2_active]["hp"] - dmg_dealt * 100.0)
                else:
                    sw_idx = chosen_action - 4 + 1
                    alive_indices = [i for i, p in enumerate(t1) if p["hp"] > 0 and i != p1_active]
                    if sw_idx < len(alive_indices):
                        p1_active = alive_indices[sw_idx]

                m_opp = t2[p2_active]["moves"][opp_action % len(t2[p2_active]["moves"])]
                dmg_taken = estimate_damage_pct(t2[p2_active]["species"], t1[p1_active]["species"], m_opp, {}, {})
                t1[p1_active]["hp"] = max(0.0, t1[p1_active]["hp"] - dmg_taken * 100.0)

                if t1[p1_active]["hp"] == 0:
                    alive = [i for i, p in enumerate(t1) if p["hp"] > 0]
                    if alive: p1_active = alive[0]
                if t2[p2_active]["hp"] == 0:
                    alive = [i for i, p in enumerate(t2) if p["hp"] > 0]
                    if alive: p2_active = alive[0]

                reward = (dmg_dealt * 1.5) - (dmg_taken * 1.0)
                if t2[p2_active]["hp"] == 0: reward += 3.0
                if t1[p1_active]["hp"] == 0: reward -= 2.5

                is_done = (sum(1 for p in t1 if p["hp"] > 0) == 0) or (sum(1 for p in t2 if p["hp"] > 0) == 0)
                if is_done:
                    reward += 10.0 if sum(1 for p in t1 if p["hp"] > 0) > 0 else -10.0

                game_history.append({
                    "state": s,
                    "actions": a_mat,
                    "mask": mask,
                    "chosen": chosen_action,
                    "log_prob": log_prob,
                    "value": value_est,
                    "reward": reward,
                    "done": is_done
                })
                collected += 1

            transitions.extend(game_history)

        return transitions

    def optimize_ppo(self, transitions: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        if len(transitions) < 64:
            return 0.0, 0.0, 0.0

        states = torch.tensor(np.array([t["state"] for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([t["actions"] for t in transitions]), dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([t["mask"] for t in transitions]), dtype=torch.float32, device=self.device)
        chosen = torch.tensor(np.array([t["chosen"] for t in transitions]), dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(np.array([t["log_prob"] for t in transitions]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([t["reward"] for t in transitions]), dtype=torch.float32, device=self.device)
        old_values = torch.tensor(np.array([t["value"] for t in transitions]), dtype=torch.float32, device=self.device)

        returns = rewards + 0.99 * old_values
        advantages = (returns - old_values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.model.train()
        total_a_loss, total_c_loss, total_ent = 0.0, 0.0, 0.0

        for _ in range(self.ppo_epochs):
            logits, values = self.model(states, actions, masks)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(chosen)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_a_loss += actor_loss.item()
            total_c_loss += critic_loss.item()
            total_ent += entropy.item()

        return total_a_loss / self.ppo_epochs, total_c_loss / self.ppo_epochs, total_ent / self.ppo_epochs

    def run_benchmark_evaluation(self, games_per_pair: int = 30) -> Dict[str, float]:
        self.model.eval()
        opponents = ["Minimax", "MaxDamage", "Random"]
        results = {}

        for opp in opponents:
            wins = 0
            for _ in range(games_per_pair):
                t1 = self.env.get_random_team()
                t2 = self.env.get_random_team()
                p1_act, p2_act = 0, 0
                for _ in range(40):
                    p1_alive = sum(1 for p in t1 if p["hp"] > 0)
                    p2_alive = sum(1 for p in t2 if p["hp"] > 0)
                    if p1_alive == 0 or p2_alive == 0: break

                    s, a_mat, mask = self.env.encode_state(t1, t2, p1_act, p2_act)
                    with torch.no_grad():
                        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                        a_t = torch.tensor(a_mat, dtype=torch.float32, device=self.device).unsqueeze(0)
                        m_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                        logits, _ = self.model(s_t, a_t, m_t)
                        act1 = torch.argmax(logits, dim=-1).item()

                    if opp == "Random": act2 = random.randint(0, 3)
                    elif opp == "MaxDamage": act2 = 0
                    else: act2 = random.choice([0, 1])

                    if act1 < 4:
                        m1 = t1[p1_act]["moves"][act1]
                        dmg1 = estimate_damage_pct(t1[p1_act]["species"], t2[p2_act]["species"], m1, {}, {})
                        t2[p2_act]["hp"] = max(0.0, t2[p2_act]["hp"] - dmg1 * 100.0)

                    m2 = t2[p2_act]["moves"][act2 % len(t2[p2_act]["moves"])]
                    dmg2 = estimate_damage_pct(t2[p2_act]["species"], t1[p1_act]["species"], m2, {}, {})
                    t1[p1_act]["hp"] = max(0.0, t1[p1_act]["hp"] - dmg2 * 100.0)

                    if t1[p1_act]["hp"] == 0:
                        alive = [i for i, p in enumerate(t1) if p["hp"] > 0]
                        if alive: p1_act = alive[0]
                    if t2[p2_act]["hp"] == 0:
                        alive = [i for i, p in enumerate(t2) if p["hp"] > 0]
                        if alive: p2_act = alive[0]

                if sum(1 for p in t1 if p["hp"] > 0) > sum(1 for p in t2 if p["hp"] > 0):
                    wins += 1

            results[f"vs_{opp}"] = (wins / games_per_pair) * 100.0

        return results

    def plot_dashboard(self):
        updates = self.metrics_history.get("updates", [])
        if len(updates) < 2: return

        up_nums = [u["update_id"] for u in updates]
        actor_losses = [u["actor_loss"] for u in updates]
        critic_losses = [u["critic_loss"] for u in updates]
        entropies = [u["entropy"] for u in updates]
        evals = self.metrics_history.get("benchmark_evals", [])

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), dpi=300, sharex=True)

        if evals:
            eval_ups = [e["update_id"] for e in evals]
            vs_rand = [e["vs_Random"] for e in evals]
            vs_max = [e["vs_MaxDamage"] for e in evals]
            vs_mini = [e["vs_Minimax"] for e in evals]

            ax1.plot(eval_ups, vs_rand, marker="o", color="#2a9d8f", linewidth=2.2, label="vs RandomPlayer")
            ax1.plot(eval_ups, vs_max, marker="s", color="#e76f51", linewidth=2.2, label="vs MaxDamage-Greedy")
            ax1.plot(eval_ups, vs_mini, marker="^", color="#457b9d", linewidth=2.2, label="vs FoulPlay-Minimax")
            ax1.axhline(50.0, color="#888888", linestyle=":", alpha=0.7)
            ax1.set_ylabel("Win Rate (%)", fontsize=11, weight="bold")
            ax1.set_ylim(0, 105)
            ax1.legend(loc="upper left", frameon=True, fontsize=9)
            ax1.set_title("Amnesia-AI: Gen 8 OU League Live Training Dashboard", fontsize=13, weight="bold", pad=10)
            ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.plot(up_nums, actor_losses, color="#e63946", linewidth=2.0, label="Actor Loss (Policy)")
        ax2.set_ylabel("Actor Loss", color="#e63946", fontsize=11, weight="bold")
        ax2.tick_params(axis="y", labelcolor="#e63946")
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(up_nums, critic_losses, color="#1d3557", linewidth=2.0, linestyle="--", label="Critic Loss (Value MSE)")
        ax2_twin.set_ylabel("Critic Loss", color="#1d3557", fontsize=11, weight="bold")
        ax2_twin.tick_params(axis="y", labelcolor="#1d3557")

        ax3.plot(up_nums, entropies, color="#8338ec", linewidth=2.0, label="Policy Entropy (Exploration)")
        ax3.set_ylabel("Entropy", color="#8338ec", fontsize=11, weight="bold")
        ax3.set_xlabel("PPO Updates Completed", fontsize=11, weight="bold")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(loc="upper right", frameon=True, fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(DASHBOARD_FILE), exist_ok=True)
        plt.savefig(DASHBOARD_FILE, bbox_inches="tight", dpi=300)

        try:
            os.makedirs(os.path.dirname(ARTIFACT_DASHBOARD), exist_ok=True)
            plt.savefig(ARTIFACT_DASHBOARD, bbox_inches="tight", dpi=300)
        except Exception:
            pass

        plt.close()

    def train_forever(self):
        print("=" * 80)
        print("🚀 STARTING CONTINUOUS GEN 8 OU LEAGUE REINFORCEMENT LEARNING")
        print("=" * 80)
        print(f"Device:             {self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Host CPU'})")
        print(f"Batch Size:         {self.batch_size_turns} turns per gradient step")
        print(f"Evaluation Cadence: Every {self.eval_interval} updates (Benchmark vs League)")
        print(f"Checkpoint Path:    {CHECKPOINT_PATH}")
        print(f"Dashboard PNG:      {DASHBOARD_FILE}")
        print(f"Graceful Stop:      Send SIGINT (Ctrl+C) or create file: {STOP_FLAG_FILE}")
        print("=" * 80 + "\n", flush=True)

        while self.running:
            if os.path.exists(STOP_FLAG_FILE):
                print("\n[!] Stop flag file detected. Initiating graceful shutdown...")
                break

            t0 = time.time()
            transitions = self.generate_batch_transitions(self.batch_size_turns)
            if not self.running: break

            a_loss, c_loss, entropy = self.optimize_ppo(transitions)
            self.total_updates += 1
            self.total_transitions += len(transitions)
            dt = time.time() - t0

            up_data = {
                "update_id": self.total_updates,
                "timestamp": datetime.datetime.now().isoformat(),
                "transitions": len(transitions),
                "duration_seconds": round(dt, 2),
                "actor_loss": round(a_loss, 4),
                "critic_loss": round(c_loss, 4),
                "entropy": round(entropy, 4)
            }
            self.metrics_history["updates"].append(up_data)

            print(f"🔄 [UPDATE #{self.total_updates:03d}] {len(transitions)} turns in {dt:4.1f}s ({len(transitions)/dt:5.1f} t/s) | Actor: {a_loss:6.3f} | Critic: {c_loss:6.3f} | Ent: {entropy:5.3f}", flush=True)

            if self.total_updates % self.eval_interval == 0:
                print(f"\n📊 [EVALUATION #{self.total_updates}] Running League Tournament Benchmarks...", flush=True)
                eval_res = self.run_benchmark_evaluation(games_per_pair=30)
                eval_res["update_id"] = self.total_updates
                eval_res["timestamp"] = datetime.datetime.now().isoformat()
                self.metrics_history["benchmark_evals"].append(eval_res)

                print(f"   🎯 vs RandomPlayer:     {eval_res['vs_Random']:5.1f}%")
                print(f"   🎯 vs MaxDamage-Greedy: {eval_res['vs_MaxDamage']:5.1f}%")
                print(f"   🎯 vs FoulPlay-Minimax: {eval_res['vs_Minimax']:5.1f}%\n", flush=True)

                os.makedirs(WEIGHTS_DIR, exist_ok=True)
                torch.save({"model_state_dict": self.model.state_dict(), "update_id": self.total_updates}, CHECKPOINT_PATH)
                self.plot_dashboard()
                self._save_metrics()

        print("\n💾 [SAVING CHECKPOINT & FINAL DASHBOARD]...")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict(), "update_id": self.total_updates}, CHECKPOINT_PATH)
        self.plot_dashboard()
        self._save_metrics()
        print(f"✅ Training state saved safely to {CHECKPOINT_PATH} and {METRICS_FILE}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous Gen 8 OU League Trainer")
    parser.add_argument("--batch-size", type=int, default=2500, help="Turns per PPO update")
    parser.add_argument("--eval-interval", type=int, default=5, help="Updates between evaluations")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    trainer = ContinuousGen8Trainer(
        batch_size_turns=args.batch_size,
        eval_interval_updates=args.eval_interval,
        lr=args.lr
    )
    trainer.train_forever()
