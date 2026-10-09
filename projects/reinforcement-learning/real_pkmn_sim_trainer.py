#!/usr/bin/env python3
"""
Real @pkmn/sim PPO League Trainer
=================================
Trains PPO Actor-Critic directly on authentic Pokémon Showdown transitions generated
by the @pkmn/sim TypeScript engine in Bun, playing against the real FoulPlay-Minimax,
PokeEnv-Heuristics, MaxDamage-Greedy, and Random baselines.
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../behavioral-cloning")))

from model import ContextualActionScoringNet

STATE_DIM = 76
ACTION_DIM = 16
NUM_ACTION_SLOTS = 9
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "weights"))
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "real_sim_gen8ou_ppo.pt")
METRICS_FILE = os.path.join(DATA_DIR, "real_gen8ou_metrics.json")
DASHBOARD_FILE = os.path.join(DATA_DIR, "real_gen8ou_dashboard.png")
ARTIFACT_DASHBOARD = "/home/usuario/.gemini/antigravity/brain/388067c6-5104-418e-add4-880610759fc0/real_gen8ou_dashboard.png"
STOP_FLAG_FILE = os.path.join(DATA_DIR, "stop_real_training.flag")
BUN_WORKER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulator/src/real_sim_worker.ts"))
BUN_TOURNAMENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulator/src/rigorous_gen8ou_tournament.ts"))


class RealPPOActorCritic(nn.Module):
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


class RealSimPPOTrainer:
    def __init__(self, battles_per_batch: int = 40, eval_interval: int = 5, lr: float = 3e-4):
        self.battles_per_batch = battles_per_batch
        self.eval_interval = eval_interval
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.running = True

        self.model = RealPPOActorCritic().to(self.device)
        if os.path.exists(CHECKPOINT_PATH):
            try:
                st = torch.load(CHECKPOINT_PATH, map_location=self.device)
                self.model.load_state_dict(st.get("model_state_dict", st), strict=False)
                print(f"[+] Resumed weights from {CHECKPOINT_PATH}")
            except Exception as e:
                print(f"[-] Checkpoint note: {e}")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.metrics = self._load_metrics()
        self.total_updates = len(self.metrics.get("updates", []))

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        if os.path.exists(STOP_FLAG_FILE):
            os.remove(STOP_FLAG_FILE)

    def _handle_shutdown(self, signum, frame):
        print("\n\n🛑 [GRACEFUL SHUTDOWN] Concluding batch and saving state safely...")
        self.running = False

    def _load_metrics(self) -> Dict[str, Any]:
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "format": "gen8ou_real_sim",
            "start_time": datetime.datetime.now().isoformat(),
            "updates": [],
            "rigorous_evals": []
        }

    def _save_metrics(self):
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

    def fetch_real_transitions(self, num_battles: int = 40) -> Tuple[List[Dict[str, Any]], float]:
        env = os.environ.copy()
        env["PATH"] = f"{os.path.expanduser('~')}/.bun/bin:" + env.get("PATH", "")
        sim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulator"))

        res = subprocess.run(
            ["bun", "run", BUN_WORKER_PATH, str(num_battles)],
            cwd=sim_dir,
            env=env,
            capture_output=True,
            text=True
        )

        if res.returncode != 0:
            print(f"[-] Bun worker error: {res.stderr}")
            return [], 0.0

        try:
            data = json.loads(res.stdout.strip().split("\n")[-1])
            return data.get("transitions", []), data.get("win_rate", 0.0)
        except Exception as e:
            print(f"[-] JSON parse error: {e}")
            return [], 0.0

    def optimize_ppo(self, transitions: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        if len(transitions) < 32:
            return 0.0, 0.0, 0.0

        states = torch.tensor(np.array([t["state"] for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([t["actions"] for t in transitions]), dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([t["mask"] for t in transitions]), dtype=torch.float32, device=self.device)
        chosen = torch.tensor(np.array([t["chosen"] for t in transitions]), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array([t["reward"] for t in transitions]), dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            old_logits, old_values = self.model(states, actions, masks)
            probs = F.softmax(old_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            old_log_probs = dist.log_prob(chosen)

        returns = rewards + 0.99 * old_values
        advantages = (returns - old_values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.model.train()
        total_a, total_c, total_e = 0.0, 0.0, 0.0

        for _ in range(4):
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

            total_a += actor_loss.item()
            total_c += critic_loss.item()
            total_e += entropy.item()

        return total_a / 4, total_c / 4, total_e / 4

    def run_rigorous_tournament(self, games_per_pair: int = 30) -> Dict[str, Any]:
        env = os.environ.copy()
        env["PATH"] = f"{os.path.expanduser('~')}/.bun/bin:" + env.get("PATH", "")
        sim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulator"))

        subprocess.run(
            ["bun", "run", BUN_TOURNAMENT_PATH, str(games_per_pair)],
            cwd=sim_dir,
            env=env,
            capture_output=True,
            text=True
        )

        matrix_path = os.path.join(DATA_DIR, "gen8ou_rigorous_matrix.json")
        if os.path.exists(matrix_path):
            with open(matrix_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def plot_dashboard(self):
        updates = self.metrics.get("updates", [])
        if len(updates) < 2: return

        up_nums = [u["update_id"] for u in updates]
        batch_wrs = [u["batch_win_rate"] for u in updates]
        critic_losses = [u["critic_loss"] for u in updates]
        entropies = [u["entropy"] for u in updates]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), dpi=300, sharex=True)

        ax1.plot(up_nums, batch_wrs, color="#e63946", linewidth=2.2, label="Batch Win Rate vs @pkmn/sim League")
        ax1.axhline(50.0, color="#888888", linestyle=":", alpha=0.7)
        ax1.set_ylabel("Win Rate (%)", fontsize=11, weight="bold")
        ax1.set_ylim(0, 100)
        ax1.legend(loc="upper left", frameon=True, fontsize=9)
        ax1.set_title("Amnesia-AI: Real @pkmn/sim Engine Training Dashboard (Gen 8 OU)", fontsize=13, weight="bold", pad=10)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.plot(up_nums, critic_losses, color="#1d3557", linewidth=2.0, label="Critic Value MSE Loss")
        ax2.set_ylabel("Critic Loss", color="#1d3557", fontsize=11, weight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", frameon=True, fontsize=9)

        ax3.plot(up_nums, entropies, color="#8338ec", linewidth=2.0, label="Policy Entropy")
        ax3.set_ylabel("Entropy", color="#8338ec", fontsize=11, weight="bold")
        ax3.set_xlabel("PPO Updates on Real Showdown Transitions", fontsize=11, weight="bold")
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
        print("🚀 REAL @PKMN/SIM ENGINE PPO LEAGUE TRAINER (GEN 8 OU)")
        print("=" * 80)
        print(f"Device:             {self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Host CPU'})")
        print(f"Sim Engine:         Bun + @pkmn/sim (100% Authentic Showdown Rules & Mechanics)")
        print(f"Battles per Batch:  {self.battles_per_batch} battles vs Minimax, PokeEnv, MaxDamage, Random")
        print(f"Checkpoint Path:    {CHECKPOINT_PATH}")
        print(f"Dashboard PNG:      {DASHBOARD_FILE}")
        print("=" * 80 + "\n", flush=True)

        while self.running:
            if os.path.exists(STOP_FLAG_FILE):
                print("\n[!] Stop flag detected. Shutting down gracefully...")
                break

            t0 = time.time()
            transitions, batch_wr = self.fetch_real_transitions(self.battles_per_batch)
            if not self.running or len(transitions) == 0: break

            a_loss, c_loss, entropy = self.optimize_ppo(transitions)
            self.total_updates += 1
            dt = time.time() - t0

            up_data = {
                "update_id": self.total_updates,
                "timestamp": datetime.datetime.now().isoformat(),
                "transitions": len(transitions),
                "batch_win_rate": round(batch_wr, 1),
                "duration_seconds": round(dt, 2),
                "actor_loss": round(a_loss, 4),
                "critic_loss": round(c_loss, 4),
                "entropy": round(entropy, 4)
            }
            self.metrics["updates"].append(up_data)

            print(f"🔄 [REAL SIM UPDATE #{self.total_updates:03d}] {len(transitions)} turns in {dt:4.1f}s | Batch WR: {batch_wr:4.1f}% | Actor: {a_loss:6.3f} | Critic: {c_loss:6.3f} | Ent: {entropy:5.3f}", flush=True)

            if self.total_updates % self.eval_interval == 0:
                print(f"\n📊 [EVALUATION #{self.total_updates}] Running 40-Game Rigorous @pkmn/sim Tournament...", flush=True)
                matrix_data = self.run_rigorous_tournament(games_per_pair=30)
                if matrix_data:
                    self.metrics["rigorous_evals"].append({
                        "update_id": self.total_updates,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "matrix": matrix_data.get("matrix"),
                        "agents": matrix_data.get("agents")
                    })
                    print(f"   ✅ Real Tournament Matrix updated successfully.\n", flush=True)

                os.makedirs(WEIGHTS_DIR, exist_ok=True)
                torch.save({"model_state_dict": self.model.state_dict(), "update_id": self.total_updates}, CHECKPOINT_PATH)
                self.plot_dashboard()
                self._save_metrics()

        print("\n💾 [SAVING FINAL REAL SIM CHECKPOINT]...")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict(), "update_id": self.total_updates}, CHECKPOINT_PATH)
        self.plot_dashboard()
        self._save_metrics()
        print(f"✅ Real @pkmn/sim training state saved to {CHECKPOINT_PATH} and {METRICS_FILE}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real @pkmn/sim PPO League Trainer")
    parser.add_argument("--battles", type=int, default=40, help="Battles per gradient step")
    parser.add_argument("--eval-interval", type=int, default=5, help="Updates between full tournament evaluations")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    trainer = RealSimPPOTrainer(
        battles_per_batch=args.battles,
        eval_interval=args.eval_interval,
        lr=args.lr
    )
    trainer.train_forever()
