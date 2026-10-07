"""
Live Pokémon Showdown Multi-Worker Online RL Trainer (20 Workers + Replay Logger)
==================================================================================
Key Capabilities:
1. Runs 20 parallel worker connections to Pokemon Showdown ranked ladder (Blitz format).
2. Collects real transitions (s, a_mat, mask, chosen_action, log_prob, value, reward, s', done).
3. Evaluates transitions using the Dynamic Potential Matrix Reward Engine (PBRS).
4. Persists complete official Showdown battle replays (.json) for offline behavioral analysis.
5. Asynchronously optimizes the PPO Actor-Critic Network on NVIDIA GPU (RTX 3050).
6. Broadcasts updated neural weights to all active workers seamlessly.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
import requests
import websocket
import numpy as np
import torch
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
from mechanics import (
    ShowdownData,
    clean_id,
    estimate_damage_pct,
    get_type_multiplier,
    TYPE_TO_IDX,
    TYPES
)

STATE_DIM = 76
ACTION_DIM = 16
NUM_ACTION_SLOTS = 9
SHOWDOWN_WS_URL = "wss://sim3.psim.us/showdown/websocket"
ACTION_URL = "https://play.pokemonshowdown.com/action.php"


def encode_types(types: List[str]) -> np.ndarray:
    vec = np.zeros(len(TYPES), dtype=np.float32)
    for t in types:
        idx = TYPE_TO_IDX.get(t.lower())
        if idx is not None:
            vec[idx] = 1.0
    return vec


class TransitionBuffer:
    """Thread-safe experience replay buffer storing live online ladder transitions."""
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.lock = threading.Lock()
        self.buffer = []

    def push(self, transition: Dict[str, Any]):
        with self.lock:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
            self.buffer.append(transition)

    def sample_all(self) -> List[Dict[str, Any]]:
        with self.lock:
            batch = list(self.buffer)
            self.buffer.clear()
            return batch

    def size(self) -> int:
        with self.lock:
            return len(self.buffer)


class LiveLadderWorker:
    def __init__(
        self,
        worker_id: int,
        shared_model: PPOActorCriticNet,
        model_lock: threading.Lock,
        buffer: TransitionBuffer,
        reward_engine: DynamicMatrixRewardEngine,
        replays_dir: str,
        format_id: str = "gen9randombattleblitz",
        server_url: str = SHOWDOWN_WS_URL
    ):
        self.worker_id = worker_id
        self.model = shared_model
        self.model_lock = model_lock
        self.buffer = buffer
        self.reward_engine = reward_engine
        self.replays_dir = replays_dir
        self.format_id = format_id
        self.server_url = server_url
        self.data = ShowdownData.get()

        self.username = f"Amnesia_{random.randint(100, 999)}_{worker_id}"
        self.ws: Optional[websocket.WebSocketApp] = None
        self.rooms: Dict[str, List[str]] = {}
        self.room_requests: Dict[str, Any] = {}
        self.active_battles: List[str] = []
        self._logged_in = False
        self.last_challstr = None

        # Per-room state memory for PPO transitions
        self.room_prev_transition: Dict[str, Dict[str, Any]] = {}
        self.room_action_history: Dict[str, List[str]] = {}

    def send(self, message: str, room: str = ""):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(f"{room}|{message}")

    def search_ladder(self):
        if not self.active_battles:
            time.sleep(random.uniform(0.5, 2.0))
            self.send(f"/search {self.format_id}")

    def handle_login(self, challstr: str):
        self.last_challstr = challstr
        try:
            res = requests.post(ACTION_URL, data={
                "act": "getassertion",
                "userid": re.sub(r"[^a-zA-Z0-9]", "", self.username).lower(),
                "challstr": challstr
            }, timeout=10)
            assertion = res.text.strip()
            if assertion.startswith(";;"):
                data = json.loads(assertion[1:])
                assertion = data.get("assertion", "")
            self.send(f"/trn {self.username},0,{assertion}")
            self.send("/avatar red")
        except Exception:
            pass

    def compute_features(self, req: Dict[str, Any], history: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
        side = req.get("side", {})
        my_side_id = side.get("id", "p1")
        opp_side_id = "p2" if my_side_id == "p1" else "p1"
        opp_prefix = f"{opp_side_id}a:"

        pokemon_list = side.get("pokemon", [])
        active_req = req.get("active", [{}])[0] if req.get("active") else {}
        moves_req = active_req.get("moves", [])

        my_active = None
        my_bench = []

        for i, p in enumerate(pokemon_list):
            species = p.get("details", "").split(",")[0].strip()
            hp_str = p.get("condition", "100/100")
            fainted = "fnt" in hp_str or hp_str.startswith("0")
            hp_val = 1.0
            if "/" in hp_str:
                num, den = hp_str.split()[0].split("/")
                hp_val = float(num) / float(den) if float(den) > 0 else 0.0

            mon_info = {
                "species": species, "hp": hp_val, "fainted": fainted, "slot": i + 1,
                "moves": [m.get("move", m.get("id", "")) for m in moves_req] if p.get("active") else p.get("moves", [])
            }
            if p.get("active"): my_active = mon_info
            else: my_bench.append(mon_info)

        opp_species = "Pikachu"
        opp_hp = 1.0
        opp_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
        my_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
        opp_has_status = False

        for line in history:
            parts = line.split("|")
            if len(parts) < 2: continue
            cmd = parts[1]
            if cmd in ["switch", "drag"]:
                tag = parts[2].strip()
                if tag.startswith(opp_prefix) or tag.startswith(f"{opp_side_id}:"):
                    opp_species = parts[3].split(",")[0].strip()
                    hp_raw = parts[4].split()[0] if len(parts) > 4 else "100/100"
                    if "/" in hp_raw:
                        n, d = hp_raw.split("/")
                        opp_hp = float(n) / float(d) if float(d) > 0 else 1.0
                    opp_has_status = any(st in parts[4] for st in ["brn", "psn", "tox", "par", "slp", "frz"])
            elif cmd in ["-damage", "-heal"]:
                tag = parts[2].strip()
                hp_raw = parts[3].split()[0] if len(parts) > 3 else "100/100"
                if "fnt" in hp_raw:
                    if tag.startswith(opp_prefix): opp_hp = 0.0
                elif "/" in hp_raw:
                    n, d = hp_raw.split("/")
                    val = float(n) / float(d) if float(d) > 0 else 0.0
                    if tag.startswith(opp_prefix):
                        opp_hp = val
                        opp_has_status = any(st in parts[3] for st in ["brn", "psn", "tox", "par", "slp", "frz"])
            elif cmd == "-boost":
                tag, stat, amt = parts[2].strip(), parts[3].strip(), int(parts[4]) if len(parts) > 4 else 1
                if tag.startswith(opp_prefix) and stat in opp_boosts: opp_boosts[stat] = min(6, opp_boosts[stat] + amt)
                elif tag.startswith(f"{my_side_id}a:") and stat in my_boosts: my_boosts[stat] = min(6, my_boosts[stat] + amt)

        opp_poke = self.data.get_pokemon(opp_species)
        opp_types = opp_poke.get("types", ["Normal"]) if opp_poke else ["Normal"]
        opp_spe = opp_poke.get("baseStats", {}).get("spe", 80) if opp_poke else 80

        my_poke = self.data.get_pokemon(my_active["species"]) if my_active else None
        my_types = my_poke.get("types", ["Normal"]) if my_poke else ["Normal"]
        my_spe = my_poke.get("baseStats", {}).get("spe", 80) if my_poke else 80
        i_am_faster = my_spe > opp_spe

        # Action Matrix
        a_mat = np.zeros((NUM_ACTION_SLOTS, ACTION_DIM), dtype=np.float32)
        mask = np.zeros(NUM_ACTION_SLOTS, dtype=np.float32)
        status_moves = ["willowisp", "toxic", "thunderwave", "spore", "hypnosis", "yawn", "sing", "glare"]

        move_damages = []
        for i, m in enumerate(moves_req[:4]):
            if not m.get("disabled", False) and m.get("pp", 1) > 0:
                m_name = m.get("move", m.get("id", ""))
                dmg = estimate_damage_pct(my_active["species"] if my_active else "Pikachu", opp_species, m_name, my_boosts, opp_boosts)
                move_damages.append((i, m_name, dmg))

        max_dmg = max([d[2] for d in move_damages], default=0.01)

        for i, m_name, dmg in move_damages:
            mid = clean_id(m_name)
            m_data = self.data.get_move(m_name)
            if not m_data: continue
            if mid in status_moves and opp_has_status: continue

            mask[i] = 1.0
            a_mat[i, 0] = 1.0
            a_mat[i, 2] = min(2.0, dmg)
            a_mat[i, 3] = 1.0 if (dmg >= opp_hp and dmg > 0) else 0.0
            cat = m_data.get("category", "Status")
            if cat == "Physical": a_mat[i, 4] = 1.0
            elif cat == "Special": a_mat[i, 5] = 1.0
            else: a_mat[i, 6] = 1.0
            m_type = m_data.get("type", "Normal")
            a_mat[i, 7] = get_type_multiplier(m_type, opp_types) / 4.0
            a_mat[i, 8] = 1.5 if m_type in my_types else 1.0
            acc = m_data.get("accuracy", 100)
            a_mat[i, 9] = (acc if isinstance(acc, (int, float)) else 100) / 100.0
            a_mat[i, 10] = (m_data.get("priority", 0) + 6.0) / 12.0
            a_mat[i, 13] = dmg / max(max_dmg, 0.01)
            a_mat[i, 14] = 1.0 if (dmg == max_dmg and dmg > 0.05) else 0.0

        for i, b in enumerate(my_bench[:5]):
            slot = 4 + i
            if not b["fainted"] and b["hp"] > 0:
                mask[slot] = 1.0
                a_mat[slot, 1] = 1.0
                a_mat[slot, 2] = b["hp"]
                b_poke = self.data.get_pokemon(b["species"])
                b_types = b_poke.get("types", ["Normal"]) if b_poke else ["Normal"]
                res = [get_type_multiplier(ot, b_types) for ot in opp_types]
                a_mat[slot, 7] = (min(res) if res else 1.0) / 4.0

        if mask.sum() == 0:
            mask[0] = 1.0

        s_vec = np.zeros(STATE_DIM, dtype=np.float32)
        s_vec[0] = my_active["hp"] if my_active else 0.0
        s_vec[1] = opp_hp
        s_vec[2:8] = [my_boosts.get(k, 0) / 6.0 for k in ["atk", "def", "spa", "spd", "spe", "acc"]]
        s_vec[8:14] = [opp_boosts.get(k, 0) / 6.0 for k in ["atk", "def", "spa", "spd", "spe", "acc"]]
        my_alive = (1 if my_active and not my_active["fainted"] else 0) + sum(1 for b in my_bench if not b["fainted"])
        s_vec[14] = my_alive / 6.0
        s_vec[15] = 0.8
        s_vec[34:52] = encode_types(my_types)
        s_vec[52:70] = encode_types(opp_types)

        state_meta = {
            "my_alive": my_alive,
            "opp_alive": int(opp_hp > 0) + 5,
            "my_hp": s_vec[0],
            "opp_hp": opp_hp,
            "i_am_faster": i_am_faster,
            "type_resistance_factor": min([get_type_multiplier(ot, my_types) for ot in opp_types], default=1.0)
        }

        action_context = {
            "active_req": active_req,
            "my_bench": my_bench,
            "move_damages": move_damages,
            "opp_hp": opp_hp,
            "i_am_faster": i_am_faster
        }

        return s_vec, a_mat, mask, state_meta, action_context

    def choose_and_record_action(self, room: str, req: Dict[str, Any]) -> str:
        s_vec, a_mat, mask, curr_state_meta, ctx = self.compute_features(req, self.rooms.get(room, []))

        # Check and compute reward for previous transition in this room
        if room in self.room_prev_transition:
            prev_data = self.room_prev_transition[room]
            dmg_dealt = max(0.0, prev_data["state_meta"]["opp_hp"] - curr_state_meta["opp_hp"])
            dmg_taken = max(0.0, prev_data["state_meta"]["my_hp"] - curr_state_meta["my_hp"])
            
            history = self.room_action_history.get(room, [])
            rep_count = history[-2:].count(prev_data["action_cmd"])

            r = self.reward_engine.calculate_turn_reward(
                prev_data["state_meta"],
                curr_state_meta,
                prev_data["action_cmd"],
                dmg_dealt=dmg_dealt,
                dmg_taken=dmg_taken,
                repeated_action_count=rep_count,
                is_terminal=False
            )

            # Push transition into experience buffer
            self.buffer.push({
                "state": prev_data["s_vec"],
                "actions": prev_data["a_mat"],
                "mask": prev_data["mask"],
                "chosen": prev_data["chosen_idx"],
                "log_prob": prev_data["log_prob"],
                "value": prev_data["value"],
                "reward": r,
                "done": False
            })

        # Neural Inference (Sample action according to policy distribution)
        device = next(self.model.parameters()).device
        s_t = torch.tensor(s_vec, dtype=torch.float32, device=device).unsqueeze(0)
        a_t = torch.tensor(a_mat, dtype=torch.float32, device=device).unsqueeze(0)
        m_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

        with self.model_lock:
            with torch.no_grad():
                logits, value = self.model(s_t, a_t, m_t)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                chosen_idx = int(dist.sample().item())
                log_prob = float(dist.log_prob(torch.tensor(chosen_idx, device=device)).item())
                val_float = float(value.item())

        # Determine Showdown Action Command
        can_tera = ctx["active_req"].get("canTerastallize")
        if chosen_idx < 4:
            move_slot = chosen_idx + 1
            action_cmd = f"move {move_slot} terastallize" if (can_tera and a_mat[chosen_idx, 2] >= 0.70) else f"move {move_slot}"
        else:
            bench_idx = chosen_idx - 4
            if bench_idx < len(ctx["my_bench"]):
                action_cmd = f"switch {ctx["my_bench"][bench_idx]['slot']}"
            else:
                action_cmd = "move 1"

        if room not in self.room_action_history:
            self.room_action_history[room] = []
        self.room_action_history[room].append(action_cmd)

        self.room_prev_transition[room] = {
            "s_vec": s_vec,
            "a_mat": a_mat,
            "mask": mask,
            "chosen_idx": chosen_idx,
            "log_prob": log_prob,
            "value": val_float,
            "state_meta": curr_state_meta,
            "action_cmd": action_cmd
        }

        return action_cmd

    def save_replay(self, room: str, winner: str):
        """Saves battle logs into structured replay storage."""
        try:
            os.makedirs(self.replays_dir, exist_ok=True)
            replay_path = os.path.join(self.replays_dir, f"{room}.json")
            data = {
                "id": room,
                "winner": winner,
                "bot_username": self.username,
                "log": "\n".join(self.rooms.get(room, [])),
                "timestamp": int(time.time())
            }
            with open(replay_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def on_message(self, ws, raw_message: str):
        lines = raw_message.split("\n")
        room = ""
        if lines[0].startswith(">"):
            room = lines[0][1:].strip()
            lines = lines[1:]

        if room and room not in self.rooms:
            self.rooms[room] = []

        for line in lines:
            if not line: continue
            if room: self.rooms[room].append(line)

            parts = line.split("|")
            if len(parts) < 2: continue
            msg_type = parts[1]

            if msg_type == "challstr":
                self.handle_login("|".join(parts[2:]))

            elif msg_type == "nametaken":
                self.username = f"Amnesia_{random.randint(100, 999)}_{self.worker_id}"
                if self.last_challstr: self.handle_login(self.last_challstr)

            elif msg_type == "updateuser":
                name = parts[2].strip()
                if name and not name.startswith("Guest") and not self._logged_in:
                    self._logged_in = True
                    self.username = name
                    self.search_ladder()

            elif msg_type == "updatesearch":
                try:
                    search_data = json.loads(parts[2])
                    games = search_data.get("games")
                    if games:
                        for b_room in games:
                            if b_room not in self.active_battles:
                                self.active_battles.append(b_room)
                                self.send("/timer on", room=b_room)
                except Exception:
                    pass

            elif msg_type == "request":
                req_str = parts[2].strip()
                if req_str:
                    try:
                        req = json.loads(req_str)
                        if req.get("teamPreview"):
                            self.send("/team 123456", room=room)
                        elif not req.get("wait"):
                            action_cmd = self.choose_and_record_action(room, req)
                            self.send(f"/choose {action_cmd}", room=room)
                    except Exception:
                        self.send("/choose default", room=room)

            elif msg_type == "error":
                err_text = parts[2] if len(parts) > 2 else ""
                if "Invalid choice" in err_text or "trapped" in err_text:
                    self.send("/choose default", room=room)

            elif msg_type == "win":
                winner = parts[2].strip()
                clean_bot = re.sub(r"[^a-zA-Z0-9]", "", self.username).lower()
                clean_win = re.sub(r"[^a-zA-Z0-9]", "", winner).lower()
                is_win = (clean_win == clean_bot)

                # Terminal transition for buffer
                if room in self.room_prev_transition:
                    prev_data = self.room_prev_transition[room]
                    r_term = self.reward_engine.calculate_turn_reward(
                        prev_data["state_meta"],
                        prev_data["state_meta"],
                        prev_data["action_cmd"],
                        dmg_dealt=0.0,
                        dmg_taken=0.0,
                        is_terminal=True,
                        won=is_win
                    )
                    self.buffer.push({
                        "state": prev_data["s_vec"],
                        "actions": prev_data["a_mat"],
                        "mask": prev_data["mask"],
                        "chosen": prev_data["chosen_idx"],
                        "log_prob": prev_data["log_prob"],
                        "value": prev_data["value"],
                        "reward": r_term,
                        "done": True
                    })
                    del self.room_prev_transition[room]

                # Save official replay file
                self.save_replay(room, winner)

                self.send(f"/leave {room}")
                if room in self.active_battles:
                    self.active_battles.remove(room)
                if room in self.room_action_history:
                    del self.room_action_history[room]

                time.sleep(1.0)
                self.search_ladder()

    def run(self):
        self.ws = websocket.WebSocketApp(
            self.server_url,
            on_message=self.on_message
        )
        self.ws.run_forever()


class OnlineRLTrainer:
    def __init__(
        self,
        num_workers: int = 20,
        checkpoint_path: str = "projects/reinforcement-learning/weights/ppo_model.pt",
        replays_dir: str = "data/replays/ladder_rl",
        update_interval_transitions: int = 300,
        lr: float = 2e-4,
        ppo_epochs: int = 4
    ):
        self.num_workers = num_workers
        self.checkpoint_path = checkpoint_path
        self.replays_dir = replays_dir
        self.update_interval = update_interval_transitions
        self.lr = lr
        self.ppo_epochs = ppo_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("=" * 80)
        print("⚡ AMNESIA-AI ON-LADDER REINFORCEMENT LEARNING TRAINER")
        print("=" * 80)
        print(f"Device:               {self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print(f"Active Ladder Workers: {self.num_workers} simultaneous online battle connections")
        print(f"Replay Directory:     {self.replays_dir}")
        print(f"Policy Checkpoint:    {self.checkpoint_path}")
        print("=" * 80 + "\n", flush=True)

        self.model = PPOActorCriticNet(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=128,
            num_actions=NUM_ACTION_SLOTS
        ).to(self.device)

        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            st = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(st, strict=False)
            print(f"[+] Loaded Initial Weights from {self.checkpoint_path}")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.model_lock = threading.Lock()
        self.buffer = TransitionBuffer(capacity=10000)
        self.reward_engine = DynamicMatrixRewardEngine()

        self.total_updates = 0
        self.total_transitions_collected = 0

    def optimize_ppo(self, transitions: List[Dict[str, Any]]):
        if len(transitions) < 32:
            return

        states = torch.tensor(np.array([t["state"] for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([t["actions"] for t in transitions]), dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([t["mask"] for t in transitions]), dtype=torch.float32, device=self.device)
        chosen = torch.tensor(np.array([t["chosen"] for t in transitions]), dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(np.array([t["log_prob"] for t in transitions]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([t["reward"] for t in transitions]), dtype=torch.float32, device=self.device)
        old_values = torch.tensor(np.array([t["value"] for t in transitions]), dtype=torch.float32, device=self.device)

        # Simple advantage estimation
        returns = rewards + 0.99 * old_values
        advantages = (returns - old_values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with self.model_lock:
            self.model.train()
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

                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.total_updates += 1
            print(f"🔄 [PPO GPU UPDATE #{self.total_updates}] Batch: {len(transitions)} turns | Loss: {total_loss.item():.4f} (Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f})", flush=True)

            # Persist updated weights
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            torch.save({"model_state_dict": self.model.state_dict()}, self.checkpoint_path)

    def start(self):
        workers = [
            LiveLadderWorker(
                worker_id=i + 1,
                shared_model=self.model,
                model_lock=self.model_lock,
                buffer=self.buffer,
                reward_engine=self.reward_engine,
                replays_dir=self.replays_dir,
                format_id="gen9randombattleblitz"
            )
            for i in range(self.num_workers)
        ]

        print(f"🚀 Spawning {self.num_workers} Ladder Worker Threads (Random Battle Blitz)...", flush=True)
        for w in workers:
            t = threading.Thread(target=w.run, daemon=True)
            t.start()
            time.sleep(0.3)  # Stagger handshakes

        # Training Loop: monitors buffer and triggers CUDA PPO updates
        loop_counter = 0
        while True:
            time.sleep(2.0)
            loop_counter += 1
            cur_size = self.buffer.size()
            if loop_counter % 5 == 0:
                print(f"📊 [LADDER RL STATUS] Buffer: {cur_size}/{self.update_interval} transitions | Total Turns Collected: {self.total_transitions_collected} | PPO Updates: {self.total_updates}", flush=True)

            if cur_size >= self.update_interval:
                transitions = self.buffer.sample_all()
                self.total_transitions_collected += len(transitions)
                self.optimize_ppo(transitions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Ladder Online RL Trainer")
    parser.add_argument("--workers", type=int, default=20, help="Number of concurrent ladder workers")
    parser.add_argument("--batch", type=int, default=300, help="Transitions per PPO update")
    parser.add_argument("--lr", type=float, default=2e-4, help="PPO learning rate")
    args = parser.parse_args()

    trainer = OnlineRLTrainer(
        num_workers=args.workers,
        update_interval_transitions=args.batch,
        lr=args.lr
    )
    trainer.start()
