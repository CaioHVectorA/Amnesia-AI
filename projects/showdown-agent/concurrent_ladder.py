"""
Concurrent Pokémon Showdown Live Ladder Climber & Benchmark (Multi-Battle Engine)
================================================================================
Plays multiple ranked matches concurrently on Pokémon Showdown ladder:
- Format: gen9randombattleblitz (Random Battle Blitz)
- Multi-Worker: Runs N parallel bot accounts/connections simultaneously (e.g. 5 concurrent battles)
- Shared Neural Policy: All workers share the same PyTorch Neural Model (GPU-accelerated)
- Aggregated Benchmark Metrics: Win Rate (%), Elo trajectory, Turns/match, Game duration
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from typing import Dict, List, Optional
import requests
import websocket

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from neural_brain import NeuralBrain

SHOWDOWN_WS_URL = "wss://sim3.psim.us/showdown/websocket"
ACTION_URL = "https://play.pokemonshowdown.com/action.php"


class ConcurrentLadderWorker:
    def __init__(
        self,
        worker_id: int,
        shared_brain: NeuralBrain,
        stats_tracker: "LadderStatsTracker",
        format_id: str = "gen9randombattleblitz",
        max_battles: int = 10,
        server_url: str = SHOWDOWN_WS_URL
    ):
        self.worker_id = worker_id
        self.shared_brain = shared_brain
        self.stats = stats_tracker
        self.format_id = format_id
        self.max_battles = max_battles
        self.server_url = server_url

        self.username = f"Amnesia_{random.randint(100, 999)}_{worker_id}"
        self.ws: Optional[websocket.WebSocketApp] = None
        self.rooms: Dict[str, List[str]] = {}
        self.active_battles: List[str] = []
        self.games_played = 0
        self._logged_in = False
        self.last_challstr = None

    def send(self, message: str, room: str = ""):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(f"{room}|{message}")

    def search_ladder(self):
        if self.stats.is_complete():
            if self.ws:
                self.ws.close()
            return

        if not self.active_battles:
            print(f"[Worker #{self.worker_id} - {self.username}] 🔎 Searching match in '{self.format_id}'...", flush=True)
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
        except Exception as e:
            print(f"[Worker #{self.worker_id}] Login Error: {e}", flush=True)

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
                if self.last_challstr:
                    self.handle_login(self.last_challstr)

            elif msg_type == "updateuser":
                logged_name = parts[2].strip()
                if logged_name and not logged_name.startswith("Guest") and not self._logged_in:
                    self._logged_in = True
                    self.username = logged_name
                    print(f"✅ [Worker #{self.worker_id}] Ready as '{logged_name}'", flush=True)
                    self.search_ladder()

            elif msg_type == "updatesearch":
                try:
                    search_data = json.loads(parts[2])
                    games = search_data.get("games")
                    if games:
                        for battle_room in games:
                            if battle_room not in self.active_battles:
                                self.active_battles.append(battle_room)
                                print(f"⚔️ [Worker #{self.worker_id}] Joined: {battle_room}", flush=True)
                                self.send("/timer on", room=battle_room)
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
                            action_cmd = self.shared_brain.choose_action(req, self.rooms.get(room, []))
                            print(f"[Worker #{self.worker_id} | {room}] Action: /choose {action_cmd}", flush=True)
                            self.send(f"/choose {action_cmd}", room=room)
                    except Exception as e:
                        print(f"[-] Worker #{self.worker_id} Error choosing: {e}", flush=True)
                        self.send("/choose default", room=room)

            elif msg_type == "error":
                err_text = parts[2] if len(parts) > 2 else ""
                print(f"[Worker #{self.worker_id} | {room}] ⚠️ Showdown Error: {err_text}", flush=True)
                if "Invalid choice" in err_text or "trapped" in err_text:
                    self.send("/choose default", room=room)

            elif msg_type == "win":
                winner = parts[2].strip()
                clean_bot = re.sub(r"[^a-zA-Z0-9]", "", self.username).lower()
                clean_win = re.sub(r"[^a-zA-Z0-9]", "", winner).lower()
                is_victory = (clean_win == clean_bot)

                self.games_played += 1
                self.stats.record_match(is_victory, winner, room, self.worker_id)
                self.send(f"/leave {room}")
                if room in self.active_battles:
                    self.active_battles.remove(room)

                if not self.stats.is_complete():
                    time.sleep(1.0)
                    self.search_ladder()
                else:
                    if self.ws: self.ws.close()

            elif msg_type == "raw" and "ladder" in line:
                clean_line = re.sub(r"<[^>]+>", " ", line)
                elo_matches = re.findall(r"(\d{3,4})\s*(?:&rarr;|->|to)\s*(\d{3,4})", clean_line)
                if elo_matches:
                    old_e, new_e = elo_matches[0]
                    self.stats.update_elo(int(new_e))

    def run(self):
        self.ws = websocket.WebSocketApp(
            self.server_url,
            on_message=self.on_message
        )
        self.ws.run_forever()


class LadderStatsTracker:
    def __init__(self, target_games: int):
        self.target_games = target_games
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.current_elo = 1000
        self.lock = threading.Lock()
        self.start_time = time.time()

    def record_match(self, is_victory: bool, winner: str, room: str, worker_id: int):
        with self.lock:
            self.total_games += 1
            if is_victory:
                self.wins += 1
                icon = "🏆 VICTORY"
            else:
                self.losses += 1
                icon = "💀 DEFEAT"

            wr = (self.wins / self.total_games) * 100.0
            elapsed = time.time() - self.start_time
            print("\n" + "=" * 75, flush=True)
            print(f"[{icon}] Worker #{worker_id} | Battle: {room} | Winner: {winner}", flush=True)
            print(f"📊 LIVE AGGREGATED LADDER: {self.wins}W - {self.losses}L ({wr:.1f}% WR) | Total: {self.total_games}/{self.target_games} | Elo: {self.current_elo} | Time: {elapsed:.1f}s", flush=True)
            print("=" * 75 + "\n", flush=True)

    def update_elo(self, elo: int):
        with self.lock:
            self.current_elo = elo

    def is_complete(self) -> bool:
        with self.lock:
            return self.total_games >= self.target_games


def run_concurrent_ladder_benchmark(
    concurrency: int = 5,
    total_battles: int = 20,
    format_id: str = "gen9randombattleblitz",
    checkpoint: str = "projects/reinforcement-learning/weights/ppo_model.pt"
):
    print("=" * 80)
    print("⚡ AMNESIA-AI MULTI-WORKER LIVE LADDER BENCHMARK (RANDOM BATTLE BLITZ)")
    print("=" * 80)
    print(f"Concurrent Battles:   {concurrency} simultaneous matches")
    print(f"Total Battles Goal:   {total_battles} ranked games")
    print(f"Format Target:        {format_id}")
    print(f"Neural Checkpoint:    {checkpoint}")
    print("=" * 80 + "\n", flush=True)

    shared_brain = NeuralBrain(checkpoint_path=checkpoint)
    stats = LadderStatsTracker(target_games=total_battles)

    workers = [
        ConcurrentLadderWorker(
            worker_id=i + 1,
            shared_brain=shared_brain,
            stats_tracker=stats,
            format_id=format_id,
            max_battles=total_battles
        )
        for i in range(concurrency)
    ]

    threads = []
    for w in workers:
        t = threading.Thread(target=w.run, daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)

    # Monitor loop
    while not stats.is_complete():
        time.sleep(1.0)

    print("\n" + "=" * 80)
    print("🏆 MULTI-WORKER BLITZ LADDER BENCHMARK COMPLETED")
    print("=" * 80)
    final_wr = (stats.wins / max(1, stats.total_games)) * 100.0
    duration = time.time() - stats.start_time
    print(f"Total Ranked Games: {stats.total_games}")
    print(f"Record:             {stats.wins}W - {stats.losses}L")
    print(f"Final Win Rate:     {final_wr:.2f}%")
    print(f"Peak Elo Rating:    {stats.current_elo}")
    print(f"Total Time:         {duration:.1f}s ({(stats.total_games / duration) * 60:.1f} games/min)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-worker concurrent Showdown ladder climber")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent matches")
    parser.add_argument("--battles", type=int, default=20, help="Total battles to play")
    parser.add_argument("--format", type=str, default="gen9randombattleblitz", help="Format ID")
    parser.add_argument("--checkpoint", type=str, default="projects/reinforcement-learning/weights/ppo_model.pt")
    args = parser.parse_args()

    run_concurrent_ladder_benchmark(
        concurrency=args.concurrency,
        total_battles=args.battles,
        format_id=args.format,
        checkpoint=args.checkpoint
    )
