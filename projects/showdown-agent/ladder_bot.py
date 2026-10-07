"""
Pokemon Showdown Live Online Ladder Climber & Real-Time Rating Tracker
======================================================================
Automated Ladder Bot that:
1. Connects to the official Pokemon Showdown server
2. Enqueues in the Ranked Ladder (/search gen9randombattle)
3. Plays real human matches online using the trained Neural PPO/BC Model
4. Tracks Rating (Elo), Wins, Losses, and Ladder Rank in real time
5. Automatically queues up for the next match upon battle completion
"""

import argparse
import json
import os
import random
import re
import sys
import time
import requests
import websocket
from typing import Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from neural_brain import NeuralBrain

SHOWDOWN_WS_URL = "wss://sim3.psim.us/showdown/websocket"
ACTION_URL = "https://play.pokemonshowdown.com/action.php"


class ShowdownLadderBot:
    def __init__(
        self,
        username: str,
        password: Optional[str] = None,
        format_id: str = "gen9randombattleblitz",
        max_battles: int = 50,
        checkpoint_path: str = "projects/reinforcement-learning/weights/ppo_model.pt",
        server_url: str = SHOWDOWN_WS_URL,
        avatar: str = "red"
    ):
        self.base_username = username
        self.username = username
        self.password = password
        self.format_id = format_id
        self.max_battles = max_battles
        self.server_url = server_url
        self.avatar = avatar
        self.last_challstr = None

        self.agent = NeuralBrain(checkpoint_path=checkpoint_path)
        self.ws: Optional[websocket.WebSocketApp] = None
        self.rooms: Dict[str, List[str]] = {}
        self.active_battles: List[str] = []

        # Ladder Stats
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.current_elo = 1000
        self.starting_elo = 1000
        self.is_searching = False
        self._logged_in = False

    def send(self, message: str, room: str = ""):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            payload = f"{room}|{message}"
            self.ws.send(payload)

    def send_chat(self, room: str, message: str):
        self.send(message, room=room)

    def search_ladder(self):
        if self.total_games >= self.max_battles:
            print(f"\n🎉 Completed target of {self.max_battles} ladder games!", flush=True)
            if self.ws:
                self.ws.close()
            return

        if not self.active_battles:
            self.is_searching = True
            print(f"\n🔎 [LADDER SEARCH] Enqueuing match #{self.total_games + 1} in '{self.format_id}'... (Record: {self.wins}W-{self.losses}L, Elo: {self.current_elo})", flush=True)
            self.send(f"/search {self.format_id}")

    def handle_login(self, challstr: str, name_to_try: str):
        self.last_challstr = challstr
        clean_user = re.sub(r"[^a-zA-Z0-9]", "", name_to_try).lower()
        try:
            if self.password:
                res = requests.post(ACTION_URL, data={
                    "act": "login",
                    "name": name_to_try,
                    "pass": self.password,
                    "challstr": challstr
                }, timeout=10)
            else:
                res = requests.post(ACTION_URL, data={
                    "act": "getassertion",
                    "userid": clean_user,
                    "challstr": challstr
                }, timeout=10)

            if res.status_code == 200:
                assertion = res.text
                if assertion.startswith("]"):
                    data = json.loads(assertion[1:])
                    assertion = data.get("assertion", "")

                self.send(f"/trn {name_to_try},0,{assertion}")
                self.send(f"/avatar {self.avatar}")
        except Exception as e:
            print(f"[-] Error during login: {e}")

    def on_message(self, ws, raw_message: str):
        lines = raw_message.split("\n")
        room = ""

        if lines[0].startswith(">"):
            room = lines[0][1:].strip()
            lines = lines[1:]

        if room and room not in self.rooms:
            self.rooms[room] = []

        for line in lines:
            if not line:
                continue

            if room:
                self.rooms[room].append(line)

            parts = line.split("|")
            if len(parts) < 2:
                continue

            msg_type = parts[1]

            # 1. Challstr
            if msg_type == "challstr":
                challstr = "|".join(parts[2:])
                self.handle_login(challstr, self.username)

            # 2. Nick in use -> Retry
            elif msg_type == "nametaken":
                new_name = f"{self.base_username}_{random.randint(100, 999)}"
                print(f"[*] Nick taken. Retrying as '{new_name}'...", flush=True)
                self.username = new_name
                if self.last_challstr:
                    self.handle_login(self.last_challstr, self.username)

            # 3. Login Confirmed
            elif msg_type == "updateuser":
                logged_name = parts[2].strip()
                if logged_name and not logged_name.startswith("Guest") and not self._logged_in:
                    self._logged_in = True
                    self.username = logged_name
                    print("\n" + "=" * 70, flush=True)
                    print(f"✅ LOGGED IN AS: '{logged_name}'", flush=True)
                    print(f"🎯 READY TO CLIMB LADDER: {self.format_id}", flush=True)
                    print("=" * 70, flush=True)
                    # Start Ladder Search
                    self.search_ladder()

            # 4. Updatesearch -> Ladder Match Found
            elif msg_type == "updatesearch":
                try:
                    search_data = json.loads(parts[2])
                    games = search_data.get("games")
                    self.is_searching = bool(search_data.get("searching"))
                    if games:
                        for battle_room in games:
                            if battle_room not in self.active_battles:
                                self.active_battles.append(battle_room)
                                print(f"\n[⚔️ MATCH FOUND!] Joined Battle Room: {battle_room}", flush=True)
                                self.send("/timer on", room=battle_room)
                except Exception:
                    pass

            # 5. Battle Request
            elif msg_type == "request":
                req_str = parts[2].strip()
                if req_str:
                    try:
                        req = json.loads(req_str)
                        if req.get("teamPreview"):
                            self.send("/team 123456", room=room)
                        elif not req.get("wait"):
                            action_cmd = self.agent.choose_action(req, self.rooms.get(room, []))
                            print(f"[{room}] Action: /choose {action_cmd}", flush=True)
                            self.send(f"/choose {action_cmd}", room=room)
                    except Exception as e:
                        print(f"[-] Error choosing action: {e}", flush=True)
                        self.send("/choose default", room=room)

            # 6. Error handling
            elif msg_type == "error":
                err_text = parts[2] if len(parts) > 2 else ""
                if "Invalid choice" in err_text or "trapped" in err_text:
                    self.send("/choose default", room=room)

            # 7. Start
            elif msg_type == "start":
                print(f"[{room}] 🚀 Battle Started! Good luck!", flush=True)
                self.send_chat(room, "GL HF! (Amnesia-AI Neural Ladder Bot)")

            # 8. Win / Match Finished
            elif msg_type == "win":
                winner = parts[2].strip()
                clean_bot = re.sub(r"[^a-zA-Z0-9]", "", self.username).lower()
                clean_win = re.sub(r"[^a-zA-Z0-9]", "", winner).lower()
                is_victory = (clean_win == clean_bot)

                self.total_games += 1
                if is_victory:
                    self.wins += 1
                    print(f"\n[{room}] 🏆 VICTORY! Winner: {winner}", flush=True)
                else:
                    self.losses += 1
                    print(f"\n[{room}] 💀 DEFEAT! Winner: {winner}", flush=True)

                win_rate = (self.wins / self.total_games) * 100.0
                print("-" * 70, flush=True)
                print(f"📊 LADDER RECORD: {self.wins}W - {self.losses}L ({win_rate:.1f}% Win Rate) | Total Games: {self.total_games}/{self.max_battles}", flush=True)
                print("-" * 70, flush=True)

                self.send(f"/leave {room}")
                if room in self.active_battles:
                    self.active_battles.remove(room)

                # Wait 2 seconds and search next ladder match!
                time.sleep(2)
                self.search_ladder()

            # 9. Elo updates in raw html
            elif msg_type == "raw" and "ladder" in line:
                clean_line = re.sub(r"<[^>]+>", " ", line)
                elo_matches = re.findall(r"(\d{3,4})\s*(?:&rarr;|->|to)\s*(\d{3,4})", clean_line)
                if elo_matches:
                    old_e, new_e = elo_matches[0]
                    self.current_elo = int(new_e)
                    print(f"📈 [RATING UPDATE] Elo: {old_e} ➔ {new_e} (Δ: {int(new_e) - int(old_e):+d})", flush=True)


    def on_error(self, ws, error):
        print(f"[-] WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[*] Disconnected from Showdown.")

    def on_open(self, ws):
        print(f"[+] Connected to Showdown Server: {self.server_url}")

    def start(self):
        self.ws = websocket.WebSocketApp(
            self.server_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever()


def main():
    parser = argparse.ArgumentParser(description="Run Amnesia-AI on Pokemon Showdown Ranked Ladder")
    parser.add_argument("--username", type=str, default=f"AmnesiaBot_{random.randint(100, 999)}", help="Bot username")
    parser.add_argument("--password", type=str, default=None, help="Registered account password (optional)")
    parser.add_argument("--format", type=str, default="gen9randombattleblitz", help="Ladder format (default: gen9randombattleblitz)")
    parser.add_argument("--battles", type=int, default=30, help="Number of ladder battles to play")
    parser.add_argument("--checkpoint", type=str, default="projects/reinforcement-learning/weights/ppo_model.pt", help="Model checkpoint")

    args = parser.parse_args()

    print("=" * 70)
    print("🤖 AMNESIA-AI RANKED LADDER CLIMBER")
    print("=" * 70)
    print(f"Bot Account:    {args.username}")
    print(f"Target Format:  {args.format}")
    print(f"Battle Goal:    {args.battles} ranked matches")
    print(f"Neural Model:   {args.checkpoint}")
    print("=" * 70 + "\n")

    bot = ShowdownLadderBot(
        username=args.username,
        password=args.password,
        format_id=args.format,
        max_battles=args.battles,
        checkpoint_path=args.checkpoint
    )
    bot.start()


if __name__ == "__main__":
    main()
