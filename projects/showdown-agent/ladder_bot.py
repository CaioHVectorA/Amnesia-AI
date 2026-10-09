"""
Pokemon Showdown Live Online Ladder Climber & Real-Time Rating Tracker
======================================================================
Automated Ladder Bot that:
1. Connects to the official Pokemon Showdown server (or custom server)
2. Enqueues in the Ranked Ladder with team upload (/utm) for OU or Random
3. Plays real human matches online using the trained Neural PPO/BC Model
4. Tracks Rating (Elo), Wins, Losses, and Ladder Rank in real time
5. Emits clickable live spectator and replay links: https://play.pokemonshowdown.com/<room>
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
        format_id: str = "gen8ou",
        max_battles: int = 50,
        checkpoint_path: str = "projects/reinforcement-learning/weights/real_sim_gen8ou_ppo.pt",
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

        if not os.path.exists(checkpoint_path):
            checkpoint_path = "projects/reinforcement-learning/weights/ppo_model.pt"

        self.agent = NeuralBrain(checkpoint_path=checkpoint_path)
        self.ws: Optional[websocket.WebSocketApp] = None
        self.rooms: Dict[str, List[str]] = {}
        self.active_battles: List[str] = []

        # Load Gen 8 OU teams if needed
        self.teams: List[str] = []
        teams_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/gen8ou_teams.json"))
        if os.path.exists(teams_path):
            try:
                with open(teams_path, "r", encoding="utf-8") as f:
                    tdata = json.load(f)
                    self.teams = [t["team"] if isinstance(t["team"], str) else "]".join(t["team"]) for t in tdata]
            except Exception as e:
                print(f"[-] Teams load error: {e}")

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

    def get_team_pack(self) -> str:
        if "random" in self.format_id.lower():
            return "null"
        if self.teams:
            return random.choice(self.teams)
        return "null"

    def search_ladder(self):
        if self.total_games >= self.max_battles:
            print(f"\n🎉 Completed target of {self.max_battles} ladder games!", flush=True)
            if self.ws:
                self.ws.close()
            return

        if not self.active_battles:
            self.is_searching = True
            team_pack = self.get_team_pack()
            self.send(f"/utm {team_pack}")
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

            # 1. Challenge String for Login
            if msg_type == "challstr":
                challstr = "|".join(parts[2:])
                self.handle_login(challstr, self.username)

            # 2. Name already taken
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
                    print("\n" + "=" * 75, flush=True)
                    print(f"✅ LOGGED IN AS: '{logged_name}'", flush=True)
                    print(f"🎯 READY TO CLIMB LADDER: {self.format_id}", flush=True)
                    print("=" * 75, flush=True)
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
                                print("\n" + "🔥" * 38, flush=True)
                                print(f"⚔️ [MATCH FOUND!] Entrou na Batalha: {battle_room}", flush=True)
                                print(f"🔗 ASSISTIR AO VIVO: https://play.pokemonshowdown.com/{battle_room}", flush=True)
                                print(f"📹 LINK DO REPLAY:   https://replay.pokemonshowdown.com/{battle_room}", flush=True)
                                print("🔥" * 38 + "\n", flush=True)
                                self.send("/timer on", room=battle_room)
                                self.send("/savereplay", room=battle_room)
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
                print(f"[{room}] 🚀 Battle Started!", flush=True)
                self.send_chat(room, "GL HF! (Amnesia-AI Live Bot)")
                self.send("/savereplay", room=room)

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

                self.send("/savereplay", room=room)
                print(f"📊 Live Score: {self.wins}W - {self.losses}L | Win Rate: {(self.wins/self.total_games)*100:.1f}%", flush=True)
                print(f"📹 Replay final salvo em: https://replay.pokemonshowdown.com/{room}\n", flush=True)

                # Leave and search next
                time.sleep(2)
                self.send(f"/leave {room}")
                self.active_battles.clear()
                self.is_searching = False
                self.search_ladder()

            # 9. Elo update
            elif "raw" in msg_type and "rating:" in line.lower():
                try:
                    match = re.search(r"(\d{3,4})&lt;!--", line)
                    if match:
                        self.current_elo = int(match.group(1))
                        print(f"📈 [RATING UPDATE] Current Elo: {self.current_elo}", flush=True)
                except Exception:
                    pass

    def on_error(self, ws, error):
        print(f"[-] WebSocket Error: {error}", flush=True)

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[*] Connection Closed ({close_status_code}): {close_msg}", flush=True)

    def on_open(self, ws):
        print("[+] Connected to Showdown Server! Waiting for challstr...", flush=True)

    def run(self):
        self.ws = websocket.WebSocketApp(
            self.server_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amnesia-AI Live Ladder Climber")
    parser.add_argument("--format", type=str, default="gen9randombattleblitz", help="Format to search (e.g. gen9randombattleblitz, gen8ou, gen9ou)")
    parser.add_argument("--battles", type=int, default=20, help="Max battles to play")
    parser.add_argument("--username", type=str, default=f"Amnesia_{random.randint(100,999)}", help="Bot username")
    parser.add_argument("--weights", type=str, default="projects/reinforcement-learning/weights/real_sim_gen8ou_ppo.pt", help="Path to weights")
    args = parser.parse_args()

    bot = ShowdownLadderBot(
        username=args.username,
        format_id=args.format,
        max_battles=args.battles,
        checkpoint_path=args.weights
    )
    bot.run()
