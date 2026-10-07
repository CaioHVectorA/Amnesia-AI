"""
Pokemon Showdown Native WebSocket Bot Client (Direct Challenge & Auto-Accept)
=============================================================================
Directly connects to Pokemon Showdown, handles login, can auto-challenge
any player, and plays in real-time.
"""

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


class ShowdownBotClient:
    def __init__(
        self,
        username: str,
        password: Optional[str] = None,
        challenge_user: Optional[str] = None,
        format_id: str = "gen9randombattle",
        checkpoint_path: str = "projects/behavioral-cloning/weights/bc_model.pt",
        server_url: str = SHOWDOWN_WS_URL,
        avatar: str = "red"
    ):
        self.base_username = username
        self.username = username
        self.password = password
        self.challenge_user = challenge_user
        self.format_id = format_id
        self.server_url = server_url
        self.avatar = avatar
        self.last_challstr = None

        self.agent = NeuralBrain(checkpoint_path=checkpoint_path)
        self.ws: Optional[websocket.WebSocketApp] = None
        self.rooms: Dict[str, List[str]] = {}
        self.active_battles: List[str] = []
        self._logged_in = False
        self._challenge_sent = False

    def send(self, message: str, room: str = ""):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            payload = f"{room}|{message}"
            self.ws.send(payload)

    def send_chat(self, room: str, message: str):
        self.send(message, room=room)

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

            # 1. Challstr from Server
            if msg_type == "challstr":
                challstr = "|".join(parts[2:])
                self.handle_login(challstr, self.username)

            # 2. Name Taken -> Retry with unique suffix
            elif msg_type == "nametaken":
                taken_name = parts[2] if len(parts) > 2 else self.username
                new_name = f"{self.base_username}_{random.randint(100, 999)}"
                print(f"[*] Nick '{taken_name}' was in use. Retrying as '{new_name}'...")
                self.username = new_name
                if self.last_challstr:
                    self.handle_login(self.last_challstr, self.username)

            # 3. Login Confirmed
            elif msg_type == "updateuser":
                logged_name = parts[2].strip()
                if logged_name and not logged_name.startswith("Guest") and not self._logged_in:
                    self._logged_in = True
                    self.username = logged_name
                    print("\n" + "=" * 65)
                    print(f"✅ BOT ONLINE: '{logged_name}'")

                    # If configured to challenge user directly, send challenge now!
                    if self.challenge_user and not self._challenge_sent:
                        self._challenge_sent = True
                        print(f"⚡ SENDING CHALLENGE TO: '{self.challenge_user}' ({self.format_id})...")
                        self.send(f"/challenge {self.challenge_user}, {self.format_id}")
                        print(f"🎮 Pop-up sent! Check your Showdown screen and click 'Accept'!")
                    else:
                        print(f"🎮 Ready for challenges on Showdown!")
                    print("=" * 65 + "\n")

            # 4. Incoming PM
            elif msg_type == "pm":
                sender = parts[2].strip()
                msg_body = "|".join(parts[4:]).strip() if len(parts) > 4 else ""
                if "/challenge" in msg_body:
                    print(f"\n[⚡ PM CHALLENGE] from '{sender}'! Accepting...")
                    self.send(f"/accept {sender}")

            # 5. Incoming Challenge
            elif msg_type == "updatechallenges":
                try:
                    challs = json.loads(parts[2])
                    challenges_from = challs.get("challengesFrom", {})
                    for challenger, format_id in challenges_from.items():
                        print(f"\n[⚡ CHALLENGE RECEIVED] from '{challenger}' ({format_id})! Accepting...")
                        self.send(f"/accept {challenger}")
                except Exception:
                    pass

            # 6. Updatesearch -> Battle Room Created
            elif msg_type == "updatesearch":
                try:
                    search_data = json.loads(parts[2])
                    games = search_data.get("games")
                    if games:
                        for battle_room in games:
                            if battle_room not in self.active_battles:
                                self.active_battles.append(battle_room)
                                print(f"[+] Battle Room Connected: {battle_room}")
                                self.send("/timer on", room=battle_room)
                except Exception:
                    pass

            # 7. Battle Request
            elif msg_type == "request":
                req_str = parts[2].strip()
                if req_str:
                    try:
                        req = json.loads(req_str)
                        if req.get("teamPreview"):
                            self.send("/team 123456", room=room)
                        elif not req.get("wait"):
                            action_cmd = self.agent.choose_action(req, self.rooms.get(room, []))
                            print(f"[{room}] Action -> /choose {action_cmd}")
                            self.send(f"/choose {action_cmd}", room=room)
                    except Exception as e:
                        print(f"[-] Error processing request: {e}")
                        self.send("/choose default", room=room)

            # 8. Error
            elif msg_type == "error":
                err_text = parts[2] if len(parts) > 2 else ""
                if "Invalid choice" in err_text:
                    self.send("/choose default", room=room)

            # 9. Start
            elif msg_type == "start":
                print(f"[{room}] ⚔️ BATTLE STARTED!")
                self.send_chat(room, "Hello! I am Amnesia-AI (Behavioral Cloning on Showdown Master replays). Good luck!")

            # 10. Win
            elif msg_type == "win":
                winner = parts[2].strip()
                clean_bot = re.sub(r"[^a-zA-Z0-9]", "", self.username).lower()
                clean_win = re.sub(r"[^a-zA-Z0-9]", "", winner).lower()
                print(f"\n[{room}] 🏆 BATTLE OVER! Winner: {winner}")
                if clean_win == clean_bot:
                    self.send_chat(room, "GG! Well played!")
                else:
                    self.send_chat(room, "GG! Great match!")
                self.send(f"/leave {room}")

    def on_error(self, ws, error):
        print(f"[-] WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[*] WebSocket connection closed.")

    def on_open(self, ws):
        print(f"[+] Connected to Showdown WebSocket: {self.server_url}")

    def start(self):
        self.ws = websocket.WebSocketApp(
            self.server_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever()
