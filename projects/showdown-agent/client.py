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

        # Load standard competitive team for constructed formats (e.g. gen8ou)
        self.team_packed = None
        teams_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/gen8ou_teams.json"))
        if os.path.exists(teams_file):
            try:
                with open(teams_file, "r", encoding="utf-8") as f:
                    t_data = json.load(f)
                    if t_data and "team" in t_data[0]:
                        self.team_packed = "]".join(t_data[0]["team"])
                        print(f"[+] Loaded competitive team for Gen 8 OU: {t_data[0].get('name', 'Gen 8 OU Team')}")
            except Exception as e:
                print(f"[-] Team load note: {e}")

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
                if self.team_packed:
                    self.send(f"/utm {self.team_packed}")
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
                        if self.team_packed:
                            self.send(f"/utm {self.team_packed}")
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
                if sender == "~":
                    print(f"\n📢 [AVISO DO SERVIDOR SHOWDOWN]: {msg_body}", flush=True)
                elif "/challenge" in msg_body:
                    print(f"\n[⚡ PM CHALLENGE] de '{sender}' recebido via PM! Aceitando...", flush=True)
                    if self.team_packed:
                        self.send(f"/utm {self.team_packed}")
                    self.send(f"/accept {sender}")
                else:
                    print(f"[💬 Mensagem de {sender}]: {msg_body}", flush=True)

            # 5. Incoming Challenge
            elif msg_type == "updatechallenges":
                try:
                    challs = json.loads(parts[2])
                    challenges_from = challs.get("challengesFrom", {})
                    for challenger, format_id in challenges_from.items():
                        print(f"\n[⚡ DESAFIO OFICIAL DETECTADO] de '{challenger}' ({format_id})! Aceitando...", flush=True)
                        if "random" not in format_id.lower() and self.team_packed:
                            self.send(f"/utm {self.team_packed}")
                        elif "random" in format_id.lower():
                            self.send("/utm null")
                        self.send(f"/accept {challenger}")
                except Exception as e:
                    print(f"[-] Error in updatechallenges: {e}", flush=True)

            # 6. Room Initialized
            elif msg_type == "init":
                if room.startswith("battle-"):
                    if room not in self.active_battles:
                        self.active_battles.append(room)
                    print("\n" + "🔥" * 40, flush=True)
                    print(f"⚔️ [SALA DE BATALHA CRIADA!]", flush=True)
                    print(f"🔗 ACESSE NO NAVEGADOR: https://play.pokemonshowdown.com/{room}", flush=True)
                    print("🔥" * 40 + "\n", flush=True)
                    self.send("/timer on", room=room)

            # 7. Battle Request
            elif msg_type == "request":
                req_str = parts[2].strip()
                if req_str:
                    try:
                        req = json.loads(req_str)
                        if req.get("teamPreview"):
                            print(f"[{room}] 📋 Team Preview iniciado! Bot enviou ordem de time. Clique em 'Start Battle' no seu navegador!", flush=True)
                            self.send("/team 123456", room=room)
                        elif not req.get("wait"):
                            action_cmd = self.agent.choose_action(req, self.rooms.get(room, []))
                            print(f"[{room}] 🎯 Amnesia-AI jogou: /choose {action_cmd}", flush=True)
                            self.send(f"/choose {action_cmd}", room=room)
                        else:
                            print(f"[{room}] ⏳ Aguardando você fazer sua jogada no Showdown...", flush=True)
                    except Exception as e:
                        print(f"[-] Error processing request: {e}", flush=True)
                        self.send("/choose default", room=room)

            # 8. Popup from Server
            elif msg_type == "popup":
                popup_msg = "|".join(parts[2:])
                print(f"\n⚠️ [SHOWDOWN POPUP/AVISO]: {popup_msg}", flush=True)

            # 9. Error
            elif msg_type == "error":
                err_text = "|".join(parts[2:])
                print(f"\n❌ [ERRO SHOWDOWN]: {err_text}", flush=True)
                if "Invalid choice" in err_text:
                    self.send("/choose default", room=room)

            # 10. Start
            elif msg_type == "start":
                print(f"[{room}] ⚔️ A BATALHA COMEÇOU!", flush=True)
                self.send_chat(room, "Boa sorte! Sou o Amnesia-AI (Rede Neural PPO treinada em Gen 8 OU).")

            # 11. Win
            elif msg_type == "win":
                winner = parts[2].strip()
                clean_bot = re.sub(r"[^a-zA-Z0-9]", "", self.username).lower()
                clean_win = re.sub(r"[^a-zA-Z0-9]", "", winner).lower()
                print(f"\n[{room}] 🏆 FIM DE PARTIDA! Vencedor: {winner}", flush=True)
                if clean_win == clean_bot:
                    self.send_chat(room, "GG! Boa partida!")
                else:
                    self.send_chat(room, "GG! Você jogou muito bem!")
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
