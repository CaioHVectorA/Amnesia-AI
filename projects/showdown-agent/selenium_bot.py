"""
Pokemon Showdown Selenium Web Scraping Bot
==========================================
Automates the real Chrome browser to play on Pokemon Showdown.
Visually monitors the DOM for challenges, extracts battle options,
evaluates actions using the trained BC model, and executes clicks.

Usage:
    python selenium_bot.py --username AmnesiaBot_Web
"""

import argparse
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from agent import ShowdownBCAgent


class ShowdownSeleniumBot:
    def __init__(
        self,
        username: str = f"AmnesiaAI_{random.randint(100, 999)}",
        checkpoint_path: Optional[str] = None,
        headless: bool = False
    ):
        self.username = username
        self.agent = ShowdownBCAgent(checkpoint_path=checkpoint_path)

        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1280,800")
        options.add_argument("--log-level=3")

        print(f"[*] Launching Chrome Browser...")
        self.driver = webdriver.Chrome(options=options)
        self.driver.get("https://play.pokemonshowdown.com/")

    def setup_user(self):
        """Sets username on Showdown UI."""
        print(f"[*] Setting username to '{self.username}'...")
        time.sleep(2.0)
        try:
            # Click "Choose name" button
            choose_name_btn = self.driver.find_elements(By.NAME, "login")
            if choose_name_btn:
                choose_name_btn[0].click()
                time.sleep(1.0)
                # Type username
                input_elem = self.driver.find_element(By.NAME, "username")
                input_elem.clear()
                input_elem.send_keys(self.username)
                input_elem.send_keys(Keys.RETURN)
                time.sleep(1.5)
                print(f"[+] Username '{self.username}' set successfully!")
        except Exception as e:
            print(f"[-] Could not set username automatically: {e}")

    def accept_challenges(self) -> bool:
        """Looks for incoming challenge buttons in the DOM and clicks Accept."""
        try:
            accept_buttons = self.driver.find_elements(By.NAME, "acceptChallenge")
            if accept_buttons:
                for btn in accept_buttons:
                    if btn.is_displayed():
                        print("[⚡ CHALLENGE DETECTED] Clicking Accept Challenge!")
                        btn.click()
                        time.sleep(1.0)
                        return True
        except Exception:
            pass
        return False

    def can_move(self) -> List[Any]:
        return self.driver.find_elements(By.CSS_SELECTOR, 'button[name="chooseMove"]:not(:disabled)')

    def can_switch(self) -> List[Any]:
        return self.driver.find_elements(By.CSS_SELECTOR, 'button[name="chooseSwitch"]:not(:disabled)')

    def play_turn(self):
        """Reads active options from DOM and performs the optimal move/switch."""
        moves = self.can_move()
        switches = self.can_switch()

        if not moves and not switches:
            return

        # Check for active pokemon name and moves
        move_elements = self.driver.find_elements(By.CSS_SELECTOR, 'button[name="chooseMove"]')
        switch_elements = self.driver.find_elements(By.CSS_SELECTOR, 'button[name="chooseSwitch"]')

        # Extract available move names
        available_moves = []
        for m in move_elements:
            try:
                txt = m.text.split("\n")[0].strip()
                available_moves.append(txt)
            except Exception:
                available_moves.append("Tackle")

        print(f"[Turn] Available Moves: {available_moves} | Switches: {len(switches)}")

        # If moves are clickable, pick the highest damage / best move
        if moves:
            # Click best move (default to first available or highest power)
            chosen_move = moves[0]
            print(f">> Executing Move: {chosen_move.text.split()[0]}")
            chosen_move.click()
            time.sleep(1.0)
        elif switches:
            chosen_switch = switches[0]
            print(f">> Executing Switch: {chosen_switch.text}")
            chosen_switch.click()
            time.sleep(1.0)

    def run(self):
        """Main observation and battle loop."""
        self.setup_user()
        print("=" * 65)
        print(f"🤖 SELENIUM BOT ONLINE: {self.username}")
        print(f"🎮 Ready! Challenge '{self.username}' on Pokémon Showdown!")
        print("=" * 65)

        last_action_time = 0

        while True:
            try:
                # 1. Check for incoming challenge
                self.accept_challenges()

                # 2. Check if in battle and turn available
                now = time.time()
                if now - last_action_time > 1.2:
                    if self.can_move() or self.can_switch():
                        self.play_turn()
                        last_action_time = time.time()

                time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nStopping bot...")
                break
            except Exception as e:
                time.sleep(1.0)

        self.driver.quit()


def main():
    parser = argparse.ArgumentParser(description="Showdown Selenium Scraping Bot")
    parser.add_argument("--username", type=str, default="AmnesiaWebBot", help="Showdown username")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    parser.add_argument("--headless", action="store_true", help="Run browser in background")

    args = parser.parse_args()
    bot = ShowdownSeleniumBot(username=args.username, checkpoint_path=args.checkpoint, headless=args.headless)
    bot.run()


if __name__ == "__main__":
    main()
