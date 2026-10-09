"""
Play with Amnesia-AI on Pokemon Showdown
========================================
CLI runner to connect the Amnesia-AI Behavioral Cloning agent to Pokemon Showdown.

Usage:
    # Option 1 (Recommended): Send direct challenge to your username!
    python play.py --challenge SeuNickNoShowdown

    # Option 2: Run with registered bot account
    python play.py --username MyBotAccount --password mySecretPassword
"""

import argparse
import random
from client import ShowdownBotClient


def main():
    parser = argparse.ArgumentParser(description="Run Amnesia-AI Bot on Pokemon Showdown")
    parser.add_argument("--challenge", type=str, default=None, help="Target username to challenge automatically (Recommended!)")
    parser.add_argument("--username", type=str, default=f"AmnesiaAI_{random.randint(100, 999)}", help="Showdown username")
    parser.add_argument("--password", type=str, default=None, help="Password (optional for registered accounts)")
    parser.add_argument("--format", type=str, default="gen8ou", help="Battle format (default: gen8ou, e.g. gen8ou, gen8randombattle, gen9randombattle)")
    parser.add_argument("--checkpoint", type=str, default="projects/reinforcement-learning/weights/gen8ou_ppo.pt", help="Path to trained model weights")
    parser.add_argument("--avatar", type=str, default="red", help="Showdown avatar name")

    args = parser.parse_args()

    print("=" * 65)
    print("🤖 STARTING AMNESIA-AI SHOWDOWN BOT")
    print("=" * 65)
    print(f"Bot Username:    {args.username}")
    if args.challenge:
        print(f"Target Opponent: {args.challenge} (Will send challenge on connect!)")
    print(f"Format:          {args.format}")
    print(f"Model:           {args.checkpoint}")
    print("=" * 65)
    print("\nConnecting to Pokémon Showdown...")

    bot = ShowdownBotClient(
        username=args.username,
        password=args.password,
        challenge_user=args.challenge,
        format_id=args.format,
        checkpoint_path=args.checkpoint,
        avatar=args.avatar
    )
    bot.start()


if __name__ == "__main__":
    main()
