"""
Showdown Replay Scraper & Clustering Pipeline (High-Throughput)
==============================================================
Industrial-grade, multi-threaded scraper for Pokémon Showdown replays.
Features:
- Time-bounded continuous scraping (--max-duration-minutes)
- Gzip-compressed storage (.json.gz) for 85%+ disk savings
- Chronological backward pagination (using Showdown's `before` timestamp)
- Real-time rating bracket clustering and Behavioral Cloning sample weighting
- SQLite metadata index for deduplication and rapid dataset querying

Usage:
    python scraper.py --format gen9randombattle --max-duration-minutes 40 --concurrency 8 --compress
"""

import argparse
import concurrent.futures
import gzip
import json
import os
import re
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Constants
BASE_SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
BASE_REPLAY_URL = "https://replay.pokemonshowdown.com/{id}.json"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AmnesiaAI-BC-Research/1.0"
DB_FILENAME = "replays_metadata.sqlite"


@dataclass
class ReplayMetadata:
    id: str
    format: str
    uploadtime: int
    p1_name: str
    p2_name: str
    p1_rating: Optional[int]
    p2_rating: Optional[int]
    winner_name: Optional[str]
    winner_side: Optional[str]  # 'p1', 'p2', or 'tie'
    winner_rating: Optional[int]
    rating_bracket: str
    turn_count: int
    is_forfeit: bool
    weight: float
    file_path: str


def get_rating_bracket(rating: Optional[int]) -> str:
    """Categorize rating into high-utility clusters."""
    if rating is None or rating == 0:
        return "unrated"
    if rating >= 1800:
        return "elite_1800plus"
    if rating >= 1650:
        return "high_1650_1799"
    if rating >= 1500:
        return "mid_1500_1649"
    if rating >= 1300:
        return "low_1300_1499"
    return "sub1300"


def calculate_sample_weight(winner_rating: Optional[int], turn_count: int, is_forfeit: bool) -> float:
    """
    Calculate sample importance weight for Behavioral Cloning training.
    High-Elo matches get higher weight; trivial instant forfeits (<4 turns) get 0.
    """
    if turn_count < 4:
        return 0.0

    rating = winner_rating if (winner_rating and winner_rating > 0) else 1000
    elo_weight = max(0.1, (rating - 900) / 900.0)

    if is_forfeit and rating < 1500:
        elo_weight *= 0.8

    return round(elo_weight, 4)


class ReplayDB:
    """Thread-safe SQLite index for metadata and deduplication."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=60.0)

    def _init_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS replays (
                    id TEXT PRIMARY KEY,
                    format TEXT,
                    uploadtime INTEGER,
                    p1_name TEXT,
                    p2_name TEXT,
                    p1_rating INTEGER,
                    p2_rating INTEGER,
                    winner_name TEXT,
                    winner_side TEXT,
                    winner_rating INTEGER,
                    rating_bracket TEXT,
                    turn_count INTEGER,
                    is_forfeit INTEGER,
                    weight REAL,
                    file_path TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_format ON replays(format)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bracket ON replays(rating_bracket)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_weight ON replays(weight)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_uploadtime ON replays(uploadtime)")

    def is_downloaded(self, replay_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM replays WHERE id = ?", (replay_id,))
            return cursor.fetchone() is not None

    def insert_batch(self, metas: List[ReplayMetadata]):
        if not metas:
            return
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO replays (
                    id, format, uploadtime, p1_name, p2_name, p1_rating, p2_rating,
                    winner_name, winner_side, winner_rating, rating_bracket,
                    turn_count, is_forfeit, weight, file_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    m.id, m.format, m.uploadtime, m.p1_name, m.p2_name,
                    m.p1_rating, m.p2_rating, m.winner_name, m.winner_side,
                    m.winner_rating, m.rating_bracket, m.turn_count,
                    1 if m.is_forfeit else 0, m.weight, m.file_path
                )
                for m in metas
            ])

    def get_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(weight), AVG(turn_count), AVG(winner_rating) FROM replays")
            total, avg_weight, avg_turns, avg_elo = cursor.fetchone()
            cursor.execute("SELECT rating_bracket, COUNT(*) FROM replays GROUP BY rating_bracket ORDER BY COUNT(*) DESC")
            brackets = dict(cursor.fetchall())
            return {
                "total_replays": total or 0,
                "avg_weight": round(avg_weight or 0, 3),
                "avg_turns": round(avg_turns or 0, 1),
                "avg_elo": round(avg_elo or 0, 1),
                "by_bracket": brackets
            }


class ShowdownReplayScraper:
    def __init__(
        self,
        output_dir: str = "data/replays",
        format_id: str = "gen9randombattle",
        min_rating: int = 0,
        compress: bool = True,
        user_agent: str = DEFAULT_USER_AGENT,
        request_delay: float = 0.08
    ):
        self.output_dir = output_dir
        self.format_id = format_id.lower()
        self.min_rating = min_rating
        self.compress = compress
        self.user_agent = user_agent
        self.request_delay = request_delay

        self.db_path = os.path.join(output_dir, DB_FILENAME)
        self.db = ReplayDB(self.db_path)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def search_replays(self, before_timestamp: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query replay search API."""
        params: Dict[str, Any] = {"format": self.format_id}
        if self.min_rating > 0:
            params["rating"] = self.min_rating
        if before_timestamp:
            params["before"] = before_timestamp

        for attempt in range(3):
            try:
                time.sleep(self.request_delay)
                res = self.session.get(BASE_SEARCH_URL, params=params, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list):
                        return data
                elif res.status_code == 429:
                    sleep_time = 4.0 * (attempt + 1)
                    print(f"[Rate Limit] HTTP 429 on search. Backing off for {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    time.sleep(1.0)
            except Exception as e:
                time.sleep(1.0)
        return []

    def fetch_and_parse_replay(self, replay_summary: Dict[str, Any]) -> Optional[ReplayMetadata]:
        """Download full replay JSON, extract features, and save compressed/clustered on disk."""
        replay_id = replay_summary["id"]

        if self.db.is_downloaded(replay_id):
            return None

        url = BASE_REPLAY_URL.format(id=replay_id)
        data = None
        for attempt in range(3):
            try:
                time.sleep(self.request_delay)
                res = self.session.get(url, timeout=12)
                if res.status_code == 200:
                    data = res.json()
                    break
                elif res.status_code == 429:
                    time.sleep(3.0 * (attempt + 1))
                elif res.status_code == 404:
                    return None
            except Exception:
                time.sleep(0.5)

        if not data or not data.get("log"):
            return None

        log = data["log"]

        # Parse player names
        p1_match = re.search(r"\|player\|p1\|([^|\n\r]+)", log)
        p2_match = re.search(r"\|player\|p2\|([^|\n\r]+)", log)
        players = data.get("players", ["", ""])
        p1_name = p1_match.group(1).strip() if p1_match else (players[0] if len(players) > 0 else "")
        p2_name = p2_match.group(1).strip() if p2_match else (players[1] if len(players) > 1 else "")

        # Parse ratings
        p1_rating = None
        p2_rating = None
        rating_matches = re.findall(r"([^\n\r]+)'s rating:\s*(\d+)", log)
        for name, r in rating_matches:
            if name.strip().lower() == p1_name.lower():
                p1_rating = int(r)
            elif name.strip().lower() == p2_name.lower():
                p2_rating = int(r)

        if not p1_rating and replay_summary.get("rating"):
            p1_rating = replay_summary["rating"]
        if not p2_rating and replay_summary.get("rating"):
            p2_rating = replay_summary["rating"]

        # Parse Winner
        win_match = re.search(r"\|win\|([^\n\r]+)", log)
        winner_name = win_match.group(1).strip() if win_match else None

        winner_side = None
        winner_rating = None
        if winner_name:
            if winner_name.lower() == p1_name.lower():
                winner_side = "p1"
                winner_rating = p1_rating
            elif winner_name.lower() == p2_name.lower():
                winner_side = "p2"
                winner_rating = p2_rating

        # Parse turn count and forfeit
        turn_matches = re.findall(r"\|turn\|(\d+)", log)
        turn_count = int(turn_matches[-1]) if turn_matches else 0
        is_forfeit = bool(re.search(r"forfeited\.", log, re.IGNORECASE))

        if turn_count < 2:
            return None

        # Compute cluster bracket & weight
        effective_rating = winner_rating or max(filter(None, [p1_rating, p2_rating, replay_summary.get("rating")]), default=0)
        bracket = get_rating_bracket(effective_rating)
        weight = calculate_sample_weight(effective_rating, turn_count, is_forfeit)

        # Save partitioned on disk
        cluster_dir = os.path.join(self.output_dir, self.format_id, bracket)
        os.makedirs(cluster_dir, exist_ok=True)

        payload = {
            "id": replay_id,
            "format": self.format_id,
            "uploadtime": data.get("uploadtime", 0),
            "p1": p1_name,
            "p2": p2_name,
            "p1_rating": p1_rating,
            "p2_rating": p2_rating,
            "winner": winner_name,
            "winner_side": winner_side,
            "winner_rating": winner_rating,
            "rating_bracket": bracket,
            "turn_count": turn_count,
            "is_forfeit": is_forfeit,
            "weight": weight,
            "inputlog": data.get("inputlog"),
            "log": log
        }

        if self.compress:
            file_path = os.path.join(cluster_dir, f"{replay_id}.json.gz")
            with gzip.open(file_path, "wt", encoding="utf-8") as f:
                json.dump(payload, f)
        else:
            file_path = os.path.join(cluster_dir, f"{replay_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        return ReplayMetadata(
            id=replay_id,
            format=self.format_id,
            uploadtime=data.get("uploadtime", 0),
            p1_name=p1_name,
            p2_name=p2_name,
            p1_rating=p1_rating,
            p2_rating=p2_rating,
            winner_name=winner_name,
            winner_side=winner_side,
            winner_rating=winner_rating,
            rating_bracket=bracket,
            turn_count=turn_count,
            is_forfeit=is_forfeit,
            weight=weight,
            file_path=file_path
        )

    def run(
        self,
        max_duration_seconds: Optional[int] = None,
        max_replays: Optional[int] = None,
        max_workers: int = 8
    ):
        start_time = time.time()
        end_time = (start_time + max_duration_seconds) if max_duration_seconds else None

        print("=" * 70)
        print("🚀 AMNESIA-AI MASS REPLAY SCRAPER")
        print("=" * 70)
        print(f"Format:             {self.format_id}")
        print(f"Storage Format:     {'.json.gz (compressed)' if self.compress else '.json (raw)'}")
        print(f"Workers:            {max_workers} threads")
        if max_duration_seconds:
            print(f"Time Budget:        {max_duration_seconds // 60} minutes ({max_duration_seconds}s)")
        if max_replays:
            print(f"Replay Target:      {max_replays:,} replays")
        print(f"Output Directory:   {self.output_dir}\n")

        total_downloaded = 0
        batch_count = 0
        oldest_timestamp = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while not self._stop_requested:
                now = time.time()
                if end_time and now >= end_time:
                    print(f"\n⏰ Time limit of {max_duration_seconds // 60} minutes reached! Wrapping up...")
                    break
                if max_replays and total_downloaded >= max_replays:
                    print(f"\n🎯 Target of {max_replays:,} replays reached!")
                    break

                batch_count += 1
                summaries = self.search_replays(before_timestamp=oldest_timestamp)
                if not summaries:
                    print("No more replays available on this timeline or temporary backoff. Waiting 5s...")
                    time.sleep(5.0)
                    continue

                candidates = [s for s in summaries if not self.db.is_downloaded(s["id"])]
                oldest_timestamp = min(s.get("uploadtime", int(time.time())) for s in summaries) - 1

                if not candidates:
                    continue

                futures = [executor.submit(self.fetch_and_parse_replay, s) for s in candidates]
                batch_metas: List[ReplayMetadata] = []

                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res:
                        batch_metas.append(res)
                        total_downloaded += 1

                # Save batch to SQLite
                if batch_metas:
                    self.db.insert_batch(batch_metas)

                # Periodic progress reporting
                elapsed = time.time() - start_time
                rate = (total_downloaded / elapsed) if elapsed > 0 else 0
                remaining = (end_time - time.time()) if end_time else 0
                rem_str = f"{int(remaining // 60)}m {int(remaining % 60)}s" if end_time else "N/A"

                print(
                    f"[{batch_count:04d}] Downloaded: {total_downloaded:6,d} | "
                    f"Speed: {rate:4.1f} replays/s | "
                    f"Elapsed: {int(elapsed//60):02d}m{int(elapsed%60):02d}s | "
                    f"Remaining Time: {rem_str}"
                )

        total_elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("✅ SCRAPING SESSION COMPLETED")
        print("=" * 70)
        print(f"Total Time:         {int(total_elapsed // 60)}m {int(total_elapsed % 60)}s")
        print(f"Downloaded This Run: {total_downloaded:,} replays")
        print("\nCumulative Database Stats:")
        stats = self.db.get_stats()
        print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="High-throughput Showdown Replay Scraper")
    parser.add_argument("--format", type=str, default="gen9randombattle", help="Format ID (e.g. gen9randombattle, gen9ou)")
    parser.add_argument("--min-rating", type=int, default=0, help="Minimum rating filter")
    parser.add_argument("--max-replays", type=int, default=None, help="Maximum number of replays to download")
    parser.add_argument("--max-duration-minutes", type=int, default=40, help="Time budget in minutes (default: 40)")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent workers (default: 8)")
    parser.add_argument("--output-dir", type=str, default="data/replays", help="Output directory")
    parser.add_argument("--compress", action="store_true", default=True, help="Save as .json.gz for high disk efficiency")
    parser.add_argument("--no-compress", dest="compress", action="store_false", help="Save as raw .json")

    args = parser.parse_args()

    duration_seconds = (args.max_duration_minutes * 60) if args.max_duration_minutes else None

    scraper = ShowdownReplayScraper(
        output_dir=args.output_dir,
        format_id=args.format,
        min_rating=args.min_rating,
        compress=args.compress
    )

    # Handle graceful exit on Ctrl+C
    def sig_handler(sig, frame):
        print("\nGracefully stopping scraper on interrupt signal...")
        scraper.request_stop()

    signal.signal(signal.SIGINT, sig_handler)

    scraper.run(
        max_duration_seconds=duration_seconds,
        max_replays=args.max_replays,
        max_workers=args.concurrency
    )


if __name__ == "__main__":
    main()
