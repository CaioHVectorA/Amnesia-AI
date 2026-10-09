"""
Curated Replay Harvester for Gen 8 OU
=====================================
Downloads authentic high-quality Showdown replays for training Behavioral Cloning.
"""

import concurrent.futures
import json
import os
import time
import urllib.request

REPLAY_DIR = "data/replays/gen8ou_curated"
os.makedirs(REPLAY_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "AmnesiaAI-ResearchBot/1.0"
}

def fetch_page(page: int) -> list:
    url = f"https://replay.pokemonshowdown.com/search.json?format=gen8ou&page={page}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [item.get("id") for item in data if item.get("id")]
    except Exception:
        return []

def download_single_replay(rid: str) -> bool:
    out_file = os.path.join(REPLAY_DIR, f"{rid}.json")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 100:
        return True

    url = f"https://replay.pokemonshowdown.com/{rid}.json"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = json.loads(resp.read().decode("utf-8"))
            log = content.get("log", "")
            if len(log.split("\n")) >= 20:
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(content, f)
                return True
    except Exception:
        pass
    return False

def main(target_count: int = 250):
    t0 = time.time()
    print("[*] Discovering Gen 8 OU replays across 10 pages...")
    all_rids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        pages = list(range(1, 11))
        results = executor.map(fetch_page, pages)
        for r in results:
            all_rids.extend(r)

    unique_rids = list(dict.fromkeys(all_rids))[:target_count]
    print(f"[+] Discovered {len(unique_rids)} unique replays. Downloading in parallel (12 workers)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        download_results = list(executor.map(download_single_replay, unique_rids))

    successful = sum(1 for r in download_results if r)
    print(f"[+] Successfully harvested {successful} curated Gen 8 OU replays in {time.time()-t0:.2f}s!")

if __name__ == "__main__":
    main(target_count=200)
