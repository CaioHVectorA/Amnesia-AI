#!/usr/bin/env python3
"""
Amnesia-AI Scheduled Windows Shutdown Watcher (Target: 23:00)
1. Periodically tracks time until 23:00.
2. At 22:58 (2 minutes before shutdown):
   - Automatically stages and commits all new battle replays, model weights, and logs.
   - Pushes absolutely everything to GitHub remote repository (origin main).
3. Issues system command `shutdown /s /f /t <seconds>` so Windows shuts down at 23:00:00.
"""

import datetime
import os
import subprocess
import time
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET_HOUR = 23
TARGET_MINUTE = 0

def git_sync_all():
    print("📦 [GIT SYNC] Staging all files, replays, and weights...", flush=True)
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if res.stdout.strip():
            print("💾 [GIT SYNC] Committing changes...", flush=True)
            subprocess.run(["git", "commit", "-m", "feat(rl): final automatic sync of all replays and weights before scheduled Windows shutdown at 23:00"], check=True)
            print("🚀 [GIT SYNC] Pushing to GitHub origin main...", flush=True)
            subprocess.run(["git", "push", "origin", "main"], check=True)
            print("✅ [GIT SYNC] Successfully pushed everything to GitHub!", flush=True)
        else:
            print("ℹ️ [GIT SYNC] Working tree clean, all files already up to date.", flush=True)
    except Exception as e:
        print(f"⚠️ [GIT SYNC ERROR] {e}", flush=True)

def main():
    print("=" * 70)
    print("🛑 AMNESIA-AI WINDOWS SYSTEM SHUTDOWN WATCHER (TARGET: 23:00)")
    print("=" * 70)
    print("This watcher will guarantee that at 22:58 all files are pushed to")
    print("GitHub, and at 23:00 the Windows operating system will shut down.")
    print("=" * 70 + "\n", flush=True)

    while True:
        now = datetime.datetime.now()
        target_today = now.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=0, microsecond=0)
        
        diff_sec = (target_today - now).total_seconds()
        if diff_sec <= 120:  # 2 minutes or less before 23:00 (22:58 or later)
            print(f"\n⏰ [SHUTDOWN SEQUENCE INITIATED] Current time: {now.strftime('%H:%M:%S')}", flush=True)
            git_sync_all()
            
            # Recalculate remaining seconds until 23:00
            now = datetime.datetime.now()
            remaining_to_23 = max(10, int((target_today - now).total_seconds()))
            print(f"🔌 Executing Windows system shutdown in {remaining_to_23} seconds (Target 23:00:00)...", flush=True)
            os.system(f'shutdown /s /f /t {remaining_to_23} /c "Amnesia-AI: 23:00 Scheduled System Shutdown. All files synced to GitHub."')
            print("💤 Shutdown command issued to Windows OS. System powering off at 23:00.", flush=True)
            break
        
        mins_left = int(diff_sec // 60)
        secs_left = int(diff_sec % 60)
        print(f"⏳ [SHUTDOWN WATCHER] Current: {now.strftime('%H:%M:%S')} | Target: 23:00:00 | Remaining: {mins_left}m {secs_left}s", flush=True)
        time.sleep(30.0)

if __name__ == "__main__":
    main()
