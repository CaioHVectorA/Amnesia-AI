#!/bin/bash
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../data" && pwd)"
FLAG_FILE="$DATA_DIR/stop_real_training.flag"

echo "[!] Sending graceful shutdown signal to Real @pkmn/sim Trainer..."
touch "$FLAG_FILE"
pkill -f "real_pkmn_sim_trainer.py" 2>/dev/null

echo "✅ Stop signal sent. The real sim trainer will finish the current batch and save weights."
