#!/bin/bash
# ==============================================================================
# Amnesia-AI: Graceful Shutdown Trigger
# ==============================================================================

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../data" && pwd)"
FLAG_FILE="$DATA_DIR/stop_training.flag"

echo "[!] Sending graceful shutdown signal to Amnesia-AI Trainer..."
touch "$FLAG_FILE"
pkill -f "continuous_gen8ou_trainer.py" 2>/dev/null

echo "✅ Stop signal sent. The trainer will finish the current batch, save weights, and export the dashboard."
