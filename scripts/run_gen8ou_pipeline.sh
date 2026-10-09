#!/bin/bash
# ==============================================================================
# Amnesia-AI: Gen 8 OU League & RL Pipeline Runner
# ==============================================================================

export PATH="$HOME/.bun/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "=============================================================================="
echo "⚡ AMNESIA-AI: GEN 8 OU TRAINING & LEAGUE PIPELINE"
echo "=============================================================================="

VENV_PYTHON="/home/usuario/develop/reinforcement-cases/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

MODE="${1:-benchmark}"

if [ "$MODE" == "benchmark" ]; then
    GAMES="${2:-50}"
    echo "[+] Running Gen 8 OU Headless League Benchmark ($GAMES games per pair)..."
    cd projects/simulator && bun run src/gen8ou_league.ts "$GAMES"
elif [ "$MODE" == "train" ]; then
    BATCH="${2:-2500}"
    UPDATES="${3:-50}"
    echo "[+] Launching Gen 8 OU League PPO Trainer (Batch: $BATCH turns, Updates: $UPDATES)..."
    $VENV_PYTHON projects/reinforcement-learning/gen8ou_league_trainer.py --batch-size "$BATCH" --updates "$UPDATES"
else
    echo "Usage: ./scripts/run_gen8ou_pipeline.sh [benchmark <num_games> | train <batch_size> <updates>]"
fi
