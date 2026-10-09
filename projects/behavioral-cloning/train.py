"""
Behavioral Cloning Training Pipeline (GPU-Accelerated / 100 Epochs / 100% Replays)
==================================================================================
Trained on 100% of available Showdown replays (~36,668 matches / ~750,000+ turns).
Evaluates empirical Win Rate Benchmark against baselines every 5 epochs.

Optimizations:
- CUDA In-VRAM Tensor Residency (zero PCIe transfer overhead per batch)
- Automatic Mixed Precision (AMP FP16) via NVIDIA Tensor Cores
- Ampere TF32 & cuDNN Benchmark acceleration
- Multithreaded parsing (20 threads) + full dataset caching
- Benchmark every 5 epochs (epochs 5, 10, 15, ..., 100)
- Automated Training Curves & Benchmark Progression Plots
"""

import argparse
import concurrent.futures
import json
import os
import random
import sqlite3
import sys
import time
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from model import ActionScoringPolicyNet
from parser import parse_replay_file, STATE_DIM, ACTION_DIM, NUM_ACTION_SLOTS


class FastGPUDataset:
    """Preloaded, fully vectorized in-VRAM dataset for maximum GPU throughput."""
    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, int, float, np.ndarray]], device: torch.device):
        self.device = device
        states_np = np.array([s[0] for s in samples], dtype=np.float32)
        actions_np = np.array([s[1] for s in samples], dtype=np.float32)
        targets_np = np.array([s[2] for s in samples], dtype=np.int64)
        weights_np = np.array([s[3] for s in samples], dtype=np.float32)
        masks_np = np.array([s[4] for s in samples], dtype=np.float32)

        self.states = torch.from_numpy(states_np).to(device)
        self.actions = torch.from_numpy(actions_np).to(device)
        self.targets = torch.from_numpy(targets_np).to(device)
        self.weights = torch.from_numpy(weights_np).to(device)
        self.masks = torch.from_numpy(masks_np).to(device)
        self.n = len(self.targets)

    def __len__(self):
        return self.n

    def get_batches(self, batch_size: int, shuffle: bool = True):
        if shuffle:
            indices = torch.randperm(self.n, device=self.device)
        else:
            indices = torch.arange(self.n, device=self.device)

        for start_idx in range(0, self.n, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield (
                self.states[batch_idx],
                self.actions[batch_idx],
                self.targets[batch_idx],
                self.weights[batch_idx],
                self.masks[batch_idx]
            )


def load_all_replays(replay_root: str = "data/replays", db_path: str = "data/replays/replays_metadata.sqlite") -> List[str]:
    """Discovers 100% of all replay files available on disk and in SQLite."""
    discovered: Set[str] = set()

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM replays WHERE turn_count >= 5")
        for r in cursor.fetchall():
            p = r[0]
            if os.path.exists(p):
                discovered.add(os.path.abspath(p))
        conn.close()

    for root, _, files in os.walk(replay_root):
        for f in files:
            if f.endswith(".json") or f.endswith(".json.gz"):
                discovered.add(os.path.abspath(os.path.join(root, f)))

    path_list = list(discovered)
    print(f"🔥 Discovered 100% of available replays: {len(path_list):,} total files on disk!")
    return path_list


def load_balanced_replays(
    db_path: str,
    elite_count: int = 4000,
    high_count: int = 1500,
    mid_count: int = 1500,
    common_count: int = 1500
) -> List[str]:
    """Queries SQLite for a balanced replay dataset across skill tiers."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    def get_paths(bracket: str, limit: int) -> List[str]:
        cursor.execute("SELECT file_path FROM replays WHERE rating_bracket = ? AND turn_count >= 5 ORDER BY RANDOM() LIMIT ?", (bracket, limit))
        return [r[0] for r in cursor.fetchall() if os.path.exists(r[0])]

    elite_paths = get_paths("elite_1800plus", elite_count)
    high_paths = get_paths("high_1650_1799", high_count)
    mid_paths = get_paths("mid_1500_1649", mid_count)
    common_paths = get_paths("low_1300_1499", common_count // 2) + get_paths("sub1300", common_count // 2)

    conn.close()

    total_paths = elite_paths + high_paths + mid_paths + common_paths
    print(f"Loaded replay paths: {len(elite_paths)} Elite, {len(high_paths)} High, {len(mid_paths)} Mid, {len(common_paths)} Common | Total: {len(total_paths)}")
    return total_paths


def extract_dataset(replay_paths: List[str], max_workers: int = 20) -> List[Tuple[np.ndarray, np.ndarray, int, float, np.ndarray]]:
    """Parses replay files in parallel into state-action tuples."""
    print(f"Parsing {len(replay_paths):,} replay files with {max_workers} threads...")
    all_samples = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(parse_replay_file, p) for p in replay_paths]
        for f in concurrent.futures.as_completed(futures):
            samples = f.result()
            if samples:
                all_samples.extend(samples)

    print(f"Extraction completed! Total training turns extracted: {len(all_samples):,}")
    return all_samples


def evaluate_benchmark(model: torch.nn.Module, games_per_opp: int = 20) -> Dict[str, Any]:
    """Runs headless simulated battles of the model against baseline bots to determine empirical Win Rate."""
    agent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../showdown-agent"))
    if agent_dir not in sys.path:
        sys.path.append(agent_dir)

    from agent import ShowdownBCAgent
    from benchmark import BaselinePlayer, HeadlessSimGame

    was_training = model.training
    model.eval()

    agent = ShowdownBCAgent(model=model)
    results = {}
    total_wins = 0
    total_games = 0

    for opp_name in ["max_damage", "random"]:
        opp = BaselinePlayer(mode=opp_name)
        wins = 0
        turns = []
        for _ in range(games_per_opp):
            sim = HeadlessSimGame(agent, opp)
            w, t, _ = sim.play()
            if w == "p1":
                wins += 1
            turns.append(t)

        wr = (wins / games_per_opp) * 100.0
        results[opp_name] = {
            "win_rate": wr,
            "wins": wins,
            "total": games_per_opp,
            "avg_turns": float(np.mean(turns)) if turns else 0.0
        }
        total_wins += wins
        total_games += games_per_opp

    if was_training:
        model.train()

    overall_wr = (total_wins / total_games) * 100.0
    results["overall_win_rate"] = overall_wr
    results["total_wins"] = total_wins
    results["total_games"] = total_games
    return results


def plot_benchmark_progression(benchmarks: List[dict], output_path: str):
    """Plots the empirical Win Rate progression of Amnesia-AI across benchmark evaluation checkpoints."""
    if not benchmarks:
        return

    epochs = [b["epoch"] for b in benchmarks]
    overall_wr = [b["overall_win_rate"] for b in benchmarks]
    md_wr = [b["max_damage_win_rate"] for b in benchmarks]
    rnd_wr = [b["random_win_rate"] for b in benchmarks]
    top1 = [b.get("val_top1", 0) for b in benchmarks]

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor('#0d1117')
    ax1.set_facecolor('#161b22')
    ax2.set_facecolor('#161b22')

    # Subplot 1: Win Rate progression
    ax1.plot(epochs, overall_wr, marker='o', linewidth=2.2, color='#e3b341', label="Win Rate Geral (Overall)", zorder=4)
    ax1.plot(epochs, md_wr, marker='s', linewidth=1.8, color='#f78166', linestyle='--', label="vs MaxDamage Player", zorder=3)
    ax1.plot(epochs, rnd_wr, marker='^', linewidth=1.8, color='#58a6ff', linestyle=':', label="vs Random Player", zorder=3)

    for ep, wr in zip(epochs, overall_wr):
        ax1.annotate(f"{wr:.0f}%", (ep, wr), textcoords="offset points", xytext=(0, 7), ha='center',
                     fontsize=9, fontweight='bold', color='#f0f6fc')

    ax1.axhline(50, color='#8b949e', linestyle='--', linewidth=1.0, alpha=0.6, label='50% Paridade')
    ax1.set_title("Evolução do Win Rate a Cada 5 Épocas (100 Épocas)", fontsize=12, fontweight='bold', color='#f0f6fc', pad=12)
    ax1.set_xlabel("Época de Treino", fontsize=11, color='#c9d1d9')
    ax1.set_ylabel("Win Rate (%) em Partidas", fontsize=11, color='#c9d1d9')
    ax1.set_xticks(epochs)
    ax1.set_xticklabels([str(ep) for ep in epochs], fontsize=9, color='#f0f6fc')
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=":", alpha=0.3, color='#8b949e')
    ax1.legend(loc='lower right', facecolor='#21262d', edgecolor='#30363d', fontsize=9)

    # Subplot 2: Win Rate vs Top-1 Accuracy Correlation
    ax2.plot(epochs, top1, marker='D', linewidth=2.0, color='#3fb950', label="Top-1 Val Accuracy", zorder=3)
    ax2.plot(epochs, overall_wr, marker='o', linewidth=2.0, color='#e3b341', linestyle='--', label="Win Rate Empírico", zorder=4)
    ax2.set_title("Correlação: Acurácia Top-1 vs Win Rate Real", fontsize=12, fontweight='bold', color='#f0f6fc', pad=12)
    ax2.set_xlabel("Época de Treino", fontsize=11, color='#c9d1d9')
    ax2.set_ylabel("Porcentagem (%)", fontsize=11, color='#c9d1d9')
    ax2.set_xticks(epochs)
    ax2.set_xticklabels([str(ep) for ep in epochs], fontsize=9, color='#f0f6fc')
    ax2.grid(True, linestyle=":", alpha=0.3, color='#8b949e')
    ax2.legend(loc='lower right', facecolor='#21262d', edgecolor='#30363d', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"📈 Benchmark progression plot saved to: {output_path}")


def plot_training_curves(history: List[dict], output_path: str):
    """Generates a high-quality visualization of the full training progression."""
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_top1 = [h["val_top1"] for h in history]
    val_top2 = [h["val_top2"] for h in history]

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#0d1117')
    ax1.set_facecolor('#161b22')
    ax2.set_facecolor('#161b22')

    # Loss curve
    ax1.plot(epochs, train_losses, label="Train Loss", color="#58a6ff", linewidth=2.0)
    ax1.plot(epochs, val_losses, label="Val Loss", color="#f78166", linewidth=2.0, linestyle="--")
    ax1.set_title("Progression of Loss Across 100 Epochs", fontsize=12, fontweight='bold', color='#f0f6fc', pad=12)
    ax1.set_xlabel("Epoch", fontsize=11, color='#c9d1d9')
    ax1.set_ylabel("Weighted CrossEntropy Loss", fontsize=11, color='#c9d1d9')
    ax1.grid(True, linestyle=":", alpha=0.3, color='#8b949e')
    ax1.legend(facecolor='#21262d', edgecolor='#30363d', fontsize=10)

    # Accuracy curve
    ax2.plot(epochs, val_top1, label="Val Top-1 Accuracy", color="#3fb950", linewidth=2.2)
    ax2.plot(epochs, val_top2, label="Val Top-2 Accuracy", color="#d29922", linewidth=2.0, linestyle="-.")
    best_idx = np.argmax(val_top1)
    ax2.scatter([epochs[best_idx]], [val_top1[best_idx]], color="#56d364", s=90, zorder=5, label=f"Best Top-1: {val_top1[best_idx]:.2f}% (Ep {epochs[best_idx]})")
    ax2.set_title("Validation Accuracy (Top-1 & Top-2)", fontsize=12, fontweight='bold', color='#f0f6fc', pad=12)
    ax2.set_xlabel("Epoch", fontsize=11, color='#c9d1d9')
    ax2.set_ylabel("Accuracy (%)", fontsize=11, color='#c9d1d9')
    ax2.grid(True, linestyle=":", alpha=0.3, color='#8b949e')
    ax2.legend(facecolor='#21262d', edgecolor='#30363d', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"📊 Training progression plot saved to: {output_path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable Ampere Tensor Core & cuDNN benchmark optimizations
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
    else:
        vram_gb = 0
        gpu_name = "CPU"

    print("=" * 80)
    print(f"🧠 BEHAVIORAL CLONING 100-EPOCH GPU TRAINER (FULL REPLAY CORPUS)")
    print(f"Device: {gpu_name} ({vram_gb:.2f} GB VRAM) | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print(f"Benchmark Interval: Every {args.benchmark_interval} Epochs ({args.benchmark_games * 2} Battles per Interval)")
    print("=" * 80)

    # 1. Dataset Loading (with caching support)
    cache_file = args.cache_path
    if getattr(args, "use_all_replays", True) and cache_file == "data/replays/bc_dataset_cache.pt":
        cache_file = "data/replays/bc_all_replays_cache.pt"

    samples = None

    if cache_file and os.path.exists(cache_file) and not getattr(args, "force_reextract", False):
        print(f"⚡ Loading cached parsed dataset from: {cache_file}")
        t_cache = time.time()
        samples = torch.load(cache_file, map_location="cpu", weights_only=False)
        print(f"Loaded {len(samples):,} turns in {time.time() - t_cache:.2f}s!")
    else:
        if getattr(args, "use_all_replays", True):
            paths = load_all_replays(replay_root="data/replays", db_path=args.db)
        elif hasattr(args, "replay_dir") and args.replay_dir and os.path.exists(args.replay_dir):
            paths = [
                os.path.join(args.replay_dir, f)
                for f in os.listdir(args.replay_dir)
                if f.endswith(".json") or f.endswith(".json.gz")
            ]
            print(f"Loaded {len(paths)} replays directly from directory: {args.replay_dir}")
        elif os.path.exists(args.db):
            paths = load_balanced_replays(
                args.db,
                elite_count=args.elite_samples,
                high_count=args.high_samples,
                mid_count=args.mid_samples,
                common_count=args.common_samples
            )
        else:
            paths = load_all_replays(replay_root="data/replays", db_path=args.db)

        workers = args.workers or (os.cpu_count() or 16)
        samples = extract_dataset(paths, max_workers=workers)

        if samples and cache_file:
            print(f"💾 Caching extracted samples to {cache_file} for instant future runs...")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save(samples, cache_file)

    if not samples:
        print("[-] Error: No training samples available.")
        return

    # 2. Train / Val Split (85% / 15%)
    np.random.seed(42)
    np.random.shuffle(samples)
    split_idx = int(len(samples) * 0.85)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"Dataset split: {len(train_samples):,} Training turns | {len(val_samples):,} Validation turns")
    print(f"🚀 Moving entire dataset tensors directly into GPU VRAM for zero-PCIe bottleneck...")
    t_gpu_load = time.time()
    train_data = FastGPUDataset(train_samples, device=device)
    val_data = FastGPUDataset(val_samples, device=device)
    print(f"Tensors resident in VRAM! Transfer took {time.time() - t_gpu_load:.2f}s.")

    # 3. Model, Optimizer, Loss, AMP Scaler
    model = ActionScoringPolicyNet(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
        num_actions=NUM_ACTION_SLOTS,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_epoch = 0
    save_path = os.path.join(args.output_dir, "bc_model.pt")
    history_path = os.path.join(args.output_dir, "training_history.json")
    benchmark_history_path = os.path.join(args.output_dir, "benchmark_decades.json")
    plot_path = os.path.join(args.output_dir, "training_curves.png")
    bench_plot_path = os.path.join(args.output_dir, "benchmark_decades_winrate.png")
    os.makedirs(args.output_dir, exist_ok=True)

    history = []
    benchmarks_history = []
    total_start_time = time.time()

    # 4. Training Loop across 100 Epochs
    print("\n" + "=" * 80)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Val Loss':<10} | {'Top-1 Acc':<11} | {'Top-2 Acc':<11} | {'LR':<8} | {'Time'}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0

        for states, actions, targets, weights, masks in train_data.get_batches(args.batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
                logits = model(states, actions, masks)
                loss_per_sample = F.cross_entropy(logits, targets, reduction="none")
                weighted_loss = (loss_per_sample * weights).mean()

            scaler.scale(weighted_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += weighted_loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(1, train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_top1 = 0
        correct_top2 = 0
        total_val = 0

        with torch.no_grad():
            for states, actions, targets, weights, masks in val_data.get_batches(args.batch_size, shuffle=False):
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
                    logits = model(states, actions, masks)
                    loss_per_sample = F.cross_entropy(logits, targets, reduction="none")
                    val_loss += (loss_per_sample * weights).mean().item()

                _, top2_preds = logits.topk(2, dim=-1)
                correct_top1 += (top2_preds[:, 0] == targets).sum().item()
                correct_top2 += ((top2_preds[:, 0] == targets) | (top2_preds[:, 1] == targets)).sum().item()
                total_val += targets.size(0)
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        val_top1 = (correct_top1 / total_val) * 100.0 if total_val else 0.0
        val_top2 = (correct_top2 / total_val) * 100.0 if total_val else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_top1)
        epoch_dur = time.time() - epoch_t0

        is_best = val_top1 > best_val_acc
        best_marker = " 🌟" if is_best else ""
        print(f"{epoch:<6} | {avg_train_loss:<11.4f} | {avg_val_loss:<10.4f} | {val_top1:<10.2f}% | {val_top2:<10.2f}% | {current_lr:<8.1e} | {epoch_dur:5.2f}s{best_marker}")

        # Save Best Model Checkpoint
        if is_best:
            best_val_acc = val_top1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_top1_acc": val_top1,
                "val_top2_acc": val_top2,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "hidden_dim": args.hidden_dim,
                "num_actions": NUM_ACTION_SLOTS
            }, save_path)

        # 5. Benchmark Evaluation (Every args.benchmark_interval epochs)
        epoch_bench = None
        if epoch % args.benchmark_interval == 0:
            print(f"  ⚔️ [BENCHMARK EPOCH {epoch}] Executing {args.benchmark_games * 2} battles against baselines...", end=" ", flush=True)
            bench_t0 = time.time()
            epoch_bench = evaluate_benchmark(model, games_per_opp=args.benchmark_games)
            b_dur = time.time() - bench_t0
            owr = epoch_bench["overall_win_rate"]
            md_wr = epoch_bench["max_damage"]["win_rate"]
            rnd_wr = epoch_bench["random"]["win_rate"]
            print(f"Win Rate: {owr:.1f}% (vs MaxDamage: {md_wr:.1f}%, vs Random: {rnd_wr:.1f}%) in {b_dur:.2f}s!")

            # Save Checkpoint for this benchmark interval
            interval_ckpt_path = os.path.join(args.output_dir, f"bc_model_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_top1_acc": val_top1,
                "val_top2_acc": val_top2,
                "benchmark": epoch_bench,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM
            }, interval_ckpt_path)

            benchmarks_history.append({
                "epoch": epoch,
                "overall_win_rate": float(owr),
                "max_damage_win_rate": float(md_wr),
                "random_win_rate": float(rnd_wr),
                "val_top1": float(val_top1),
                "val_top2": float(val_top2)
            })

        history.append({
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "val_top1": float(val_top1),
            "val_top2": float(val_top2),
            "lr": float(current_lr),
            "epoch_sec": float(epoch_dur),
            "benchmark": epoch_bench
        })

    total_training_sec = time.time() - total_start_time
    print("=" * 80)
    print(f"🎉 Training 100 Epochs & Benchmark Complete in {total_training_sec:.1f}s ({total_training_sec/60:.2f} min)!")
    print(f"🏆 Best Checkpoint: Epoch {best_epoch} with Val Top-1: {best_val_acc:.2f}%")
    print(f"💾 Checkpoint saved to: {save_path}")

    # Save history json & plot curves
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"📄 Training history recorded to: {history_path}")

    with open(benchmark_history_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks_history, f, indent=2)
    print(f"📄 Benchmark history recorded to: {benchmark_history_path}")

    try:
        plot_training_curves(history, plot_path)
    except Exception as e:
        print(f"[-] Training curves plotting warning: {e}")

    try:
        plot_benchmark_progression(benchmarks_history, bench_plot_path)
    except Exception as e:
        print(f"[-] Benchmark plotting warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train 100-Epoch Behavioral Cloning Policy on 100% Replays with Interval Benchmarking")
    parser.add_argument("--replay-dir", type=str, default="", help="Directory with replay JSON files")
    parser.add_argument("--db", type=str, default="data/replays/replays_metadata.sqlite", help="Replays DB path")
    parser.add_argument("--use-all-replays", action="store_true", default=True, help="Load 100% of all 36k+ replays on disk")
    parser.add_argument("--cache-path", type=str, default="data/replays/bc_all_replays_cache.pt", help="Cached parsed dataset file")
    parser.add_argument("--force-reextract", action="store_true", help="Force re-extract dataset instead of using cache")
    parser.add_argument("--elite-samples", type=int, default=4000, help="Number of elite replays")
    parser.add_argument("--high-samples", type=int, default=1500, help="Number of high replays")
    parser.add_argument("--mid-samples", type=int, default=1500, help="Number of mid replays")
    parser.add_argument("--common-samples", type=int, default=1500, help="Number of common replays")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for GPU")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Initial learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--workers", type=int, default=20, help="Worker threads for parsing")
    parser.add_argument("--benchmark-interval", type=int, default=5, help="Epoch interval to run benchmark (e.g. 5)")
    parser.add_argument("--benchmark-games", type=int, default=20, help="Games per opponent in interval benchmark")
    parser.add_argument("--output-dir", type=str, default="projects/behavioral-cloning/weights", help="Output directory")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
