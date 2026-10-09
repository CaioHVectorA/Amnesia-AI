"""
Behavioral Cloning Training Pipeline
====================================
Balanced multi-tier replay training for Pokemon Showdown.
Combines Elite (Elo 1800+) with Common/Mid-tier matches, applying sample weighting.
"""

import argparse
import concurrent.futures
import os
import sqlite3
import sys
import time
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from model import ActionScoringPolicyNet
from parser import parse_replay_file, STATE_DIM, ACTION_DIM, NUM_ACTION_SLOTS


class ShowdownBCDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, int, float, np.ndarray]]):
        self.states = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
        self.actions = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
        self.targets = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.long)
        self.weights = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32)
        self.masks = torch.tensor(np.array([s[4] for s in samples]), dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.targets[idx],
            self.weights[idx],
            self.masks[idx]
        )


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


def extract_dataset(replay_paths: List[str], max_workers: int = 8) -> List[Tuple[np.ndarray, np.ndarray, int, float, np.ndarray]]:
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


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"🧠 TRAINING BEHAVIORAL CLONING POLICY NETWORK (Device: {device})")
    print("=" * 65)

    # 1. Load Replays & Extract Samples
    if hasattr(args, "replay_dir") and args.replay_dir and os.path.exists(args.replay_dir):
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
        # Fallback to scanning data/replays
        replay_root = "data/replays"
        paths = []
        for root, _, files in os.walk(replay_root):
            for f in files:
                if f.endswith(".json") or f.endswith(".json.gz"):
                    paths.append(os.path.join(root, f))
        print(f"Fallback: Discovered {len(paths)} replays under {replay_root}")

    samples = extract_dataset(paths, max_workers=args.workers)

    if not samples:
        print("Error: No training samples extracted. Check dataset and DB paths.")
        return

    # 2. Train / Val Split (85% / 15%)
    np.random.seed(42)
    np.random.shuffle(samples)
    split_idx = int(len(samples) * 0.85)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_dataset = ShowdownBCDataset(train_samples)
    val_dataset = ShowdownBCDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model, Optimizer, Loss
    model = ActionScoringPolicyNet(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
        num_actions=NUM_ACTION_SLOTS,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    save_path = os.path.join(args.output_dir, "bc_model.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Training Loop
    print("\nStarting Training...")
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Val Loss':<10} | {'Top-1 Acc':<11} | {'Top-2 Acc':<11} | {'LR'}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for states, actions, targets, weights, masks in train_loader:
            states, actions = states.to(device), actions.to(device)
            targets, weights, masks = targets.to(device), weights.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(states, actions, masks)

            # Sample-weighted CrossEntropy
            loss_per_sample = F.cross_entropy(logits, targets, reduction="none")
            weighted_loss = (loss_per_sample * weights).mean()

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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
            for states, actions, targets, weights, masks in val_loader:
                states, actions = states.to(device), actions.to(device)
                targets, weights, masks = targets.to(device), weights.to(device), masks.to(device)

                logits = model(states, actions, masks)
                loss_per_sample = F.cross_entropy(logits, targets, reduction="none")
                val_loss += (loss_per_sample * weights).mean().item()
                val_batches += 1

                # Top-1 and Top-2 accuracy
                _, top2_preds = logits.topk(2, dim=-1)
                correct_top1 += (top2_preds[:, 0] == targets).sum().item()
                correct_top2 += ((top2_preds[:, 0] == targets) | (top2_preds[:, 1] == targets)).sum().item()
                total_val += targets.size(0)

        avg_val_loss = val_loss / max(1, val_batches)
        val_top1 = (correct_top1 / total_val) * 100.0 if total_val else 0.0
        val_top2 = (correct_top2 / total_val) * 100.0 if total_val else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_top1)

        print(f"{epoch:<6} | {avg_train_loss:<11.4f} | {avg_val_loss:<10.4f} | {val_top1:<10.2f}% | {val_top2:<10.2f}% | {current_lr:.1e}")

        # Save Best Model Checkpoint
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_top1_acc": val_top1,
                "val_top2_acc": val_top2,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM
            }, save_path)

    print("\n" + "=" * 65)
    print(f"🎉 Training Complete! Best Validation Top-1 Acc: {best_val_acc:.2f}%")
    print(f"💾 Checkpoint saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Behavioral Cloning Policy on Showdown Replays")
    parser.add_argument("--replay-dir", type=str, default="", help="Directory with replay JSON files")
    parser.add_argument("--db", type=str, default="data/replays/replays_metadata.sqlite", help="Replays DB path")
    parser.add_argument("--elite-samples", type=int, default=4000, help="Number of elite replays (Elo 1800+)")
    parser.add_argument("--high-samples", type=int, default=1500, help="Number of high replays (Elo 1650-1799)")
    parser.add_argument("--mid-samples", type=int, default=1500, help="Number of mid replays (Elo 1500-1649)")
    parser.add_argument("--common-samples", type=int, default=1500, help="Number of common replays (sub1500)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--workers", type=int, default=8, help="Worker threads for parsing")
    parser.add_argument("--output-dir", type=str, default="projects/behavioral-cloning/weights", help="Output directory")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
