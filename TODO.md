# TODO.md — Phased Execution Roadmap (v2)

This roadmap outlines the prioritized engineering phases to implement, train, evaluate, and deploy **Amnesia-AI v2**.

---

## 📋 Status Overview

- [x] **Phase 0: Clean Slate & Foundation**
  - [x] Archive v1 legacy codebase to `it_failed` branch and push to GitHub.
  - [x] Create clean `v2-selfplay-ppo` branch.
  - [x] Purge all legacy multi-runtime bloat and monorepo files.
  - [x] Install and verify official `poke-env` and `gymnasium` dependencies.
  - [x] Establish foundational documentation: `AGENTS.md`, `CONTEXT.md`, `METAS.md`, `TODO.md`.

---

## 🚀 Phase 1: State Representation & Environment (`src/env.py`)

- [ ] **1.1 Canonical Vocabulary & Entity Mappings**:
  - [ ] Extract canonical integer ID mappings from `poke-env` data for:
    - 1,023 Species
    - 731 Moves
    - 368 Items
    - 238 Abilities
  - [ ] Handle unrevealed/unknown tokens (`<UNK>`, `<NONE>`) for partial observability.
- [ ] **1.2 Hierarchical State Tensor Extractor**:
  - [ ] Extract active Pokémon categorical & continuous features.
  - [ ] Extract 5 bench Pokémon categorical & continuous features.
  - [ ] Extract opponent active Pokémon (revealed features) & 5 opponent bench members.
  - [ ] Extract global field conditions (weather, terrain, side conditions, hazards).
- [ ] **1.3 Dynamic Action Masking**:
  - [ ] Build exact 9-dimensional binary mask $s \in \{0, 1\}^9$ (4 moves + 5 switches).
  - [ ] Zero out fainted switches, trapped Pokémon switches, and PP-depleted moves.
- [ ] **1.4 Gymnasium Environment Wrapper**:
  - [ ] Wrap `poke-env.player.Player` into a clean Gymnasium `Env` interface.
  - [ ] Support both `gen7randombattle` and `gen8randombattle`.

---

## 🧠 Phase 2: 1.3M Parameter Neural Architecture (`src/model.py`)

- [ ] **2.1 Entity Embedding Modules**:
  - [ ] `nn.Embedding(1023, 128)` for Species.
  - [ ] `nn.Embedding(731, 128)` for Moves.
  - [ ] `nn.Embedding(368, 128)` for Items.
  - [ ] `nn.Embedding(238, 128)` for Abilities.
- [ ] **2.2 Team-Level Permutation Invariant Pooling**:
  - [ ] Combine continuous stats with embedding vectors for each Pokémon.
  - [ ] Apply Max-Pooling over bench members to achieve order invariance.
- [ ] **2.3 Dual Actor-Critic Heads**:
  - [ ] **Actor Head**: Candidate action scoring matrix producing logits $p \in \mathbb{R}^9$.
  - [ ] **Action Masker**: Renormalized masked softmax layer guaranteeing zero illegal choice probability.
  - [ ] **Critic Head**: Feedforward MLP producing scalar board state advantage estimate $V(s) \in [-1.0, +1.0]$.
- [ ] **2.4 Sanity & Gradient Verification**:
  - [ ] Write unit test verifying forward pass tensor shapes, device placement (CUDA TF32), and backward gradients.

---

## ⚡ Phase 3: Pure Self-Play PPO Engine (`src/train.py`)

- [ ] **3.1 High-Throughput Self-Play Match Runner**:
  - [ ] Configure two `poke-env` players sharing the same model parameters $f_\theta$.
  - [ ] Run parallel asynchronous self-play battles.
  - [ ] Collect complete episode trajectories from both player perspectives.
- [ ] **3.2 Episode-Level Generalized Advantage Estimation (GAE)**:
  - [ ] Implement full trajectory backward discounting ($\gamma=0.99, \lambda=0.95$).
  - [ ] Ensure terminal win/loss ($\pm 1.0$) propagates back through all game turns.
  - [ ] Incorporate auxiliary shaping ($-0.0125$ own faint, $+0.0025$ super-effective).
- [ ] **3.3 PPO Optimization Loop**:
  - [ ] Implement clipped surrogate loss ($L^{\text{CLIP}}$ with $\epsilon=0.2$).
  - [ ] Implement Critic MSE loss ($L^{\text{VF}}$).
  - [ ] Implement Entropy bonus ($S[\pi]$) for exploration control.
  - [ ] Gradient clipping ($\text{max\_norm}=1.0$) and AdamW optimizer.
- [ ] **3.4 Telemetry & Checkpointing**:
  - [ ] Real-time logging: turns/sec, actor loss, critic loss, entropy, win rate.
  - [ ] Checkpoint manager saving best model weights (`weights/amnesia_v2.pt`).

---

## 🏆 Phase 4: Tournament & Baseline Evaluation (`src/evaluate.py`)

- [ ] **4.1 Baseline Benchmarking Suite**:
  - [ ] Automated 500-match head-to-head tournament vs `RandomPlayer` (Goal: >99%).
  - [ ] Automated 500-match head-to-head tournament vs `MaxBasePowerPlayer` (Goal: >88%).
  - [ ] Automated 500-match head-to-head tournament vs `SimpleHeuristicsPlayer` (Goal: >65%).
- [ ] **4.2 Glicko-1 & Elo Tracking**:
  - [ ] Compute official Glicko-1 rating curves across training iterations.
  - [ ] Track win rates across historical checkpoint pool to prevent cyclic forgetting.

---

## 🌐 Phase 5: Live Showdown Ladder Client (`src/client.py`)

- [ ] **5.1 Production Ladder Bot**:
  - [ ] Connect directly to official Pokémon Showdown server (`sim3.psim.us`).
  - [ ] Handle automated login, authentication assertion, and avatar configuration.
  - [ ] Enqueue in ranked `gen7randombattle` / `gen8randombattle` ladder.
- [ ] **5.2 Match Logging & Public Spectator Links**:
  - [ ] Real-time spectator URLs (`https://play.pokemonshowdown.com/battle-...`).
  - [ ] Save full match replays locally for post-match tactical analysis.
  - [ ] Reach and sustain Target Milestone: **>1600 Glicko-1 rating**.
