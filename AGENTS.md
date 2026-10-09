# AGENTS.md — AI Agent Guidelines & Engineering Standards (v2)

Welcome to **Amnesia-AI v2**. This document defines the engineering standards, architecture patterns, development workflows, and behavioral expectations for AI agents contributing to this repository.

---

## 1. Project Mission & Core Philosophy

**Amnesia-AI v2** is a ground-up reinvention of the Amnesia competitive Pokémon AI project. Its mission is to achieve **human-master competitive performance (>1600 Glicko-1 / Top 500 Ladder)** on Pokémon Showdown without search trees or hardcoded heuristics, strictly adhering to the seminal paper:

> **"A Self-Play Policy Optimization Approach to Battling Pokémon"**  
> *Dan Huang & Scott Lee — IEEE Conference on Games (CoG 2019)*

### Non-Negotiable Core Principles

1. **Radical Simplicity & Zero Monorepo Bloat**:
   - The entire v2 system lives in a clean, focused Python 3.12+ package (`src/`).
   - No polyglot sprawl (no Bun/TypeScript/Selenium alongside Python).
   - No fragmented subprojects (no independent scrapers, team builders, or type classifiers).

2. **Authentic Engine via `poke-env`**:
   - **NEVER** build or use a handcrafted/approximate battle simulator. Handcrafted simulators create a fatal reality gap.
   - All mechanics, damage calculations, weather, items, stat boosts, and abilities are evaluated strictly through the official Pokémon Showdown engine via `poke-env`.

3. **Rich Semantic Entity Embeddings (128-d)**:
   - **NEVER** collapse the game state into a tiny flat vector of raw floats (the fatal flaw of v1).
   - Use 128-dimensional learned entity embeddings for species, moves, abilities, items, and volatile statuses (Figure 1 & Table 1 of Huang & Lee 2019).

4. **Symmetric Pure Self-Play**:
   - Train the agent primarily through symmetric self-play ($f_\theta$ vs $f_\theta$).
   - Terminal zero-sum reward ($\pm 1.0$ for win/loss) with minimal auxiliary shaping ($\pm 0.01$).

---

## 2. Technology Stack & Environment

| Component | Technology | Rationale |
|---|---|---|
| **Runtime & Language** | Python 3.12+ | Native async support, high-speed ecosystem |
| **Deep Learning Framework** | PyTorch 2.6+ (CUDA 12.4, TF32 enabled) | Accelerated neural tensor execution |
| **Environment & Protocol** | `poke-env` 0.16+ & `gymnasium` 1.3+ | Official Showdown WebSocket wrapper with Gymnasium compliance |
| **RL Algorithm** | PPO (Proximal Policy Optimization) + GAE | Clipped surrogate objective with generalized advantage estimation |
| **Target Battle Format** | `gen7randombattle` / `gen8randombattle` | Balanced levels, maximum tactical diversity, eliminates team-matchup bias |

---

## 3. Architecture Blueprint (`src/`)

```
Amnesia-AI/
├── AGENTS.md                  # This file (Agent instructions & standards)
├── CONTEXT.md                 # Deep domain context, theoretical foundation & v1 post-mortem
├── METAS.md                   # Quantitative milestones, Elo targets & success criteria
├── TODO.md                    # Prioritized, phased execution roadmap
├── requirements.txt           # Minimal pinned dependencies
└── src/
    ├── __init__.py
    ├── model.py               # Actor-Critic network with 128-d Entity Embeddings & Action Masking
    ├── env.py                 # poke-env wrapper translating Battle state into embedding tensors
    ├── train.py               # Pure Self-Play PPO training loop (Algorithm 1)
    ├── evaluate.py            # Head-to-head tournament evaluator against baseline bots
    └── client.py              # Live Showdown WebSocket ladder client
```

---

## 4. Engineering Standards for AI Agents

When reading, modifying, or creating code in this repository, follow these rules strictly:

### 4.1 Strict Type Annotations
- All functions and class methods must feature complete Python type hints (`typing.Dict`, `typing.List`, `typing.Tuple`, `torch.Tensor`).
- Document tensor shapes in docstrings: `# [batch_size, num_actions]` or `# [batch_size, 6, 128]`.

### 4.2 Action Masking Integrity
- Pokémon battle action spaces contain invalid choices depending on the state (fainted Pokémon cannot be switched to; disabled moves cannot be used).
- Agents must **always** enforce dynamic action masking ($s \in \{0, 1\}^n$) before softmax sampling to guarantee zero probability for illegal choices.

### 4.3 GPU Optimization & TF32
- Maintain TF32 tensor core acceleration on Ampere+ GPUs:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = True
  ```

### 4.4 Reproducibility
- Always support deterministic seeding (`torch.manual_seed`, `np.random.seed`, `random.seed`) in test and evaluation scripts.

---

## 5. Agent Personas

When executing tasks in this codebase, assume one of these specialized personas:

1. **RL Systems & Environment Engineer**:
   - Focus: `poke-env` integrations, async WebSocket battle throughput, state feature extraction, action masking.
   - Files: `src/env.py`, `src/client.py`.

2. **Neural Architect & PyTorch Specialist**:
   - Focus: 128-d categorical embedding layers, pooling modules, two-headed Actor-Critic nets, numerical stability.
   - Files: `src/model.py`.

3. **Self-Play & Training Strategist**:
   - Focus: PPO loss formulation, GAE backward return calculation, entropy scheduling, empirical Glicko/Elo tracking.
   - Files: `src/train.py`, `src/evaluate.py`.

---

## 6. Commit & Branch Conventions

- Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `perf:`, `test:`.
- **Active v2 Branch**: `v2-selfplay-ppo`.
- **Archived v1 Legacy Branch**: `it_failed` (do not merge or contaminate `v2-selfplay-ppo` with v1 code).
