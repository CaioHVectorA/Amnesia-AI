# AGENTS.md — AI Agent Guidelines, Architecture Standards & Operating Manual (v2)

Welcome to **Amnesia-AI v2**. This document defines the engineering standards, architecture patterns, development workflows, and behavioral expectations for AI agents contributing to this repository.

---

## 1. Project Mission & Non-Negotiable Invariants

**Amnesia-AI v2** is a ground-up reinvention of the Amnesia competitive Pokémon AI project. Its mission is to achieve **human-master competitive performance (>1600 Glicko-1 / Top Ladder)** on Pokémon Showdown without search trees or hardcoded heuristics, strictly adhering to:

> **"A Self-Play Policy Optimization Approach to Battling Pokémon"**  
> *Dan Huang & Scott Lee — IEEE Conference on Games (CoG 2019)*

### Non-Negotiable Core Principles (The Four Invariants)

1. **INVARIANT 1: Zero Custom Simulators**:
   - **NEVER** write, import, or use a handcrafted battle simulator. Handcrafted engines inevitably fail on complex abilities, items, and priority interactions, creating an insurmountable reality gap.
   - All state transitions, damage calculations, weather, items, stat boosts, and abilities are evaluated strictly through the official Pokémon Showdown engine via `poke-env`.

2. **INVARIANT 2: Rich Semantic Entity Embeddings (128-d)**:
   - **NEVER** collapse the game state into a small flat vector of floats (the fatal flaw of v1).
   - All categorical variables (1,023 Species, 731 Moves, 368 Items, 238 Abilities) must be mapped through 128-dimensional learned entity embeddings.

3. **INVARIANT 3: Strict Action Masking**:
   - In Pokémon, legal actions change dynamically every turn.
   - The neural network must **always** enforce dynamic action masking ($s \in \{0, 1\}^9$) before softmax normalization. An illegal move probability must be mathematically guaranteed to be **exactly 0.0**.

4. **INVARIANT 4: Pure Symmetric Self-Play Objective**:
   - Train the agent primarily through symmetric self-play ($f_\theta$ vs $f_\theta$).
   - The primary reward must be the terminal zero-sum battle outcome ($\pm 1.0$), with auxiliary shaping constrained to minute magnitudes ($\pm 0.01$) so it cannot overpower the win condition.

---

## 2. Technology Stack & Environment

| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Runtime & Language** | Python | 3.12+ | Async battle orchestration, type safety, performance |
| **Deep Learning** | PyTorch | 2.6+ (CUDA 12.4) | Tensor execution, embedding layers, PPO loss computation |
| **Hardware Mode** | CUDA TF32 | Ampere+ | Tensor core acceleration (`matmul.allow_tf32 = True`) |
| **Showdown Protocol** | `poke-env` | 0.16+ | Official Showdown WebSocket wrapper with Gymnasium compliance |
| **Environment Standard** | `gymnasium` | 1.3+ | Standardized step, reset, observation, and action spaces |
| **Target Battle Format** | `gen7randombattle` / `gen8randombattle` | Standard | Level-balanced, procedural diversity, zero team-matchup bias |

---

## 3. Architecture Blueprint (`src/`)

```text
Amnesia-AI/
├── AGENTS.md                  # This file (Operating manual & engineering standards)
├── CONTEXT.md                 # Deep domain context, theoretical foundation & v1 post-mortem
├── METAS.md                   # Quantitative milestones, Elo targets & success criteria
├── TODO.md                    # Prioritized, phased execution roadmap
├── requirements.txt           # Minimal pinned dependencies
└── src/
    ├── __init__.py            # Package root & exports
    ├── model.py               # Actor-Critic network with 128-d Entity Embeddings & Action Masking
    ├── env.py                 # poke-env wrapper translating Battle state into embedding tensors
    ├── train.py               # Pure Self-Play PPO training loop (Algorithm 1)
    ├── evaluate.py            # Head-to-head tournament evaluator against baseline bots
    └── client.py              # Live Showdown WebSocket ladder client
```

---

## 4. Coding Standards & Implementation Rules

### 4.1 Strict Python Typing & Docstring Tensor Shapes
- All functions and class methods must feature complete Python type annotations (`typing.Dict`, `typing.List`, `typing.Tuple`, `torch.Tensor`, `Optional`).
- Tensor shapes **must be documented** in comments and docstrings using batch notation:
  ```python
  def forward(
      self, 
      state_tokens: torch.Tensor,     # [batch_size, num_tokens]
      continuous_stats: torch.Tensor, # [batch_size, num_features]
      action_mask: torch.Tensor       # [batch_size, 9]
  ) -> Tuple[torch.Tensor, torch.Tensor]: # policy: [batch_size, 9], value: [batch_size]
  ```

### 4.2 Numerical Stability Best Practices
- Never compute raw probabilities before computing log probabilities. Use `F.log_softmax` or PyTorch's `torch.distributions.Categorical` directly.
- In Advantage normalization, always include numerical stability epsilon:
  ```python
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  ```
- In Action Masking, set illegal logits to `-1e8` (not `-inf`) to avoid `NaN` propagation during backpropagation:
  ```python
  masked_logits = logits.masked_fill(~mask.bool(), -1e8)
  ```

### 4.3 Asynchronous Execution & `poke-env` Integrity
- `poke-env` runs on Python's `asyncio` event loop.
- Never place blocking, synchronous disk I/O or heavy CPU training loops directly inside the asynchronous event loop without delegating to `asyncio.to_thread` or running battle collection in dedicated worker tasks.
- Always handle graceful disconnection and reconnection: if the Showdown WebSocket drops, the client must safely reconnect without crashing the training run.

### 4.4 GPU Optimization & Memory Management
- Maintain TF32 tensor core acceleration on Ampere+ GPUs:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = True
  ```
- Use pinned memory (`pin_memory=True`) when moving transition batches from host RAM to CUDA device.

---

## 5. Agent Personas & File Ownership

When contributing to this repository, adopt one of the following specialized personas depending on the task:

### Persona 1: RL Systems & Environment Engineer
* **Primary Ownership**: `src/env.py`, `src/client.py`.
* **Focus**:
  * Extracting clean categorical indices and continuous stat tensors from `poke-env.battle.Battle`.
  * Dynamic action masking verification (handling trapped Pokémon, taunt, choice locks, and fainted members).
  * High-throughput asynchronous battle collection and connection handling.
* **Anti-Patterns to Avoid**:
  * Writing custom damage estimation formulas inside the environment (delegate damage and state resolution to Showdown).
  * Letting unrevealed moves crash the parser (always map unknown tokens to `<UNK>` index 0).

### Persona 2: Neural Architect & PyTorch Specialist
* **Primary Ownership**: `src/model.py`.
* **Focus**:
  * 128-dimensional entity embedding modules (`nn.Embedding(vocab_size, 128)`).
  * Permutation-invariant team pooling across bench members (`max_pooling`).
  * Actor-Critic dual head computation with shared candidate action scoring.
  * Ensuring the model parameter count precisely conforms to Huang & Lee's architecture (~1.32M parameters).
* **Anti-Patterns to Avoid**:
  * Using monolithic flat Linear layers that bypass categorical embeddings.
  * Hardcoding action choices or building rule-based override branches inside the network forward pass.

### Persona 3: Self-Play & Training Strategist
* **Primary Ownership**: `src/train.py`, `src/evaluate.py`.
* **Focus**:
  * Symmetric pure self-play trajectory collection ($2m$ samples from $m$ games).
  * Trajectory-level GAE calculation with backward episodic discounting.
  * PPO clipped surrogate objective, value loss weighting, and entropy scheduling.
  * Automated 500-match tournament evaluations and Glicko-1 rating tracking.
* **Anti-Patterns to Avoid**:
  * Using turn-level single-step TD error without backward return propagation.
  * Flooding the reward function with dense immediate damage terms.

---

## 6. Pre-Commit Verification Checklist

Before pushing any commit on `v2-selfplay-ppo`, ensure:
1. [ ] **No Legacy Code**: No files from v1 monorepo (bun, selenium, fast simulator) are introduced.
2. [ ] **Syntax & Type Cleanliness**: Python code passes syntax checking with complete type hints.
3. [ ] **Tensor Shape Consistency**: All PyTorch forward passes run without shape mismatches.
4. [ ] **Action Mask Zero-Check**: Softmax output on illegal actions is mathematically verified to be `0.0`.
5. [ ] **Conventional Commit**: Commit message follows `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`.
