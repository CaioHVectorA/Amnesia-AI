# TODO.md — Phased Execution Roadmap & Engineering Task Tracker (v2)

This roadmap defines the prioritized, atomic engineering tasks required to build, train, evaluate, and deploy **Amnesia-AI v2**.

---

## 📋 High-Level Phase Progression

| Phase | Focus | Status | Primary Artifacts |
|---|---|:---:|---|
| **Phase 0** | Clean Slate & Foundational Standards | **COMPLETED ✅** | `AGENTS.md`, `CONTEXT.md`, `METAS.md`, `TODO.md` |
| **Phase 1** | State Representation & Environment | **IN PROGRESS ⏳** | `src/env.py`, `tests/test_env.py` |
| **Phase 2** | 1.3M Neural Architecture & Masking | **PLANNED 📅** | `src/model.py`, `tests/test_model.py` |
| **Phase 3** | Pure Symmetric Self-Play PPO Engine | **PLANNED 📅** | `src/train.py` |
| **Phase 4** | Baseline Tournaments & Glicko-1 Suite | **PLANNED 📅** | `src/evaluate.py` |
| **Phase 5** | Live Showdown Ladder Client & Replays | **PLANNED 📅** | `src/client.py` |

---

## 🚀 Phase 1: Environment & State Representation (`src/env.py`)

*Goal: Build the state extraction pipeline that translates raw `poke-env` Battle objects into PyTorch embedding index tensors and continuous stat vectors conforming strictly to Table I of Huang & Lee (2019).*

- [ ] **Task 1.1: Canonical Tokenizer & Vocabulary Mappings**
  - [ ] Extract canonical entity dictionaries from `poke-env.data.GenData` for:
    - Species vocabulary: 1,023 tokens (indices $1 \dots 1022$, index $0 = \text{<UNK>}$).
    - Moves vocabulary: 731 tokens (indices $1 \dots 730$, index $0 = \text{<UNK>}$).
    - Items vocabulary: 368 tokens (indices $1 \dots 367$, index $0 = \text{<UNK>}$, index $1 = \text{<NONE>}$).
    - Abilities vocabulary: 238 tokens (indices $1 \dots 237$, index $0 = \text{<UNK>}$).
  - [ ] Implement fast string-to-index hashing with fallback to token $0$ for unrevealed or unknown entities.

- [ ] **Task 1.2: Single Pokémon Feature Extractor**
  - [ ] Implement `extract_pokemon_features(mon: Pokemon) -> Dict[str, torch.Tensor]`:
    - Categorical IDs: `species_id`, `item_id`, `ability_id`, `moveset_ids` (4-element vector), `last_move_id`.
    - Continuous Stats (normalized to $[0, 1]$): Base stats vector (HP, Atk, Def, SpA, SpD, Spe).
    - Stat Boosts: Stage modifiers for 6 stats from $-6$ to $+6$, normalized to $[-1.0, +1.0]$.
    - Hitpoints: Current HP fraction $h_t \in [0.0, 1.0]$.
    - Move PP Used: Normalized PP count consumed per move slot.
    - Status Indicators: 28-dimensional multi-hot vector (sleep, burn, poison, paralysis, freeze, etc.).
    - Type Indicators: 18-dimensional multi-hot elemental type indicator.
    - Volatile Statuses: 23-dimensional indicator vector (Leech Seed, Taunt, Substitute, Confusion, etc.).

- [ ] **Task 1.3: Team & Bench Extractor with Permutation Invariance**
  - [ ] Extract Player Active Pokémon features ($1$ instance).
  - [ ] Extract Player Bench Pokémon features ($5$ instances), padding fainted slots with neutral representations.
  - [ ] Extract Opponent Active Pokémon features (revealed features only; unrevealed attributes mapped to `<UNK>`).
  - [ ] Extract Opponent Bench Pokémon features ($5$ instances).

- [ ] **Task 1.4: Global Field & Side Conditions Extractor**
  - [ ] Weather indicator & remaining turn counter (Rain, Sun, Sandstorm, Hail).
  - [ ] Terrain indicator & remaining turn counter (Electric, Grassy, Misty, Psychic).
  - [ ] Side hazards for both sides: Stealth Rock (0/1), Spikes (0-3), Toxic Spikes (0-2), Sticky Web (0/1).
  - [ ] Side screens for both sides: Reflect, Light Screen, Aurora Veil, Tailwind.

- [ ] **Task 1.5: Dynamic Action Mask Construction**
  - [ ] Construct exact binary action mask $s \in \{0, 1\}^9$:
    - Indices $0, 1, 2, 3$: Legal active moves (mask out disabled moves, 0-PP moves, or choice-locked moves).
    - Indices $4, 5, 6, 7, 8$: Legal switches to bench Pokémon (mask out fainted Pokémon, currently active Pokémon, and illegal switches when trapped by *Shadow Tag* or *Mean Look*).
  - [ ] Handle `force_switch` phases (mask out all moves $0-3$, allowing only legal switches).

- [ ] **Task 1.6: Gymnasium Environment Wrapper (`AmnesiaEnv`)**
  - [ ] Wrap `poke-env` Player into Gymnasium `Env` interface with `reset()` and `step(action)`.
  - [ ] Configure support for both `gen7randombattle` and `gen8randombattle`.
  - [ ] Verify that step observation returns structured dict: `{"tokens": ..., "continuous": ..., "mask": ...}`.

- [ ] **Task 1.7: Unit Testing & Integrity Certification**
  - [ ] Create `tests/test_env.py` verifying:
    - Zero tensor exceptions across 100 simulated battles.
    - Action mask accurately reflects legal moves from `battle.available_moves` and `battle.available_switches`.
    - Zero illegal choices permitted by mask.

---

## 🧠 Phase 2: 1.3M Parameter Neural Architecture (`src/model.py`)

*Goal: Implement the Actor-Critic network with 128-d Entity Embeddings, Permutation-Invariant Team Pooling, and Action Masking conforming precisely to Figure 1 of Huang & Lee (2019).*

- [ ] **Task 2.1: Entity Embedding Modules**
  - [ ] `self.species_embed = nn.Embedding(1023, 128, padding_idx=0)`
  - [ ] `self.moves_embed = nn.Embedding(731, 128, padding_idx=0)`
  - [ ] `self.items_embed = nn.Embedding(368, 128, padding_idx=0)`
  - [ ] `self.abilities_embed = nn.Embedding(238, 128, padding_idx=0)`
  - [ ] Average moveset embedding: compute mean of embeddings for known moves: $\bar{E}_{\text{moves}} = \frac{1}{4} \sum_{k=1}^4 E_{\text{move}_k}$.

- [ ] **Task 2.2: Pokémon Representation Subnetwork**
  - [ ] Concatenate: Species (128) + Item (128) + Ability (128) + Mean Moves (128) + Last Move (128) + Continuous Features ($C$).
  - [ ] Project through Fully-Connected layer with ReLU: $\text{Linear}(\text{in\_dim}, 128) \rightarrow \text{ReLU} \rightarrow h_{\text{mon}} \in \mathbb{R}^{128}$.

- [ ] **Task 2.3: Permutation-Invariant Bench Pooling**
  - [ ] Pass all 5 bench Pokémon through the shared representation subnetwork: $[h_1, h_2, h_3, h_4, h_5] \in \mathbb{R}^{5 \times 128}$.
  - [ ] Apply element-wise Max-Pooling across bench slots: $h_{\text{bench}} = \max_{j=1}^5 (h_j) \in \mathbb{R}^{128}$.
  - [ ] Repeat for opponent bench representation: $h_{\text{opp\_bench}} = \max_{j=1}^5 (h_{\text{opp}, j}) \in \mathbb{R}^{128}$.

- [ ] **Task 2.4: Core Board State Fusion**
  - [ ] Concatenate: $h_{\text{active}} \ (128) + h_{\text{bench}} \ (128) + h_{\text{opp\_active}} \ (128) + h_{\text{opp\_bench}} \ (128) + h_{\text{global}} \ (64)$.
  - [ ] Project through core board MLP: $\text{Linear}(576, 256) \rightarrow \text{ReLU} \rightarrow H_{\text{state}} \in \mathbb{R}^{256}$.

- [ ] **Task 2.5: Dual Heads (Actor & Critic)**
  - [ ] **Critic Head (Value Estimator)**:
    - $\text{Linear}(256, 128) \rightarrow \text{ReLU} \rightarrow \text{Linear}(128, 1) \rightarrow \text{Tanh} \rightarrow V(s) \in [-1.0, +1.0]$.
  - [ ] **Actor Head (Candidate Action Scorer)**:
    - For each Move $k \in \{0, 1, 2, 3\}$: Concat $H_{\text{state}}$ with Move Embedding $E_{\text{move}_k} \rightarrow \text{Linear} \rightarrow p_k$.
    - For each Switch $j \in \{0, 1, 2, 3, 4\}$: Concat $H_{\text{state}}$ with Bench Pokémon representation $h_{\text{bench}, j} \rightarrow \text{Linear} \rightarrow p_{4+j}$.
    - Produces raw action logit vector $p \in \mathbb{R}^9$.

- [ ] **Task 2.6: Masked Softmax Layer**
  - [ ] Implement `masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`:
    - Set illegal logits to $-10^8$: `logits.masked_fill(~mask.bool(), -1e8)`.
    - Apply `F.softmax(..., dim=-1)` guaranteeing $0.0$ probability on illegal choices.

- [ ] **Task 2.7: Sanity & Architecture Verification Suite**
  - [ ] Create `tests/test_model.py`:
    - Verify total trainable parameter count is approximately **1,327,618 parameters** ($\pm 5\%$).
    - Test forward pass latency on CUDA (< 2.0 ms per batch).
    - Verify backward gradient propagation without `NaN` or `Inf`.
    - Verify mathematically that masked illegal actions receive probability exactly `0.0`.

---

## ⚡ Phase 3: Pure Symmetric Self-Play PPO Engine (`src/train.py`)

*Goal: Implement Algorithm 1 from Huang & Lee (2019): concurrent self-play match execution, full-trajectory backward GAE computation, clipped PPO loss, and entropy scheduling.*

- [ ] **Task 3.1: Asynchronous Self-Play Match Generator**
  - [ ] Configure two `poke-env` player instances sharing the current neural network weights $f_\theta$.
  - [ ] Execute parallel asynchronous self-play battles (target batch size: $m = 128$ matches per batch).
  - [ ] Collect transitions from **BOTH player perspectives** ($2m$ complete trajectories per batch).

- [ ] **Task 3.2: Episode Trajectory Storage & Ingestion**
  - [ ] Store complete battle rollouts: $(s_t, a_t, r_t, s_{t+1}, \text{mask}_t, \text{done}_t, \log \pi(a_t \mid s_t), V(s_t))$.
  - [ ] Assign terminal zero-sum rewards ($\pm 1.0$) upon battle completion.
  - [ ] Assign auxiliary shaping rewards ($-0.0125$ own Pokémon faints, $+0.0025$ super-effective move hits).

- [ ] **Task 3.3: Trajectory-Level Generalized Advantage Estimation (GAE)**
  - [ ] Implement backward recursive GAE calculation across chronological episodes:
    $$\delta_t = r_t + \gamma V(s_{t+1}) (1 - d_t) - V(s_t)$$
    $$\hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1} (1 - d_t)$$
    $$R_t = \hat{A}_t + V(s_t)$$
  - [ ] Set hyperparameters: $\gamma = 0.99$, $\lambda = 0.95$.
  - [ ] Normalize advantages across batch: $\hat{A} = (\hat{A} - \mu) / (\sigma + 10^{-8})$.

- [ ] **Task 3.4: PPO Optimization Objective**
  - [ ] Implement Clipped Surrogate Actor Loss:
    $$r_t(\theta) = \exp(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{old}}(a_t \mid s_t))$$
    $$L^{\text{CLIP}}(\theta) = -\mathbb{E} \left[ \min(r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right] \quad (\epsilon = 0.2)$$
  - [ ] Implement Clipped Value Function Loss:
    $$L^{\text{VF}}(\theta) = \frac{1}{2} \mathbb{E} \left[ (V_\theta(s_t) - R_t)^2 \right]$$
  - [ ] Implement Policy Entropy Bonus:
    $$S[\pi_\theta](s_t) = -\sum_{i=1}^9 \pi_\theta(a_i \mid s_t) \log \pi_\theta(a_i \mid s_t)$$
  - [ ] Combine total loss: $L_{\text{total}} = L^{\text{CLIP}} + 0.5 \cdot L^{\text{VF}} - 0.01 \cdot S[\pi]$.

- [ ] **Task 3.5: Training Telemetry & Live Checkpointing**
  - [ ] Track real-time metrics: turns/sec, actor loss, critic loss ($R^2$), policy entropy, clipping fraction.
  - [ ] Periodically save best model weights to `weights/amnesia_v2.pt`.
  - [ ] Maintain an archive of frozen historical checkpoints for evaluation tournaments.

---

## 🏆 Phase 4: Benchmarking & Tournament Evaluation (`src/evaluate.py`)

*Goal: Validate tactical competence against baseline bots and calculate empirical Glicko-1 rating progression.*

- [ ] **Task 4.1: Automated 500-Match Tournament vs `RandomPlayer`**
  - [ ] Run 500 independent matches with fixed random seeds.
  - [ ] Compute win rate and Wilson Score 95% Confidence Interval.
  - [ ] Gate: Achieve $\ge 99.0\%$ win rate (Milestone 2.1).

- [ ] **Task 4.2: Automated 500-Match Tournament vs `MaxBasePowerPlayer`**
  - [ ] Run 500 independent matches with fixed random seeds.
  - [ ] Compute win rate and Wilson Score 95% Confidence Interval.
  - [ ] Gate: Achieve $\ge 88.0\%$ win rate (Milestone 2.2).

- [ ] **Task 4.3: Automated 500-Match Tournament vs `SimpleHeuristicsPlayer`**
  - [ ] Run 500 independent matches with fixed random seeds.
  - [ ] Compute win rate and Wilson Score 95% Confidence Interval.
  - [ ] Gate: Achieve $\ge 65.0\%$ win rate (Milestone 2.3).

- [ ] **Task 4.4: Fictitious Play Historical Checkpoint League**
  - [ ] Maintain a pool of 10 frozen historical checkpoints.
  - [ ] Compute round-robin tournament matrix between current model and historical versions.
  - [ ] Calculate Glicko-1 rating progression across training epochs.

- [ ] **Task 4.5: Telemetry Dashboard Generation**
  - [ ] Generate publication-quality 3-panel dashboard:
    1. Win Rate vs Baselines across training iterations.
    2. Actor Loss & Critic Value Explained Variance ($R^2$).
    3. Policy Entropy smooth decay curve.

---

## 🌐 Phase 5: Live Showdown Ladder Client (`src/client.py`)

*Goal: Deploy Amnesia-AI v2 to the public Pokémon Showdown server and compete against human players.*

- [ ] **Task 5.1: Asynchronous WebSocket Showdown Ladder Client**
  - [ ] Connect directly to official Pokémon Showdown server (`wss://sim3.psim.us/showdown/websocket`).
  - [ ] Automated authentication assertion via `play.pokemonshowdown.com/action.php`.
  - [ ] Automatic reconnection and session keep-alive handling.

- [ ] **Task 5.2: Ranked Ladder Matchmaking Loop**
  - [ ] Enqueue automatically in ranked `gen7randombattle` / `gen8randombattle` matchmaking.
  - [ ] Automatically activate timer (`/timer on`) in every battle.
  - [ ] Real-time spectator links: print `https://play.pokemonshowdown.com/battle-...` to terminal.

- [ ] **Task 5.3: Match Replay Serializer & Analysis Logger**
  - [ ] Save full match replays and turn decisions locally to `data/replays/`.
  - [ ] Parse opponent ratings and track live Glicko-1 ladder progress.
  - [ ] Reach and sustain Target Milestone: **>1600 Glicko-1 rating** against human players.
