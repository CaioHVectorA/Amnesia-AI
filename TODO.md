# TODO.md — Amnesia-AI Roadmap & Task Backlog

This backlog tracks the development roadmap of **Amnesia-AI**, prioritized from foundational infrastructure to high-level model architectures and competitive ladder deployment.

---

## Roadmap Summary & Progress

- [ ] **Phase 1**: Simulation Engine & State Parser *(In Progress)*
- [ ] **Phase 2**: State Encoding & Feature Pipeline *(Next Up)*
- [ ] **Phase 3**: Predictive Sub-Models (Moveset, Matchup, Classifier)
- [ ] **Phase 4**: Agent Decision Architectures (Heuristic $\rightarrow$ Search $\rightarrow$ RL)
- [ ] **Phase 5**: Showdown Client Integration & WebSocket Bot
- [ ] **Phase 6**: Evaluation, Telemetry & Ladder Progression

---

## 1. Phase 1: Simulation Engine & State Parser (`projects/simulator`)

- [x] Integrate `@pkmn/sim` and `pokemon-showdown` with Bun runtime.
- [x] Implement team parsing and packing using `@pkmn/sets` (`src/sets.ts`).
- [x] Build multi-stream execution test bench (`src/sim.ts`).
- [ ] **Stream Normalization & Type Definitions** `[HIGH PRIORITY]`:
  - [ ] Create TypeScript types for all Showdown battle protocol messages (`|request|`, `|move|`, `|switch|`, `|-damage|`, `|-boost|`, `|-weather|`, `|-fieldstart|`, `|faint|`).
  - [ ] Write a stream state accumulator that turns a raw battle log stream into a deterministic `BattleState` object.
  - [ ] Correctly parse `|request|` JSON payloads to determine legal moves, legal switches, and forced trapped/struggle states.
- [ ] **Gymnasium / RL Environment Wrapper** `[HIGH PRIORITY]`:
  - [ ] Implement an OpenAI Gym / Gymnasium compatible API (`reset()`, `step(action)`, `render()`).
  - [ ] Define standardized discrete action space (Moves 1–4, Switches 1–6, Mega/Z/Dynamax/Tera flags).
  - [ ] Support self-play battle environments (Agent vs. Agent, Agent vs. RandomPlayerAI, Agent vs. Baseline Heuristic).
- [ ] **Performance Benchmarking**:
  - [ ] Profile and achieve $\ge 1,000$ battles/sec for headless self-play rollouts.

---

## 2. Phase 2: State Representation & Feature Engineering

- [ ] **Numerical Vectorization / Tensor Encoding** `[HIGH PRIORITY]`:
  - [ ] Active Pokémon features: Current HP %, Status condition, Boost stages (-6 to +6 for Atk, Def, SpA, SpD, Spe, Acc, Eva), Known moves & PP, Types.
  - [ ] Bench Pokémon features: Alive/Fainted status, HP %, Known moves, Item status (knocked off vs active).
  - [ ] Field & Global state: Active weather & turns remaining, Active terrain & turns remaining, Entry hazards on each side (Stealth Rock, Spikes count, Toxic Spikes, Sticky Web), Screens (Reflect, Light Screen, Aurora Veil), Trick Room / Tailwind.
  - [ ] Opponent belief state: Encoded probabilistic distribution over unrevealed Pokémon, items, and moves.
- [ ] **Embedding Layer**:
  - [ ] Entity embeddings for Pokémon species, types, moves, items, and abilities.

---

## 3. Phase 3: Analytical Sub-Models

### 3.1 Type Classifier (`projects/type-classifier`)
- [x] Seed pokedex dataset (`seed_data.py`).
- [x] MultiLabel binarization of abilities, exploratory analysis, and baseline Random Forest classifier.
- [ ] Refactor notebook into clean, repeatable training CLI script (`train.py`, `model.pkl`).
- [ ] Evaluate model on Smogon CAP (Create-A-Pokémon) and community Fakemons.
- [ ] Add dual-type multi-label classification (predicting both Primary and Secondary types).

### 3.2 Matchup Calculator (`projects/matchup-calculator`)
- [ ] Port/integrate `@smogon/calc` damage calculation library for deterministic min/max damage roll computation.
- [ ] Build a fast pairwise matchup advantage lookup table (offensive/defensive matchup score between any two Pokémon).
- [ ] Speed tier analysis (which Pokémon outspeeds considering EVs, natures, Choice Scarf, and speed boosts).

### 3.3 Moveset Recommendation (`projects/moveset-recomendation`)
- [ ] Scrape Smogon monthly format usage statistics (Chaos / JSON dumps from `smogon.com/stats`).
- [ ] Build a Bayesian prior or Neural predictor: Given an opponent Pokémon in format $F$, output probabilities for:
  - Top 4 moves.
  - Held item (e.g., Choice Specs vs Choice Scarf vs Leftovers).
  - Ability (e.g., Static vs Lightning Rod).
  - EV spread archetype (Offensive vs Bulky/Defensive).

### 3.4 Team Optimizer (`projects/team-optimizer`)
- [ ] Calculate defensive weakness/resistance overlap matrix for a 6-Pokémon roster.
- [ ] Role compression metric (Hazards + Hazard Removal + Physical Wall + Special Wall + Wincon / Sweeper).
- [ ] Suggest optimal 6th Pokémon to complete an incomplete core.

---

## 4. Phase 4: Battle Agent Decision Architectures

### 4.1 Tier 1: Rule-Based & Heuristic Agent
- [ ] Implement a damage-maximizing greedy agent (chooses highest expected damage move).
- [ ] Add basic switch heuristics (switch out if facing 4x weakness or negligible damage output; switch into resistant wall).
- [ ] Add hazard and setup logic (prioritize Stealth Rock turn 1 if opponent has high rock weakness).

### 4.2 Tier 2: Search-Based Agent (Expectiminimax / Monte Carlo Tree Search)
- [ ] Implement Expectiminimax tree search considering simultaneous turn matrices and damage roll probability distributions.
- [ ] Implement MCTS with belief state sampling (determinization for hidden information).
- [ ] Pruning: Limit branch expansion to top-3 viable moves and sensible switches.

### 4.3 Tier 3: Reinforcement Learning & Deep Neural Network
- [ ] Policy & Value Network architecture using PyTorch / ONNX Runtime.
- [ ] **Behavioral Cloning (BC) Offline Pre-training Pipeline** `[HIGH PRIORITY]`:
  - [ ] Treinamento supervisionado com ponderação amostral por Elo ($w$) sobre os replays indexados em SQLite.
  - [ ] Implementação de DAgger e IQL (Implicit Q-Learning) para mitigar erro composto e estados fora da distribuição.
  - [ ] Avaliação de métricas de acurácia Top-1 e Top-3 frente a jogadas de especialistas humanos ($\ge 1800$ Elo).
- [ ] **Model Merging & Checkpoint Blending Engine**:
  - [ ] Implementação de utilitário de fusão de pesos via SLERP e Linear Weight Averaging.
  - [ ] Merge de especialistas por fase de jogo (Early Game vs Late Game) e por formato (`Random Battles` vs `OU`).
  - [ ] Roteamento dinâmico via Mixture of Experts (MoE) baseado no número de Pokémon vivos e hazards em campo.
- [ ] Deep Q-Learning (DQN) / Proximal Policy Optimization (PPO) training loop against self-play pools.
- [ ] Opponent modeling module adapting policy based on observed opponent playstyle (aggressive vs stall/conservative).

---

## 5. Phase 5: Showdown Client & Infrastructure

- [x] Prototype Selenium-based automation (`main.ipynb`).
- [ ] **Native WebSocket Showdown Bot** `[HIGH PRIORITY]`:
  - [ ] Implement lightweight WebSocket client connecting to `sim3.psim.us:80/showdown/websocket`.
  - [ ] Automated authentication / challenge acceptance via Showdown action commands (`/challenge`, `/accept`, `/join`).
  - [ ] Automatic reconnection, rate-limit adherence, and anti-ban safeguards.
- [ ] **Battle Replay Recorder**:
  - [ ] Export played battle replays and logs in JSON format for offline dataset training and retrospective analysis.

---

## 6. Phase 6: Evaluation & Ladder Milestones

- [ ] **Benchmark vs RandomPlayerAI**: Achieve $>95\%$ win rate.
- [ ] **Benchmark vs Smogon Baseline Bots** (e.g. standard rule-based Showdown bots): Achieve $>75\%$ win rate.
- [ ] **Random Battles Ladder Deployment**:
  - [ ] Reach 1300 Elo.
  - [ ] Reach 1500 Elo.
  - [ ] Reach 1700+ Elo.
- [ ] **OverUsed (OU) Ladder Deployment**:
  - [ ] Reach Top 500 leaderboard.
