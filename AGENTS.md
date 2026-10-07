# AGENTS.md — AI Agent Guidelines & Engineering Standards

Welcome to the **Amnesia-AI** project repository. This document defines the engineering standards, architecture patterns, development workflows, and behavioral expectations for AI agents contributing to this codebase.

---

## 1. Project Mission & Identity

**Amnesia-AI** is an artificial intelligence research and engineering initiative aimed at creating an autonomous agent that plays **Competitive Pokémon** (specifically via Pokémon Showdown) at a human-master level (targeting >1700 Elo / Top 500 ladder standing).

Competitive Pokémon presents unique challenges:
- **Imperfect information** (unrevealed movesets, items, abilities, EVs/IVs).
- **Simultaneous turns** with dynamic prediction and risk trade-offs.
- **Massive discrete state & action spaces** ($>10^{358}$ branching combinations).
- **Stochastic outcomes (RNG)** (accuracy checks, critical hits, secondary status triggers, damage rolls).

---

## 2. Monorepo Structure & Tech Stack

The repository is structured as a multi-package polyglot monorepo:

```
Amnesia-AI/
├── AGENTS.md                  # Instructions for AI agents (this file)
├── CONTEXT.md                 # Deep domain context & system architecture
├── TODO.md                    # Prioritized roadmap and task tracking
├── project.md                 # Original thesis & theoretical motivation
├── README.md                  # Project introduction
├── main.ipynb                 # Prototype Selenium web client for Showdown
├── data/                      # Static metadata (dex.json, moves.json)
└── projects/
    ├── simulator/             # TypeScript / Bun engine powered by @pkmn/sim & pokemon-showdown
    ├── type-classifier/       # Python scikit-learn model for Pokémon type inference
    ├── matchup-calculator/    # Pairwise matchup & advantage engine
    ├── moveset-recomendation/ # Smogon usage-based moveset prediction
    └── team-optimizer/        # Synergistic team builder & tier coverage analyzer
```

### Technology Matrix

| Subproject / Layer | Primary Language | Runtimes / Libraries | Purpose |
|---|---|---|---|
| **Simulator** (`projects/simulator`) | TypeScript (ESNext) | Bun, `@pkmn/sim`, `@pkmn/sets`, `@pkmn/randoms`, `pokemon-showdown` | Fast headless simulations, RL environments, protocol parsing |
| **ML / Analytics** (`projects/*`) | Python 3.10+ | Pandas, NumPy, Scikit-learn, PyTorch (planned), Matplotlib, Seaborn | Feature engineering, classification models, data scraping |
| **Protocol / Client** | TypeScript / Python | WebSockets (`ws`), Selenium (prototype) | Direct battle interaction with Pokémon Showdown servers |

---

## 3. Guiding Principles for AI Agents

When modifying or extending this codebase, adhere to the following principles:

### 3.1 Prefer WebSocket Protocol over Web Scraping (Selenium)
- Early prototypes used Selenium in `main.ipynb`. For all production and training tasks, prefer native **Showdown WebSocket protocol** or the headless stream interface in `@pkmn/sim`.
- Scraping DOM elements is fragile and too slow for large-scale RL training.

### 3.2 Strict Typing & Data Integrity
- In TypeScript modules, ensure complete strictness (`noImplicitAny`, exact interface definitions for battle states, player requests, and actions).
- Keep data schemas (`dex.json`, `moves.json`, CSV datasets) standardized with canonical Pokémon IDs (lowercase, alphanumeric without punctuation, e.g., `landorustherian`, `dragapult`).

### 3.3 Separation of Concerns
1. **Engine Layer**: Direct battle state execution and rule evaluation (`@pkmn/sim`).
2. **State / Representation Layer**: Translation of raw stream logs (`|move|`, `|switch|`, `|request|`) into normalized numerical tensors or strongly-typed state structs.
3. **Decision Layer**: Agents (Heuristic/Minimax, Random, Neural Network / MCTS, Reinforcement Learning Policy).
4. **I/O Layer**: WebSocket Showdown client or local battle stream runner.

### 3.4 Reproducibility and Headless Execution
- All scripts must run headlessly and support non-interactive CLI arguments or automated test runners.
- Seed random number generators (`PRNG` in Showdown simulator, `random_state` in scikit-learn / PyTorch) when writing deterministic unit and integration tests.

---

## 4. Development Workflows & Commands

### 4.1 TypeScript Simulator (`projects/simulator`)
- **Runtime**: [Bun](https://bun.sh/)
- **Install dependencies**:
  ```bash
  cd projects/simulator
  bun install
  ```
- **Run simulator / test scripts**:
  ```bash
  bun run src/index.ts       # Showdown battle stream sandbox
  bun run src/sim.ts         # Headless custom team simulation
  bun run src/sets.ts        # Team import/export packer
  ```

### 4.2 Python Analytics & Machine Learning
- **Virtual Environment**: Python 3.10+ recommended
- **Execute Type Classifier pipeline**:
  ```bash
  cd projects/type-classifier
  python seed_data.py        # Fetch latest pokedex data into pokedex.csv
  python show_graphic.py     # Generate statistical plots
  ```

---

## 5. Agent Specialized Roles & Personas

When tackling tasks in this repository, identify which persona is active:

1. **Simulator & Systems Engineer**:
   - Focus: Stream normalization, state decoding from Showdown protocol, Gymnasium-compatible RL environment wrappers, fast batch simulation.
   - Files: `projects/simulator/*`, `data/*`.

2. **Machine Learning / RL Researcher**:
   - Focus: Reward shaping, state vectorization, opponent moveset belief distribution, MCTS / Expectiminimax search, policy networks (PPO / DQN).
   - Files: `projects/team-optimizer/*`, `projects/moveset-recomendation/*`, `projects/matchup-calculator/*`.

3. **Domain & Data Specialist**:
   - Focus: Competitive metagame analysis (Smogon tiers, OU, Random Battles), damage calculation formula accuracy, dataset harvesting from Showdown logs.
   - Files: `data/*`, `projects/type-classifier/*`.

---

## 6. Code Style & Commit Conventions

- **Docstrings & Comments**: Maintain existing docstrings and comments. Explain non-obvious game mechanics (e.g., why a specific ability bypasses speed priority or damage rolls).
- **File Links**: When referencing code files or lines in markdown documentation, use relative links or clickable file links.
- **Commit Messages**: Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `perf:`).
