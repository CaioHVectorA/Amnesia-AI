# CONTEXT.md — Amnesia-AI Architectural & Domain Context

---

## 1. Executive Summary

**Amnesia-AI** is an AI research codebase designed to develop, evaluate, and deploy intelligent agents capable of mastering **Competitive Pokémon Battles** on the Pokémon Showdown platform.

The end goal is an autonomous battle engine capable of achieving a competitive ladder rating:
- **Top 500 (~1800+ Elo)**: Master / superhuman tier.
- **1700 Elo**: High competitive standard.
- **1500 Elo**: Baseline competent player.

---

## 2. Competitive Pokémon as an AI Problem

Pokémon is a turn-based, two-player zero-sum game that combines elements of imperfect information (like Poker) and high combinatoric branching (like Chess or Go), with stochastic mechanics.

### Key Complexity Drivers

```
+-------------------------------------------------------------------------+
|                          COMPLEXITY FACTORS                             |
+------------------------------------+------------------------------------+
|  1. State Space Explosion          |  2. Simultaneous Moves             |
|     - >1000 Pokémon species        |     - Turn actions chosen secretly |
|     - >800 Moves, >300 Abilities   |     - Yomi / Double-switches       |
|     - Items, EVs, IVs, Natures     |     - Dynamic payoff matrix        |
+------------------------------------+------------------------------------+
|  3. Imperfect Information          |  4. Stochastic RNG Dynamics        |
|     - Hidden opponent movesets     |     - Move accuracy (e.g. Focus    |
|     - Hidden items & abilities     |       Blast 70% vs Moonblast 100%) |
|     - Unrevealed bench Pokémon     |     - Critical hits & damage rolls |
|     - Requires belief modeling     |     - Status procs (freeze/paralyze)|
+------------------------------------+------------------------------------+
|  5. Metagame & Team Dynamics       |  6. Battle Modifiers               |
|     - Roles: Sweeper, Wall, Pivot  |     - Weather (Sun, Rain, Sand)    |
|     - Synergy & Defensive Cores    |     - Terrains & Entry Hazards     |
|     - Tier-specific viability      |     - Gimmicks (Tera, Mega, Z)     |
+------------------------------------+------------------------------------+
```

---

## 3. System Architecture & Component Landscape

Amnesia-AI is structured into modular subprojects separating data engineering, headless battle simulation, specialized sub-models, and the active playing agent:

```mermaid
flowchart TD
    subgraph Data Layer
        D1[dex.json & moves.json]
        D2[Smogon Usage Stats / Replays]
    end

    subgraph Analytics & Sub-Models
        M1[type-classifier<br/><i>Type & Trait Prediction</i>]
        M2[matchup-calculator<br/><i>Damage & Advantage Matrix</i>]
        M3[moveset-recomendation<br/><i>Belief Distribution on Sets</i>]
        M4[team-optimizer<br/><i>Synergy & Coverage Builder</i>]
    end

    subgraph Simulation & Environment
        S1[projects/simulator<br/><i>@pkmn/sim + Bun Engine</i>]
        S2[State Encoder / Tensor Normalizer]
        S3[Gymnasium / RL Environment Interface]
    end

    subgraph Decision Engine / Agents
        A1[Heuristic / Rule-based Agent]
        A2[Expectiminimax / MCTS Search Agent]
        A3[Neural Policy / Value Network (PPO/DQN)]
    end

    subgraph Interaction Layer
        I1[Showdown WebSocket Client]
        I2[Headless Local Battle Stream]
    end

    D1 --> S1
    D2 --> M3
    D1 --> M1
    M1 & M2 & M3 --> S2
    S1 --> S2 --> S3
    S3 --> A1 & A2 & A3
    A1 & A2 & A3 --> I1 & I2
    I1 & I2 --> S1
```

> [!NOTE]
> Para aprofundamento nos pipelines de algoritmos, consulte os guias em [`docs/`](docs/README.md):
> - **[docs/behavioral_cloning.md](docs/behavioral_cloning.md)**: Formulação matemática do BC, DAgger, IQL e Decision Transformers.
> - **[docs/model_merging.md](docs/model_merging.md)**: Fusão de checkpoints (SLERP, TIES, Task Arithmetic) e Mixture of Experts (MoE).
> - **[docs/state_representation.md](docs/state_representation.md)**: Especificação formal da vetorização e Action Masking.
> - **[docs/battle_protocol.md](docs/battle_protocol.md)**: Parsing determinístico de mensagens e ciclo de vida Showdown.

---

## 4. Detailed Subprojects Breakdown

### 4.1 `projects/simulator/` (Headless Simulation Engine)
- **Tech**: TypeScript, Bun, `@pkmn/sim`, `@pkmn/sets`, `@pkmn/randoms`.
- **Purpose**: High-throughput headless Pokémon battle simulator allowing thousands of games per second for heuristic evaluation, rollouts, or reinforcement learning.
- **Key Modules**:
  - `src/sim.ts`: Simulates full games using `@pkmn/sim` BattleStreams between custom teams (e.g., Sun vs. Rain).
  - `src/sets.ts`: Uses `@pkmn/sets` to parse Showdown exportable text format and pack into fast compact strings.
  - `src/index.ts`: Low-level stream reader interfacing with `pokemon-showdown` battle protocol messages (`|request|`, `|move|`, `|switch|`, `|faint|`).
  - `src/plan.md`: Roadmap for stream normalization, logging, and typing.

### 4.2 `projects/type-classifier/` (Pokémon Feature Learning)
- **Tech**: Python 3, Pandas, Scikit-Learn (Random Forest, SGD), Matplotlib, Seaborn.
- **Purpose**: Machine learning model that predicts/classifies a Pokémon's primary type based on physical attributes, base stats, ability distributions, evolutionary capability (`canEvo`), and egg groups.
- **Key Files**:
  - `seed_data.py`: Fetches and extracts full pokedex data from Pokémon Showdown into `pokedex.csv`.
  - `main.ipynb`: Exploratory data analysis, multi-label binarization of abilities, feature encoding, and Random Forest classifier training.
  - `show_graphic.py`: Visualizations of mean base stats across primary Pokémon types.

### 4.3 `projects/matchup-calculator/` (Advantage & Matchup Engine)
- **Status**: Planning / Initial specification (`todo.md`).
- **Purpose**: Compute offensive/defensive matchup metrics between two given Pokémon or team combinations (type chart multipliers, speed tier comparison, STAB damage potential, survival odds).

### 4.4 `projects/moveset-recomendation/` (Belief & Prediction Engine)
- **Status**: Planning / Inception (`.gitkeep`).
- **Purpose**: Given unrevealed or partially revealed opponent Pokémon, build a probabilistic prior over possible moves, items, and abilities based on Smogon format usage statistics.

### 4.5 `projects/team-optimizer/` (Team Builder & Evaluator)
- **Status**: Inception (`todo.md`).
- **Purpose**: Assemble competitive 6-Pokémon teams with balanced defensive cores, offensive coverage, hazard control, speed control, and high synergy scores.

### 4.6 Root Components
- `project.md`: The foundational research proposal outlining the AI's goals, complexity analysis, and ladder benchmarks.
- `main.ipynb`: Early prototype utilizing Selenium WebDriver to automate logging into `play.pokemonshowdown.com`, accepting challenges, and making random moves via DOM inspection.
- `data/dex.json` & `data/moves.json`: Canonical Pokémon Showdown Pokédex and Move data tables.

---

## 5. Pokémon Showdown Protocol Essentials

Communication with Pokémon Showdown occurs via a line-delimited message protocol over standard streams or WebSockets:

| Protocol Directive | Description | Example Payload |
|---|---|---|
| `|request|` | Sent to a player when it is their turn to act. Contains active moves with PP/disabled state, and bench switch choices. | `|request|{"active":[{"moves":[...]}],"side":{...}}` |
| `|move|` | Notification that a Pokémon used a move. | `|move|p1a: Dragapult|Shadow Ball|p2a: Ferrothorn` |
| `|switch|` | Notification that a Pokémon was switched in with current HP/status. | `|switch|p2a: Pelipper|Pelipper, L82, M|100/100` |
| `|-damage|` / `|-heal|` | HP changes following an attack or passive effect. | `|-damage|p2a: Ferrothorn|64/100` |
| `|-weather|` / `|-fieldstart|` | Global weather or terrain alterations. | `|-weather|RainDance|[from] ability: Drizzle` |
| `|faint|` | Triggered when a Pokémon's HP reaches 0. | `|faint|p1a: Dragapult` |
| `|win|` | Signals end of battle and the winning player. | `|win|Alice` |

### Action Output Format
- Move command: `>p1 move <move_slot>` (e.g. `>p1 move 1` or `>p1 move shadowball`)
- Switch command: `>p1 switch <slot>` (e.g. `>p1 switch 3`)
- Special mechanic: `>p1 move 1 mega`, `>p1 move 1 dynamax`, `>p1 move 1 terastallize`

---

## 6. Target Metagame Formats

1. **Generation 8/9 Random Battles (`[gen8randombattle]`, `[gen9randombattle]`)**:
   - Random teams with level balancing.
   - Ideal starting environment because team-building complexity is eliminated and movesets are constrained by known generation sets.
2. **Standard OverUsed (`[gen9ou]`, `[gen8ou]`)**:
   - 6v6 tier with pre-built teams.
   - Requires full metagame knowledge, team synergy, hazard control, and setup sweeping strategies.
