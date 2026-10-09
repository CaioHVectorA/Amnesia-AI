# CONTEXT.md — Deep Domain Context & Theoretical Foundations (v2)

This document provides the theoretical foundation, domain architecture, mathematical modeling, and engineering post-mortem for **Amnesia-AI v2**.

---

## 1. Post-Mortem: Why Amnesia-AI v1 Failed

Before engineering v2, we conducted a rigorous diagnostic of v1's failure mode to ensure no previous errors are repeated:

### 1.1 The "Toy Simulator" Trap (The Reality Gap)
In v1, we attempted to build a handcrafted battle engine (`Gen8FastSimEnvironment` / `mechanics.py`) in Python to maximize simulation speed (~250 turns/sec).  
**The Failure**: Competitive Pokémon is among the most mechanically nuanced games in existence. Handcrafted simulators miss entry hazard interactions, nuanced ability triggers (*Magic Bounce*, *Regenerator*, *Intimidate*, *Levitate*), secondary effect rates, item mechanics (*Choice items*, *Assault Vest*), and switch priority mechanics. The agent learned optimal policies for a simplified *toy game*, which became completely dysfunctional when exposed to the real Pokémon Showdown engine.

### 1.2 The "Flat Vector" Bottleneck
In v1, we reduced the entire battle state into a flat vector of **76 numerical floats**.  
**The Failure**: A 76-dimensional vector cannot represent categorical identities. The neural net had no way to distinguish whether a switch was into a defensive wall (*Ferrothorn*) or a frail sweeper (*Weavile*); it only saw normalized HP and speed. It was fundamentally blind to the semantic meaning of moves, abilities, and items.

### 1.3 The "Myopic Reward" Trap
In v1, we shaped rewards using immediate per-turn damage:
$$R_t = 1.5 \times \text{DamageDealt} - 1.0 \times \text{DamageTaken} + 3.0 \times \text{KO}$$
**The Failure**: In competitive Pokémon, many optimal moves deal zero damage (setting entry hazards, stat boosting, pivoting with *U-turn*, sacrificing a Pokémon to maintain momentum). By heavily rewarding immediate damage without deep episodic credit assignment, the v1 agent degenerated into a greedy bot that merely chased instant damage, unable to plan endgames.

### 1.4 Architectural Sprawl & Fragmented Focus
v1 was structured as a sprawling monorepo containing 10 loosely-connected subprojects:
- `projects/simulator` (Bun / TypeScript / `@pkmn/sim`)
- `projects/behavioral-cloning` (Python)
- `projects/type-classifier`
- `projects/moveset-recomendation`
- `projects/team-optimizer`
- `projects/showdown-agent` (Selenium + custom WebSockets)
**The Failure**: Effort was diluted across infrastructure plumbing instead of mastering the core decision-making policy.

---

## 2. Theoretical Grounding: The Huang & Lee (CoG 2019) Blueprint

Amnesia-AI v2 directly implements the architecture and methodology proven in:

> **"A Self-Play Policy Optimization Approach to Battling Pokémon"**  
> *Dan Huang & Scott Lee (IEEE Conference on Games 2019)*  
> *Achieved **1677 Glicko-1** rating on live Pokémon Showdown ladder without search trees or heuristics.*

```mermaid
flowchart TD
    subgraph Showdown Engine
        PS[Pokémon Showdown Server / poke-env]
    end

    subgraph State Encoding
        PS -->|Raw Battle Object| Extract[Feature Extractor]
        Extract --> Cat[Categorical Tokens: Species, Moves, Items, Abilities]
        Extract --> Cont[Continuous Stats: HP, Boosts, Status, Weather]
    end

    subgraph Neural Representation
        Cat --> Embed[128-d Entity Embeddings]
        Cont --> Concat1[State Concatenation]
        Embed --> Concat1
        Concat1 --> Pool[Permutation-Invariant Team Pooling]
    end

    subgraph Dual Heads
        Pool --> ActorHead[Actor: Action Scoring Matrix]
        Pool --> CriticHead[Critic: State Value V(s)]
    end

    subgraph Action Selection
        ActorHead --> Mask[Dynamic Action Masking]
        Mask --> Softmax[Action Probabilities π]
        Softmax --> Choose[Sampled Action: Move / Switch]
        Choose -->|WebSocket Action| PS
    end
```

---

## 3. Mathematical Formulation

### 3.1 Battle as a POMDP
A Pokémon battle is formalized as a Partially Observable Markov Decision Process:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

- $\mathcal{S}$: True hidden state (opponents' unrevealed movesets, exact EV spreads, abilities, and items).
- $\Omega$: Observation space visible to the player (current revealed moves, visible HP %, active status).
- $\mathcal{A}$: Discrete action space of size $n = 9$ (4 moves + 5 bench switches).
- $\mathcal{R}$: Terminal zero-sum reward with minimal auxiliary shaping:
  $$R_{\text{terminal}} = \begin{cases} +1.0 & \text{if battle won} \\ -1.0 & \text{if battle lost} \end{cases}$$
  Auxiliary shaping:
  $$r_{\text{faint}} = -0.0125 \quad (\text{own Pokémon faints}), \quad r_{\text{supereffective}} = +0.0025$$

### 3.2 Action Masking
Not every action is legal in every state (e.g., fainted Pokémon cannot be switched into; disabled moves cannot be executed).  
A dynamic binary mask $s \in \{0, 1\}^n$ is constructed for each state. The policy distribution $\pi$ is normalized strictly over legal actions:
$$\pi_i = \frac{s_i \cdot \exp(p_i)}{\sum_{j=1}^n s_j \cdot \exp(p_j)}$$
This guarantees **zero probability** of taking illegal actions without relying on heuristic fallbacks.

---

## 4. The 1.3-Million Parameter Embedding Architecture

Unlike flat-vector approaches, Huang & Lee (2019) represent game entities through **128-dimensional learned embedding spaces**:

### 4.1 Feature Dimensionality (Table I from paper)
| Feature | Type | Dimensionality | Description |
|---|---|---|---|
| `species` | Categorical | $1 \times 1023$ | Embedding layer for every Pokémon species |
| `item` | Categorical | $1 \times 368$ | Embedding layer for competitive items |
| `ability` | Categorical | $1 \times 238$ | Embedding layer for abilities |
| `moveset` | Categorical | $4 \times 731$ | Learned embeddings for known moves |
| `lastmove` | Categorical | $1 \times 731$ | Latest move executed by active Pokémon |
| `stats` | Continuous | 6 | HP, Atk, Def, SpA, SpD, Spe |
| `boosts` | Continuous | 6 | Stat stage modifiers ($-6$ to $+6$) |
| `hp` / `maxhp` | Continuous | 2 | Current and max hitpoints |
| `status` | Indicator | 28 | Sleep, Burn, Paralysis, Poison, Freeze, etc. |
| `types` | Indicator | 18 | Elemental typing binary flags |
| `volatiles` | Indicator | 23 | Leech Seed, Taunt, Substitute, Confusion, etc. |

### 4.2 Why Entity Embeddings Work
By learning a 128-dimensional embedding matrix for items, moves, and abilities:
- The network discovers latent semantic relationships (e.g., *Leftovers* and *Black Sludge* map close together in vector space; *Flamethrower* and *Fire Blast* share similar directional vectors).
- The network learns these dynamics purely from win/loss gradients without requiring millions of lines of hardcoded rule trees.

---

## 5. Training Paradigm: Pure Symmetric Self-Play

Following Algorithm 1 of Huang & Lee (2019):
1. **Initialize** network parameters $\theta_0$ randomly.
2. **Simulate** $m$ self-play matches using $f_{\theta_i}$ as both player 1 and player 2.
3. **Collect** all $2m$ trajectories (both perspectives provide valid transitions).
4. **Compute** Generalized Advantage Estimation (GAE) across full game trajectories:
   $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
   $$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
5. **Optimize** network parameters via PPO clipped surrogate objective:
   $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right] - c_1 L^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t)$$
6. **Iterate** indefinitely, periodically checkpointing and measuring rating progression against fixed baseline bots.
