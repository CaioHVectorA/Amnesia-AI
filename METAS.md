# METAS.md — Quantitative Milestones & Success Criteria (v2)

This document establishes the measurable benchmarks, performance targets, and validation criteria for **Amnesia-AI v2**.

---

## 1. The North Star

> **Reproduce and exceed the IEEE CoG 2019 paper's benchmark**:  
> Achieve a sustained **>1600 Glicko-1 rating** on the official Pokémon Showdown `gen7randombattle` / `gen8randombattle` ladder, demonstrating human-master level competitive play purely via Reinforcement Learning and Self-Play without heuristic search trees.

---

## 2. Phased Quantitative Milestones

### Milestone 1: Engine Integrity & Zero-Error Pipeline
*Objective: Prove that the `poke-env` environment, 128-d state extractor, and action masker function without protocol or tensor exceptions.*

| Metric | Target | Measurement Method |
|---|---|---|
| **Action Validity** | **100.0%** | Zero illegal move errors (`|error|[Invalid choice]`) across 1,000 battles |
| **State Encoding Latency** | **< 1.5 ms / turn** | PyTorch tensor collation and feature extraction on CPU/GPU |
| **Simulation Throughput** | **> 120 turns / sec** | Parallel self-play matches executed via `poke-env` local/remote instances |
| **Mask Compliance** | **0% illegal choice probability** | Exact binary masking verification before softmax in `src/model.py` |

---

### Milestone 2: Baseline Bot Dominance
*Objective: Systematically surpass all standard benchmark agents provided in `poke-env`.*

| Opponent Agent | Target Win Rate | Training Horizon | Paper Reference (Table II) |
|---|---|---|---|
| **`RandomPlayer`** | **≥ 99.0%** | Within 50,000 matches | 99.5% in Huang & Lee 2019 |
| **`MaxBasePowerPlayer`** | **≥ 88.0%** | Within 250,000 matches | 92.9% in Huang & Lee 2019 |
| **`SimpleHeuristicsPlayer`** | **≥ 65.0%** | Within 1,000,000 matches | ~61.2% vs `pmariglia` tree-search |

*Evaluation Protocol: Evaluated over 500 independent matches per baseline with fixed random seeds.*

---

### Milestone 3: Convergence & Policy Stability (Self-Play)
*Objective: Ensure stable optimization dynamics during extended self-play training.*

| Training Diagnostic | Target Range | Interpretation |
|---|---|---|
| **Policy Entropy** | **0.45 – 0.65 nats** | Confident tactical execution without stochastic collapse |
| **Critic Loss (MSE)** | **< 0.08** (on $\pm 1.0$ scale) | Accurate board state advantage estimation |
| **PPO Policy Clip Ratio** | **< 15% clipped** | Stable policy updates without destructive distribution shifts |
| **Self-Play Win Rate** | **48% – 52%** | Symmetric equilibrium against current self |
| **Win Rate vs Historical Checkpoint** | **> 60%** | Continual strategic improvement over past generations |

---

### Milestone 4: Live Human Ladder Performance
*Objective: Deploy Amnesia-AI v2 to the live Pokémon Showdown server and compete against human players.*

| Milestone Level | Glicko-1 Rating | Human Win Rate | Status |
|---|---|---|---|
| **Tier 1: Competent** | **1300+** | > 50% vs Humans | Baseline competency |
| **Tier 2: Advanced** | **1450+** | > 60% vs Humans | Consistent tactical play |
| **Tier 3: Master (Paper Goal)** | **1600+** | **> 70% vs Humans** | **Primary Project Target** |
| **Tier 4: Grandmaster** | **1750+ (Top 500)** | > 75% vs Humans | Stretch Goal |

---

## 3. Verification Protocol

Each milestone is strictly verified before advancing:
1. **Automated Unit Tests**: Tensor shapes, embedding dictionary coverage, action masking math.
2. **Head-to-Head Evaluation Tournaments**: 500-match batches against `RandomPlayer` and `MaxBasePowerPlayer`.
3. **Live Replay Logging**: Every ladder match must serialize raw logs and action choices to disk for human inspection.
