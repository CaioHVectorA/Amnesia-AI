# METAS.md — Quantitative Milestones, Rating Targets & Verification Framework (v2)

This document establishes the rigorous quantitative benchmarks, statistical verification protocols, failure mitigation triggers, and success criteria for **Amnesia-AI v2**.

---

## 1. The Project North Star

> **Reproduce and exceed the IEEE CoG 2019 reference benchmark**:  
> Achieve a sustained **>1600 Glicko-1 rating** on the official Pokémon Showdown `gen7randombattle` / `gen8randombattle` ladder, proving human-master competitive performance purely via Reinforcement Learning and Symmetric Self-Play without search trees or hardcoded heuristics.

---

## 2. Statistical Framework & Evaluation Metrics

To eliminate evaluation illusions (such as the small-sample noise that plagued v1), all quantitative evaluations in v2 are bound to formal statistical criteria:

### 2.1 Wilson Score Confidence Intervals
For any win-rate metric $\hat{p}$ evaluated over $N$ matches, performance is certified using the **Wilson Score Interval** at a $95\%$ confidence level:
$$\text{CI}_{95\%} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}} \quad (z = 1.96)$$
* **Evaluation Standard**: Standard benchmarks must run across **$N = 500$ independent battles**, bounding the margin of error to $\le \pm 4.3\%$.
* A win rate of $60\%$ over 30 matches (v1 standard) has a noisy margin of error of $\pm 17.5\%$ and is **statistically invalid** for certifying milestone progression.

### 2.2 Glicko-1 Rating Framework
Matchmaking skill is measured via the Glicko-1 rating system ($r, RD$), where $r$ denotes skill estimate and $RD$ denotes rating deviation (uncertainty):
* Default initial rating: $r_0 = 1500$, $RD_0 = 350$.
* A milestone is officially achieved only when:
  $$r_{\text{bot}} - 2 \cdot RD_{\text{bot}} \ge \text{Target Rating}$$
  (i.e., the agent's conservative lower bound exceeds the threshold with $95\%$ certainty).

---

## 3. Phased Quantitative Milestones

### Milestone 1: Engine Integrity & Zero-Error Pipeline
*Objective: Prove that the `poke-env` environment, 128-d state extractor, and action masker function without protocol, serialization, or tensor exceptions.*

| Metric | Target | Failure Threshold | Measurement Method |
|---|---|---|---|
| **Action Validity** | **100.0%** | $< 100.0\%$ (Zero tolerance) | Zero `|error|[Invalid choice]` across 1,000 matches |
| **State Encoding Latency** | **< 1.5 ms / turn** | $> 5.0$ ms / turn | Python `time.perf_counter()` on CPU batch |
| **CUDA Tensor Transfer** | **< 0.5 ms / turn** | $> 2.0$ ms / turn | PyTorch CUDA event timer |
| **Simulation Throughput** | **> 120 turns / sec** | $< 50$ turns / sec | Parallel self-play matches executed via `poke-env` |
| **Mask Compliance** | **$0.000$ probability** | $> 10^{-7}$ probability | Softmax output on illegal action indices |

---

### Milestone 2: Baseline Bot Dominance
*Objective: Systematically surpass all standard benchmark bots provided by `poke-env` and the reference literature.*

| Opponent Agent | Target Win Rate ($N=500$) | Literature Benchmark (Huang & Lee 2019 Table II) | Tactical Capability Certified |
|---|---|---|---|
| **`RandomPlayer`** | **≥ 99.0%** (CI: $[97.8\%, 99.8\%]$) | 99.5% (995 wins / 5 losses) | Basic move selection, zero self-sabotage |
| **`MaxBasePowerPlayer`** | **≥ 88.0%** (CI: $[84.8\%, 90.6\%]$) | 92.9% (929 wins / 71 losses) | Type matchups, defensive switching, threat awareness |
| **`SimpleHeuristicsPlayer`** | **≥ 65.0%** (CI: $[60.7\%, 69.1\%]$) | ~61.2% vs `pmariglia` tree-search | Multi-turn planning, hazard control, momentum |

---

### Milestone 3: Convergence & Policy Stability (Self-Play Dynamics)
*Objective: Ensure mathematically stable optimization during extended symmetric self-play.*

| Training Diagnostic | Target Range | Ideal Trajectory | Anomaly Indicator |
|---|---|---|---|
| **Policy Entropy** | **0.45 – 0.65 nats** | Smooth exponential decay from $\ln(9) \approx 2.19$ | Entropy $< 0.20$ (premature collapse) |
| **Critic Loss (MSE)** | **< 0.08** (scale $[-1, +1]$) | Decreasing toward $\approx 0.04$ | Divergence above $> 0.25$ |
| **Value Explained Variance ($R^2$)** | **$R^2 > 0.55$** | Increasing toward $0.70$ | $R^2 < 0.20$ (Critic failure) |
| **PPO Policy Clip Ratio** | **< 15%** of transitions | Bounded between $3\%$ and $12\%$ | $> 25\%$ (policy steps too aggressive) |
| **Symmetric Self-Play Win Rate** | **48.0% – 52.0%** | Stable oscillation around $50.0\%$ | Drift beyond $45\% - 55\%$ |
| **Win Rate vs Past Checkpoints** | **> 60.0%** | Transitive dominance over older models | $< 45\%$ (catastrophic forgetting / loops) |

---

### Milestone 4: Live Human Ladder Performance
*Objective: Deploy Amnesia-AI v2 to the live Pokémon Showdown server and compete against real human players.*

| Milestone Level | Target Glicko-1 | Human Win Rate | Ladder Standing Equivalent |
|---|---|---|---|
| **Tier 1: Competent** | **1300+** | > 50% vs Humans | Average human ladder player |
| **Tier 2: Advanced** | **1450+** | > 60% vs Humans | Above-average competitive player |
| **Tier 3: Master (Paper Goal)** | **1600+** | **> 70% vs Humans** | **Top 5% Competitive Ladder (Published Paper Target)** |
| **Tier 4: Grandmaster** | **1750+** | > 75% vs Humans | Top 500 Global Ladder |

---

## 4. Failure Mode Detection & Mitigation Matrix

If an anomaly is detected during development or training, execute the corresponding mitigation protocol:

| Failure Symptom | Root Cause Diagnostic | Immediate Engineering Mitigation |
|---|---|---|
| **Illegal Move Exception (`|error|`)** | Masking was not applied prior to softmax or index mapping misaligned. | 1. Verify `logits.masked_fill(~mask, -1e8)`.<br>2. Unit-test action slot mapping against `battle.available_moves` and `battle.available_switches`. |
| **Premature Policy Collapse (Entropy < 0.20)** | Entropy regularization coefficient $c_2$ too low; learning rate too high. | 1. Increase entropy bonus coefficient $c_2$ from $0.01$ to $0.02$.<br>2. Halve policy learning rate to $\text{lr} = 1.0 \times 10^{-4}$. |
| **Critic Loss Exploding ($R^2 < 0$)** | GAE return targets scale mismatched or learning rate too high for Critic. | 1. Normalize advantage estimates $(\hat{A} - \mu)/\sigma$.<br>2. Ensure value predictions are clipped or bounded in $[-1.0, +1.0]$. |
| **Self-Play Cycling (Rock-Paper-Scissors Loop)** | Agent overfits to its most recent checkpoint and unlearns prior counters. | 1. Introduce **Fictitious Self-Play**: sample $20\%$ of training opponents from historical checkpoint pool.<br>2. Evaluate regularly against frozen baselines to detect regressions. |
| **Reality Gap against Humans** | Discrepancy in handling timer, simultaneous choice resolution, or team preview. | 1. Audit `src/client.py` protocol handling.<br>2. Inspect human replay logs to identify tactical edge cases. |
