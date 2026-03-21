# Data Generation: Methodology vs. This Repo

Comparison between the data generation approach described in the target methodology paper and the actual implementation in this repo.

---

## Overview

- **Layer 1 (deterministic TSPTW instances):** Both use the official PIP (Bi et al., NeurIPS 2024) data generator as the backbone — `POMO+PIP/envs/TSPTWEnv.py` is identical to the PIP-constraint repo, so Layer 1 matches exactly.

- **Layer 2 (stochastic overlay):** The methodology uses Gamma and two-point distributions with explicit CV control. This repo's v1 env (`STSPTWEnv`) instead uses a time- and distance-dependent lognormal-like delay controlled by `delay_scale`. The v2 env (`STSPTWEnv_v2`) matches the methodology design.

- **Feasibility certification:** The methodology filters instances into Regime A/B/C using Dumas DP + Monte Carlo. This repo has no such pre-certification — all generated instances are used for training and evaluation as-is.

---

## Layer 1: Instance Backbone (TSPTW)

### Node coordinates

Both sample n nodes uniformly from `[0, 1]^2` (this repo internally samples from `[0, 100]^2` and normalizes, yielding the same distribution).

### Mean travel time

Both use Euclidean distance: `mu_ij = ||x_i - x_j||_2` (speed = 1.0).

### Time window generation

This repo uses the same code as the official PIP-constraint repo:

| Hardness | Method | Window width |
|---|---|---|
| hard | Da Silva & Urrutia (2010) / Cappart hybrid-cp-rl — random feasible tour, TW sampled around cumulative distances | Tight |
| medium | JAMPR (Falkner) style — `dura_region = [0.1, 0.2]` | Narrow |
| easy | JAMPR style — `dura_region = [0.5, 0.75]` | Wide |

---

## Layer 2: Stochastic Overlay

### Methodology design (implemented in STSPTWEnv_v2)

Mean-preserving noise applied directly to travel times, with CV as the explicit control variable:

- **Gamma** (Taş et al., EJOR 2014):
  `t_ij ~ Gamma(k, mu_ij/k)`, where `k = 1/CV²`
  → `E[t] = mu_ij`, `CV(t) = CV`
  CV sweep: {1.0, 0.5, 0.25} (k = {1, 4, 16}); k→∞ recovers deterministic TSPTW.

- **Two-point** (Zhang et al.):
  `t_ij = mu_ij * (1 - delta)` with prob p, else `mu_ij * (1 + epsilon)` (epsilon chosen to preserve mean).
  Fixed `delta=0.3, p=0.5`.

### This repo's v1 design (STSPTWEnv)

Additive, time-dependent delay on top of deterministic travel:

```
t_ij = d_ij + delay_scale * base_delay(d_ij, current_time) * lognormal_factor(current_time)
```

- `base_delay`: two Gaussian rush-hour peaks at 30% and 70% of the time horizon, multiplied by a saturating distance factor `1 - exp(-d/0.5)`
- `lognormal_factor`: `exp(N(mu, sigma²))` where sigma inflates during rush hours
- **Not mean-preserving** — delay is always positive, so E[t] > d_ij
- **State-dependent** — noise magnitude depends on current_time
- Effective CV ≈ 0.03–0.12 (much lower than v2)

---

## Feasibility Certification and Regimes

### Methodology

Instances are classified into three regimes:

| Regime | Deterministic feasibility | Stochastic feasibility |
|---|---|---|
| A | Feasible | At least one policy satisfies chance constraints |
| B | Feasible | No policy can satisfy chance constraints |
| C | Infeasible | — |

Classification uses Dumas et al. (1995) DP for deterministic feasibility, and Monte Carlo simulation of the best deterministic tour for stochastic feasibility. **Evaluation uses only Regime A instances**, isolating model quality from inherent instance infeasibility.

### This repo

No Dumas DP, no regime classification, no Monte Carlo pre-certification. All generated instances are used directly. This means a fraction of evaluated instances may be inherently infeasible (Regime B/C) — infeasibility rates reported in results reflect both model limitations and instance difficulty.
