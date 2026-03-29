# S-PIP Logic and Integration

This document is **self-contained**: it explains the key logic of S-PIP (Stochastic Proactive Infeasibility Prevention) and how it is integrated into this codebase. It does not assume familiarity with external papers.

---

## 1. Overview

**S-PIP** extends TSP with Time Windows (TSPTW) and **PIP** (Proactive Infeasibility Prevention) with:

- **Optional stochastic transitions:** realized travel time = distance + bounded noise ξ, so training and evaluation can account for travel-time uncertainty.
- **Path-level buffered PIP masking:** when computing which nodes are still feasible, a safety buffer proportional to path length is added so that time-window feasibility holds with high probability under that uncertainty.

In this repo:

- **Problem type:** `TSPTW_SPIP`
- **Environment class:** `TSPTWEnv_SPIP` in [src/envs/TSPTWEnv_SPIP.py](src/envs/TSPTWEnv_SPIP.py)

The base PIP method is from the NeurIPS 2024 paper *Learning to Handle Complex Constraints for Vehicle Routing Problems*; this fork adds the stochastic transition and path-level buffer logic above.

---

## 2. Stochastic transition (travel-time noise)

### Idea

In the deterministic setting, moving from the current node to the selected node uses travel time = Euclidean distance / speed. In **stochastic mode**, the **realized** travel is distance + a bounded noise term ξ. The agent observes and is rewarded with respect to this realized travel, so it learns to be robust to travel-time uncertainty.

### Formulas

- **Deterministic:**  
  `realized_length = d` (Euclidean distance), `travel_time = d / speed`.

- **Stochastic:**  
  `realized_length = d + ξ`, with ξ bounded and scale proportional to √d:
  - **Uniform:** ξ ∈ [-b√d, +b√d], where `b` is `spip_noise_bound`.
  - **Clipped Gaussian:** ξ ~ N(0, σ₀²d), then clipped to ±b√d; σ₀ is `spip_sigma0`, b is `spip_noise_bound`.

Distance `d` is clamped away from zero (e.g. min 1e-8) before computing scale to avoid degenerate noise.

### Pseudocode: bounded noise sampling

```
function SampleBoundedNoise(distance):
    d ← max(distance, 1e-8)
    scale ← sqrt(d)
    if noise_dist == "uniform" then
        half_width ← bound * scale
        ξ ← (random_uniform(0,1) * 2 - 1) * half_width
    else if noise_dist == "clipped_gaussian" then
        σ ← sigma0 * scale
        ξ ← random_normal(0, σ²)
        ξ ← clip(ξ, -bound*scale, +bound*scale)
    return ξ
```

### Pseudocode: step with stochastic transition

```
function Step(selected):
    current_coord ← node_xy[selected]
    new_length ← ||current_coord - current_coord_prev||₂
    if stochastic_transition then
        ξ ← SampleBoundedNoise(new_length)
        new_length ← max(new_length + ξ, 1e-8)
    length ← length + new_length
    travel_time ← new_length / speed
    cumulative_realized_travel_time ← cumulative_realized_travel_time + travel_time
    current_time ← max(current_time + travel_time, tw_start[selected]) + service_time[selected]
    ... (update masks, PIP, reward when done)
```

### Code

- **Noise sampling:** [src/envs/TSPTWEnv_SPIP.py](src/envs/TSPTWEnv_SPIP.py) — `_sample_bounded_noise(distance)` (lines 129–145). It returns ξ with the same shape/device/dtype as `distance`.
- **Application in step:** In `step()`, after computing `new_length` (Euclidean segment length), if `spip_stochastic_transition` is True and noise is enabled, the env samples ξ via `_sample_bounded_noise(new_length)`, sets `realized_length = (d_ij + ξ).clamp(min=1e-8)`, and uses that for `new_length` (lines 348–353). That value is then used to update `self.length` and `travel_time_for_step`, and accumulated in `cumulative_realized_travel_time`.

### Reward

At episode end, the distance component of the reward is **negative cumulative realized travel time** (i.e. negative `cumulative_realized_travel_time`). So both deterministic and stochastic runs minimize the same quantity: total realized travel time.

---

## 3. PIP mask with S-PIP path-level buffer

### Idea

PIP marks nodes that cannot be reached in time (infeasible). **S-PIP** adds a **path-level buffer** so that, under uncertain travel, we still satisfy time windows with high probability. The buffer is added to arrival-time estimates when deciding feasibility.

### Path buffer (first-step)

- **Path length to candidate j:**  
  L_j = current path length + d_ij (distance from current node to j), clamped away from zero.  
  In code: `L_k = self.length`, `L_j = (L_k.unsqueeze(-1) + d_ij).clamp(min=1e-8)` (shape batch × pomo × problem).

- **Buffer:**  
  δ_path = z · σ₀ · √L_j  
  where z = Φ⁻¹(1 − ε) is the standard normal quantile for confidence (1 − ε). ε is `spip_epsilon`; z is precomputed once in `__init__` as `self.z_factor` (using `scipy.stats.norm.ppf(1.0 - spip_epsilon)`).

- **Buffered arrival at j:**  
  t̂₁_j = max(current_time + d_ij/speed + δ_path, tw_start_j)

- **Feasibility:**  
  Node j is marked infeasible (masked) if t̂₁_j > tw_end_j (with a small numerical tolerance).

### Two-step lookahead (pip_step=1)

For each unvisited candidate j, the env considers going j → m (another unvisited node). Path length to m via j is L_jm = L_j + d_jm. The **incremental buffer** for the second leg is:

- δ_increment = z · σ₀ · (√L_jm − √L_j)

Second-step arrival at m is computed from the first-step buffered arrival at j, plus travel time j→m, plus δ_increment, plus service at j, then max with tw_start_m. If for any such (j, m) this arrival exceeds tw_end_m, j is marked infeasible. So a node j is **selectable** only if there exists at least one feasible second-step node m after visiting j.

### Pseudocode: PIP mask with path-level buffer (pip_step = 0 or 1)

```
function CalculatePIPMask(selected):
    d_ij ← distances from current node to all nodes j
    L_k ← current path length
    L_j ← (L_k + d_ij) clamped to min 1e-8 for all j

    // First-step: path buffer and buffered arrival
    δ_path[j] ← z * sigma0 * sqrt(L_j)
    arrival_raw[j] ← current_time + d_ij/speed + δ_path[j]
    t_hat_1[j] ← max(arrival_raw[j], tw_start[j])
    out_of_tw_1step[j] ← (t_hat_1[j] > tw_end[j])

    if pip_step == 0 then
        for each j: simulated_ninf_flag[j] ← -∞ if out_of_tw_1step[j] else 0
        return

    // pip_step == 1: two-step lookahead
    unvisited ← set of indices not yet visited
    for each j in unvisited:
        selectable[j] ← false
        for each m in unvisited, m ≠ j:
            L_jm ← L_j[j] + distance(j, m)
            δ_increment ← z * sigma0 * (sqrt(L_jm) - sqrt(L_j[j]))
            arrival_2nd ← max(t_hat_1[j] + d_jm/speed + δ_increment, tw_start[m]) + service[m]
            if arrival_2nd ≤ tw_end[m] then
                selectable[j] ← true
                break
        simulated_ninf_flag[j] ← 0 if selectable[j] else -∞
```

### Pseudocode: depot tw_end (load_problems, normalize)

```
function SetDepotTWEnd(node_xy, tw_end):
    depot_xy ← node_xy[:, 0]
    for each non-depot node j:
        dist_depot_j ← ||depot_xy - node_xy[j]||₂
    tw_end[depot] ← max over j of (tw_end[j] + dist_depot_j)
```

### Code

All of the above is in [src/envs/TSPTWEnv_SPIP.py](src/envs/TSPTWEnv_SPIP.py), method `_calculate_PIP_mask`:

- Path buffer and first-step check: lines 461–471 (`delta_path`, `t_hat_1_j`, `out_of_tw_local`).
- **pip_step=0:** one-step only; nodes failing the first-step buffered check get `simulated_ninf_flag = -inf` (lines 473–477).
- **pip_step=1:** two-step lookahead with `delta_increment` and second-step arrival; selectable nodes get `simulated_ninf_flag = 0`, others `-inf` (lines 478–536).
- **pip_step=2:** deterministic (no S-PIP buffer); uses plain arrival times without δ_path/δ_increment (lines 538+).

---

## 4. Data and depot tw_end

- **Data layout:** Same as TSPTW: `(node_xy, service_time, tw_start, tw_end)`. The env sets `self.problem = "TSPTW"`, so the same data paths are used, e.g. `data/TSPTW/tsptw{N}_{hardness}.pkl`.

- **Depot time window (tw_end for node 0):** In `load_problems`, when `normalize=True`, the depot’s tw_end is set to a **tight upper bound** so that returning to the depot is feasible whenever the tour is feasible:  
  tw_end[depot] = max over nodes j of (tw_end[j] + distance(depot, j)).  
  Implemented as: `tw_end[:, 0] = (torch.cdist(node_xy[:, None, 0], node_xy[:, 1:]).squeeze(1) + tw_end[:, 1:]).max(dim=-1)[0]` (with a guard for empty batch).

---

## 5. Integration: training and entrypoints

### Env registration

[src/utils.py](src/utils.py): `get_env(problem)` maps `'TSPTW_SPIP'` → `TSPTWEnv_SPIP.TSPTWEnv_SPIP`. Train and test code use this to instantiate the correct env class.

### CLI and env_params

[src/train.py](src/train.py):

- Problem: `--problem TSPTW_SPIP`.
- S-PIP flags: `--spip_stochastic_transition`, `--spip_noise_dist`, `--spip_noise_bound`, `--spip_sigma0`, `--spip_epsilon`.

These are passed into `env_params` in `args2dict` (lines 17–22) and thence to the env constructor.

### Validation dataset

When `--val_dataset` is not set, train sets it for TSPTW_SPIP to `[f"tsptw{problem_size}_{hardness}.pkl"]` (lines 165–166), so validation uses the same TSPTW data path pattern.

### Trainer

[src/Trainer.py](src/Trainer.py): Uses `problem in ("TSPTW", "TSPTW_SPIP")` when passing `tw_end` into the model and for validation/grid scaling. The env is created via the same `get_env(args.problem)` pattern (e.g. in validation and in Tester).

### Model

[src/models/SINGLEModel.py](src/models/SINGLEModel.py): `TSPTW_PROBLEMS = {"TSPTW", "TSPTW_SPIP"}`; the model treats both the same (e.g. for `tw_end` and masking). There is no separate S-PIP branch in the model; only the env implements S-PIP logic.

---

## 6. Summary table

| Parameter | Default | Where used | Meaning |
|-----------|---------|------------|--------|
| `spip_stochastic_transition` | False | Env init, `step()` | If True, realized travel = d + ξ in `step()`. |
| `spip_noise_dist` | `"uniform"` | `_sample_bounded_noise` | `"uniform"` or `"clipped_gaussian"`. |
| `spip_noise_bound` | 2.0 | `_sample_bounded_noise` | Bound b: ξ in ±b√d. |
| `spip_sigma0` | 0.3 | Env init, `_sample_bounded_noise` (clipped_gaussian), `_calculate_PIP_mask` | Scale σ₀ for noise and for path buffer δ_path. |
| `spip_epsilon` | 0.05 | Env init (`z_factor`) | Confidence for z = Φ⁻¹(1−ε) in PIP mask buffer. |
| `pip_step` | 1 | `_calculate_PIP_mask` | 0: one-step buffered; 1: two-step with δ_increment; 2: deterministic no buffer. |

All of these are set from [src/train.py](src/train.py) (and optionally [src/test.py](src/test.py)) and passed via `env_params` into `TSPTWEnv_SPIP.__init__`.
