# STSPTWEnv_v2 Changes: Fully Stochastic Environment with Fixed-Noise Reproducibility

The diff contains **four conceptually distinct changes**. Together they shift the environment from a "stochastic travel for the agent, deterministic lookahead for feasibility/masking" design toward a **fully stochastic** design with a **fixed-noise reproducibility mechanism** for fair testing.

---

## 1. `out_of_tw` now uses stochastic travel times (was deterministic)

**Old behavior:**
The time-window feasibility check (`out_of_tw`) computed `next_arrival_time` using **deterministic** L2 distances divided by speed. The rationale in the old docstring was "monotonic accumulation is correct" — once a node became unreachable under deterministic distances, it stayed unreachable.

**New behavior:**
```python
pairwise_travel = self._sample_travel_time(pairwise_dist)
next_arrival_time = current_time + pairwise_travel  # stochastic
```
A **fresh stochastic sample** is drawn at every step to estimate arrival times at unvisited nodes.

### Impact on training

- The masking signal is now **noisier**. A node that appears out-of-window in one step might not appear out-of-window in the next (and vice versa), because each step draws a new random travel time. The old deterministic mask was monotone (once masked, always masked); the new one is **non-monotone**.
- This means the model must learn a **risk-aware policy** — it can no longer rely on the mask being a perfectly safe guarantee. The mask becomes a probabilistic hint rather than a hard certificate.
- Gradient variance will increase because the reward landscape depends on two layers of randomness (actual travel + mask sampling). Training may need more episodes or a lower learning rate to converge.

### Impact on testing

- The mask is now *consistent* with the actual stochastic dynamics the agent experiences. Under the old design, the agent could experience a stochastic travel time that made it late, yet the deterministic mask never flagged the risk. Now the mask and the dynamics speak the same language.
- However, because the mask is sampled independently each step, **test-time results have higher variance**. This is partially addressed by Change 3 below.

---

## 2. PIP mask uses stochastic lookahead (was deterministic)

**Old behavior:**
The PIP (Proactive Infeasibility Prevention) mask was computed using `det_pairwise_dist` — deterministic distances. The Monte Carlo forward simulations inside `_calculate_PIP_mask` used deterministic travel estimates as their base.

**New behavior:**
```python
self._calculate_PIP_mask(pip_step, selected, next_arrival_time, pairwise_dist)
```
`pairwise_dist` (raw distances) is still passed, but `next_arrival_time` is now computed from stochastic travel. The PIP rollouts now account for travel time variability.

### Impact on training

- PIP masks will be more conservative on average — stochastic lookahead sometimes produces longer-than-expected travel times, causing the mask to pre-emptively block nodes that *might* become infeasible. The model learns with a "safety margin" baked in.
- However, the PIP mask can also randomly be *less* conservative when the sampled travel times happen to be shorter. This bidirectional noise can make PIP-guided exploration less stable during early training.

### Impact on testing

- The PIP mask becomes a better predictor of actual reachability under stochastic travel, leading to fewer "surprise" infeasibilities at test time. The mask quality (precision/recall of predicting true infeasibility) should improve in expectation.

---

## 3. Fixed noise multipliers for reproducible test-time evaluation

This is the largest structural change. Three new methods are introduced:

| Method | Purpose |
|---|---|
| `_sample_noise_multipliers` | Samples distance-independent multipliers `m` with `E[m]=1`, so `travel = m * distance / speed` |
| `_get_fixed_travel` | Looks up pre-sampled `m` for a specific `(from, to)` pair |
| `_get_fixed_pairwise_travel` | Looks up pre-sampled `m` for all destination nodes at once |

### How it works

- At `reset()`, if `test_noise_seed` is set and `cv > 0`, a matrix of shape `(original_batch, N, N)` is pre-sampled using the fixed seed. The RNG state is saved and restored so subsequent randomness is unaffected.
- During `step()`, when `noise_multipliers is not None`, travel times are computed as `m[from, to] * distance / speed` instead of drawing a fresh sample. This means **every model sees the exact same stochastic realization** for a given problem instance.
- The multipliers are indexed by `orig_idx = batch_idx % original_batch_size`, so augmented copies (rotations/flips from `aug_factor=8`) share the same noise matrix as their base instance.

### Old approach (removed)

The old code pre-sampled a full `N x N` travel time matrix using NumPy in `load_problems()` and stored it as `self.pre_sampled_travel_matrix`. This was disabled when `aug_factor > 1`. The new approach:

1. Stores **multipliers** (distance-independent) rather than travel times — more memory-efficient and works correctly with augmentation.
2. Samples at `reset()` rather than `load_problems()`, which is the correct lifecycle point.
3. Works with `aug_factor > 1` by mapping augmented indices back to original instances.

### Impact on training

- **No change.** When `test_noise_seed` is `None` (the training default), `noise_multipliers` stays `None` and all travel times are sampled fresh as before.

### Impact on testing

- This is the key improvement for **fair model comparison**. Previously, two models evaluated on the same problem set would experience different random travel times, making comparison noisy. Now, with a fixed seed, the stochastic realization is deterministic across models.
- Augmentation (`aug_factor=8`) now works correctly with fixed noise — the old code simply disabled pre-sampling when augmentation was on.
- The pre-sampling happens in `step_select_action` too: the pairwise travel lookahead (used for PIP/masking) also goes through `_get_fixed_pairwise_travel`, ensuring consistency between the lookahead and the actual travel experienced.

---

## 4. Explicit "actually late" infeasibility check

**Old behavior:**
Infeasibility was detected only when all unvisited nodes became unreachable (the `out_of_tw` mask blocked everything remaining).

**New behavior:**
```python
arrival_at_selected = current_time - service_time[selected]
actually_late = arrival_at_selected > tw_end[selected] + epsilon
newly_infeasible = newly_infeasible | actually_late
```
The environment now also checks whether **the node just visited** was arrived at after its time window closed. This is a direct consequence of the stochastic `out_of_tw` mask: because the mask is no longer monotone, it's possible for the agent to select a node that was not masked, but the actual stochastic travel time caused a late arrival. The old deterministic mask would have prevented this, but the new stochastic mask cannot guarantee it.

### Impact on training

- The model receives a more **honest infeasibility signal**. Under the old design, the model could arrive late at a node but not be flagged as infeasible (because the deterministic mask said it was reachable). Now, late arrivals are always caught.
- This creates a stronger learning signal for the model to build in time buffers and avoid tight-deadline nodes when travel time variance is high.

### Impact on testing

- Test-time infeasibility counts will be more accurate. Any comparison of "feasibility rate" across models will now reflect true time-window violations rather than only mask-based detection.

---

## Summary

| Aspect | Old Design | New Design | Training Effect | Testing Effect |
|---|---|---|---|---|
| `out_of_tw` mask | Deterministic distances | Stochastic travel samples | Noisier mask, harder learning | Consistent with actual dynamics |
| PIP mask | Deterministic lookahead | Stochastic lookahead | More realistic pruning | Better infeasibility prevention |
| Cross-model reproducibility | NumPy pre-sampled matrix (no aug support) | Noise multipliers with fixed seed (aug-aware) | None (disabled in training) | Fair apple-to-apple comparison |
| Infeasibility detection | Only "all nodes blocked" | Also "arrived late at selected node" | Stronger penalty signal | More accurate feasibility reporting |

The overall direction is toward a **fully stochastic, internally consistent** environment where the mask, the PIP lookahead, and the actual travel dynamics all operate under the same stochastic model — with a controlled mechanism to freeze the randomness at test time for reproducible evaluation.
