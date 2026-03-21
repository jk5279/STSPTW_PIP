# STSPTWenv

Stochastic TSP with Time Windows (STSPTW) environment and POMO+PIP training/evaluation. All commands below are run from the **`POMO+PIP/`** directory.

```bash
cd POMO+PIP
```

---

## Environments

### STSPTW (v1) (Realistic Version, Test later)

Defined in `envs/STSPTWEnv.py`. Uses a **time-dependent, additive delay** model:

```
travel_time(i→j) = d_ij + delay_scale · base_delay(d_ij, current_time) · lognormal_factor(current_time)
```

- Delay is always positive — **not mean-preserving** (expected travel time > d_ij)
- Noise magnitude depends on **current_time**: two Gaussian rush-hour peaks at 30% and 70% of the time horizon inflate both the delay and its variance
- Longer edges get proportionally more delay (via a saturating distance factor)
- Controlled by `--delay_scale` (0.1 / 0.3 / 0.5 in sweep); effective CV ≈ 0.03–0.12

### STSPTW v2 (Theoretical Version, Test first)

Defined in `envs/STSPTWEnv_v2.py`. Uses **edge-level, time-independent, mean-preserving noise**:

```
travel_time(i→j) ~ F(mean=d_ij, CV=cv)
```

Two supported distributions (set via `--noise_type`):

| `--noise_type` | Distribution | CV control |
|---|---|---|
| `gamma` | Gamma(k, k/d) with k = 1/CV² | `--cv` directly sets CV |
| `two_point` | d·0.7 or d·1.3 with equal probability | fixed CV ≈ 0.3; `--cv` has no effect |

- **Mean-preserving**: E[t] = d_ij exactly; noise cancels on average
- **State-independent**: same distribution regardless of when the edge is traversed
- `out_of_tw` lookahead uses deterministic distances (consistent and monotonic)
- Substantially higher variance than v1: CV ∈ {0.25, 0.5, 1.0} vs v1's effective CV of 0.03–0.12

### Key differences

| | V1 (STSPTW) | V2 (STSPTW_v2) |
|---|---|---|
| Noise model | Additive delay on d | Multiplicative perturbation of d |
| Mean-preserving | No (always slower) | Yes |
| Time-dependent | Yes (rush hours) | No |
| Control parameter | `--delay_scale` | `--cv` |
| Effective CV range | 0.03 – 0.12 | 0.25 – 1.0 |
| `out_of_tw` lookahead | Re-samples noisy travel | Uses deterministic d |

### Pre- vs post-decision noise

Both environments support:
- **Post-decision** (default): noise is realized *after* the agent picks a node — the agent never sees it
- **Pre-decision** (`--reveal_delay_before_action`): travel times to all candidates are sampled and revealed to the agent *before* action selection

---

## Model types

| `--model_type` | LM reward | PIP mask | Run name suffix |
|---|---|---|---|
| `POMO` | no | no | *(none)* |
| `POMO_STAR` | yes | no | `_LM` |
| `POMO_STAR_PIP` | yes | yes (during training) | `_LM_PIMask_1Step` |

**LM** = Lagrangian Multiplier: adds timeout penalty terms (`timeout_reward`, `timeout_node_reward`) to the reward signal, driving the model toward feasibility.
**PIP** = Proactive Infeasibility Prevention: a 1-step deterministic lookahead mask applied at each action step to eliminate nodes that would make the remaining tour infeasible.

PIP can also be applied **post-hoc at inference** on a `POMO_STAR` checkpoint (no retraining needed) by passing `--generate_PI_mask --pip_step 1` to `test.py`.

---

## Training

Entry point: **`train.py`**

**Example — STSPTW v1, n=10, hard, post-decision, POMO_STAR_PIP:**

```bash
python train.py --problem STSPTW --problem_size 10 --pomo_size 10 --hardness hard \
  --model_type POMO_STAR_PIP --delay_scale 0.3 \
  --epochs 10000 --train_episodes 10000 --train_batch_size 1024 \
  --val_episodes 10000 --validation_batch_size 1000 --validation_interval 500 --model_save_interval 50
```

**Example — STSPTW v2, n=10, hard, gamma noise, CV=0.5, POMO_STAR_PIP:**

```bash
python train.py --problem STSPTW_v2 --problem_size 10 --pomo_size 10 --hardness hard \
  --model_type POMO_STAR_PIP --noise_type gamma --cv 0.5 \
  --reveal_delay_before_action \
  --epochs 10000 --train_episodes 10000 --train_batch_size 1024 \
  --val_episodes 10000 --validation_batch_size 1000 --validation_interval 500 --model_save_interval 50
```

**Key training flags:**

| Flag | Meaning |
|---|---|
| `--problem` | `TSPTW`, `STSPTW` (v1), or `STSPTW_v2` |
| `--model_type` | `POMO`, `POMO_STAR`, `POMO_STAR_PIP` |
| `--delay_scale` | V1 noise magnitude (default `0.1`) |
| `--noise_type` | V2 distribution: `gamma` or `two_point` |
| `--cv` | V2 coefficient of variation (default `0.5`) |
| `--reveal_delay_before_action` | Pre-decision noise (omit for post-decision) |

Checkpoints are saved under `results/<run_name>/epoch-N.pt`. The run name encodes all key settings (problem, size, hardness, noise params, `_LM`, `_PIMask_1Step`).

---

## Evaluation

Entry point: **`test.py`**

**Example — evaluate POMO_STAR_PIP checkpoint on STSPTW v1:**

```bash
python test.py --problem STSPTW --problem_size 10 --hardness hard \
  --checkpoint results/<run_name>/epoch-10000.pt \
  --delay_scale 0.3 --aug_factor 8 --no_opt_sol
```

**Example — evaluate POMO_STAR checkpoint with PIP applied post-hoc (v2, gamma):**

```bash
python test.py --problem STSPTW_v2 --problem_size 10 --hardness hard \
  --checkpoint results/<run_name>/epoch-10000.pt \
  --noise_type gamma --cv 0.5 --reveal_delay_before_action \
  --aug_factor 8 --no_opt_sol \
  --generate_PI_mask --pip_step 1
```

If `--test_set_path` is omitted, the default dataset under `../data/TSPTW/tsptw<N>_<hardness>.pkl` is used (both v1 and v2 use the same deterministic TSPTW instance files).

---

## Using the Python API directly

All commands above go through `train.py` / `test.py`. If you want to use the environments or model in a script or notebook, there are several non-obvious requirements.

### Environment API

**Required constructor parameter: `k_sparse`**

All three environments (`TSPTWEnv`, `STSPTWEnv`, `STSPTWEnv_v2`) accept `**env_params` and require `k_sparse` to be present. Set it to `problem_size + 1` to disable sparsification (use all neighbors):

```python
env_params = dict(
    problem_size=10,
    pomo_size=10,
    hardness='hard',
    k_sparse=11,          # required — set > problem_size to disable sparsification
    device=torch.device('cpu'),
)
```

**Correct call sequence: `load_problems` → `reset` → `pre_step` → loop `step`**

```python
env = TSPTWEnv(**env_params)
problems = env.get_random_problems(batch_size, problem_size)
env.load_problems(batch_size, problems=problems)
reset_state, _, _ = env.reset()   # returns (reset_state, None, False)
env.pre_step()

done = False
while not done:
    # ... compute `selected` (shape: batch × pomo) ...
    step_state, reward, done, infeasible = env.step(selected)
    # reward is None until done=True
```

**`env.step()` returns 4 values, not 3**

```python
step_state, reward, done, infeasible = env.step(selected)
# reward: (batch, pomo) tensor — only non-None when done=True
# infeasible: (batch, pomo) bool tensor — True for any pomo that violated a TW
```

### Model API

**`SINGLEModel` required parameters**

Beyond the standard architecture params, the model requires several flags that `train.py` sets from CLI args:

```python
model_params = dict(
    problem='TSPTW',          # or 'STSPTW', 'STSPTW_v2'
    embedding_dim=128,
    sqrt_embedding_dim=128**0.5,
    encoder_layer_num=6,
    qkv_dim=16,
    head_num=8,
    logit_clipping=10,
    ff_hidden_dim=512,
    eval_type='softmax',
    model_type='POMO_STAR_PIP',
    norm='instance',
    norm_loc='norm_last',     # required — 'norm_first' or 'norm_last'
    tw_normalize=True,        # required — normalize TW values by max TW end
    pip_decoder=False,        # required — True only for auxiliary PIP decoder variant
    W_q_sl=True,              # required — weight flags for selective attention
    W_kv_sl=True,
    W_out_sl=True,
    device=torch.device('cpu'),
)
```

**`model.forward()` requires `tw_end` for TSPTW problems**

The decoder uses `tw_end` to normalize the current-time context feature. Pass `env.node_tw_end`:

```python
model.pre_forward(reset_state)
env.pre_step()

done = False
while not done:
    selected, prob = model(env.step_state, tw_end=env.node_tw_end)
    step_state, reward, done, infeasible = env.step(selected)
```

### Minimal working example

```python
import torch
from envs.TSPTWEnv import TSPTWEnv
from models.SINGLEModel import SINGLEModel

batch_size, n = 4, 10
device = torch.device('cpu')

env_params = dict(problem_size=n, pomo_size=n, hardness='hard', k_sparse=n+1, device=device)
model_params = dict(
    problem='TSPTW', embedding_dim=128, sqrt_embedding_dim=128**0.5,
    encoder_layer_num=6, qkv_dim=16, head_num=8, logit_clipping=10,
    ff_hidden_dim=512, eval_type='softmax', model_type='POMO_STAR_PIP',
    norm='instance', norm_loc='norm_last', tw_normalize=True,
    pip_decoder=False, W_q_sl=True, W_kv_sl=True, W_out_sl=True, device=device,
)

env = TSPTWEnv(**env_params)
env.load_problems(batch_size, problems=env.get_random_problems(batch_size, n))
reset_state, _, _ = env.reset()

model = SINGLEModel(**model_params)
model.pre_forward(reset_state)
env.pre_step()

done = False
while not done:
    selected, prob = model(env.step_state, tw_end=env.node_tw_end)
    step_state, reward, done, infeasible = env.step(selected)

print(reward.shape)      # (batch, pomo)
print(infeasible.shape)  # (batch, pomo)
```

### Note on high infeasibility with random / untrained models

With random actions or an untrained model on `hardness='hard'` instances, `infeasible` will be `True` for nearly all POMO rollouts. This is expected — hard instances have tight TW constraints that are almost impossible to satisfy without a trained policy.

---

## Sweep experiments

All sweep scripts live in `scripts/` and are run from the **project root** (`STSPTWenv/`).

| Script | What it runs | Output CSV |
|---|---|---|
| `scripts/train/sweep_v1.sh` | 54 v1 training runs (SLURM array) | — |
| `scripts/train/sweep_v2.sh` | 54 v2 training runs (SLURM array) | — |
| `scripts/test/run_test_sweep_v1_v2.py` | Evaluate all 108 trained models | `results/csv/test_sweep_v1_v2.csv` |
| `scripts/test/run_test_pomo_star_posthoc_pip.py` | POMO* checkpoints + PIP post-hoc | `results/csv/test_pomo_star_posthoc_pip.csv` |

Submit test jobs via the corresponding `.sh` wrappers:

```bash
sbatch scripts/test/run_test_sweep_v1_v2.sh
sbatch scripts/test/run_test_pomo_star_posthoc_pip.sh
```

Plots are generated by scripts in `scripts/plot/` and saved to `results/figures/`.
