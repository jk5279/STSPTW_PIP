# OTA for TSPTW

Offline Temporally-Abstracted (OTA) reinforcement learning applied to the
**Travelling Salesman Problem with Time Windows (TSPTW)**.

The method learns a hierarchical actor–critic from a fixed offline trajectory
buffer (no environment interaction at training time).  At inference it uses
8-fold symmetry augmentation and an optional PIP constraint mask to improve
feasibility.

---

## Problem

| Setting | Value |
|---|---|
| Problem | TSPTW |
| Nodes | 10 customers + 1 depot (problem\_size = 11) |
| Hardness | easy / medium / hard (tighter time windows) |
| Coordinates | uniform $[0, 100]^2$ |
| Travel time | Euclidean distance + stochastic noise $\sim U(0, \sqrt{2})$ per edge |

---

## Data

All files live in `data/TSPTW/` (gitignored, generated locally).

| File | Size | Purpose |
|---|---|---|
| `tsptw10_{h}_buffer.pkl` | 10 000 trajectories | Offline RL training buffer |
| `tsptw10_{h}_train.pkl` | 2 000 instances | Instance pool used during buffer generation |
| `tsptw10_{h}_val.pkl` | 500 instances | Validation during training |
| `tsptw10_{h}_test.pkl` | 10 000 instances | Held-out test set |

`{h}` ∈ {`easy`, `medium`, `hard`}.  One separate model is trained per hardness.

**Generate data:**
```bash
cd OTA/
python gen_tsptw_dataset.py   # train / val / buffer splits
python gen_tsptw_test.py      # test set (seed 9999)
```

---

## Model

Transformer encoder + hierarchical low/high actor–critic heads.

| Hyperparameter | Value |
|---|---|
| Encoder layers | 6 |
| Embedding dim | 128 |
| Attention heads | 8 |
| FF hidden dim | 512 |
| Norm | BatchNorm, `norm_first` |
| Parameters | ~2.49 M |

---

## Training

| Hyperparameter | Value |
|---|---|
| Epochs | 200 |
| Batch size | 64 |
| Optimiser | Adam, lr = 1e-4 |
| LR schedule | ×0.1 at epochs 80, 160 |
| Abstraction factor $K_{abs}$ | 5 |
| Subgoal steps $K_{sub}$ | 25 |
| Expectile $\tau$ | 0.7 |
| Discount $\gamma$ (low / high) | 0.99 |
| IQL temperature $\alpha$ (low / high) | 3.0 |
| EMA $\tau$ (target net) | 0.005 |
| Seed | 2023 |

**Train one model:**
```bash
cd OTA/
python train_ota.py \
    --hardness easy \
    --dataset  data/TSPTW/tsptw10_easy_buffer.pkl \
    --val_dataset data/TSPTW/tsptw10_easy_val.pkl
```

---

## Evaluation

Inference uses **best-of-8 symmetry augmentation** (`--aug_only`).
`--pip_step 1` additionally applies a 1-step Propagation In the future (PIP)
feasibility mask that filters moves which would strand a remaining node.

**Run all 6 evaluations in parallel** (easy/medium/hard × pip\_step 0/1):
```bash
bash run_test_eval_parallel.sh      # from repo root
```

Logs are written to `OTA/results/eval_aug8_{hardness}_pip{step}.log`.

**Single evaluation:**
```bash
cd OTA/
python train_ota.py \
    --val_only --aug_only \
    --hardness easy \
    --pip_step 1 \
    --dataset  data/TSPTW/tsptw10_easy_buffer.pkl \
    --val_dataset data/TSPTW/tsptw10_easy_test.pkl \
    --checkpoint results/<run_dir>/model_epoch_200.pt
```

---

## Results (test set, 10 000 instances, aug-8)

| Hardness | pip\_step | Feasible (Inst.) | Tour dist (mean) |
|---|---|---|---|
| Easy   | 0 | 82%  | 3.84 |
| Easy   | 1 | 99%  | 3.85 |
| Medium | 0 |  8%  | 4.58 |
| Medium | 1 | 29%  | 4.82 |
| Hard   | 0 |  0%  | —    |
| Hard   | 1 | 57%  | 5.71 |

*Inst. feasible = at least one of the 8 augmented rollouts is feasible.*

---

## Checkpoints

Trained checkpoints are stored in `results/` (gitignored):

| Hardness | Run dir |
|---|---|
| Easy   | `260305_094703_OTA_TSPTW/model_epoch_200.pt` |
| Medium | `260305_232641_OTA_TSPTW/model_epoch_200.pt` |
| Hard   | `260305_232647_OTA_TSPTW/model_epoch_200.pt` |
