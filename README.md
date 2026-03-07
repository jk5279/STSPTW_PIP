# STSPTW_PIP: TSP with Time Windows — S-PIP, OTA, and Primal-Dual Implementations

This repository provides the implementation of **S-PIP (Stochastic Proactive Infeasibility Prevention)** for the **TSP with Time Windows (TSPTW)**. It extends the PIP framework ([Bi et al., NeurIPS 2024](https://arxiv.org/abs/2410.21066)) with optional **stochastic travel-time transitions** (realized travel = distance + bounded noise) and **path-level buffered feasibility masking**. The repository also includes an **Offline Temporally-Abstracted (OTA)** reinforcement learning implementation for TSPTW (in `OTA/`) and a **primal-dual** approach for stochastic TSPTW (in `POMO+PD/`). The following sections describe S-PIP with pointers to OTA and POMO+PD where relevant. The codebase is a fork of [PIP-constraint](https://github.com/jieyibi/PIP-constraint), focused on TSPTW and S-PIP for reproducibility of the reported experiments.

---

## Problem and Method

**TSP with Time Windows (TSPTW).** The agent must visit each node exactly once, minimize total travel time, and respect a time window [a_i, b_i] at each node i (arrival must fall within the window).

**PIP (baseline).** Proactive Infeasibility Prevention uses lookahead masking to exclude nodes that would lead to time-window violations; see the original paper *Learning to Handle Complex Constraints for Vehicle Routing Problems* (NeurIPS 2024).

**S-PIP (this work).**

- **Stochastic transitions:** Realized travel time = d + \xi, with \xi drawn from a **bounded** distribution (e.g. uniform on [-b\sqrt{d}, +b\sqrt{d}] or clipped Gaussian), so the agent is trained and evaluated under travel-time uncertainty.
- **Path-level buffer:** In the feasibility mask, a buffer proportional to \sqrt{L_j} (path length to candidate j) is applied so that time-window feasibility holds with high probability under that uncertainty.

For equations and integration details, see [docs/S-PIP_LOGIC_AND_INTEGRATION.md](docs/S-PIP_LOGIC_AND_INTEGRATION.md).

---

## Repository Contents and Scope

**Repository layout**

- `POMO+PIP/` — S-PIP and PIP training/evaluation (this README).
- `POMO+PD/` — Primal-dual approach for stochastic TSPTW; workflow and commands [below](#primal-dual-pomopd) and in [POMO+PD/Readme.md](POMO+PD/Readme.md).
- `OTA/` — Offline temporally-abstract RL for TSPTW; workflow and commands [below](#ota-offline-temporally-abstracted) and in [OTA/README.md](OTA/README.md).
- `trillium_scripts/` — Cluster setup and S-PIP job scripts.
- `docs/` — S-PIP logic and integration notes.

This repository includes:

- **POMO+PIP (S-PIP):** Implementation of the **TSPTW_SPIP** problem and environment ([POMO+PIP/envs/TSPTWEnv_SPIP.py](POMO+PIP/envs/TSPTWEnv_SPIP.py)), training and evaluation scripts (`train.py`, `test.py`), and data generation (`generate_data.py`) for TSPTW.
- **OTA:** Offline temporally-abstract RL for TSPTW: hierarchical actor–critic trained from a fixed offline trajectory buffer; lives under [OTA/](OTA/) with [OTA/README.md](OTA/README.md) for full details.
- **POMO+PD (Primal-Dual):** Primal-dual training for (stochastic) TSPTW: dual variables (lambdas) for total timeout and late-node count, updated by subgradient; optional PI masking. Lives under [POMO+PD/](POMO+PD/) with [POMO+PD/Readme.md](POMO+PD/Readme.md) for training and evaluation.

It is a fork of PIP-constraint: POMO+PIP training and TSPTW PIP/PIP-D are retained; the S-PIP environment and bounded-noise transitions are added. For full multi-problem PIP experiments, see the [original repository](https://github.com/jieyibi/PIP-constraint).

**Methods implemented**

- **S-PIP:** Online RL (POMO+PIP), optional stochastic travel-time transitions and path-level buffered PIP masking; problem type `TSPTW_SPIP`, training via `POMO+PIP/train.py`.
- **OTA:** Offline RL with temporal abstraction; no environment interaction at training time; inference with 8-fold augmentation and optional PIP mask; training via `OTA/train_ota.py`. Full setup and hyperparameters in [OTA/README.md](OTA/README.md).
- **Primal-Dual (POMO+PD):** TSPTW training with dual variables (lambdas) for timeout and late-node constraints, updated by subgradient (`--penalty_mode primal_dual`); supports PI masking. Training via `POMO+PD/train.py`.

---

## Primal-Dual (POMO+PD)

Workflow: generate datasets for training and validation → train per hardness → test per hardness. Replication can use saved datasets and trained models and run only test.

**Train** (from `POMO+PD/`):

```bash
cd POMO+PD
python train.py --problem TSPTW --problem_size 10 --hardness {easy,medium,hard} \
  --train_episodes 10000 --val_episodes 10000 --train_batch_size 64 \
  --epochs 10000 --generate_PI_mask --penalty_mode primal_dual
```

**Generate data** (for both validation and test):

```bash
cd POMO+PD
python generate_data.py --problem TSPTW --problem_size 10 --pomo_size 1 \
  --hardness {easy,medium,hard} --num_samples 10000 --dir ../data
```

**Test** (use the checkpoint and test set paths for your run):

```bash
cd POMO+PD
python test.py --problem TSPTW --hardness hard --problem_size 10 \
  --checkpoint path/to/epoch-1000.pt \
  --test_set_path ../data/TSPTW/tsptw10_hard.pkl --test_episodes 10000
```

Use `--hardness easy` or `--hardness medium` and the corresponding checkpoint and test set for other difficulty levels. Full options: [POMO+PD/Readme.md](POMO+PD/Readme.md).

---

## OTA (Offline Temporally-Abstracted)

OTA learns a hierarchical actor–critic from a fixed offline trajectory buffer (no environment interaction at training time). Inference uses 8-fold symmetry augmentation and an optional PIP feasibility mask.

**Problem:** TSPTW, 10 customers + 1 depot (`problem_size = 11`), hardness easy/medium/hard, coordinates uniform in [0,100]^2, travel time = Euclidean distance + stochastic noise per edge.

**Data** (under `OTA/data/TSPTW/` or `data/TSPTW/`): `tsptw10_{h}_buffer.pkl` (training buffer), `tsptw10_{h}_train.pkl`, `tsptw10_{h}_val.pkl`, `tsptw10_{h}_test.pkl` with `{h}` ∈ {easy, medium, hard}. One model per hardness.

**Generate data:**

```bash
cd OTA
python gen_tsptw_dataset.py   # train / val / buffer splits
python gen_tsptw_test.py      # test set (seed 9999)
```

**Model:** Transformer encoder (6 layers, 128-dim, 8 heads, 512 FFN) + hierarchical low/high actor–critic (~2.49M parameters).

**Train** (one model per hardness):

```bash
cd OTA
python train_ota.py --hardness easy \
  --dataset data/TSPTW/tsptw10_easy_buffer.pkl \
  --val_dataset data/TSPTW/tsptw10_easy_val.pkl
```

**Evaluation:** Best-of-8 augmentation; optional `--pip_step 1` for PIP feasibility mask. Single run:

```bash
cd OTA
python train_ota.py --val_only --aug_only --hardness easy --pip_step 1 \
  --dataset data/TSPTW/tsptw10_easy_buffer.pkl \
  --val_dataset data/TSPTW/tsptw10_easy_test.pkl \
  --checkpoint results/<run_dir>/model_epoch_200.pt
```

**Representative results** (test set, 10k instances, aug-8): Easy pip_step 0/1 → 82% / 99% feasible; Medium → 8% / 29%; Hard → 0% / 57%. Trained checkpoints live under `OTA/results/` (gitignored). Full tables and options: [OTA/README.md](OTA/README.md).

---

## Reproducibility

### Software

- Python 3.8+
- PyTorch (install first, then remaining dependencies)

The same stack (PyTorch, scipy, ortools, etc.) is used by POMO+PIP, OTA, and POMO+PD. For OTA and POMO+PD, see their READMEs for component-specific data paths and commands.

From the repository root:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install scipy matplotlib tqdm pytz scikit-learn pandas wandb tensorboard_logger ortools
```

### Data

TSPTW instances are stored under `data/TSPTW/tsptw{N}_{hardness}.pkl`. S-PIP uses the same layout (`env.problem = "TSPTW"`). Generate data from `POMO+PIP/`:

```bash
cd POMO+PIP
python generate_data.py --problem TSPTW_SPIP --problem_size 10 --hardness hard --num_samples 10000 --dir ../data
```

Use `--hardness easy` or `--hardness medium` for other difficulty levels. Output is written to `../data/TSPTW/tsptw10_{hardness}.pkl`.

OTA uses its own data layout under `OTA/data/TSPTW/` (see [OTA/README.md](OTA/README.md)). POMO+PD uses TSPTW datasets; see [POMO+PD/Readme.md](POMO+PD/Readme.md) for generation and paths.

### Seeds

Training uses a fixed seed via `seed_everything(args.seed)`; the default is `--seed 2023` in [POMO+PIP/train.py](POMO+PIP/train.py). For reproducible validation under stochastic mode, the same process and seed are used; see the code for details.

---

## Experimental Setup (Training and Evaluation)

### Training

Default S-PIP protocol:

- Problem: **TSPTW_SPIP**, 10 nodes, hardness **easy** / **medium** / **hard**
- 10,000 epochs; 10,000 train and 10,000 validation episodes; batch size 128; PI masking enabled

**Deterministic S-PIP** (no travel-time noise):

```bash
cd POMO+PIP
python train.py --problem TSPTW_SPIP --problem_size 10 --hardness hard \
  --epochs 10000 --train_episodes 10000 --val_episodes 10000 \
  --train_batch_size 128 --generate_PI_mask
```

**Stochastic S-PIP** (bounded noise):

```bash
python train.py --problem TSPTW_SPIP --problem_size 10 --hardness hard \
  --epochs 10000 --train_episodes 10000 --val_episodes 10000 \
  --train_batch_size 128 --generate_PI_mask --spip_stochastic_transition True
```

Optional: `--spip_noise_dist {uniform|clipped_gaussian}`, `--spip_noise_bound`, `--spip_sigma0` (see table below).

For **OTA** and **primal-dual** workflows (data, training, evaluation), see the [Primal-Dual (POMO+PD)](#primal-dual-pomopd) and [OTA (Offline Temporally-Abstracted)](#ota-offline-temporally-abstracted) sections above.

### Evaluation

```bash
cd POMO+PIP
python test.py --problem TSPTW_SPIP --hardness hard --problem_size 10 --checkpoint path/to/model.pt
```

For custom test sets and optimality gaps: `--test_set_path` and `--test_set_opt_sol_path` (same interface as upstream PIP-constraint). OTA and primal-dual evaluation commands are in the [OTA](#ota-offline-temporally-abstracted) and [Primal-Dual (POMO+PD)](#primal-dual-pomopd) sections above.

---

## S-PIP Hyperparameters


| Parameter                    | Default   | Description                                                                                               |
| ---------------------------- | --------- | --------------------------------------------------------------------------------------------------------- |
| `spip_stochastic_transition` | `False`   | If `True`, realized travel = distance + bounded noise \xi.                                                |
| `spip_noise_dist`            | `uniform` | Noise distribution: `uniform` or `clipped_gaussian`.                                                      |
| `spip_noise_bound`           | `2.0`     | Bound on                                                                                                  |
| `spip_sigma0`                | `0.3`     | Scale for Gaussian before clipping (used only for `clipped_gaussian`).                                    |
| `spip_epsilon`               | `0.05`    | Used to compute z-factor \Phi^{-1}(1-\varepsilon) once at init for the path-level buffer in the PIP mask. |


---

## Cluster and Optional Scripts

For runs on the **Trillium** cluster, [trillium_scripts/setup_venv.sh](trillium_scripts/setup_venv.sh) sets up the environment; [trillium_scripts/run_train_spip_n10.sh](trillium_scripts/run_train_spip_n10.sh) and SLURM scripts (e.g. `run_spip_n10_*_cglee.slurm`) run the same training protocol. Account and paths are cluster-specific; account-specific scripts are listed in `.gitignore`.

Example submissions from the repo root:

```bash
# Hard, deterministic
sbatch trillium_scripts/run_spip_n10_cglee.slurm

# Hard, stochastic
sbatch --export=ALL,STOCHASTIC=1,HARDNESS=hard trillium_scripts/run_spip_n10_cglee.slurm

# Easy hardness
sbatch trillium_scripts/run_spip_n10_easy_cglee.slurm
```

---

## Citation

If you use this codebase in your research, please cite the original PIP paper:

```text
@inproceedings{
    bi2024learning,
    title={Learning to Handle Complex Constraints for Vehicle Routing Problems},
    author={Bi, Jieyi and Ma, Yining and Zhou, Jianan and Song, Wen and Cao,
    Zhiguang and Wu, Yaoxin and Zhang, Jie},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2024}
}
```

If you use the S-PIP extension in this repository, please also cite this codebase (and any associated paper or preprint, if applicable).

---

## Acknowledgements

This project builds on and reuses code from:

- [POMO](https://github.com/yd-kwon/POMO)
- [Attention Model (AM)](https://github.com/wouterkool/attention-learn-to-route)
- [GFACS](https://github.com/ai4co/gfacs)
- [Routing-MVMoE](https://github.com/RoyalSkye/Routing-MVMoE)
- [PIP-constraint](https://github.com/jieyibi/PIP-constraint)

---

## Troubleshooting and Known Issues

- **tensorboard_logger and protobuf:** Newer `protobuf` versions can trigger `TypeError: Descriptors cannot be created directly.` This repository sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` in [POMO+PIP/train.py](POMO+PIP/train.py) and in the Trillium scripts, so no manual setting is needed for standard workflows.
- **GPU memory:** Adjust `--train_batch_size`, `--aug_batch_size`, and `--test_batch_size` according to your GPU; defaults assume a reasonably large GPU as in the original PIP experiments.

