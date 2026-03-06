## Learning to Handle Complex Constraints for VRPs (STSPTW_PIP fork)

This repository is a **specialized fork** of the original [PIP-constraint](https://github.com/jieyibi/PIP-constraint) codebase.
It focuses on **TSP with Time Windows (TSPTW)** and an extended **Stochastic Proactive Infeasibility Prevention (S-PIP)** environment,
with a training workflow tailored to the **Trillium GPU cluster**.

The original method is described in the NeurIPS 2024 paper
[*Learning to Handle Complex Constraints for Vehicle Routing Problems*](https://arxiv.org/abs/2410.21066).

---

## Scope of this fork

- **Kept from upstream:**
  - POMO+PIP training code in `[POMO+PIP](POMO+PIP/)`
  - TSPTW PIP/PIP-D logic and general CLI structure (`train.py`, `test.py`)
- **Added in this fork:**
  - `TSPTWEnv_SPIP` environment: `[POMO+PIP/envs/TSPTWEnv_SPIP.py](POMO+PIP/envs/TSPTWEnv_SPIP.py)`
  - New problem type `TSPTW_SPIP` (S-PIP on TSPTW)
  - Bounded-noise **stochastic transitions** for S-PIP
  - Trillium-specific training scripts in `[trillium_scripts/](trillium_scripts/)`
- **Not used here:**
  - Any `vrp_bench`-related delay logic from the original project

For multi-problem experiments and the full AM/GFACS integration, please refer to the
[original PIP-constraint repository](https://github.com/jieyibi/PIP-constraint).

---

## Installation and environment setup

### Local (generic)

From the repo root:

```bash
python -m venv venv
source venv/bin/activate

pip install --upgrade pip

# Core PyTorch stack (adapt CUDA index-url to your machine if needed)
pip install torch torchvision torchaudio

# PIP / S-PIP dependencies
pip install scipy matplotlib tqdm pytz scikit-learn pandas wandb tensorboard_logger ortools
```

These packages match what the Trillium setup script installs (see below).

### Trillium cluster setup

On Trillium (e.g., `trig-login01`), from the repo root:

```bash
cd /scratch/kimjong/STSPTW_PIP
bash trillium_scripts/setup_venv.sh
```

This script (`[trillium_scripts/setup_venv.sh](trillium_scripts/setup_venv.sh)`):

- Loads the Trillium module stack (`StdEnv/2023`, `python/3.11.5`, `cuda/12.6`, etc.)
- Creates a venv at `./venv` if it does not exist
- Installs all required Python dependencies (including `torch`, `scipy`, `tensorboard_logger`, `ortools`)

> **Note:** `setup_venv.sh` assumes the Trillium environment. On other clusters or machines, adjust the `module load` section as needed.

---

## Data layout and generation (TSPTW / TSPTW_SPIP)

This fork uses the same TSPTW data layout as upstream:

- **Data path pattern:** `data/TSPTW/tsptw{N}_{hardness}.pkl`
  - Example: `data/TSPTW/tsptw10_hard.pkl`
- The S-PIP environment `TSPTWEnv_SPIP` sets `env.problem = "TSPTW"`, so
  `TSPTW_SPIP` reuses the **TSPTW** data directory and filenames.

To generate data for 10-city TSPTW/TSPTW_SPIP:

```bash
cd POMO+PIP
python generate_data.py --problem TSPTW_SPIP --problem_size 10 --hardness hard --num_samples 10000 --dir ../data
```

For `TSPTW_SPIP`, this command writes:

- `../data/TSPTW/tsptw10_hard.pkl`

Repeat with `--hardness easy` or `--hardness medium` to create the other difficulty levels.

---

## S-PIP problem and configuration

**Documentation:** For a self-contained description of S-PIP logic (stochastic transitions, path-level buffer, PIP mask) and how it is integrated into the codebase, see [docs/S-PIP_LOGIC_AND_INTEGRATION.md](docs/S-PIP_LOGIC_AND_INTEGRATION.md).

### Problem type and environment

- **Problem name:** `TSPTW_SPIP`
- **Environment:** `[POMO+PIP/envs/TSPTWEnv_SPIP.py](POMO+PIP/envs/TSPTWEnv_SPIP.py)`
- **Deterministic mode:**
  - Realized travel time = Euclidean distance / speed (no noise)
- **Stochastic mode:**
  - Realized travel time = \( d + \xi \) where \( \xi \) is **bounded noise** (unknown bounded distribution)

### S-PIP flags in `train.py`

Defined in `[POMO+PIP/train.py](POMO+PIP/train.py)`:

- **Problem selection**
  - `--problem TSPTW_SPIP`
- **Stochastic transition**
  - `--spip_stochastic_transition {True,False}` (default `False`)
    - `False`: deterministic
    - `True`: use bounded stochastic noise in transitions
- **Noise distribution**
  - `--spip_noise_dist {uniform,clipped_gaussian}` (default `uniform`)
  - `--spip_noise_bound` (default `2.0`)
    - Uniform: \( \xi \sim \mathcal{U}\left[-b\sqrt{d}, +b\sqrt{d}\right] \)
    - Clipped Gaussian: \( \xi \sim \mathcal{N}(0, \sigma_0^2 d) \) then clipped to ±\( b\sqrt{d} \)
  - `--spip_sigma0` (default `0.3`) – used only for `clipped_gaussian`
- **PIP lookahead (z-factor)**
  - `--spip_epsilon` (default `0.05`)
    - In `[POMO+PIP/envs/TSPTWEnv_SPIP.py](POMO+PIP/envs/TSPTWEnv_SPIP.py)`, the z-score is computed once:
      \( z = \Phi^{-1}(1 - \varepsilon) \) and used in `_calculate_PIP_mask`.

In the environment, all S-PIP noise tensors are created on the correct device
(`torch.randn_like(distance)` etc.), and the reward is **negative realized travel time** so the
trainer’s reward maximization corresponds to minimizing travel time.

---

## Training usage (generic CLI)

All training commands are run under `[POMO+PIP](POMO+PIP/)`:

```bash
cd POMO+PIP
```

### Deterministic S-PIP (10-city TSPTW, hard)

```bash
python train.py --problem TSPTW_SPIP --problem_size 10 --hardness hard \
  --epochs 10000 --train_episodes 10000 --val_episodes 10000 \
  --train_batch_size 128 --generate_PI_mask
```

### Stochastic S-PIP (bounded noise)

```bash
python train.py --problem TSPTW_SPIP --problem_size 10 --hardness hard \
  --epochs 10000 --train_episodes 10000 --val_episodes 10000 \
  --train_batch_size 128 --generate_PI_mask --spip_stochastic_transition True
```

You can further configure the noise with:

- `--spip_noise_dist {uniform,clipped_gaussian}`
- `--spip_noise_bound 3.0` (for a wider or narrower bound)
- `--spip_sigma0 0.3` (Gaussian scale)

All other PIP/PIP-D options (`--pip_decoder`, `--generate_PI_mask`, etc.) behave as in the upstream project.

---

## Trillium S-PIP training scripts (10-city TSPTW)

The folder `[trillium_scripts](trillium_scripts/)` contains scripts specialized for Trillium.

### Shared trainer: `run_train_spip_n10.sh`

- Reads:
  - `HARDNESS` env var (`easy|medium|hard`, default `hard`)
  - `STOCHASTIC` env var (`0` or `1`, default `0`)
- Generates validation data (if missing) and runs S-PIP training with:
  - `problem = TSPTW_SPIP`
  - `problem_size = 10`
  - `epochs = 10000`, `train_episodes = 10000`, `val_episodes = 10000`
  - `train_batch_size = 128`
- Uses `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` internally to work around
  the `tensorboard_logger` + protobuf 4+ incompatibility.

### SLURM job scripts (account `rrg-cglee`)

All of these live in `[trillium_scripts](trillium_scripts/)` and are executable:

- `run_spip_n10_easy_cglee.slurm` – 10-city TSPTW, **easy** hardness
- `run_spip_n10_medium_cglee.slurm` – 10-city TSPTW, **medium** hardness
- `run_spip_n10_cglee.slurm` – 10-city TSPTW, **hard** hardness

Examples from the repo root:

```bash
# Hard, deterministic
sbatch trillium_scripts/run_spip_n10_cglee.slurm

# Hard, stochastic
sbatch --export=ALL,STOCHASTIC=1,HARDNESS=hard trillium_scripts/run_spip_n10_cglee.slurm

# Easy hardness, deterministic
sbatch trillium_scripts/run_spip_n10_easy_cglee.slurm

# Medium hardness, stochastic
sbatch --export=ALL,STOCHASTIC=1,HARDNESS=medium trillium_scripts/run_spip_n10_medium_cglee.slurm
```

> These job scripts are **cluster-specific**. Account-specific variants
> (e.g., `*_cglee.slurm`) are already listed in `.gitignore`.

---

## Evaluation

### Using `test.py` on TSPTW_SPIP

Basic evaluation on 10-city TSPTW S-PIP using the default data:

```bash
cd POMO+PIP
python test.py --problem TSPTW_SPIP --hardness hard --problem_size 10 \
  --checkpoint path/to/model.pt
```

To evaluate on a custom dataset and compute optimality gaps:

```bash
python test.py --test_set_path path/to/your_dataset.pkl \
  --checkpoint path/to/model.pt \
  --test_set_opt_sol_path path/to/opt_solutions.pkl
```

The interface mirrors the upstream PIP-constraint `test.py`:
- `--problem / --hardness / --problem_size` when using built-in data
- `--test_set_path` and `--test_set_opt_sol_path` for custom data and gap computation

---

## Troubleshooting

- **tensorboard_logger + protobuf error**  
  If you see:

  > `TypeError: Descriptors cannot be created directly.`

  this is due to newer `protobuf` versions. This fork sets:

  - `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` inside `[POMO+PIP/train.py](POMO+PIP/train.py)`
  - and in the Trillium scripts under `[trillium_scripts](trillium_scripts/)`

  so you should not need to set it manually for standard workflows.

- **GPU memory**  
  Adjust:
  - `--train_batch_size` in `train.py`
  - `--aug_batch_size` / `--test_batch_size` / `--eval_batch_size`

  according to your GPU memory. The defaults assume a reasonably large GPU (as in the original PIP experiments).

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

You may also be interested in the follow-up work
[CaR: Constraint-aware Routing](https://github.com/jieyibi/CaR-constraint), accepted at ICLR 2026.

---

## Acknowledgements

This project builds on and reuses code from:

- [POMO](https://github.com/yd-kwon/POMO)
- [Attention Model (AM)](https://github.com/wouterkool/attention-learn-to-route)
- [GFACS](https://github.com/ai4co/gfacs)
- [Routing-MVMoE](https://github.com/RoyalSkye/Routing-MVMoE)
- [PIP-constraint](https://github.com/jieyibi/PIP-constraint)
