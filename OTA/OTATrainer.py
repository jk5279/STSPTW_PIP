import torch
import torch.nn as nn
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import os
import sys
import numpy as np
import pickle
_OTA_DIR = os.path.dirname(os.path.abspath(__file__))
if _OTA_DIR not in sys.path:
    sys.path.insert(0, _OTA_DIR)
from utils import TimeEstimator, num_param, get_env
from models.OTAModel import OTAModel
from models.OTAAgent import OTAAgent

# ── Module-level state for fork-based parallel eval ─────────────────────────
# Set by _validate before Pool.map; inherited by forked workers via COW fork.
_EVAL_WORKER_TRAINER = None


def _eval_worker_chunk(iids):
    """Fork-based worker: evaluates a slice of val_instances with aug-8 rollout.

    Called inside a forked subprocess — inherits _EVAL_WORKER_TRAINER which
    already has the model moved to CPU.  torch.set_num_threads(1) prevents
    each worker from spawning its own OpenMP thread-pool and fighting over cores.
    """
    import torch as _t
    _t.set_num_threads(1)
    trainer = _EVAL_WORKER_TRAINER
    results = {}
    for iid in iids:
        idata = trainer.val_instances.get(iid)
        r = trainer._greedy_rollout_aug(
            iid, aug_factor=8, idata=idata, pip_step=trainer.pip_step
        )
        results[iid] = r
    return results


class OTATrainer:
    """
    Trainer for OTA (Option-aware Temporally Abstracted) method on STSPTW.
    Uses offline RL approach: learns from trajectory dataset.
    """

    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params):
        # save arguments
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.problem = self.args.problem

        self.device = args.device
        self.log_path = args.log_path
        self.result_log = {"val_score": [], "val_gap": [], "val_infsb_rate": [],
                           "val_buffer_gap_mean": [], "val_buffer_gap_median": [],
                           "val_buffer_gap_A": [], "val_buffer_gap_C": [], "val_buffer_gap_B": [],
                           "val_pol_gap_mean": [], "val_pol_gap_median": [],
                           "val_pol_gap_A": [], "val_pol_gap_C": [], "val_pol_gap_B": [],
                           "val_aug_gap_mean": [], "val_aug_gap_median": [],
                           "val_aug_gap_A": [], "val_aug_gap_C": [], "val_aug_gap_B": []}

        # Main Components
        self.envs = get_env(self.args.problem)
        self.model = OTAModel(**self.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        
        # Agent wrapper
        self.agent = OTAAgent(
            self.model,
            self.optimizer,
            self.scheduler,
            self.device,
            **trainer_params.get('agent_params', {})
        )

        num_param(self.model)

        # Per-instance validation helpers (populated in _load_dataset)
        self.inst_seed_cost  = {}   # instance_id -> seed tour distance
        self.inst_best_cost  = {}   # instance_id -> best feasible distance in buffer
        self.inst_group      = {}   # instance_id -> 'A' (1 unique FA), 'C' (2 unique FA), 'B' (>=3)
        self.inst_action_sets = {}  # instance_id -> {step_idx: set(actions)}
        self.inst_data       = {}   # instance_id -> dict(node_xy, service_time, tw_start, tw_end)
        self._tabu_cache     = {}   # instance_id -> tabu_rollout result dict (lazily filled)

        # Disk-persistent Tabu cache — lives next to the training dataset
        _ds = getattr(args, 'dataset', None) or ''
        self._tabu_cache_path = (os.path.splitext(_ds)[0] + '_tabu_cache.pkl') if _ds else None
        if self._tabu_cache_path and os.path.exists(self._tabu_cache_path):
            try:
                with open(self._tabu_cache_path, 'rb') as _f:
                    self._tabu_cache = pickle.load(_f)
                print(f'>> Tabu cache loaded: {len(self._tabu_cache)} entries from {self._tabu_cache_path}', flush=True)
            except Exception as _e:
                print(f'>> WARNING: could not load tabu cache ({_e}), starting fresh.', flush=True)

        # ── Val instances (separate held-out problem set) ────────────────────────────────
        self.val_instances = {}   # iid -> idata_dict
        self._val_tabu_cache = {}
        # Pre-computed reference solutions (val_opt pkl) — skips live Tabu when available
        self.val_opt = {}         # iid -> {'distance': float, 'feasible': bool}
        _val_opt_path = getattr(args, 'val_opt', None)
        if _val_opt_path and os.path.exists(_val_opt_path):
            try:
                with open(_val_opt_path, 'rb') as _f:
                    _opt_list = pickle.load(_f)
                for _entry in _opt_list:
                    _iid = int(_entry.get('instance_idx', len(self.val_opt)))
                    self.val_opt[_iid] = {
                        'distance': float(_entry['distance']),
                        'feasible': bool(_entry.get('feasible', False)),
                    }
                print(f'>> Val opt loaded: {len(self.val_opt)} entries from {_val_opt_path}', flush=True)
            except Exception as _e:
                print(f'>> WARNING: could not load val_opt ({_e})', flush=True)
        _val_ds = getattr(args, 'val_dataset', None) or ''
        self._val_tabu_cache_path = (os.path.splitext(_val_ds)[0] + '_tabu_cache.pkl') if _val_ds else None
        if self._val_tabu_cache_path and os.path.exists(self._val_tabu_cache_path):
            try:
                with open(self._val_tabu_cache_path, 'rb') as _f:
                    self._val_tabu_cache = pickle.load(_f)
                print(f'>> Val Tabu cache loaded: {len(self._val_tabu_cache)} entries from {self._val_tabu_cache_path}', flush=True)
            except Exception as _e:
                print(f'>> WARNING: could not load val tabu cache ({_e}), starting fresh.', flush=True)

        # Load offline RL dataset
        self.dataset = None
        self.dataset_size = 0
        self.expert_cost = None
        if hasattr(args, 'dataset') and args.dataset is not None:
            self._load_dataset(args.dataset)
        else:
            print(">> WARNING: No dataset provided. Using dummy batches for testing.")

        # Load val instances (held-out problem set, no trajectories needed)
        if _val_ds and os.path.exists(_val_ds):
            self._load_val_instances(_val_ds)

        # PIP lookahead step for inference masking (0=disabled, 1=1-step, 2=2-step)
        self.pip_step = int(getattr(args, 'pip_step', 0))
        if self.pip_step > 0:
            print(f'>> PIP inference masking enabled: pip_step={self.pip_step}', flush=True)

        # Aug-only mode: skip single greedy rollout, only run aug-8 during validation
        self.aug_only = bool(getattr(args, 'aug_only', False))
        if self.aug_only:
            print('>> aug_only mode: greedy rollout skipped during validation.', flush=True)

        # CPU-parallel eval: number of forked worker processes (0 = sequential on current device)
        self.num_eval_workers = int(getattr(args, 'num_eval_workers', 0))

        # Restore from checkpoint if provided
        self.start_epoch = 1
        if args.checkpoint is not None:
            checkpoint_fullname = args.checkpoint
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.start_epoch = 1 + checkpoint['epoch']
            self.scheduler.last_epoch = checkpoint['epoch'] - 1
            if self.trainer_params.get("load_optimizer", True):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(">> Optimizer (Epoch: {}) Loaded (lr = {})!".format(
                    checkpoint['epoch'], self.optimizer.param_groups[0]['lr']))
            print(">> Checkpoint (Epoch: {}) Loaded!".format(checkpoint['epoch']))

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        """Main training loop"""
        self.time_estimator.reset(self.start_epoch)

        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            print('=' * 80, flush=True)
            print('Epoch {}/{}'.format(epoch, self.trainer_params['epochs']), flush=True)
            print('=' * 80, flush=True)

            # Train
            print('>> Training...', flush=True)
            train_loss, train_info = self._train_one_epoch(epoch)
            print(f'>> Training complete - Loss: {train_loss:.4f}', flush=True)

            # Validation
            validation_interval = self.trainer_params.get('validation_interval', 500)
            if epoch == 1 or (epoch % validation_interval == 0):
                print('>> Running validation...', flush=True)
                val_score = self._validate(epoch)

            # Learning rate scheduling
            self.scheduler.step()

            # Save model
            model_save_interval = self.trainer_params.get('model_save_interval', 50)
            if epoch % model_save_interval == 0:
                self.save_model(epoch)

            # Logging
            print('>> Loss Details:', flush=True)
            for key, val in train_info.items():
                print(f"   {key}: {val:.4f}", flush=True)

            # Time estimation
            elapsed_str, remain_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            print(f">> Time Estimate: Elapsed[{elapsed_str}], Remain[{remain_str}]", flush=True)

    def _train_one_epoch(self, epoch):
        """
        Train for one epoch using offline RL from trajectory data.
        
        Returns:
            avg_loss: average loss over batches
            avg_info: averaged logging info
        """
        self.model.train()
        epoch_loss = 0
        epoch_info = {}
        batch_count = 0

        train_batch_size = self.trainer_params.get('train_batch_size', 64)
        accumulation_steps = self.trainer_params.get('accumulation_steps', 1)

        # Dataset-based epoch (shuffle once per epoch) or fallback to steps-based dummy batches
        if self.dataset is not None and len(self.dataset) > 0:
            num_samples = len(self.dataset)
            num_batches = max(1, num_samples // train_batch_size)
            indices = np.random.permutation(num_samples)
        else:
            num_batches = 1
            indices = None

        for batch_idx in range(num_batches):
            # Load batch from dataset if available, otherwise create dummy
            if self.dataset is not None and len(self.dataset) > 0:
                start = batch_idx * train_batch_size
                end = min(start + train_batch_size, len(self.dataset))
                batch_indices = indices[start:end]
                batch = self._load_batch_from_indices(batch_indices)
            else:
                batch = self._create_dummy_batch(train_batch_size)

            # Forward and loss
            loss, info = self.agent.update(batch, epoch * num_batches + batch_idx)

            epoch_loss += loss.item()
            for key, val in info.items():
                if key not in epoch_info:
                    epoch_info[key] = 0
                epoch_info[key] += val

            batch_count += 1

            if (batch_idx + 1) % 100 == 0:
                print(f">> Batch {batch_idx + 1}/{num_batches} | Loss: {loss.item():.4f}", flush=True)

        # Average
        avg_loss = epoch_loss / batch_count
        avg_info = {key: val / batch_count for key, val in epoch_info.items()}

        return avg_loss, avg_info

    def _load_dataset(self, dataset_path):
        """
        Load offline RL trajectory dataset from pickle file.
        
        Args:
            dataset_path: path to pickle file containing trajectories
        """
        if not os.path.exists(dataset_path):
            print(f">> WARNING: Dataset file not found: {dataset_path}")
            return
        
        try:
            with open(dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
            
            # Compute expert cost (best feasible tour from dataset)
            feasible_costs = []
            all_costs = []
            for traj in self.dataset:
                if isinstance(traj, dict):
                    cost = traj.get('distance', float('inf'))
                    feasible = traj.get('feasible', False)
                    all_costs.append(cost)
                    if feasible:
                        feasible_costs.append(cost)
            
            if feasible_costs:
                self.expert_cost = min(feasible_costs)
                print(f">> Expert cost (best feasible): {self.expert_cost/100:.4f}", flush=True)
                print(f">> Feasible trajectories: {len(feasible_costs)}/{len(all_costs)} ({len(feasible_costs)/len(all_costs)*100:.2f}%)", flush=True)
            else:
                self.expert_cost = min(all_costs) if all_costs else None
                print(f">> No feasible trajectories found. Best cost: {self.expert_cost:.2f if self.expert_cost else 'N/A'}")
            self.dataset_size = len(self.dataset) if isinstance(self.dataset, (list, tuple)) else 0
            print(f">> Loaded {self.dataset_size} trajectories from {dataset_path}", flush=True)

            # ── Per-instance precomputation ─────────────────────────────
            # Group trajectories by instance_id
            by_inst = {}  # instance_id -> list[traj]
            for traj in self.dataset:
                if not isinstance(traj, dict):
                    continue
                # Support both key names for backward compatibility
                iid = traj.get('instance_idx', traj.get('instance_id', -1))
                by_inst.setdefault(iid, []).append(traj)

            for iid, trajs in by_inst.items():
                # Seed cost = distance of trajectory index 0 within the group
                # (sorted by insertion order they were appended; traj 0 is seed)
                seed_traj = trajs[0]
                self.inst_seed_cost[iid] = seed_traj.get('distance', float('inf'))

                # Best feasible cost in buffer for this instance
                feas_dists = [t['distance'] for t in trajs if t.get('feasible', False)]
                self.inst_best_cost[iid] = min(feas_dists) if feas_dists else self.inst_seed_cost[iid]

                # First-action diversity → group assignment
                # A: 1 unique first action  (hardest for policy to improve)
                # C: 2 unique first actions (middle)
                # B: ≥3 unique first actions (most flexible)
                first_actions = {t.get('first_action', -1) for t in trajs
                                 if t.get('first_action', -1) >= 0}
                n_fa = len(first_actions)
                self.inst_group[iid] = 'A' if n_fa <= 1 else ('B' if n_fa >= 3 else 'C')

                # Per-step action sets (for BC gap)
                step_sets = {}
                for t in trajs:
                    tour = t.get('tour', [])
                    for step, node in enumerate(tour[1:], start=1):  # skip depot
                        step_sets.setdefault(step, set()).add(node)
                self.inst_action_sets[iid] = step_sets

            counts = {'A': 0, 'C': 0, 'B': 0}
            for g in self.inst_group.values():
                counts[g] = counts.get(g, 0) + 1
            print(f">> Instance groups  A(1 FA)={counts['A']}  "
                  f"C(2 FAs)={counts['C']}  B(>=3 FAs)={counts['B']}",
                  flush=True)

            # Cache instance data (node_xy, service_time, tw_start, tw_end)
            for iid, trajs in by_inst.items():
                for t in trajs:
                    idata = t.get('instance_data')
                    if idata is not None:
                        self.inst_data[iid] = idata
                        break
            if self.inst_data:
                print(f">> Instance data cached for {len(self.inst_data)} instances "
                      f"(greedy rollout enabled)", flush=True)
            else:
                print(">> No instance_data in dataset — greedy rollout disabled. "
                      "Regenerate with gen_small_dataset.py.", flush=True)
        except Exception as e:
            print(f">> ERROR loading dataset: {e}")
            self.dataset = None

    def _load_val_instances(self, val_path):
        """Load raw STSPTW problem instances from val pkl for held-out policy evaluation."""
        try:
            with open(val_path, 'rb') as f:
                instances = pickle.load(f)
            for iid, item in enumerate(instances):
                if isinstance(item, (tuple, list)) and len(item) >= 3:
                    arrays = [np.array(x, dtype=np.float32) for x in item]
                    self.val_instances[iid] = {
                        'node_xy':      arrays[0],
                        'service_time': arrays[1] if len(arrays) > 1 else np.zeros(len(arrays[0]), dtype=np.float32),
                        'tw_start':     arrays[2] if len(arrays) > 2 else np.zeros(len(arrays[0]), dtype=np.float32),
                        'tw_end':       arrays[3] if len(arrays) > 3 else np.full(len(arrays[0]), 1e6, dtype=np.float32),
                    }
                elif isinstance(item, dict):
                    self.val_instances[iid] = {k: np.array(v, dtype=np.float32) for k, v in item.items()}
            print(f'>> Val instances loaded: {len(self.val_instances)} from {val_path}', flush=True)
        except Exception as e:
            print(f'>> ERROR loading val instances: {e}', flush=True)

    def _load_batch_from_indices(self, batch_indices):
        """
        Load a batch of trajectories using explicit indices.

        Args:
            batch_indices: 1D array of indices into the dataset

        Returns:
            batch: dict with required keys for model
        """
        if self.dataset is None or len(self.dataset) == 0:
            return self._create_dummy_batch(len(batch_indices))

        sampled_trajectories = [self.dataset[int(i)] for i in batch_indices]
        return self._trajectories_to_batch(sampled_trajectories)

    # ------------------------------------------------------------------
    # Rich observation builder
    # ------------------------------------------------------------------
    def _make_goal_obs(self, idata: dict) -> torch.Tensor:
        """
        Build a canonical, tour-order-independent goal observation.

        Semantics: "standing at depot (t=0), all nodes have been visited."
        Matches _make_rich_obs exactly: 6 global + 10 per-node features.

        Global (6):
            cur_x, cur_y   = depot / loc_factor
            cur_time       = 0.0
            frac_remaining = 0.0  (all visited)
            last_noise     = 0.0  (no noise at goal)
            cum_noise      = 0.0

        Per-node (10 each): same as _make_rich_obs, fwd_noise = 0 (all visited).
        """
        loc_factor = 100.0
        node_xy  = torch.as_tensor(idata['node_xy'],  dtype=torch.float32)
        tw_start = torch.as_tensor(idata['tw_start'], dtype=torch.float32)
        tw_end   = torch.as_tensor(idata['tw_end'],   dtype=torch.float32)
        n_nodes  = node_xy.shape[0]

        global_feats = torch.tensor([
            node_xy[0, 0] / loc_factor,
            node_xy[0, 1] / loc_factor,
            0.0,   # cur_time = 0
            0.0,   # frac_remaining = 0 (all done)
            0.0,   # last_noise = 0
            0.0,   # cum_noise  = 0
        ], dtype=torch.float32)

        visited   = torch.ones(n_nodes, dtype=torch.float32)  # all visited
        fwd_noise = torch.zeros(n_nodes, dtype=torch.float32)  # all visited → 0
        d_raw   = (node_xy - node_xy[0]).norm(dim=1)           # dist from depot
        arrival = d_raw                                          # t=0
        start   = torch.maximum(arrival, tw_start)
        wait    = torch.clamp(tw_start - arrival, min=0.0)
        slack   = tw_end - start
        feas    = (start <= tw_end).float()

        xy_s  = node_xy  / loc_factor
        tws_s = tw_start / loc_factor
        twe_s = tw_end   / loc_factor
        d_s   = d_raw    / loc_factor
        w_s   = wait     / loc_factor
        sl_s  = slack    / loc_factor

        per_node = torch.stack(
            [xy_s[:, 0], xy_s[:, 1], tws_s, twe_s, visited, d_s, w_s, sl_s, feas, fwd_noise],
            dim=1,
        ).flatten()  # 10 × n_nodes

        return torch.cat([global_feats, per_node])  # 6 + 10*n_nodes

    def _make_rich_obs(self, t: int, nodes: list, idata: dict,
                       noise_hist: list = None) -> torch.Tensor:
        """
        Build rich TW-aware, noise-aware observation at step t.

        obs = [global (6), per_node (10 × n_nodes)]  → (6 + 10*n_nodes,)

        Global (6):
            cur_x, cur_y    – current position / loc_factor
            cur_time        – current time (with accumulated noise) / loc_factor
            frac_remaining  – (K - t) / K
            last_noise      – U(0,√2) noise on the last traversed edge / loc_factor
                              (0 at depot / first step)
            cum_noise       – total accumulated noise over the prefix / loc_factor

        Per-node (10 each, depot + every customer):
            x_i, y_i                         – location / loc_factor
            a_i, b_i                         – TW start/end / loc_factor
            visited_i                        – 1 if visited (depot always 1)
            d_i   = dist(cur, i)             – Euclidean, / loc_factor
            wait_i = max(0, a_i - arrival)   – / loc_factor
            slack_i = b_i - start_i          – / loc_factor (< 0 ⟹ infeasible)
            feas_i  = 1[start_i <= b_i]
            fwd_noise_i                      – prospective noise sample U(0,√2)
                                               drawn for each *unvisited* node;
                                               0 for depot and already-visited
                                               nodes.  Lets the model estimate
                                               noisy arrival = cur_time +
                                               (d_i + fwd_noise_i) / speed.

        Noise handling:
            If noise_hist is provided (list[float], len >= t), those exact per-step
            noises are replayed (deterministic rollout / evaluation).
            Otherwise fresh U(0,√2) samples are drawn for every prefix step
            (training-time augmentation — teaches the value function about noise
            even from deterministic buffer trajectories).
        """
        import numpy as _np
        _noise_max = _np.sqrt(2)
        loc_factor = 100.0

        node_xy  = torch.as_tensor(idata['node_xy'],  dtype=torch.float32)  # (n, 2)
        tw_start = torch.as_tensor(idata['tw_start'], dtype=torch.float32)  # (n,)
        tw_end   = torch.as_tensor(idata['tw_end'],   dtype=torch.float32)  # (n,)
        n_nodes  = node_xy.shape[0]
        K        = len(nodes)

        # ── Replay prefix with noise to reconstruct current_time ─────────────
        #   arrival  = time + dist(prev, next) + noise_step
        #   time     = max(arrival, tw_start[next])      ← no service_time
        if noise_hist is None:
            # Training: sample fresh per-step noises for data augmentation
            noise_hist = [float(_np.random.uniform(0.0, _noise_max)) for _ in range(t)]
        current_time = 0.0
        current_node = 0
        for i in range(t):
            nxt          = nodes[i]
            d_step       = float((node_xy[current_node] - node_xy[nxt]).norm())
            noise_step   = float(noise_hist[i]) if i < len(noise_hist) else 0.0
            current_time = max(current_time + d_step + noise_step, float(tw_start[nxt]))
            current_node = nxt

        last_noise = (float(noise_hist[t - 1]) if t > 0 and t <= len(noise_hist)
                      else 0.0) / loc_factor
        cum_noise  = sum(float(noise_hist[i]) for i in range(min(t, len(noise_hist)))) / loc_factor

        # ── Global features (6) ──────────────────────────────────────────────
        global_feats = torch.tensor([
            node_xy[current_node, 0] / loc_factor,
            node_xy[current_node, 1] / loc_factor,
            current_time             / loc_factor,
            (K - t) / max(K, 1),
            last_noise,
            cum_noise,
        ], dtype=torch.float32)                                             # (6,)

        # ── Vectorized per-node features (10 × n_nodes) ──────────────────────
        # visited flag
        visited = torch.zeros(n_nodes, dtype=torch.float32)
        visited[0] = 1.0
        for node in nodes[:t]:
            visited[node] = 1.0

        cur_t   = torch.tensor(current_time, dtype=torch.float32)
        d_raw   = (node_xy - node_xy[current_node]).norm(dim=1)            # (n,)
        arrival = cur_t + d_raw                                            # Euclidean arrival
        start   = torch.maximum(arrival, tw_start)                         # (n,)
        wait    = torch.clamp(tw_start - arrival, min=0.0)                 # (n,)
        slack   = tw_end - start                                           # (n,)
        feas    = (start <= tw_end).float()                                # (n,)

        xy_s  = node_xy  / loc_factor                                      # (n, 2)
        tws_s = tw_start / loc_factor                                      # (n,)
        twe_s = tw_end   / loc_factor                                      # (n,)
        d_s   = d_raw    / loc_factor                                      # Euclidean / lf
        w_s   = wait     / loc_factor                                      # (n,)
        sl_s  = slack    / loc_factor                                      # (n,)

        # Prospective noise: independently sampled for each unvisited node.
        # Gives the model a forward noise signal: effective arrival ≈
        #   (cur_time + d_i + fwd_noise_i) / speed.
        # Visited nodes and depot get 0 (noise is already baked into cur_time).
        fwd_noise = torch.zeros(n_nodes, dtype=torch.float32)
        for j in range(n_nodes):
            if visited[j] == 0.0:
                fwd_noise[j] = float(_np.random.uniform(0.0, _noise_max)) / loc_factor

        per_node = torch.stack(
            [xy_s[:, 0], xy_s[:, 1], tws_s, twe_s, visited, d_s, w_s, sl_s, feas, fwd_noise],
            dim=1,
        ).flatten()                                                        # (10*n,)

        return torch.cat([global_feats, per_node])                         # (6 + 10*n,)

    def _trajectories_to_batch(self, trajectories):
        """
        Convert trajectory data to model-compatible batch format.

        Implements step-level sampling with temporal abstraction, matching the
        HGCDataset logic from the reference OTA implementation (ota-v/ota-v).

        For each trajectory we:
          1. Strip depot (node 0) to get the ordered customer sequence nodes[0..K-1].
          2. Pick a uniformly random step t ∈ [0, K-1].
          3. Build binary visited-nodes state vectors at t, t+1, t+abstraction_factor,
             t+subgoal_steps (all capped at K).
          4. Compute the average actual edge cost over abstraction_factor steps.
          5. Use the final state (all customers visited) as the goal for both
             value and actor objectives — the natural goal for TSP.

        Batch keys match the reference ota.py exactly:
            observations                   : s_t
            next_observations              : s_{t+1}
            rewards                        : r_t  (total_reward / K, per step)
            masks                          : 1 unless t is the last step
            value_goals                    : final state (goal for low value)
            low_actor_goals                : final state (goal for low actor)
            high_value_option_observations : s_{t + abstraction_factor}
            high_value_rewards             : mean edge cost (negated) over [t, t+K_abs)
            high_value_masks               : 1 unless t+abstraction_factor >= K
            high_value_goals               : final state (goal for high value)
            high_actor_targets             : s_{t + subgoal_steps}
            high_actor_goals               : final state (goal for high actor)
            actions                        : one-hot over problem_size at the action node
        """
        obs_dim      = self.model_params.get('obs_dim', self.model_params.get('embedding_dim', 128))
        problem_size = self.env_params.get('problem_size', 50)
        agent_p      = self.trainer_params.get('agent_params', {})
        K_abs        = agent_p.get('abstraction_factor', 5)
        K_sub        = agent_p.get('subgoal_steps', 25)

        obs_list, nobs_list, r_list, mask_list   = [], [], [], []
        vg_list, lag_list                         = [], []
        hvo_list, hvr_list, hvm_list, hvg_list    = [], [], [], []
        hat_list, hag_list, act_list              = [], [], []

        zero = torch.zeros(obs_dim, dtype=torch.float32)

        for traj in trajectories:
            if not isinstance(traj, dict):
                # Fallback: zero tensors
                for lst in (obs_list, nobs_list, vg_list, lag_list,
                            hvo_list, hvg_list, hat_list, hag_list, act_list):
                    lst.append(zero.clone())
                r_list.append(0.0); mask_list.append(0.0)
                hvr_list.append(0.0); hvm_list.append(0.0)
                continue

            tour     = traj.get('tour', [])
            reward   = float(traj.get('reward', 0.0))
            feasible = traj.get('feasible', False)
            idata    = traj.get('instance_data', None)

            # Strip depot (node 0) → ordered customer sequence
            nodes = [int(n) for n in tour if int(n) != 0]
            K     = len(nodes)

            if K == 0 or idata is None:
                for lst in (obs_list, nobs_list, vg_list, lag_list,
                            hvo_list, hvg_list, hat_list, hag_list, act_list):
                    lst.append(zero.clone())
                r_list.append(0.0); mask_list.append(0.0)
                hvr_list.append(0.0); hvm_list.append(0.0)
                continue

            # ── Sample a random step t ∈ [0, K-1] ──────────────────────
            t = int(np.random.randint(0, K))

            # ── Rich observations at t, t+1 ──────────────────────────────
            s_t  = self._make_rich_obs(t,     nodes, idata)
            s_t1 = self._make_rich_obs(t + 1, nodes, idata)

            # Dense reward per step (distribute total trajectory reward evenly).
            # The stored 'reward' already includes the soft lateness penalty
            # (lambda * total_lateness / REWARD_SCALE) baked in by gen_small_dataset.
            # No additional flat penalty is applied here — signal comes from
            # the proportional lateness term in the stored reward.
            r_per_step = reward / K

            # Terminal mask: 0 at last customer step, 1 otherwise
            mask_t = 0.0 if t == K - 1 else 1.0

            # Goal = canonical instance-level obs (depot, t=0, all visited).
            # Tour-order-independent: same for all trajectories on this instance,
            # and reproducible identically at rollout time.
            final_state = self._make_goal_obs(idata)

            # ── High-value: obs at t + abstraction_factor ────────────────
            t_abs    = min(t + K_abs, K)
            s_abs    = self._make_rich_obs(t_abs, nodes, idata)
            # High-value reward: average of actual edge costs over [t, t_abs).
            # Use pre-stored edge_costs from the buffer when available; fall back
            # to recomputing from coordinates for legacy trajectories.
            n_abs_steps = t_abs - t
            if n_abs_steps > 0:
                stored_ec = traj.get('edge_costs')
                if stored_ec is not None and len(stored_ec) >= t_abs:
                    r_abs = -float(sum(stored_ec[t:t_abs]) / n_abs_steps)
                else:
                    _node_xy    = torch.as_tensor(idata['node_xy'], dtype=torch.float32)
                    _LOC_FACTOR = 100.0
                    _ec = [
                        float((_node_xy[nodes[_k]] - _node_xy[0 if _k == 0 else nodes[_k - 1]]).norm())
                        / _LOC_FACTOR
                        for _k in range(t, t_abs)
                    ]
                    r_abs = -float(sum(_ec) / n_abs_steps)
            else:
                r_abs = 0.0
            mask_abs = 0.0 if t_abs >= K else 1.0

            # ── High-actor target: obs at t + subgoal_steps ──────────────
            t_sub = min(t + K_sub, K)
            s_sub = self._make_rich_obs(t_sub, nodes, idata)

            # ── Action: one-hot for the node visited at step t ──────────
            # Action space = problem_size nodes (actor output dim)
            a = torch.zeros(problem_size, dtype=torch.float32)
            idx = nodes[t] - 1   # 1-indexed → 0-indexed
            if 0 <= idx < problem_size:
                a[idx] = 1.0

            obs_list.append(s_t);   nobs_list.append(s_t1)
            r_list.append(float(r_per_step)); mask_list.append(mask_t)
            vg_list.append(final_state);  lag_list.append(final_state)
            hvo_list.append(s_abs);  hvr_list.append(float(r_abs))
            hvm_list.append(mask_abs); hvg_list.append(final_state)
            hat_list.append(s_sub);  hag_list.append(final_state)
            act_list.append(a)

        def _stack(lst):
            return torch.stack(lst, dim=0).to(self.device)
        def _scalar(lst):
            return torch.tensor(lst, dtype=torch.float32).to(self.device)

        return {
            'observations':                    _stack(obs_list),
            'next_observations':               _stack(nobs_list),
            'rewards':                         _scalar(r_list),
            'masks':                           _scalar(mask_list),
            'value_goals':                     _stack(vg_list),
            'low_actor_goals':                 _stack(lag_list),
            'high_value_option_observations':  _stack(hvo_list),
            'high_value_rewards':              _scalar(hvr_list),
            'high_value_masks':                _scalar(hvm_list),
            'high_value_goals':                _stack(hvg_list),
            'high_actor_targets':              _stack(hat_list),
            'high_actor_goals':                _stack(hag_list),
            'actions':                         _stack(act_list),
        }

    def _create_dummy_batch(self, batch_size):
        """Dummy batch for testing — uses same keys as _trajectories_to_batch."""
        obs_dim      = self.model_params.get('obs_dim', self.model_params.get('embedding_dim', 128))
        problem_size = self.env_params.get('problem_size', 50)
        s  = torch.randn(batch_size, obs_dim).to(self.device)
        sn = torch.randn(batch_size, obs_dim).to(self.device)
        g  = torch.randn(batch_size, obs_dim).to(self.device)
        return {
            'observations':                    s,
            'next_observations':               sn,
            'rewards':                         torch.randn(batch_size).to(self.device),
            'masks':                           torch.ones(batch_size).to(self.device),
            'value_goals':                     g,
            'low_actor_goals':                 g,
            'high_value_option_observations':  torch.randn(batch_size, obs_dim).to(self.device),
            'high_value_rewards':              torch.randn(batch_size).to(self.device),
            'high_value_masks':                torch.ones(batch_size).to(self.device),
            'high_value_goals':                g,
            'high_actor_targets':              torch.randn(batch_size, obs_dim).to(self.device),
            'high_actor_goals':                g,
            'actions':                         torch.randn(batch_size, problem_size).to(self.device),
        }

    @staticmethod
    def _pip_blocked(current, current_time, visited_set, n_cust, dist, tw_start, tw_end, pip_step=1):
        """
        Return the set of PIP-blocked nodes using deterministic lookahead.

        pip_step=1 — block j if visiting j makes any remaining node k directly
                     unreachable: max(t_j + dist[j][k], tw_start[k]) > tw_end[k].
                     O(n^2) per call.

        pip_step=2 — additionally block j if, after going j→k for any reachable k,
                     some third node l becomes directly unreachable from k.
                     O(n^3) per call (trivial for n≤20).

        Safety: if ALL unvisited nodes would be blocked, returns empty set so
        the caller falls through to the plain TW-only mask (no dead-lock amplification).
        """
        unvisited = [n for n in range(1, n_cust + 1) if n not in visited_set]
        if len(unvisited) <= 1:
            return set()

        blocked = set()
        for j in unvisited:
            t_j = max(current_time + dist[current][j], tw_start[j])
            remaining_after_j = [k for k in unvisited if k != j]
            if not remaining_after_j:
                continue  # last customer — nothing to strand

            # pip_step ≥ 1: block j if any remaining node is directly unreachable from j
            for k in remaining_after_j:
                if max(t_j + dist[j][k], tw_start[k]) > tw_end[k]:
                    blocked.add(j)
                    break

            # pip_step ≥ 2: additionally check one level deeper
            if pip_step >= 2 and j not in blocked:
                for k in remaining_after_j:
                    t_k = max(t_j + dist[j][k], tw_start[k])
                    if t_k > tw_end[k]:
                        continue  # k unreachable from j; already caught above
                    remaining_after_jk = [l for l in remaining_after_j if l != k]
                    if not remaining_after_jk:
                        continue
                    for l in remaining_after_jk:
                        if max(t_k + dist[k][l], tw_start[l]) > tw_end[l]:
                            blocked.add(j)
                            break
                    if j in blocked:
                        break

        # Safety: never block every candidate
        if len(blocked) == len(unvisited):
            return set()
        return blocked

    def _greedy_rollout(self, iid, idata=None, pip_step=None):
        """
        Greedily decode a tour for instance `iid` under the current policy.
        Pass `idata` directly to evaluate on val instances or augmented coords.

        TW semantics
        ------------
        arrival  = current_time + dist(cur, j)
        t_next   = max(arrival, tw_start[j])  (wait if early; service_time = 0)
        feasible iff  t_next <= tw_end[j]

        Masking: node j blocked if t_next > tw_end[j]  (uses waiting, not raw arrival).

        No fallback: if every unvisited node is TW-infeasible the rollout is
        marked infeasible, the step index is recorded, and decoding stops.

        Returns dict: tour, distance, feasible, dead_step (None if completed).
        Returns None if instance data unavailable.
        """
        if idata is None:
            idata = self.inst_data.get(iid)
        if idata is None:
            return None

        node_xy  = idata['node_xy']    # (n_nodes, 2), node 0 = depot
        tw_start = idata['tw_start']  # (n_nodes,)
        tw_end   = idata['tw_end']    # (n_nodes,)
        n_nodes  = len(node_xy)
        n_cust   = n_nodes - 1

        obs_dim      = self.model_params.get('obs_dim', self.model_params.get('embedding_dim', 128))
        problem_size = self.env_params.get('problem_size', 50)

        diff = node_xy[:, None, :] - node_xy[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))

        # Canonical goal: depot, t=0, all visited — identical to training goal
        idata_dict = {'node_xy': node_xy, 'tw_start': tw_start, 'tw_end': tw_end}
        goal_obs = self._make_goal_obs(idata_dict)
        g_in = goal_obs.to(self.device).unsqueeze(0)

        # Scale-invariant λ (same formula as STSPTWEnv.reset)
        _diffs     = node_xy[:, None, :] - node_xy[None, :, :]
        _mean_edge = float(np.sqrt((_diffs ** 2).sum(axis=-1)).mean())
        _tw_widths = np.maximum(tw_end[1:] - tw_start[1:], 1e-6)
        lam        = 5.0 * max(_mean_edge, 1e-6) / max(float(_tw_widths.mean()), 1e-6)
        _REWARD_SCALE = 2600.0

        NEG_INF = torch.finfo(torch.float32).min / 2

        self.model.eval()
        with torch.no_grad():
            tour           = [0]
            visited        = set()
            nodes_visited  = []
            current        = 0
            current_time   = 0.0
            total_dist     = 0.0
            total_lateness = 0.0
            violations     = 0
            dead_step      = None
            noise_hist     = []
            import numpy as _np_r

            def _serve(node):
                """Visit node: advance time+noise, accumulate dist/lateness/violations."""
                nonlocal current, current_time, total_dist, total_lateness, violations
                step_noise    = float(_np_r.random.uniform(0.0, _np_r.sqrt(2)))
                arrival       = current_time + dist[current][node] + step_noise
                current_time  = max(arrival, tw_start[node])
                total_dist   += dist[current][node] + step_noise
                lateness      = max(0.0, arrival - tw_end[node])
                if lateness > 1e-6:
                    violations    += 1
                    total_lateness += lateness
                current = node
                visited.add(node)
                nodes_visited.append(node)
                tour.append(node)
                noise_hist.append(step_noise)

            for step in range(1, n_cust + 1):
                obs = self._make_rich_obs(len(nodes_visited), nodes_visited,
                                          idata_dict, noise_hist=noise_hist)
                s_in = obs.to(self.device).unsqueeze(0)

                goal_rep = self.model.compute_goal_representation(s_in, g_in)
                logits   = self.model.get_low_actor_logits(s_in, goal_rep).squeeze(0)

                # Build TW-feasibility mask (step-0: direct arrival check)
                _pip = pip_step if pip_step is not None else self.pip_step
                mask     = torch.full((problem_size,), NEG_INF, dtype=torch.float32,
                                      device=self.device)
                tw_open = set()
                for node in range(1, n_cust + 1):
                    if node in visited:
                        continue
                    arrival_est = current_time + dist[current][node]
                    if max(arrival_est, tw_start[node]) <= tw_end[node]:
                        tw_open.add(node)

                # PIP lookahead filter (pip_step≥1): remove nodes that strand future nodes
                if _pip > 0 and len(tw_open) > 1:
                    pip_bl = OTATrainer._pip_blocked(
                        current, current_time, visited, n_cust,
                        dist, tw_start, tw_end, pip_step=_pip
                    )
                    tw_open -= pip_bl

                for node in tw_open:
                    idx = node - 1
                    if idx < problem_size:
                        mask[idx] = 0.0
                any_open = bool(tw_open)

                if not any_open:
                    # No TW-feasible choice → mark dead step, force nearest unvisited.
                    # Continue the loop (trajectory always completes all n_cust steps).
                    if dead_step is None:
                        dead_step = step
                    forced = min(
                        (n for n in range(1, n_cust + 1) if n not in visited),
                        key=lambda n: dist[current][n]
                    )
                    _serve(forced)
                else:
                    _serve(int((logits + mask).argmax().item()) + 1)

            # Return to depot
            total_dist += dist[current][0]

            penalized_cost = total_dist + lam * total_lateness
            reward         = -(penalized_cost / _REWARD_SCALE)
            return {
                'tour':           tour,
                'distance':       total_dist,
                'violations':     violations,
                'total_lateness': total_lateness,
                'penalized_cost': penalized_cost,
                'reward':         reward,
                'feasible':       violations == 0,
                'dead_step':      dead_step,
            }

    @staticmethod
    def _augment_instance(node_xy):
        """
        Produce 8 symmetry-augmented copies of node coordinates.
        Matches the POMO/STSPTWEnv 8-fold augmentation exactly.

        Args:
            node_xy: np.ndarray (n_nodes, 2), already zero-indexed,
                     values expected in [0, 1] after loc_factor normalisation.

        Returns:
            list of 8 np.ndarray (n_nodes, 2), one per augmentation.
        """
        x = node_xy[:, 0:1]
        y = node_xy[:, 1:2]
        return [
            np.concatenate([x,     y    ], axis=1),  # original
            np.concatenate([1 - x, y    ], axis=1),  # flip x
            np.concatenate([x,     1 - y], axis=1),  # flip y
            np.concatenate([1 - x, 1 - y], axis=1),  # flip xy
            np.concatenate([y,     x    ], axis=1),  # swap
            np.concatenate([1 - y, x    ], axis=1),  # swap + flip x
            np.concatenate([y,     1 - x], axis=1),  # swap + flip y
            np.concatenate([1 - y, 1 - x], axis=1),  # swap + flip xy
        ]

    def _tabu_rollout(self, iid):
        """
        Run vrp_bench Tabu Search on instance `iid` and cache the result.

        Requires instance_data to be present (stamped by generate_ota_dataset.py).
        Returns a dict with the same schema as _greedy_rollout, or None.
        """
        if iid in self._tabu_cache:
            return self._tabu_cache[iid]
        idata = self.inst_data.get(iid)
        if idata is None:
            return None
        # Lazy import from parent directory
        import sys as _sys, os as _os
        _parent = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _parent not in _sys.path:
            _sys.path.insert(0, _parent)
        try:
            from generate_ota_dataset import STSPTWDatasetGenerator
        except ImportError:
            return None
        gen = STSPTWDatasetGenerator(seed=0)
        node_xy  = idata['node_xy']    # raw [0,100]
        tw_start = idata['tw_start']
        tw_end   = idata['tw_end']
        svc      = idata.get('service_time', np.zeros(len(node_xy)))
        problem  = (node_xy.tolist(), svc.tolist(), tw_start.tolist(), tw_end.tolist())
        result   = gen.tabu_tour(problem, time_limit_s=2.0, seed=0)
        # Compute penalized cost (same λ formula)
        _diffs     = node_xy[:, None, :] - node_xy[None, :, :]
        _mean_edge = float(np.sqrt((_diffs ** 2).sum(axis=-1)).mean())
        _tw_widths = np.maximum(tw_end[1:] - tw_start[1:], 1e-6)
        lam        = 5.0 * max(_mean_edge, 1e-6) / max(float(_tw_widths.mean()), 1e-6)
        dist_raw   = result.get('distance', 0.0)
        # Tabu eval is deterministic (no noise); penalized = dist (feasible tours have 0 lateness)
        penalized  = dist_raw  # Tabu _eval_tour already includes VIOLATION_PENALTY if infeasible
        out = {
            'tour':           result.get('tour', []),
            'distance':       dist_raw,
            'violations':     result.get('violations', 0),
            'total_lateness': 0.0,
            'penalized_cost': penalized,
            'reward':         result.get('reward', 0.0),
            'feasible':       result.get('feasible', False),
            'dead_step':      None,
        }
        self._tabu_cache[iid] = out
        # Persist to disk so future runs skip re-solving
        if self._tabu_cache_path:
            try:
                with open(self._tabu_cache_path, 'wb') as _f:
                    pickle.dump(self._tabu_cache, _f)
            except Exception:
                pass
        return out

    def _val_tabu_rollout(self, iid):
        """Like _tabu_rollout but uses val_instances and _val_tabu_cache (disk-persistent)."""
        if iid in self._val_tabu_cache:
            return self._val_tabu_cache[iid]
        idata = self.val_instances.get(iid)
        if idata is None:
            return None
        import sys as _sys, os as _os
        _parent = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _parent not in _sys.path:
            _sys.path.insert(0, _parent)
        try:
            from generate_ota_dataset import STSPTWDatasetGenerator
        except ImportError:
            return None
        gen = STSPTWDatasetGenerator(seed=0)
        node_xy  = idata['node_xy']
        tw_start = idata['tw_start']
        tw_end   = idata['tw_end']
        svc      = idata.get('service_time', np.zeros(len(node_xy), dtype=np.float32))
        problem  = (node_xy.tolist(), svc.tolist(), tw_start.tolist(), tw_end.tolist())
        result   = gen.tabu_tour(problem, time_limit_s=2.0, seed=0)
        dist_raw = result.get('distance', 0.0)
        out = {
            'tour':           result.get('tour', []),
            'distance':       dist_raw,
            'violations':     result.get('violations', 0),
            'total_lateness': 0.0,
            'penalized_cost': dist_raw,
            'reward':         result.get('reward', 0.0),
            'feasible':       result.get('feasible', False),
            'dead_step':      None,
        }
        self._val_tabu_cache[iid] = out
        if self._val_tabu_cache_path:
            try:
                with open(self._val_tabu_cache_path, 'wb') as _f:
                    pickle.dump(self._val_tabu_cache, _f)
            except Exception:
                pass
        return out

    def _greedy_rollout_aug(self, iid, aug_factor=8, idata=None, pip_step=None):
        """
        Run _greedy_rollout on `aug_factor` symmetry-augmented copies of the
        instance and return the best result (feasible-first, then shortest).

        Pass `idata` directly to evaluate val instances or augmented coords.

        Args:
            iid: instance id
            aug_factor: 1 or 8
            idata: optional pre-fetched instance data
            pip_step: PIP lookahead depth (None = use self.pip_step)

        Returns:
            best result dict (same schema as _greedy_rollout), or None.
        """
        if idata is None:
            idata = self.inst_data.get(iid)
        if idata is None:
            return None
        if aug_factor == 1:
            return self._greedy_rollout(iid, idata=idata, pip_step=pip_step)

        node_xy_orig = idata['node_xy']   # (n_nodes, 2)
        tw_start     = idata['tw_start']
        tw_end       = idata['tw_end']

        # Pre-compute original distances for re-evaluation
        diff_orig = node_xy_orig[:, None, :] - node_xy_orig[None, :, :]
        dist_orig = np.sqrt((diff_orig ** 2).sum(axis=-1))

        aug_variants = self._augment_instance(node_xy_orig)  # list of 8 arrays

        best = None
        n_feasible_aug = 0  # count of individual aug solutions that are feasible
        for aug_xy in aug_variants:
            # Build a temporary idata with augmented coords and run rollout
            aug_idata = {'node_xy': aug_xy, 'tw_start': tw_start, 'tw_end': tw_end}
            result = self._greedy_rollout(iid, idata=aug_idata, pip_step=pip_step)

            if result is None:
                continue

            # Re-evaluate tour distance on *original* coordinates so all
            # aug variants are comparable on the same metric.
            tour = result['tour']
            orig_dist = sum(
                dist_orig[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)
            ) + dist_orig[tour[-1]][0]  # return to depot
            result['distance'] = orig_dist

            if result['feasible']:
                n_feasible_aug += 1

            if best is None:
                best = result
                continue
            # Prefer feasible; among equal feasibility prefer shorter
            if result['feasible'] and not best['feasible']:
                best = result
            elif result['feasible'] == best['feasible'] and orig_dist < best['distance']:
                best = result

        if best is not None:
            best['n_feasible_aug'] = n_feasible_aug
            best['n_aug'] = len(aug_variants)
        return best

    def _validate(self, epoch):
        """
        Validation step.

        Metrics reported
        ----------------
        [Dataset stats — constant across epochs, shown as reference]
          feasible%   : fraction of stored trajectories that are feasible
          cost<=seed% : fraction of stored trajectories with cost <= seed cost
                        NOTE: these reflect buffer quality, not policy performance.

        [Buffer-gap stats — upper bound on what imitation can achieve]
          buffer_gap = best_buffer_cost - seed_cost  (per instance)
          Δ < 0 means the buffer contains a tour better than the seed.
          Δ = 0 means seed is already the best in the buffer.
          Reports mean, median, pct_improved (=pct where Δ < 0)
          Split by groups:
            A: 1 unique first-action  (policy has no first-step diversity)
            C: 2 unique first-actions (middle)
            B: ≥3 unique first-actions (flexible; expect improvement here first)

        [BC coverage — data-coverage check, constant across epochs]
          Per step: fraction of instances where >1 action is in the buffer.
          <10%  → hard BC gap risk
          10-25%→ soft note (expected for late steps)
        """
        self.model.eval()
        print(f"\n>> Validation at epoch {epoch}", flush=True)

        if self.dataset is None or len(self.dataset) == 0:
            print(">> No dataset for validation")
            return 0.0

        # ── Dataset stats (constant reference) ─────────────────────────
        n_sample = min(len(self.dataset), len(self.dataset))  # use full dataset
        all_costs, n_feas, n_leq_seed = [], 0, 0
        for traj in self.dataset:
            if not isinstance(traj, dict):
                continue
            iid  = traj.get('instance_id', -1)
            dist = traj.get('distance', 0.0)
            all_costs.append(dist)
            if traj.get('feasible', False):
                n_feas += 1
            if iid in self.inst_seed_cost and dist <= self.inst_seed_cost[iid]:
                n_leq_seed += 1
        n_total   = max(len(all_costs), 1)
        feas_pct  = n_feas  / n_total * 100
        leq_pct   = n_leq_seed / n_total * 100
        avg_cost  = float(np.mean(all_costs)) if all_costs else 0.0

        print(f"   [Dataset — buffer quality, constant]", flush=True)
        print(f"     Trajectories : {n_total}  |  "
              f"Feasible: {feas_pct:.1f}%  |  "
              f"Cost<=seed: {leq_pct:.1f}%", flush=True)
        if self.expert_cost:
            print(f"     Expert cost (best feasible in buffer): {self.expert_cost/100:.4f}",
                  flush=True)

        # ── Buffer-gap: best_buffer_cost - seed_cost ────────────────────
        # Convention: Δ < 0  ⟹  buffer contains a tour better than seed
        #             Δ = 0  ⟹  seed is already the best in the buffer
        # This is an upper bound on what the policy can achieve via imitation.
        gaps, gaps_A, gaps_C, gaps_B = [], [], [], []
        for iid in self.inst_seed_cost:
            seed  = self.inst_seed_cost[iid]
            best  = self.inst_best_cost[iid]
            gap   = best - seed   # negative = buffer beats seed
            gaps.append(gap)
            g = self.inst_group.get(iid, 'C')
            if   g == 'A': gaps_A.append(gap)
            elif g == 'C': gaps_C.append(gap)
            elif g == 'B': gaps_B.append(gap)

        def _fmt(arr, label=""):
            """Buffer gap formatter: raw [0,100]² distance units → PIP scale absolute diff."""
            if not arr:
                return "n/a"
            arr = np.array(arr) / 100.0
            return (f"mean={np.mean(arr):+.4f}  "
                    f"median={np.median(arr):+.4f}  "
                    f"pct_improved={100*np.mean(arr < 0):.0f}%")

        def _fmt_gap(arr):
            """Policy gap formatter: values already in % units → (policy-opt)/opt*100."""
            if not arr:
                return "n/a"
            arr = np.array(arr)
            return (f"mean={np.mean(arr):+.2f}%  "
                    f"median={np.median(arr):+.2f}%  "
                    f"pct_improved={100*np.mean(arr < 0):.0f}%")

        print(f"   [Buffer gap = best_buffer_cost - seed_cost  (Δ<0 = improved)]",
              flush=True)
        print(f"     Overall ({len(gaps):3d} inst): {_fmt(gaps)}", flush=True)
        print(f"     Group A ({len(gaps_A):3d} inst, 1 FA): {_fmt(gaps_A)}", flush=True)
        print(f"     Group C ({len(gaps_C):3d} inst, 2 FA): {_fmt(gaps_C)}", flush=True)
        print(f"     Group B ({len(gaps_B):3d} inst,>=3FA): {_fmt(gaps_B)}", flush=True)

        # -- Policy rollout + Tabu baseline on held-out val instances ----
        # delta = policy_dist - tabu_dist  (delta < 0 => policy beat Tabu)
        pol_gaps, pol_gaps_A, pol_gaps_C, pol_gaps_B = [], [], [], []
        aug_gaps, aug_gaps_A, aug_gaps_C, aug_gaps_B = [], [], [], []
        if self.val_instances:
            n_feas_pol = 0
            n_feas_aug = 0
            n_feas_sol_aug = 0   # total feasible individual solutions (all 8 × N_inst)
            n_total_sol_aug = 0  # total individual solutions attempted
            n_inst_all_infeas = 0  # instances where every aug solution is infeasible
            dead_steps     = []
            dead_steps_aug = []
            tabu_dists, tabu_feas = [], []
            pol_dists, pol_pen, pol_viols, pol_lat = [], [], [], []
            aug_dists, aug_pen, aug_viols, aug_lat = [], [], [], []

            # --- Pre-compute aug-8 rollouts (sequential or CPU-parallel) ---
            iid_list = sorted(self.val_instances.keys())
            n_w = self.num_eval_workers
            if n_w > 1:
                import multiprocessing as _mp
                global _EVAL_WORKER_TRAINER
                # Move model to CPU before forking to avoid CUDA context issues
                _orig_device = self.device
                self.model.cpu()
                self.device = torch.device('cpu')
                _EVAL_WORKER_TRAINER = self
                n_w = min(n_w, len(iid_list))
                chunks = [iid_list[i::n_w] for i in range(n_w)]
                print(f"   [aug-8 rollout: {len(iid_list)} instances split across "
                      f"{n_w} CPU workers]", flush=True)
                ctx = _mp.get_context('fork')
                with ctx.Pool(n_w) as pool:
                    all_dicts = pool.map(_eval_worker_chunk, chunks)
                # Restore device and model
                self.model.to(_orig_device)
                self.device = _orig_device
                aug_results_map = {}
                for d in all_dicts:
                    aug_results_map.update(d)
            else:
                aug_results_map = None  # compute inline per-instance below

            for iid in iid_list:
                idata = self.val_instances[iid]

                # --- Reference baseline: val_opt pkl (instant) or live Tabu (slow) ---
                if self.val_opt:
                    _opt = self.val_opt.get(iid)
                    if _opt is not None:
                        tabu_dists.append(_opt['distance'])
                        tabu_feas.append(_opt['feasible'])
                        _tabu_ref = _opt['distance']
                    else:
                        _tabu_ref = None
                elif self._val_tabu_cache:
                    # Only run live Tabu if a precomputed cache was loaded for this dataset
                    tabu_r = self._val_tabu_rollout(iid)
                    if tabu_r is not None:
                        tabu_dists.append(tabu_r['distance'])
                        tabu_feas.append(tabu_r['feasible'])
                        _tabu_ref = tabu_r['distance']
                    else:
                        _tabu_ref = None
                else:
                    _tabu_ref = None  # no val_opt and no precomputed cache — skip gap

                # --- no-aug greedy rollout on val instance ---
                if not self.aug_only:
                    result = self._greedy_rollout(iid, idata=idata, pip_step=self.pip_step)
                    if result is not None:
                        if result['feasible']:
                            n_feas_pol += 1
                            pol_dists.append(result['distance'])
                            pol_pen.append(result['penalized_cost'])
                            pol_lat.append(result['total_lateness'])
                            if _tabu_ref is not None and _tabu_ref > 0:
                                pol_gaps.append(100.0 * (result['distance'] - _tabu_ref) / _tabu_ref)
                        pol_viols.append(result['violations'])
                        if result['dead_step'] is not None:
                            dead_steps.append(result['dead_step'])

                # --- 8-fold aug best-of-8 on val instance ---
                if aug_results_map is not None:
                    aug_result = aug_results_map.get(iid)
                else:
                    aug_result = self._greedy_rollout_aug(iid, aug_factor=8, idata=idata, pip_step=self.pip_step)
                if aug_result is not None:
                    # per-solution feasibility counts (solution-level infeasible rate)
                    _nf = aug_result.get('n_feasible_aug', int(aug_result['feasible']))
                    _na = aug_result.get('n_aug', 8)
                    n_feas_sol_aug   += _nf
                    n_total_sol_aug  += _na
                    if _nf == 0:
                        n_inst_all_infeas += 1
                    if aug_result['feasible']:
                        n_feas_aug += 1
                        aug_dists.append(aug_result['distance'])
                        aug_pen.append(aug_result['penalized_cost'])
                        aug_lat.append(aug_result['total_lateness'])
                        if _tabu_ref is not None and _tabu_ref > 0:
                            aug_gaps.append(100.0 * (aug_result['distance'] - _tabu_ref) / _tabu_ref)
                    aug_viols.append(aug_result['violations'])
                    if aug_result.get('dead_step') is not None:
                        dead_steps_aug.append(aug_result['dead_step'])

            n_inst = len(self.val_instances)   # total instances evaluated
            n_aug  = len(self.val_instances)

            # --- Tabu summary ---
            ref_label = "Opt (val_opt pkl)" if self.val_opt else "Tabu baseline"
            if tabu_dists:
                n_tf = sum(tabu_feas)
                print(f"   [{ref_label} -- {len(tabu_dists)} val instances]", flush=True)
                print(f"     Feasible: {n_tf}/{len(tabu_dists)} "
                      f"({100*n_tf/max(len(tabu_dists),1):.0f}%)  "
                      f"Dist: mean={np.mean(tabu_dists)/100:.4f}  "
                      f"median={np.median(tabu_dists)/100:.4f}", flush=True)

            # --- greedy policy ---
            if self.aug_only:
                print(f"   [Policy rollout -- greedy  SKIPPED (aug_only mode)]", flush=True)
            else:
                print(f"   [Policy rollout -- greedy  on {n_inst} val instances]", flush=True)
                if pol_dists:
                    print(f"     Feasible: {n_feas_pol}/{n_inst} "
                          f"({100*n_feas_pol/max(n_inst,1):.0f}%)", flush=True)
                    print(f"     Tour dist     : mean={np.mean(pol_dists)/100:.4f}  "
                          f"median={np.median(pol_dists)/100:.4f}", flush=True)
                    print(f"     Penalized cost: mean={np.mean(pol_pen)/100:.4f}  "
                          f"median={np.median(pol_pen)/100:.4f}", flush=True)
                    print(f"     Lateness: mean={np.mean(pol_lat)/100:.4f}  "
                          f"max={np.max(pol_lat)/100:.4f}", flush=True)
                    if dead_steps:
                        ds = np.array(dead_steps)
                        print(f"     TW-blocked: {len(dead_steps)}/{n_inst} inst "
                              f"at step mean={ds.mean():.1f}", flush=True)
                print(f"     Gap% = (policy-opt)/opt*100  (over feasible instances only; <0 = beats opt)",
                      flush=True)
                print(f"     Overall ({len(pol_gaps)} feasible inst): {_fmt_gap(pol_gaps)}", flush=True)

            # --- aug policy ---
            print(f"   [Policy rollout -- best-of-8 aug  on {n_aug} val instances]",
                  flush=True)
            print(f"     Feasible: {n_feas_aug}/{n_aug} "
                  f"({100*n_feas_aug/max(n_aug,1):.0f}%)", flush=True)
            # Infeasible rates (solution-level and instance-level) — always shown
            if n_total_sol_aug > 0:
                sol_infeas_pct  = 100.0 * (n_total_sol_aug - n_feas_sol_aug) / n_total_sol_aug
                inst_infeas_pct = 100.0 * n_inst_all_infeas / max(n_aug, 1)
                print(f"     Infeasible%   Sol. (all {n_total_sol_aug} aug solutions): "
                      f"{sol_infeas_pct:.2f}%  "
                      f"({n_total_sol_aug - n_feas_sol_aug}/{n_total_sol_aug})", flush=True)
                print(f"     Infeasible%   Inst. (0-of-8 feasible): "
                      f"{inst_infeas_pct:.2f}%  "
                      f"({n_inst_all_infeas}/{n_aug})", flush=True)
            if aug_dists:
                print(f"     Tour dist     : mean={np.mean(aug_dists)/100:.4f}  "
                      f"median={np.median(aug_dists)/100:.4f}", flush=True)
                print(f"     Penalized cost: mean={np.mean(aug_pen)/100:.4f}  "
                      f"median={np.median(aug_pen)/100:.4f}", flush=True)
                print(f"     Lateness: mean={np.mean(aug_lat)/100:.4f}  "
                      f"max={np.max(aug_lat)/100:.4f}", flush=True)
                if dead_steps_aug:
                    da = np.array(dead_steps_aug)
                    print(f"     TW-blocked: {len(dead_steps_aug)}/{n_aug} inst "
                          f"at step mean={da.mean():.1f}", flush=True)
            print(f"     Gap% = (aug-opt)/opt*100  (over feasible instances only; <0 = beats opt)",
                  flush=True)
            print(f"     Overall ({len(aug_gaps)} feasible inst): {_fmt_gap(aug_gaps)}", flush=True)
        else:
            print("   [Policy rollout] No val_dataset loaded -- pass --val_dataset.",
                  flush=True)

        # ── BC coverage ─────────────────────────────────────────────────
        n_steps  = max((len(v) for v in self.inst_action_sets.values()), default=0)
        step_cov = []
        for step in range(1, n_steps + 1):
            n_multi = sum(
                1 for iid, sets in self.inst_action_sets.items()
                if len(sets.get(step, set())) > 1
            )
            step_cov.append(n_multi / max(len(self.inst_action_sets), 1))

        if step_cov:
            print(f"   [BC coverage — fraction of instances with >1 action per step]",
                  flush=True)
            print("     " + "  ".join(
                f"s{s+1}={100*c:.0f}%" for s, c in enumerate(step_cov)), flush=True)
            hard  = [s + 1 for s, c in enumerate(step_cov) if c < 0.10]
            soft  = [s + 1 for s, c in enumerate(step_cov) if 0.10 <= c < 0.25]
            # Late steps naturally have fewer branches — only warn for early steps
            # (defined as first half of the problem)
            early = n_steps // 2
            hard_early = [s for s in hard if s <= early]
            soft_early = [s for s in soft if s <= early]
            if hard_early:
                print(f"     HARD WARNING  (early steps <10%): {hard_early} "
                      f"— high BC gap risk", flush=True)
            if soft_early:
                print(f"     Soft note (early steps 10-25%): {soft_early}",
                      flush=True)
            late_low = [s for s in (hard + soft) if s > early]
            if late_low:
                print(f"     Late steps with low coverage (expected): {late_low}",
                      flush=True)

        # ── Log scalars ─────────────────────────────────────────────────
        self.result_log["val_score"].append(avg_cost)
        self.result_log["val_infsb_rate"].append(100 - feas_pct)
        if gaps:
            self.result_log["val_buffer_gap_mean"].append(float(np.mean(gaps)))
            self.result_log["val_buffer_gap_median"].append(float(np.median(gaps)))
        if gaps_A:
            self.result_log["val_buffer_gap_A"].append(float(np.mean(gaps_A)))
        if gaps_C:
            self.result_log["val_buffer_gap_C"].append(float(np.mean(gaps_C)))
        if gaps_B:
            self.result_log["val_buffer_gap_B"].append(float(np.mean(gaps_B)))
        if pol_gaps:
            self.result_log["val_pol_gap_mean"].append(float(np.mean(pol_gaps)))
            self.result_log["val_pol_gap_median"].append(float(np.median(pol_gaps)))
        if pol_gaps_A:
            self.result_log["val_pol_gap_A"].append(float(np.mean(pol_gaps_A)))
        if pol_gaps_C:
            self.result_log["val_pol_gap_C"].append(float(np.mean(pol_gaps_C)))
        if pol_gaps_B:
            self.result_log["val_pol_gap_B"].append(float(np.mean(pol_gaps_B)))
        if aug_gaps:
            self.result_log["val_aug_gap_mean"].append(float(np.mean(aug_gaps)))
            self.result_log["val_aug_gap_median"].append(float(np.median(aug_gaps)))
        if aug_gaps_A:
            self.result_log["val_aug_gap_A"].append(float(np.mean(aug_gaps_A)))
        if aug_gaps_C:
            self.result_log["val_aug_gap_C"].append(float(np.mean(aug_gaps_C)))
        if aug_gaps_B:
            self.result_log["val_aug_gap_B"].append(float(np.mean(aug_gaps_B)))
        return avg_cost

    def save_model(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        checkpoint_path = os.path.join(self.log_path, f"model_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f">> Model saved to {checkpoint_path}")

    def load_model(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f">> Model loaded from {checkpoint_path}")
