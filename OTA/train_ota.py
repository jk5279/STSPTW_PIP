import sys, os
# OTA/ is a sibling of POMO+PIP/ — add both to path
_PIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'POMO+PIP')
_OTA_DIR = os.path.dirname(os.path.abspath(__file__))
for _d in (_PIP_DIR, _OTA_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import pytz
import argparse
import pprint as pp
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import wandb
import torch
import numpy as np
import random
from OTATrainer import OTATrainer
from utils import *


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def args2dict(args):
    """Convert argparse args to parameter dictionaries"""
    env_params = {
        "problem_size": args.problem_size,
        "pomo_size": args.pomo_size,
        "hardness": args.hardness,
        "pomo_start": args.pomo_start,
        "val_dataset": args.val_dataset,
        "val_episodes": args.val_episodes,
        "k_sparse": args.k_sparse,
        "pip_buffer": args.pip_buffer,
        # ── Soft lateness penalty ─────────────────────────────────────────────
        # use_soft_penalty : True  → reward = -dist - λ·Σlateness  (train mode)
        # lambda_penalty   : None  → auto-computed per batch from data scale
        #                    float → fixed value (override auto)
        # use_squared_lateness : True → λ·l²  False → λ·l  (linear default)
        "use_soft_penalty":       args.use_soft_penalty,
        "lambda_penalty":         args.lambda_penalty,      # None ⟹ auto
        "use_squared_lateness":   args.use_squared_lateness,
        # ── Violation behaviour ───────────────────────────────────────────────
        # False → episode runs to completion even after a TW violation (soft)
        # True  → episode terminates immediately on first violation (hard/eval)
        "terminate_on_violation": not args.use_soft_penalty,
    }

    model_params = {
        # Encoder parameters (shared with POMO)
        "embedding_dim": args.embedding_dim,
        "sqrt_embedding_dim": args.sqrt_embedding_dim,
        "encoder_layer_num": args.encoder_layer_num,
        "decoder_layer_num": args.decoder_layer_num,
        "qkv_dim": args.qkv_dim,
        "head_num": args.head_num,
        "logit_clipping": args.logit_clipping,
        "ff_hidden_dim": args.ff_hidden_dim,
        "norm": args.norm,
        "norm_loc": args.norm_loc,
        "eval_type": args.eval_type,
        "problem": args.problem,
        
        # OTA-specific parameters
        "tw_normalize": args.tw_normalize,
        "rep_dim": args.rep_dim,
        "value_hidden_dims": args.value_hidden_dims,
        "actor_hidden_dims": args.actor_hidden_dims,
        "action_dim": args.problem_size,  # For STSPTW, action is node selection

        # Rich noise-aware observation dimension:
        #   6 global features  (cur_x, cur_y, cur_t, frac_rem, last_noise, cum_noise)
        #   10 per-node features × problem_size nodes
        #   problem_size = total nodes = 1 depot + N customers  (e.g. 10 → 106)
        "obs_dim": 6 + 10 * args.problem_size,
        # GC-Encoder hidden width (3-layer MLP before embedding bottleneck)
        "gc_hidden_dim": args.gc_hidden_dim,
    }

    optimizer_params = {
        "optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
        "scheduler": {"milestones": args.milestones, "gamma": args.gamma}
    }

    agent_params = {
        "low_alpha": args.low_alpha,
        "high_alpha": args.high_alpha,
        "low_discount": args.low_discount,
        "high_discount": args.high_discount,
        "expectile": args.expectile,
        "tau": args.tau,
        "subgoal_steps": args.subgoal_steps,
        "abstraction_factor": args.abstraction_factor,
        "rep_dim": args.rep_dim,
    }

    trainer_params = {
        "epochs": args.epochs,
        "accumulation_steps": args.accumulation_steps,
        "train_batch_size": args.train_batch_size,
        "validation_interval": args.validation_interval,
        "validation_batch_size": args.validation_batch_size,
        "model_save_interval": args.model_save_interval,
        "load_optimizer": args.load_optimizer,
        "agent_params": agent_params,
    }

    return env_params, model_params, optimizer_params, trainer_params


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    # problem
    parser.add_argument('--problem', type=str, default='STSPTW')
    parser.add_argument('--problem_size', type=int, default=10)

    # model params
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=int, default=8)
    parser.add_argument('--encoder_layer_num', type=int, default=6)
    parser.add_argument('--decoder_layer_num', type=int, default=2)
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=int, default=10)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    parser.add_argument('--norm', type=str, default='batch', choices=['batch', 'layer', 'instance'])
    parser.add_argument('--norm_loc', type=str, default='norm_first', choices=['norm_first', 'norm_last'])
    parser.add_argument('--eval_type', type=str, default='softmax')
    parser.add_argument('--tw_normalize', type=bool, default=True)
    
    # OTA-specific model params
    parser.add_argument('--rep_dim', type=int, default=16, help='Goal representation dimension')
    parser.add_argument('--value_hidden_dims', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--actor_hidden_dims', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--gc_hidden_dim', type=int, default=256,
                        help='Hidden width of 3-layer GC-Encoder MLP (default: 256)')

    # OTA algorithm params
    parser.add_argument('--low_alpha', type=float, default=3.0, help='Low-level actor temperature')
    parser.add_argument('--high_alpha', type=float, default=3.0, help='High-level actor temperature')
    parser.add_argument('--low_discount', type=float, default=0.99)
    parser.add_argument('--high_discount', type=float, default=0.99)
    parser.add_argument('--expectile', type=float, default=0.7, help='IQL expectile')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--subgoal_steps', type=int, default=25, help='Subgoal steps for abstraction')
    parser.add_argument('--abstraction_factor', type=int, default=5, help='Temporal abstraction factor')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--milestones', type=int, nargs='+', default=[80, 160])
    parser.add_argument('--gamma', type=float, default=0.1)

    # trainer params
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--validation_interval', type=int, default=50)
    parser.add_argument('--validation_batch_size', type=int, default=1000)
    parser.add_argument('--model_save_interval', type=int, default=50)
    parser.add_argument('--load_optimizer', type=bool, default=True)

    # env params
    parser.add_argument('--pomo_size', type=int, default=1)
    parser.add_argument('--hardness', type=str, default=None)
    parser.add_argument('--pomo_start', type=bool, default=True)
    parser.add_argument('--val_dataset', type=str, default='data/TSPTW/tsptw10_easy_val.pkl')
    parser.add_argument('--val_episodes', type=int, default=10000)
    parser.add_argument('--k_sparse', type=int, default=None)
    parser.add_argument('--pip_buffer', type=int, default=10)

    # ── Soft lateness penalty ──────────────────────────────────────────────────
    # lambda_penalty = None  → auto-computed per batch as:
    #   5 * mean_edge / mean_tw_width   (scale-invariant; see STSPTWEnv.reset)
    # Override with a positive float to fix the value across all batches.
    parser.add_argument('--use_soft_penalty',    type=lambda x: x.lower() != 'false',
                        default=True,
                        help='Enable soft lateness penalty (default: True).')
    parser.add_argument('--lambda_penalty',       type=float, default=None,
                        help='Fixed penalty coefficient λ.  None = auto from data.')
    parser.add_argument('--use_squared_lateness', type=lambda x: x.lower() != 'false',
                        default=False,
                        help='Use λ·l² instead of λ·l (default: False = linear).')

    # settings
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--log_dir', type=str, default="./results")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--tb_logger', type=bool, default=True)
    parser.add_argument('--wandb_logger', type=bool, default=False)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='data/TSPTW/tsptw10_easy_buffer.pkl')
    parser.add_argument('--val_only', action='store_true',
                        help='Skip training; load checkpoint and run one validation pass then exit.')
    parser.add_argument('--pip_step', type=int, default=0, choices=[0, 1, 2],
                        help='PIP lookahead depth at inference: 0=off, 1=1-step, 2=2-step (default: 0).')
    parser.add_argument('--aug_only', action='store_true',
                        help='Skip greedy rollout during validation; run only aug-8 best-of-8.')
    parser.add_argument('--num_eval_workers', type=int, default=0,
                        help='Number of CPU workers for parallel aug-8 rollout evaluation. '
                             '0 (default) = sequential on current device. '
                             '>1 = fork N processes on CPU (Linux only). '
                             'Recommended: set to number of physical CPU cores.')
    parser.add_argument('--val_opt', type=str, default=None,
                        help='Path to val_opt pkl (pre-computed optimal/reference solutions). '
                             'When provided, used as reference instead of live Tabu during validation.')

    args = parser.parse_args()
    return args


def main():
    """Main training function"""
    args = get_args()

    # Set device
    if args.no_cuda:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda')

    # Set seed
    set_seed(args.seed)

    # Convert args to dicts
    env_params, model_params, optimizer_params, trainer_params = args2dict(args)

    # Setup logging
    now = datetime.now(pytz.timezone('America/Toronto'))
    now_str = now.strftime('%y%m%d_%H%M%S')
    args.log_path = os.path.join(args.log_dir, '{}_OTA_{}'.format(now_str, args.problem))
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Save arguments
    with open(os.path.join(args.log_path, 'args.txt'), 'w') as f:
        pp.pprint(vars(args), stream=f)

    print("=" * 80)
    print("OTA Training on {}".format(args.problem))
    print("=" * 80)
    print(f"Log path: {args.log_path}")
    print("Args:")
    pp.pprint(vars(args))
    print("=" * 80)

    # W&B logging
    if args.wandb_logger:
        wandb.init(
            project='OTA-STSPTW',
            name=f'{now_str}_OTA_{args.problem}',
            config=vars(args)
        )

    # Create trainer and run
    trainer = OTATrainer(args, env_params, model_params, optimizer_params, trainer_params)

    if args.val_only:
        print(">> Val-only mode: running validation pass then exiting.")
        trainer._validate(0)
        print(">> Validation complete.")
        return

    trainer.run()

    print("=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
