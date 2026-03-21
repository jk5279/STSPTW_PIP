import pytz
import argparse
import pprint as pp
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import wandb
from Trainer import Trainer
from utils import *


def args2dict(args):
    env_params = {"problem_size": args.problem_size, "pomo_size": args.pomo_size, "hardness": args.hardness,
                  "pomo_start": args.pomo_start, "val_dataset": args.val_dataset, "val_episodes": args.val_episodes,
                  "k_sparse": args.k_sparse,
                  "delay_scale": getattr(args, "delay_scale", 0.1),
                  "reveal_delay_before_action": getattr(args, "reveal_delay_before_action", False),
                  # STSPTW_v2 params (ignored by other envs)
                  "noise_type": getattr(args, "noise_type", "gamma"),
                  "cv": getattr(args, "cv", 0.5),
                  "alpha": getattr(args, "alpha", 0.95),
                  "n_mc_samples": getattr(args, "n_mc_samples", 32),
                  "two_point_delta": getattr(args, "two_point_delta", 0.3),
                  "two_point_p": getattr(args, "two_point_p", 0.5),
                  }

    model_params = {
                    # original parameters in MvMOE for POMO
                    "embedding_dim": args.embedding_dim, "sqrt_embedding_dim": args.sqrt_embedding_dim,
                    "encoder_layer_num": args.encoder_layer_num, "decoder_layer_num": args.decoder_layer_num,
                    "qkv_dim": args.qkv_dim, "head_num": args.head_num, "logit_clipping": args.logit_clipping,
                    "ff_hidden_dim": args.ff_hidden_dim, "norm": args.norm, "norm_loc": args.norm_loc,
                    "eval_type": args.eval_type, "problem": args.problem,
                    # PIP parameters
                    "pip_decoder": args.pip_decoder, "tw_normalize": args.tw_normalize,
                    "decision_boundary": args.decision_boundary, "detach_from_encoder": args.detach_from_encoder,
                    "W_q_sl":args.W_q_sl, "W_out_sl": args.W_out_sl, "W_kv_sl": args.W_kv_sl,
                    "use_ninf_mask_in_sl_MHA": args.use_ninf_mask_in_sl_MHA, "generate_PI_mask": args.generate_PI_mask,
                    }

    optimizer_params = {"optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
                        "scheduler": {"milestones": args.milestones, "gamma": args.gamma}}

    trainer_params = {"epochs": args.epochs, "train_episodes": args.train_episodes, "accumulation_steps": args.accumulation_steps,
                      "train_batch_size": args.train_batch_size, "validation_interval": args.validation_interval,
                      "validation_batch_size": args.validation_batch_size, "model_save_interval": args.model_save_interval,
                      # reward
                      "timeout_reward": args.timeout_reward, "timeout_node_reward": args.timeout_node_reward,
                      "fsb_dist_only": args.fsb_dist_only, "fsb_reward_only": args.fsb_reward_only,
                      "penalty_increase": args.penalty_increase, "penalty_factor": args.penalty_factor,
                      # resume
                      "checkpoint": args.checkpoint, "pip_checkpoint": args.pip_checkpoint, "load_optimizer": args.load_optimizer,
                      # loss
                      "decision_boundary": args.decision_boundary, "sl_loss":args.sl_loss,
                      "label_balance_sampling": args.label_balance_sampling, "fast_label_balance": args.fast_label_balance,
                      "fast_weight": args.fast_weight,
                      # PIP-related parameters
                      "generate_PI_mask": args.generate_PI_mask, "pip_step": args.pip_step,
                      "use_real_PI_mask": args.use_real_PI_mask, "use_predicted_PI_mask": args.use_predicted_PI_mask,
                      "lazy_pip_model": args.lazy_pip_model, "simulation_stop_epoch": args.simulation_stop_epoch,
                      "pip_update_interval": args.pip_update_interval, "pip_last_growup": args.pip_last_growup,
                      "pip_update_epoch": args.pip_update_epoch, "load_which_pip": args.load_which_pip, "pip_save": args.pip_save,
                      }

    return env_params, model_params, optimizer_params, trainer_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proactive Infeasibility Prevention (PIP) Framework for Routing Problems with Complex Constraints.")
    # env_params
    parser.add_argument('--problem', type=str, default="TSPTW", choices=["TSPTW", "STSPTW", "STSPTW_v2"])
    parser.add_argument('--hardness', type=str, default="hard", choices=["hard", "medium", "easy"], help="Different levels of constraint hardness")
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--pomo_size', type=int, default=50, help="the number of start node, should <= problem size")
    parser.add_argument('--pomo_start', type=bool, default=False)
    parser.add_argument('--val_dataset', type=str, nargs='+', default=None, help="use the default one if set to None")
    parser.add_argument('--delay_scale', type=float, default=0.1, help='STSPTW delay weight (relative to deterministic travel)')
    parser.add_argument(
        '--reveal_delay_before_action',
        action='store_true',
        help='For STSPTW: if set, sample and reveal stochastic travel times before action selection (pre-decision noise).'
    )
    # STSPTW_v2 params
    parser.add_argument('--noise_type', type=str, default='gamma', choices=['gamma', 'two_point'],
                        help='STSPTW_v2: stochastic distribution family')
    parser.add_argument('--cv', type=float, default=0.5, help='STSPTW_v2: coefficient of variation (0 = deterministic)')
    parser.add_argument('--alpha', type=float, default=0.95, help='STSPTW_v2: chance constraint level for MC PIP mask')
    parser.add_argument('--n_mc_samples', type=int, default=32, help='STSPTW_v2: MC samples for PIP probability estimation')
    parser.add_argument('--two_point_delta', type=float, default=0.3, help='STSPTW_v2: delta for two-point distribution')
    parser.add_argument('--two_point_p', type=float, default=0.5, help='STSPTW_v2: p(low) for two-point distribution')
    parser.add_argument(
        '--model_type',
        type=str,
        default="POMO_STAR",
        choices=["POMO", "POMO_STAR", "POMO_STAR_PIP"],
        help="Backbone + PIP variant: POMO (no LM, no PI mask), "
             "POMO_STAR (LM only), POMO_STAR_PIP (LM + real PI mask; no PIP-D)."
    )

    # model_params
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=float, default=128**(1/2))
    parser.add_argument('--encoder_layer_num', type=int, default=6, help="the number of MHA in encoder")
    parser.add_argument('--decoder_layer_num', type=int, default=1, help="the number of MHA in decoder")
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=float, default=10)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    parser.add_argument('--eval_type', type=str, default="argmax", choices=["argmax", "softmax"])
    parser.add_argument('--norm', type=str, default="instance", choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--norm_loc', type=str, default="norm_last", choices=["norm_first", "norm_last"], help="whether conduct normalization before MHA/FFN/MOE")
    
    # PIP params
    # model
    parser.add_argument('--tw_normalize', type=bool, default=True)
    parser.add_argument('--generate_PI_mask', action='store_true', help="whether to generates PI masking")
    parser.add_argument('--use_real_PI_mask', type=bool, default=True, help="whether to use PI masking")
    parser.add_argument('--pip_step', type=int, default=1)
    parser.add_argument('--k_sparse', type=int, default=500)
    parser.add_argument("--use_predicted_PI_mask", type=bool, default=True, help="whether to use PIP-D masking")
    parser.add_argument('--pip_decoder', action='store_true')
    parser.add_argument('--W_q_sl', type=bool, default=True)
    parser.add_argument('--W_out_sl', type=bool, default=True)
    parser.add_argument('--W_kv_sl', type=bool, default=True)
    parser.add_argument('--detach_from_encoder', type=bool, default=False)
    parser.add_argument('--use_ninf_mask_in_sl_MHA', type=bool, default= False)
    parser.add_argument('--lazy_pip_model', type=bool, default=True)
    # update frequency
    parser.add_argument('--simulation_stop_epoch', type=int, default=200) # use 100 when N=100
    parser.add_argument('--pip_update_interval', type=int, default=1000)
    parser.add_argument('--pip_update_epoch', type=int, default=50) # use 20 when N=100
    parser.add_argument('--pip_last_growup', type=int, default=50)
    parser.add_argument('--pip_save', type=str, default="epoch")
    parser.add_argument('--load_which_pip', type=str, default="train_fsb_bsf", choices=["last_epoch", "train_fsb_bsf", "train_infsb_bsf", "train_accuracy_bsf"])

    # optimizer_params
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--milestones', type=int, nargs='+', default=[9001, ], help='when to decay lr')
    parser.add_argument('--gamma', type=float, default=0.1, help='new_lr = lr * gamma')

    # trainer_params
    parser.add_argument('--epochs', type=int, default=10000, help="total training epochs")
    parser.add_argument('--train_episodes', type=int, default=10000, help="the num. of training instances per epoch")
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--validation_interval', type=int, default=500)
    parser.add_argument('--validation_batch_size', type=int, default=1000)
    parser.add_argument('--val_episodes', type=int, default=10000)
    parser.add_argument('--model_save_interval', type=int, default=50)
    # reward params
    parser.add_argument('--timeout_reward', type=bool, default=True)
    parser.add_argument("--timeout_node_reward",type=bool,default=True)
    parser.add_argument('--fsb_dist_only', type=bool, default=True)
    parser.add_argument('--fsb_reward_only', type=bool, default=True) # activate only if no penalty
    parser.add_argument('--penalty_increase', type=bool, default=False)
    parser.add_argument('--penalty_factor', type=float, default=1.)
    # resume params
    parser.add_argument('--resume_path', type=str, default=None, help='path to the old run')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pip_checkpoint', type=str, default=None)
    parser.add_argument('--load_optimizer', type=bool, default=True)
    # sl loss params
    parser.add_argument('--decision_boundary', type=float, default=0.5)
    parser.add_argument('--sl_loss', type=str, default="BCEWithLogitsLoss", choices=["BCEWithLogitsLoss", "BCELoss", "FL", "CE"], help="FL: focal loss; CE: cross entropy loss")
    parser.add_argument('--label_balance_sampling', type=bool, default=True)
    parser.add_argument('--fast_label_balance', type=bool, default=True)
    parser.add_argument('--fast_weight', type=bool, default=True)

    # settings (e.g., GPU)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--log_dir', type=str, default="./results")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument("--multiple_gpu", type=bool, default=False)
    parser.add_argument('--occ_gpu', type=float, default=0., help="occupy (X)% GPU memory in advance, please use sparingly.")
    parser.add_argument('--tb_logger', type=bool, default=True)
    parser.add_argument('--wandb_logger', type=bool, default=False)

    args = parser.parse_args()

    # Map high-level model_type to concrete training options.
    # POMO:       no LM, no PI masking, no PIP-D
    # POMO_STAR:  LM only (timeout_reward etc.), no PI masking, no PIP-D
    # POMO_STAR_PIP: LM + real PI masking (generate_PI_mask), no PIP-D
    if args.model_type == "POMO":
        # No Lagrangian-based timeout reward
        args.timeout_reward = False
        args.timeout_node_reward = False
        args.fsb_reward_only = False
        args.penalty_increase = False
        # No PI masking / PIP-D
        args.generate_PI_mask = False
        args.use_real_PI_mask = False
        args.use_predicted_PI_mask = False
        args.pip_decoder = False
        args.lazy_pip_model = False
    elif args.model_type == "POMO_STAR":
        # Lagrangian-based timeout reward on, but no PI masking
        args.timeout_reward = True
        args.timeout_node_reward = True
        args.penalty_increase = False
        # Keep penalty_factor as provided
        args.generate_PI_mask = False
        args.use_real_PI_mask = False
        args.use_predicted_PI_mask = False
        args.pip_decoder = False
        args.lazy_pip_model = False
    elif args.model_type == "POMO_STAR_PIP":
        # LM on + real PI masking (PIP), no PIP-D
        args.timeout_reward = True
        args.timeout_node_reward = True
        args.penalty_increase = False
        args.generate_PI_mask = True
        args.use_real_PI_mask = True
        args.use_predicted_PI_mask = False
        args.pip_decoder = False
        args.lazy_pip_model = False

    seed_everything(args.seed)

    if args.val_dataset is None:
        if args.problem in ("STSPTW", "STSPTW_v2"):
            args.val_dataset = [f"tsptw{args.problem_size}_{args.hardness}.pkl"]
        else:
            args.val_dataset = [f"{args.problem.lower()}{args.problem_size}_{args.hardness}.pkl"]

    # set log
    run_name = f"_{args.problem}{args.problem_size}_{args.hardness}"
    if args.problem == "STSPTW":
        run_name += f"_dw{getattr(args, 'delay_scale', 0.1)}"
    if args.problem == "STSPTW_v2":
        run_name += f"_{args.noise_type}_cv{args.cv}"
    if args.timeout_reward:
        run_name += "_LM"
    if args.generate_PI_mask:
        run_name += f"_PIMask_{args.pip_step}Step"
    if args.pip_decoder:
        run_name += f"_PIPDecoder_UpdateFrequency_{args.simulation_stop_epoch}_{args.pip_update_interval}_{args.pip_update_epoch}_{args.pip_last_growup}"
    # Use Toronto time for run naming / logging
    process_start_time = datetime.now(pytz.timezone("America/Toronto"))
    if args.resume_path is not None:
        args.log_path = args.resume_path
    else:
        name = process_start_time.strftime("%Y%m%d_%H%M%S") + run_name
        args.log_path = os.path.join(args.log_dir, name)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    print(">> Log Path: {}".format(args.log_path))
    # set gpu
    if args.multiple_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not args.no_cuda and torch.cuda.is_available():
        if args.multiple_gpu and torch.cuda.device_count() > 1:
            args.device = torch.device('cuda')
        else:
            occumpy_mem(args) if args.occ_gpu != 0. else print(">> No occupation needed")
            # args.device = torch.device('cuda')
            args.gpu_id = int(args.gpu_id)
            args.device = torch.device('cuda', args.gpu_id)
            torch.cuda.set_device(args.gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
    print(">> USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(not args.no_cuda, args.gpu_id))

    pp.pprint(vars(args))
    env_params, model_params, optimizer_params, trainer_params = args2dict(args)

    if args.wandb_logger:
        wandb.init(project="PIP", name=name,
                   config={**env_params, **model_params, **optimizer_params, **trainer_params})
    create_logger(filename="run_log", log_path=args.log_path)
    torch.set_printoptions(threshold=1000000)

    # start training
    print(">> Start {} Training ...".format(args.problem))

    trainer = Trainer(args=args, env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params)
    copy_all_src(args.log_path)
    trainer.run()

    print(">> Finish {} Training ...".format(args.problem))

