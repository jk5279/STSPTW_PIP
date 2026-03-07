"""
Generate a test set of N instances for TSPTW (no buffer, no optimal).
Saves: data/TSPTW/tsptw{N}_{hardness}_test.pkl
"""
import argparse
import os
import pickle
import sys

_OTA_DIR = os.path.abspath(os.path.dirname(__file__))
if _OTA_DIR not in sys.path:
    sys.path.insert(0, _OTA_DIR)

from envs.TSPTWEnv import TSPTWEnv


def main():
    p = argparse.ArgumentParser(description="Generate TSPTW test set")
    p.add_argument('--problem_size', type=int, default=10)
    p.add_argument('--hardness', type=str, default='easy',
                   choices=['easy', 'medium', 'hard'])
    p.add_argument('--n_test', type=int, default=10000)
    p.add_argument('--seed', type=int, default=9999,
                   help='Use a different seed from train/val (default 2024)')
    p.add_argument('--output_dir', type=str, default=os.path.join(_OTA_DIR, 'data', 'TSPTW'))
    args = p.parse_args()

    import torch
    torch.manual_seed(args.seed)

    n = args.problem_size
    n_cust = n - 1  # customers only (node 0 is depot)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    # Use customer count in filename (matching convention: tsptw10_easy = 10 customers)
    out_path = os.path.join(out_dir, f'tsptw{n_cust}_{args.hardness}_test.pkl')

    print(f"Generating {args.n_test} test instances  "
          f"(n_cust={n_cust}, total_nodes={n}, hardness={args.hardness}, seed={args.seed})")

    env = TSPTWEnv(
        problem_size=n,
        pomo_size=n,
        hardness=args.hardness,
        k_sparse=n + 1,
        device='cpu',
    )

    node_xy_t, svc_t, tws_t, twe_t = env.get_random_problems(
        args.n_test, n, coord_factor=100, max_tw_size=100
    )

    instances = []
    for i in range(args.n_test):
        instances.append((
            node_xy_t[i].numpy().tolist(),
            svc_t[i].numpy().tolist(),
            tws_t[i].numpy().tolist(),
            twe_t[i].numpy().tolist(),
        ))

    with open(out_path, 'wb') as f:
        pickle.dump(instances, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(instances)} instances → {out_path}")


if __name__ == '__main__':
    main()
