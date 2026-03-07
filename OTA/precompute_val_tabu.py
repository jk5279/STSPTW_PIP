"""
Pre-compute and persist Tabu expert solutions for the val instance set.

Run once before training:
    conda run -n pip python3 precompute_val_tabu.py \
        --val_dataset ../data/STSPTW/stsptw10_val.pkl

Saves: ../data/STSPTW/stsptw10_val_tabu_cache.pkl
"""
import argparse, os, sys, pickle, time
import numpy as np

# OTA/ for generate_ota_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_ota_dataset import STSPTWDatasetGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dataset', default='../data/STSPTW/stsptw10_val.pkl')
    parser.add_argument('--time_limit', type=float, default=2.0,
                        help='Tabu time limit per instance (seconds)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    val_path = args.val_dataset
    cache_path = os.path.splitext(val_path)[0] + '_tabu_cache.pkl'

    # Load val instances
    with open(val_path, 'rb') as f:
        instances = pickle.load(f)
    print(f'Loaded {len(instances)} val instances from {val_path}')

    # Load existing cache if any
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f'Existing cache: {len(cache)} entries')

    gen = STSPTWDatasetGenerator(seed=args.seed)
    t0 = time.time()

    for iid, item in enumerate(instances):
        if iid in cache:
            continue  # already done

        # Parse instance
        arrays = [np.array(x, dtype=np.float32) for x in item]
        node_xy  = arrays[0]
        svc      = arrays[1] if len(arrays) > 1 else np.zeros(len(arrays[0]), dtype=np.float32)
        tw_start = arrays[2] if len(arrays) > 2 else np.zeros(len(arrays[0]), dtype=np.float32)
        tw_end   = arrays[3] if len(arrays) > 3 else np.full(len(arrays[0]), 1e6, dtype=np.float32)

        problem = (node_xy.tolist(), svc.tolist(), tw_start.tolist(), tw_end.tolist())
        result  = gen.tabu_tour(problem, time_limit_s=args.time_limit, seed=args.seed)

        dist_raw = result.get('distance', 0.0)
        cache[iid] = {
            'tour':           result.get('tour', []),
            'distance':       dist_raw,
            'violations':     result.get('violations', 0),
            'total_lateness': 0.0,
            'penalized_cost': dist_raw,
            'reward':         result.get('reward', 0.0),
            'feasible':       result.get('feasible', False),
            'dead_step':      None,
        }

        # Save incrementally so progress is never lost
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

        elapsed = time.time() - t0
        remaining = (elapsed / (iid + 1)) * (len(instances) - iid - 1)
        feas_str = 'F' if cache[iid]['feasible'] else 'X'
        print(f'  [{iid+1:4d}/{len(instances)}] {feas_str} dist={dist_raw:.1f}  '
              f'elapsed={elapsed:.0f}s  ETA={remaining:.0f}s', flush=True)

    # Final summary
    dists  = [v['distance'] for v in cache.values()]
    n_feas = sum(v['feasible'] for v in cache.values())
    print(f'\nDone. {len(cache)} instances in cache.')
    print(f'Feasible: {n_feas}/{len(cache)} ({100*n_feas/max(len(cache),1):.1f}%)')
    print(f'Dist: mean={np.mean(dists):.2f}  median={np.median(dists):.2f}  '
          f'min={np.min(dists):.2f}  max={np.max(dists):.2f}')
    print(f'Saved to {cache_path}')


if __name__ == '__main__':
    main()
