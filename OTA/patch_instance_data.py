"""
Patch instance_data into an existing offline RL buffer without regenerating.

Reproduces the exact same random.seed(2023) → sample(2500) → split_train_test
path used by generate_ota_dataset.py, then stamps instance_data (node_xy,
service_time, tw_start, tw_end) on every trajectory using the stored instance_idx.

Usage:
    python3 patch_instance_data.py
    # or with explicit paths:
    python3 patch_instance_data.py \
        --instances data/STSPTW/stsptw10_train.pkl \
        --train_buf data/offline_rl_buffer/stsptw10_easy_train.pkl \
        --test_buf  data/offline_rl_buffer/stsptw10_easy_test.pkl \
        --seed 2023 --max_instances 2500 --train_ratio 0.8
"""
import argparse
import pickle
import random
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances',    default='data/STSPTW/stsptw10_train.pkl')
    parser.add_argument('--train_buf',    default='data/offline_rl_buffer/stsptw10_easy_train.pkl')
    parser.add_argument('--test_buf',     default='data/offline_rl_buffer/stsptw10_easy_test.pkl')
    parser.add_argument('--seed',         type=int, default=2023)
    parser.add_argument('--max_instances',type=int, default=2500)
    parser.add_argument('--train_ratio',  type=float, default=0.8)
    args = parser.parse_args()

    # ── Step 1: Reproduce the exact instance split ─────────────────────────────
    print(f"Loading instances from {args.instances} …")
    with open(args.instances, 'rb') as f:
        all_instances = pickle.load(f)
    print(f"  Total instances in file: {len(all_instances)}")

    # Reproduce max_instances sampling (same as main() in generate_ota_dataset.py)
    _rnd = random
    _rnd.seed(args.seed)
    instances = _rnd.sample(all_instances, min(args.max_instances, len(all_instances)))
    print(f"  Sampled {len(instances)} instances (seed={args.seed})")

    # Reproduce split_train_test (same code as STSPTWDatasetGenerator.split_train_test)
    # Note: STSPTWDatasetGenerator.__init__ calls random.seed(seed) first, then main()
    # calls _rnd.seed(args.seed) again — net effect: random state at split time is
    # determined by seed=2023 + len(sample)=2500 draws from random.sample.
    idx_list = list(range(len(instances)))
    random.shuffle(idx_list)   # same global random state
    split_pt     = int(len(idx_list) * args.train_ratio)
    train_set    = set(idx_list[:split_pt])

    train_instances = [instances[i] for i in range(len(instances)) if i in train_set]
    test_instances  = [instances[i] for i in range(len(instances)) if i not in train_set]
    print(f"  Train: {len(train_instances)}  |  Test: {len(test_instances)}")

    def _make_idata(problem):
        node_xy, svc, tws, twe = problem
        return {
            'node_xy':      np.asarray(node_xy, dtype=np.float64),
            'service_time': np.asarray(svc,     dtype=np.float64),
            'tw_start':     np.asarray(tws,     dtype=np.float64),
            'tw_end':       np.asarray(twe,     dtype=np.float64),
        }

    # ── Step 2: Patch train buffer ─────────────────────────────────────────────
    for buf_path, inst_list, split_name in [
        (args.train_buf, train_instances, 'train'),
        (args.test_buf,  test_instances,  'test'),
    ]:
        print(f"\nPatching {split_name} buffer: {buf_path} …")
        with open(buf_path, 'rb') as f:
            trajs = pickle.load(f)
        print(f"  Trajectories: {len(trajs)}")

        already   = sum(1 for t in trajs if t.get('instance_data') is not None)
        print(f"  Already have instance_data: {already}")
        if already == len(trajs):
            print("  Nothing to do — all trajectories already patched.")
            continue

        missing_idx = set()
        patched = 0
        # Build per-instance cache so we don't re-create np arrays for every traj
        idata_cache = {}
        for t in trajs:
            iid = t.get('instance_idx', t.get('instance_id'))
            if iid is None:
                continue
            if t.get('instance_data') is not None:
                patched += 1
                continue
            if iid not in idata_cache:
                if 0 <= iid < len(inst_list):
                    idata_cache[iid] = _make_idata(inst_list[iid])
                else:
                    missing_idx.add(iid)
                    continue
            t['instance_data'] = idata_cache[iid]
            patched += 1

        print(f"  Patched: {patched}/{len(trajs)}")
        if missing_idx:
            print(f"  WARNING: {len(missing_idx)} unknown instance_idx values: "
                  f"{sorted(missing_idx)[:10]} …")

        print(f"  Saving back to {buf_path} …")
        with open(buf_path, 'wb') as f:
            pickle.dump(trajs, f)
        print(f"  Done.")

    print("\nAll buffers patched. Ready for training.")


if __name__ == '__main__':
    main()
