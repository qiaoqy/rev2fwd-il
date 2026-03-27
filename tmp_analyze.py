#!/usr/bin/env python3
"""Quick analysis of ee_pose movement patterns."""
import numpy as np

print("Loading NPZ (this may take a while)...")
data = np.load('data/pick_place_isaac_lab_simulation/exp21/rollout_A_200.npz', allow_pickle=True)
eps = list(data['episodes'])
print(f'Total episodes: {len(eps)}')
print(f'Episode lengths (first 10): {[len(e["images"]) for e in eps[:10]]}')
print(f'Keys in episode: {list(eps[0].keys())}')

all_diffs = []
all_static_ratios = []
all_max_runs = []

for i in range(min(10, len(eps))):
    ep = eps[i]
    ee = ep['ee_pose']
    T = len(ee)
    diffs = np.linalg.norm(np.diff(ee[:, :3], axis=0), axis=1)
    all_diffs.extend(diffs.tolist())

    print(f'\nEp {i} (T={T}):')
    print(f'  min={diffs.min():.8f}, max={diffs.max():.8f}')
    print(f'  mean={diffs.mean():.8f}, median={np.median(diffs):.8f}')

    for thresh in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        static = np.sum(diffs < thresh)
        if thresh == 1e-4:
            all_static_ratios.append(static / (T - 1))
        print(f'  frames < {thresh:.0e}: {static}/{T-1} ({100*static/(T-1):.1f}%)')

    is_static = diffs < 1e-4
    max_run = cur_run = 0
    for s in is_static:
        if s:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    all_max_runs.append(max_run)
    print(f'  longest static run (1e-4): {max_run} frames')

    g = ep.get('gripper', ep['action'][:, 7])
    if hasattr(g, 'flatten'):
        g = g.flatten()
    transitions = np.where(np.abs(np.diff(g)) > 0.5)[0]
    print(f'  gripper transitions at: {transitions.tolist()[:10]}')
    print(f'  gripper open={np.sum(g > 0)}, close={np.sum(g < 0)}')

print(f'\n=== SUMMARY (first 10 eps) ===')
all_diffs = np.array(all_diffs)
print(f'Overall movement percentiles:')
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f'  p{p}: {np.percentile(all_diffs, p):.8f}')
print(f'Avg static ratio (1e-4): {np.mean(all_static_ratios):.3f}')
print(f'Max static runs: {all_max_runs}')
print("Done!")
