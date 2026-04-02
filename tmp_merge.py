import numpy as np
from pathlib import Path
import time, sys

base_dir = "data/pick_place_isaac_lab_simulation/exp33/cylinder"
iter_num = int(sys.argv[1]) if len(sys.argv) > 1 else 4
num_parts = 4

for task in ['A', 'B']:
    t0 = time.time()
    parts = []
    for j in range(num_parts):
        f = Path(base_dir) / f"iter{iter_num}_collect_{task}_p{j}.npz"
        if f.exists():
            data = np.load(f, allow_pickle=True)
            if 'episodes' in data:
                eps = list(data['episodes'])
                parts.extend(eps)
                print(f"  Part {j} task {task}: {len(eps)} episodes ({time.time()-t0:.1f}s)")
    out = Path(base_dir) / f"iter{iter_num}_collect_{task}.npz"
    if parts:
        np.savez(str(out), episodes=np.array(parts, dtype=object))
        print(f"Merged task {task}: {len(parts)} episodes -> {out.name} ({time.time()-t0:.1f}s)")
    else:
        print(f"WARNING: No episodes collected for task {task}!")

# Verify
for t in ['A', 'B']:
    f = Path(base_dir) / f"iter{iter_num}_collect_{t}.npz"
    d = np.load(f, allow_pickle=True)
    print(f"Verified task {t}: {len(d['episodes'])} episodes")

print(f"Merge iter {iter_num} complete!")
