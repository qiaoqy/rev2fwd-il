import numpy as np
from pathlib import Path
p = Path('/mnt/dongxu-fs1/data-ssd/qiyuanqiao/workspace/rev2fwd-il/data/pick_place_isaac_lab_simulation/exp55/iter1_collect_A.npz')
d = np.load(p, allow_pickle=True)
print('top keys:', list(d.keys()))
eps = list(d['episodes'])
print('n_eps:', len(eps))
e = eps[0]
print('episode keys:', list(e.keys()))
for k,v in e.items():
    if isinstance(v, np.ndarray):
        print(f'  {k}: shape={v.shape} dtype={v.dtype}')
    else:
        print(f'  {k}: {type(v).__name__} = {v if not hasattr(v,"__len__") else f"len={len(v)}"}')
print('ee_pose[:3]:', e['ee_pose'][:3])
print('action[:3]:', e['action'][:3])
print('success:', e.get('success'))
