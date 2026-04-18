import json
from pathlib import Path
base = Path('/mnt/dongxu-fs1/data-ssd/qiyuanqiao/workspace/rev2fwd-il/data/pick_place_isaac_lab_simulation/exp55')

print(f"{'iter':>4} {'stat_in':>8} {'stat_out':>8} {'stat_keep':>9} {'spd_in':>8} {'spd_out':>8} {'spd_x':>6} {'adv_in_ep':>9} {'adv_out_ep':>10} {'adv_in_fr':>9} {'adv_out_fr':>10} {'adv_keep':>8} {'total_keep':>10}")
for i in range(1,11):
    sf = json.load(open(base / f'iter{i}_collect_B_reversed_static_filtered.stats.json'))
    sp = json.load(open(base / f'iter{i}_collect_B_reversed_static_speed.stats.json'))
    af = json.load(open(base / f'iter{i}_advantage_filter' / 'filter_stats.json'))
    if i == 1:
        print('STATIC keys:', list(sf.keys()))
        print('SPEED keys:', list(sp.keys()))
        if isinstance(af, list):
            print('ADV first ep keys:', list(af[0].keys()))
        else:
            print('ADV keys:', list(af.keys()))
    # try common keys
    def g(d, *keys):
        for k in keys:
            if k in d: return d[k]
        return None
    stat_in = g(sf,'original_total_frames','total_original_frames','total_input_frames','input_total_frames')
    stat_out = g(sf,'filtered_total_frames','total_filtered_frames','total_output_frames','output_total_frames','total_kept_frames')
    spd_in = g(sp,'original_total_frames','total_original_frames','total_input_frames','input_total_frames')
    spd_out = g(sp,'adjusted_total_frames','total_adjusted_frames','total_output_frames','output_total_frames')
    if isinstance(af, list):
        adv_in_ep = len(af)
        adv_out_ep = sum(1 for s in af if s.get('kept_length',0) > 0)
        adv_in_fr = sum(s.get('original_length',0) for s in af)
        adv_out_fr = sum(s.get('kept_length',0) for s in af)
    else:
        adv_in_ep = af.get('num_input_episodes')
        adv_out_ep = af.get('num_output_episodes')
        adv_in_fr = af.get('total_original_frames')
        adv_out_fr = af.get('total_kept_frames')
    sk = stat_out/stat_in if stat_in else 0
    sx = spd_out/spd_in if spd_in else 0
    ak = adv_out_fr/adv_in_fr if adv_in_fr else 0
    tk = adv_out_fr/stat_in if stat_in else 0
    print(f"{i:>4} {stat_in:>8} {stat_out:>8} {sk:>9.3f} {spd_in:>8} {spd_out:>8} {sx:>6.3f} {adv_in_ep:>9} {adv_out_ep:>10} {adv_in_fr:>9} {adv_out_fr:>10} {ak:>8.3f} {tk:>10.3f}")
