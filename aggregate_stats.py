import json
import os

base_path = "/mnt/dongxu-fs1/data-ssd/qiyuanqiao/workspace/rev2fwd-il/data/pick_place_isaac_lab_simulation/exp55/"

print(f"{'iter':<5} | {'st_in_f':<8} | {'st_out_f':<8} | {'st_keep':<8} | {'sp_in_f':<8} | {'sp_out_f':<8} | {'sp_fact':<8} | {'adv_in_e':<8} | {'adv_out_e':<8} | {'adv_in_f':<8} | {'adv_out_f':<8} | {'adv_keep':<8} | {'tot_keep':<8}")
print("-" * 125)

for i in range(1, 11):
    static_file = os.path.join(base_path, f"iter{i}_collect_B_reversed_static_filtered.stats.json")
    speed_file = os.path.join(base_path, f"iter{i}_collect_B_reversed_static_speed.stats.json")
    adv_file = os.path.join(base_path, f"iter{i}_advantage_filter/filter_stats.json")
    
    # Load static stats
    try:
        with open(static_file, 'r') as f:
            static_data = json.load(f)
        st_in_f = static_data.get('original_total_frames', 0)
        st_out_f = static_data.get('filtered_total_frames', 0)
        st_keep = st_out_f / st_in_f if st_in_f > 0 else 0
    except Exception as e:
        print(f"Error reading {static_file}: {e}")
        continue

    # Load speed stats
    try:
        with open(speed_file, 'r') as f:
            speed_data = json.load(f)
        sp_in_f = speed_data.get('original_total_frames', 0)
        sp_out_f = speed_data.get('adjusted_total_frames', 0)
        sp_fact = sp_out_f / sp_in_f if sp_in_f > 0 else 0
    except Exception as e:
        print(f"Error reading {speed_file}: {e}")
        continue

    # Load adv stats
    try:
        with open(adv_file, 'r') as f:
            adv_data = json.load(f)
        adv_in_e = len(adv_data)
        adv_out_e = sum(1 for ep in adv_data if ep.get('kept_length', 0) > 0)
        adv_in_f = sum(ep.get('original_length', 0) for ep in adv_data)
        adv_out_f = sum(ep.get('kept_length', 0) for ep in adv_data)
        adv_keep = adv_out_f / adv_in_f if adv_in_f > 0 else 0
    except Exception as e:
        print(f"Error reading {adv_file}: {e}")
        continue

    tot_keep = adv_out_f / st_in_f if st_in_f > 0 else 0
    
    print(f"{i:<5} | {st_in_f:<8} | {st_out_f:<8} | {st_keep:<8.2%} | {sp_in_f:<8} | {sp_out_f:<8} | {sp_fact:<8.2f} | {adv_in_e:<8} | {adv_out_e:<8} | {adv_in_f:<8} | {adv_out_f:<8} | {adv_keep:<8.2%} | {tot_keep:<8.2%}")

