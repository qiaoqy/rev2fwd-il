# Weekly Report — March 19, 2026

## Setting

- Fixed red rectangle region (12×10 cm) + fixed green goal; smaller and simpler than last week's randomly-moving red dot
- Significant improvement over last weekend → confirms new action labeling works; constant environment change was the main convergence barrier

---

### 1. Exp 13 — Baseline: Both Tasks Forward-Collected

- Both Task A & B data collected forward via FSM (100 eps each, 200 epochs)
- **Task A: 74%** (37/50), **Task B: 78%** (39/50)
- Iter1 data collection completed; continuing iterative training

---

### 2. Exp 14 — Time-Reversed Data for Task A

- Task B: forward-collected (reuse exp13). Task A: **time-reversed** from Task B, using next-frame ee_pose as action label
- **Task B: 87.5%** (28/32, partial eval)
- **Task A: 16.7%** (3/18, partial eval, 200-epoch model)
- vs. last week's 0% → next-frame ee_pose control is effective, but still far below forward baseline (74%)
- **Failure mode:** can grasp successfully (random perturbation in data helps), but **never releases the gripper** — suspects temporal irreversibility gaps in reversed trajectories
- **Next step:** resume training 200 → 600 epochs (currently ~92K/93K steps, nearly done)

---

### 3. Exp 15 — Reverse-Replay Reset

- After each Policy B rollout, **directly replay the time-reversed trajectory** to reset env (no Policy A needed)
- **Task B:** iter1 80% (40/50), iter2 76% (38/50), iter3 85.7% (12/14, in progress)
- **Reverse-replay reset: 100%** (40/40, 38/38, 12/12) — validates that reversed trajectories are physically replayable in sim
- Task B picks to random targets within the red region → natural data randomization
- Completed 2 iterations of incremental finetuning; iter3 collection in progress
