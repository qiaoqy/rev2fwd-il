# RL-Based Rev2Fwd: 实验计划

> **目标**: 用 RL 替代当前的 rollout + DAgger finetune 循环，从 BC 预训练的 Diffusion Policy 出发，直接在 Isaac Lab 环境中在线优化。
>
> 两条技术路线：
> 1. **方案 A — TD-Learning (Critic-Guided Diffusion Policy)**: 用 TD3/SAC 风格的 Q-function 对 Diffusion Policy 的去噪过程做梯度引导，同时保留 BC 预训练的先验（DDPO / DiffusionQL 思路）。
> 2. **方案 B — SOTA: DPPO (Diffusion Policy Policy Optimization)**: 把整个 DDIM 去噪链条视为一条 multi-step MDP，用 PPO 直接优化每一步去噪（Allen Ren et al., RSS 2025 / CoRL 2024）。

---

## 0. 背景分析

### 当前 Pipeline (DAgger)
```
BC pretrain (Task A/B)
    ↓
for iter in 1..K:
    cyclic evaluation → 收集成功 rollout
    aggregate rollout data into LeRobot dataset (1:1 weighted sampling)
    finetune policy with BC loss (额外 5000 steps)
```

**局限性**：
- 只从成功 rollout 中学习，数据效率低
- BC loss 无法利用失败 episode 的信息
- 不能对 reward 做梯度优化，收敛慢
- DAgger 依赖 expert 生成修正动作——这里没有 oracle expert，用的是 policy 自身的 rollout

### 为什么 RL 适合这个场景
1. **明确的 reward**: pick-place 任务有清晰的 success criteria（object at goal within threshold）
2. **仿真环境**: Isaac Lab 提供高通量（可并行数百个 env）、可微分的环境
3. **BC 预训练提供好的初始化**: Diffusion Policy 已经学到了合理的行为先验，RL 只需 fine-tune
4. **失败 episode 也包含信息**: Q-learning 天然利用次优数据

---

## 1. 环境接口设计

### 1.1 Observation Space
与现有 pipeline 保持一致：
```python
obs = {
    "image":        (3, 128, 128),  # table camera (float32, normalized)
    "wrist_image":  (3, 128, 128),  # wrist camera
    "state":        (15,),          # ee_pose(7) + obj_pose(7) + gripper(1)
}
```

### 1.2 Action Space
```python
action = (8,)  # [goal_ee_x, goal_ee_y, goal_ee_z, goal_qw, goal_qx, goal_qy, goal_qz, gripper]
```
- 绝对 IK 目标 + 二值夹爪
- **Action chunking**: 一次预测 16 步，执行 8 步（与 eval 一致）

### 1.3 Reward Design
```python
def compute_reward(obj_pose, goal_xy, ee_pose, gripper, prev_obj_pose):
    """Dense + sparse reward for pick-place."""
    obj_xy = obj_pose[:2]
    obj_z = obj_pose[2]
    dist_to_goal = np.linalg.norm(obj_xy - goal_xy)

    # --- Sparse success reward ---
    success = (obj_z < 0.15) and (gripper > 0.5) and (dist_to_goal < 0.03)
    r_success = 10.0 if success else 0.0

    # --- Dense reaching reward (encourage gripper to approach object) ---
    ee_to_obj = np.linalg.norm(ee_pose[:3] - obj_pose[:3])
    r_reach = -1.0 * ee_to_obj  # negative distance

    # --- Dense progress reward (object moving toward goal) ---
    prev_dist = np.linalg.norm(prev_obj_pose[:2] - goal_xy)
    r_progress = (prev_dist - dist_to_goal) * 5.0  # positive if getting closer

    # --- Grasp reward (holding object above table) ---
    r_grasp = 2.0 if (obj_z > 0.05 and gripper < -0.5) else 0.0

    return r_success + r_reach + r_progress + r_grasp
```

### 1.4 RL Gym Wrapper
```python
class PickPlaceRLEnv(gym.Env):
    """Wraps Isaac Lab env into standard Gym API with reward shaping."""
    # obs_space: Dict[image, wrist_image, state]
    # action_space: Box(8,) or chunked Box(n_action_steps, 8)
    # step() returns (obs, reward, terminated, truncated, info)
    # 支持 vectorized (num_envs > 1) 高通量采样
```

---

## 2. 方案 A: TD-Learning — Critic-Guided Diffusion Policy

### 2.1 核心思想

基于 **Diffusion Q-Learning (DQL)** 和 **IDQL (Implicit Diffusion Q-Learning)** 的思路：

1. 保留 Diffusion Policy 的 **actor** 结构不变（DDIM 去噪生成 action chunk）
2. 额外训练一个 **Q-network**（critic），用 TD learning 估计 Q(s, a)
3. RL finetune 阶段：
   - **Critic 更新**：标准 TD3 / SAC 风格的 Bellman backup
   - **Actor 更新**：通过 critic 的梯度引导 diffusion actor，保持 BC regularization
   
与直接 SAC 的区别：不是训练一个 Gaussian actor，而是保留 Diffusion Policy 的表达能力。相当于用 critic 给 diffusion 的输出做"打分"和梯度引导。

### 2.2 技术细节

**Q-Network 架构**:
```python
class QNetwork(nn.Module):
    """
    Twin Q-networks for TD3-style clipped double-Q.
    Input: observation encoding (from shared vision backbone) + action chunk
    Output: scalar Q-value
    """
    def __init__(self, obs_encoder, action_dim, hidden_dims=[512, 512]):
        # 复用 Diffusion Policy 的 ResNet18 vision encoder（冻结或微调）
        # MLP head: concat(obs_embed, action_flat) → Q1, Q2
```

**Actor 更新（Critic-Guided Diffusion）**:
```python
# Sample action from diffusion: a ~ π_θ(a|s)
noisy_action = torch.randn(batch, horizon, action_dim)
for t in reversed(ddim_timesteps):
    noisy_action = ddim_step(policy, noisy_action, t, obs)

# Compute Q-value and backprop through diffusion chain
q_value = Q_phi(obs, noisy_action)
actor_loss = -q_value.mean() + α * bc_loss  # α 控制 BC regularization 强度
```

**BC Regularization**:
```python
# Prevent catastrophic forgetting of BC prior
# Option 1: KL divergence to pretrained policy
bc_loss = MSE(policy_output, bc_target_from_demo)

# Option 2: Action-space anchoring
# Keep some fraction of training batches as pure BC updates
```

### 2.3 算法伪代码
```
Initialize:
  π_θ ← load pretrained Diffusion Policy (BC)
  Q_φ1, Q_φ2 ← random twin Q-networks (share obs encoder with π)
  Q̄_φ1, Q̄_φ2 ← target Q-networks (EMA of Q)
  D_demo ← BC demonstration buffer (original LeRobot dataset)
  D_online ← empty online replay buffer

For each iteration:
  # --- Online data collection ---
  for env_step in 1..N_collect:
    a = π_θ(obs) + ε  # exploration noise (or entropy from diffusion stochasticity)
    obs', r, done = env.step(a)
    D_online.add(obs, a, r, obs', done)

  # --- Critic update (TD learning) ---
  for critic_step in 1..N_critic:
    batch ~ D_online ∪ D_demo  # mixed online + demo
    a' = π_θ(obs') + clip(noise)  # target action
    target_Q = r + γ * min(Q̄_φ1(obs', a'), Q̄_φ2(obs', a'))
    L_critic = MSE(Q_φ1(obs, a), target_Q) + MSE(Q_φ2(obs, a), target_Q)
    update φ1, φ2

  # --- Actor update (critic-guided + BC) ---
  for actor_step in 1..N_actor:
    batch_online ~ D_online
    batch_demo ~ D_demo
    
    # RL objective: maximize Q
    a_gen = run_ddim_chain(π_θ, obs_online)  # differentiable
    L_rl = -Q_φ1(obs_online, a_gen).mean()
    
    # BC objective: imitation on demos
    L_bc = diffusion_loss(π_θ, obs_demo, a_demo)  # standard denoising score matching
    
    L_actor = L_rl + λ * L_bc
    update θ

  # --- Target network EMA ---
  Q̄ ← τ * Q + (1-τ) * Q̄
```

### 2.4 关键超参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `γ` (discount) | 0.99 | |
| `τ` (target EMA) | 0.005 | |
| `λ` (BC weight) | 初始 2.0 → 退火到 0.1 | 逐渐释放 RL 控制 |
| `lr_critic` | 3e-4 | Adam |
| `lr_actor` | 1e-5 | 低 LR 防止破坏 BC prior |
| `batch_size` | 256 | |
| `replay_buffer_size` | 1M transitions | |
| `num_envs` | 64 | 并行环境数量 |
| `exploration_noise` | 0.1 (std) | Gaussian / 利用 diffusion 自身随机性 |
| `n_critic_updates_per_collect` | 50 | |
| `n_actor_updates_per_collect` | 20 | Delayed actor update |
| `demo_ratio` | 0.25 | demo buffer 占 batch 比例 |

### 2.5 文件结构
```
scripts/scripts_rl/
├── README.md                      # 本文件
├── rl_env_wrapper.py             # Isaac Lab → Gym RL wrapper (reward, obs, reset)
├── reward.py                      # Reward function definitions
├── q_network.py                   # Twin Q-network architecture
├── replay_buffer.py               # Replay buffer + demo buffer
├── train_td_diffusion.py          # 方案 A 主训练脚本
├── train_dppo.py                  # 方案 B 主训练脚本
├── eval_rl.py                     # 公平评估脚本 (同 7_eval_fair.py 接口)
├── utils.py                       # 共享工具函数
└── run_rl_pipeline.sh             # 自动化 pipeline shell 脚本
```

---

## 3. 方案 B: DPPO (Diffusion Policy Policy Optimization)

### 3.1 核心思想

**DPPO** (Allen Ren, Justin Lidard et al., 2024) 把 Diffusion Policy 的 DDIM 去噪过程本身建模为一条 MDP：

- **State**: 当前去噪步的中间 noisy action $x_t$ + conditioning observation $s$
- **Action**: denoiser 的输出（即 noise prediction $\epsilon_\theta$）
- **Transition**: DDIM update rule $x_{t-1} = f(x_t, \epsilon_\theta)$
- **Reward**: 只在最后一步（$t=0$, 完全去噪后）给 environment reward

这样，整个去噪链 $x_T \to x_{T-1} \to \cdots \to x_0$ 就是一条 "episode"，可以直接用 PPO 优化。

### 3.2 为什么 DPPO 比 TD-Learning 更适合 Diffusion Policy

1. **不需要额外的 Q-network**: 直接用 PPO 的 advantage estimation (GAE)
2. **On-policy**: 每次都用最新 policy 生成数据，避免 replay buffer 中 off-policy 数据的分布偏移
3. **自然处理 action chunking**: 整个 chunk 的生成过程就是一条 MDP trajectory
4. **实验验证**: DPPO 在 robotic manipulation benchmarks 上一致优于 IDQL、DQL 等 TD 方法（Allen Ren et al. 的实验表明）
5. **稳定性**: PPO 的 clipping 机制天然防止 policy 崩溃，适合从 BC pretrain 出发

### 3.3 技术细节

**去噪 MDP 定义**:
```python
# 对于 K 步 DDIM (K=10):
# 去噪 "episode" 长度 = K
# 每个去噪步:
#   state_k = (x_k, obs)           # noisy action + env observation
#   action_k = ε_θ(x_k, k, obs)   # denoiser output
#   x_{k-1} = DDIM_step(x_k, ε_θ, k)
# 
# Reward assignment:
#   r_k = 0  for k > 0
#   r_0 = R_env(x_0)  # environment reward after executing the clean action
```

**Value Network (Critic for PPO)**:
```python
class DenoisingValueNetwork(nn.Module):
    """
    Estimates V(x_k, obs, k) — the expected return from denoising step k.
    Architecture: same as denoiser but outputs scalar instead of action.
    """
    def __init__(self, obs_encoder, denoising_step_embed_dim, hidden_dim):
        # Shared obs encoder (can share with policy or be separate)
        # Input: concat(obs_embed, x_k_flat, step_embedding(k))
        # Output: V(x_k, obs, k)
```

**PPO 更新（去噪 MDP 上的）**:
```python
# --- Rollout collection ---
# In env, run full episode with action chunking:
for env_step in range(horizon):
    # Generate action chunk via DDIM
    x_K ~ N(0, I)
    for k in reversed(range(K)):
        ε_k = policy(x_k, k, obs)  # record log_prob
        x_{k-1} = ddim_step(x_k, ε_k, k)
    
    action_chunk = x_0  # shape: (n_action_steps, 8)
    
    # Execute in env
    for a in action_chunk[:n_execute]:
        obs', r, done = env.step(a)
    
    # Assign environment reward to the denoising chain
    denoising_trajectory = [(x_k, ε_k, log_prob_k) for k in range(K)]
    # R at step 0 = sum of env rewards during chunk execution
    # R at step k>0 = 0 (reward only flows through final clean action)

# --- PPO update ---
# GAE on denoising MDP
advantages = compute_gae(denoising_trajectories, value_network)

# Clipped PPO objective
ratio = exp(log_prob_new - log_prob_old)
L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
L_value = MSE(V(x_k, obs, k), returns)
L_entropy = -log_prob.mean()  # encourage exploration in denoising space

L_total = -L_clip + c1 * L_value - c2 * L_entropy
```

**BC Regularization (KL penalty)**:
```python
# Prevent forgetting of BC prior
L_kl = KL(π_θ_new || π_θ_pretrained)
# In practice: MSE between current denoiser output and pretrained denoiser output
# on the same (x_k, k, obs) input
L_total += β * L_kl  # β decays over training
```

### 3.4 算法伪代码
```
Initialize:
  π_θ ← load pretrained Diffusion Policy (BC) — denoiser network
  π_ref ← frozen copy of π_θ (for KL regularization)
  V_ψ ← value network (same architecture, scalar head)
  K = 10 (DDIM denoising steps)

For each PPO iteration:
  # === Phase 1: Env rollout + denoising rollout ===
  buffer = []
  for env_episode in 1..M (parallel across num_envs):
    obs = env.reset()
    for t in 0..horizon:
      # Run DDIM chain, recording each step
      x_K ~ N(0, I)
      chain = []
      for k in reversed(range(K)):
        ε_k = π_θ(x_k, k, obs)
        log_prob_k = gaussian_log_prob(ε_k, predicted_mean, std=1.0)
        x_{k-1} = ddim_step(x_k, ε_k, k)
        chain.append((k, x_k, ε_k, log_prob_k))
      
      action_chunk = x_0  # (n_action_steps, 8)
      
      # Execute chunk in env
      chunk_reward = 0
      for i, a in enumerate(action_chunk[:n_execute]):
        obs', r, done, info = env.step(a)
        chunk_reward += r * γ^i
        if done: break
        obs = obs'
      
      # Store denoising MDP transitions
      for (k, x_k, ε_k, lp_k) in chain:
        r_k = chunk_reward if k == 0 else 0.0
        buffer.append((obs, k, x_k, ε_k, lp_k, r_k, done))

  # === Phase 2: PPO update on denoising MDP ===
  compute_returns_and_advantages(buffer, V_ψ)

  for ppo_epoch in 1..E:
    for mini_batch in buffer.shuffle().chunks(batch_size):
      # Value loss
      L_v = MSE(V_ψ(obs, x_k, k), returns)
      
      # Policy loss (clipped)
      ε_new = π_θ(x_k, k, obs)
      lp_new = gaussian_log_prob(ε_new, ...)
      ratio = exp(lp_new - lp_old)
      L_clip = -min(ratio * adv, clip(ratio, 1-ε_ppo, 1+ε_ppo) * adv)
      
      # KL regularization to BC prior
      ε_ref = π_ref(x_k, k, obs)   # frozen pretrained output
      L_kl = MSE(ε_new, ε_ref)
      
      L = L_clip + c1 * L_v + β * L_kl
      update θ, ψ
```

### 3.5 关键超参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `K` (denoising steps) | 10 | 同 DDIM inference steps |
| `ε_ppo` (clip ratio) | 0.2 | PPO clipping |
| `γ` (env discount) | 0.99 | |
| `λ_gae` | 0.95 | GAE lambda |
| `β` (KL weight) | 初始 1.0 → 退火到 0.01 | BC regularization |
| `c1` (value coeff) | 0.5 | |
| `lr_policy` | 1e-5 | 低 LR 保护 BC prior |
| `lr_value` | 3e-4 | |
| `batch_size` | 256 | |
| `ppo_epochs` | 5 | |
| `num_envs` | 64 | 并行 rollout |
| `rollout_horizon` | 1500 | 同现有 eval 的 horizon |
| `n_action_steps_execute` | 8 | 每个 chunk 执行 8 步 |
| `n_action_steps_predict` | 16 | 预测 16 步 |
| `entropy_coeff` | 0.01 | |
| `max_grad_norm` | 0.5 | 梯度裁剪 |

---

## 4. 共享组件

### 4.1 RL Environment Wrapper (`rl_env_wrapper.py`)

两种方案共用同一个 Gym wrapper，封装：
- Isaac Lab env 创建 + camera setup
- Observation preprocessing（与 BC eval 一致）
- Reward shaping（§1.3）
- Episode hard-reset（复用 `scene_api.py`）
- Vectorized batching（`num_envs > 1`）
- Action chunk → env step 展开

### 4.2 Reward Function (`reward.py`)

独立模块，方便实验不同 reward designs：
- `dense_reward()`: reaching + progress + grasp + success
- `sparse_reward()`: success only (10.0)
- `distance_reward()`: negative distance to goal

### 4.3 Evaluation (`eval_rl.py`)

统一评估接口，与现有 `7_eval_fair.py` 可比较：
- 50 episodes 独立评估
- 输出 success rate + stats JSON
- 可选 video 保存

---

## 5. 实验设计

### 5.1 对比实验矩阵

| 实验 | 方法 | 训练数据 | 备注 |
|------|------|---------|------|
| **Baseline 1** | BC only | 100 demo episodes | 无 finetune |
| **Baseline 2** | BC + DAgger (现有 pipeline) | demo + rollout | 10 iterations, 5000 steps/iter |
| **Baseline 3** | BC + Ablation (无 rollout 聚合) | demo only | 10 iterations, 5000 steps/iter |
| **Ours A** | BC + TD-Diffusion (方案 A) | demo + online replay | 对标 Baseline 2 的总训练量 |
| **Ours B** | BC + DPPO (方案 B) | demo + on-policy rollout | 对标 Baseline 2 的总训练量 |

### 5.2 控制变量

- **相同 BC 预训练**: 所有方法从同一个 pretrained checkpoint 出发
- **相同环境配置**: Mode 3, goal_xy=(0.5, -0.2), 30cm×30cm red region
- **相同评估**: 50 independent episodes per task, horizon=1500
- **相同总环境交互量**: 匹配 Baseline 2 的 10 iter × 50 cycles 的总 env steps
- **相同硬件**: 2 GPU training, 1 GPU eval

### 5.3 评估指标

1. **Task A success rate** (primary): 物体从随机位置到 goal
2. **Task B success rate**: 物体从 goal 到 red region
3. **Sample efficiency**: success rate vs. env steps 曲线
4. **Wall-clock time**: 总训练时间
5. **Training stability**: reward curve, Q-value divergence (方案 A), policy loss (方案 B)

### 5.4 消融实验

| 消融 | 变量 | 目的 |
|------|------|------|
| BC weight λ/β | {0, 0.1, 1.0, 5.0} | BC regularization 的重要性 |
| Dense vs sparse reward | dense / sparse | reward shaping 的影响 |
| num_envs | {16, 64, 256} | 并行度 vs 稳定性 |
| denoising steps K | {5, 10, 20} | DPPO 去噪 MDP 长度 |

---

## 6. 实施计划

### Phase 1: 基础设施 (共享模块)
1. `rl_env_wrapper.py` — Isaac Lab Gym wrapper
2. `reward.py` — Reward functions
3. `utils.py` — 共享工具
4. `eval_rl.py` — 评估脚本

### Phase 2: 方案 A 实现
5. `q_network.py` — Twin Q-network
6. `replay_buffer.py` — Replay buffer + demo buffer
7. `train_td_diffusion.py` — TD-Learning 主训练

### Phase 3: 方案 B 实现
8. `train_dppo.py` — DPPO 主训练

### Phase 4: Pipeline 自动化
9. `run_rl_pipeline.sh` — 自动化实验 shell 脚本

---

## 7. 依赖

现有环境即可满足，无需额外安装：
```
torch >= 2.7.0        # 已有
lerobot >= 0.4.2      # 已有 (加载 pretrained Diffusion Policy)
isaaclab == 2.3.0     # 已有
numpy >= 1.26         # 已有
wandb                 # 已有
```

---

## 8. 参考文献

1. **DPPO**: Allen Ren, Justin Lidard et al. "Diffusion Policy Policy Optimization." CoRL 2024 / RSS 2025.
2. **Diffusion-QL**: Wang et al. "Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning." ICLR 2023.
3. **IDQL**: Hansen-Estruch et al. "IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies." ICML 2023.
4. **DDPO**: Black et al. "Training Diffusion Models with Reinforcement Learning." ICML 2024.
5. **DiffusionRL Survey**: Yang et al. "Diffusion Models for Reinforcement Learning: A Survey." 2024.
6. **PPO**: Schulman et al. "Proximal Policy Optimization Algorithms." 2017.
7. **TD3**: Fujimoto et al. "Addressing Function Approximation Error in Actor-Critic Methods." ICML 2018.
8. **SAC**: Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." ICML 2018.
