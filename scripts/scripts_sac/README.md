# SAC (Soft Actor-Critic) 微调 Diffusion Policy：完整实现指南

> **目标**: 使用 SAC (Soft Actor-Critic) 算法对 BC 预训练的 Diffusion Policy 进行 RL 微调，提升 pick-place 任务的成功率。
>
> **核心思想**: SAC 是一种 off-policy 的最大熵 RL 算法，在标准 RL 目标（最大化累积回报）的基础上增加了 **策略熵正则化**，鼓励探索的同时保持训练稳定性。
>
> **与 `scripts_rl` 的关系**: `scripts_rl` 实现了 TD-Diffusion (方案 A) 和 DPPO (方案 B)，本文件夹是 **方案 C — SAC**，提供两种 actor 模式：
> 1. **SAC-Gaussian**: 全新的 Squashed Gaussian Actor，从头训练
> 2. **SAC-Diffusion**: 直接微调已有的 Diffusion Policy

---

## 目录

1. [SAC 算法原理详解](#1-sac-算法原理详解)
2. [与项目中其他 RL 方法的对比](#2-与项目中其他-rl-方法的对比)
3. [文件结构说明](#3-文件结构说明)
4. [网络架构详解](#4-网络架构详解)
5. [训练流程详解](#5-训练流程详解)
6. [超参数说明](#6-超参数说明)
7. [使用方法](#7-使用方法)
8. [核心公式推导](#8-核心公式推导)
9. [调参建议](#9-调参建议)
10. [常见问题与排查](#10-常见问题与排查)

---

## 1. SAC 算法原理详解

### 1.1 最大熵 RL 框架

传统 RL 的目标是最大化累积回报：

$$J_{\text{standard}}(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) \right]$$

SAC 在此基础上增加了 **策略熵 (entropy)** 作为正则化项：

$$J_{\text{SAC}}(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \cdot \mathcal{H}(\pi(\cdot|s_t)) \right]$$

其中：
- $\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$ 是策略的熵
- $\alpha > 0$ 是温度系数 (temperature)，控制熵正则化的强度

**为什么要最大化熵？**

| 好处 | 解释 |
|------|------|
| **探索** | 高熵 = 策略更随机 = 更多探索，避免过早收敛到局部最优 |
| **鲁棒性** | 学到的策略不会过度依赖某一条动作路径，对扰动更鲁棒 |
| **多模态** | 可以维持对多个 "差不多好" 的动作的概率，不会粗暴地丢弃 |
| **训练稳定** | 熵正则化相当于一种隐式的数据增强，防止策略坍缩 (collapse) |

### 1.2 Soft Bellman 方程

在最大熵框架下，Q-function 和 V-function 的定义相应修改：

**Soft Q-function:**

$$Q^{\text{soft}}(s, a) = r(s, a) + \gamma \mathbb{E}_{s'} \left[ V^{\text{soft}}(s') \right]$$

**Soft V-function:**

$$V^{\text{soft}}(s) = \mathbb{E}_{a \sim \pi} \left[ Q^{\text{soft}}(s, a) - \alpha \log \pi(a|s) \right]$$

将两者结合，得到 **Soft Bellman Backup**:

$$y = r + \gamma (1 - d) \left[ \min(Q_1^{\bar{\theta}}(s', a'), Q_2^{\bar{\theta}}(s', a')) - \alpha \log \pi(a'|s') \right]$$

其中 $a' \sim \pi(\cdot|s')$（从当前策略采样），$Q^{\bar{\theta}}$ 是 target Q-network。

### 1.3 SAC 的三个核心组件

SAC 同时优化三个目标：

#### (1) Critic (Twin Q-Networks)

训练两个 Q-network 最小化 TD 误差：

$$L_Q(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} \left[ (Q_{\theta_i}(s, a) - y)^2 \right], \quad i \in \{1, 2\}$$

使用 clipped double-Q 防止过估计（取两个 Q-network 的最小值作为 target）。

#### (2) Actor (Policy)

最大化 Q 值同时最大化熵：

$$L_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{B}, a \sim \pi_\phi} \left[ \alpha \log \pi_\phi(a|s) - \min(Q_{\theta_1}(s, a), Q_{\theta_2}(s, a)) \right]$$

注意：action $a$ 是通过 **reparameterization trick** 从 $\pi_\phi$ 采样的，因此梯度可以反向传播到 $\phi$。

#### (3) Entropy Temperature (α)

自动调节 α 使得策略的平均熵接近目标值：

$$L_\alpha = -\alpha \mathbb{E}_{a \sim \pi} \left[ \log \pi(a|s) + \mathcal{H}_{\text{target}} \right]$$

其中 $\mathcal{H}_{\text{target}} = -\text{dim}(\mathcal{A})$（动作空间维度的负数，是论文推荐的启发式值）。

### 1.4 Squashed Gaussian Policy

SAC 使用 **tanh squashed Gaussian** 作为策略的参数化形式：

$$u = \mu_\phi(s) + \sigma_\phi(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
$$a = \tanh(u)$$

**为什么需要 tanh?**
- 将动作映射到 $[-1, 1]$ 范围内
- 使得概率密度有界（无界 Gaussian 在 RL 中容易出问题）

**Log probability 修正:**

由于 tanh 是非线性变换，需要用变量替换公式修正 log probability：

$$\log \pi(a|s) = \log \mathcal{N}(u|\mu, \sigma) - \sum_{i=1}^{D} \log(1 - \tanh^2(u_i))$$

实现中用 `log(1 - a² + ε)` 代替，其中 $\epsilon = 10^{-6}$ 防止数值不稳定。

---

## 2. 与项目中其他 RL 方法的对比

| 特性 | TD-Diffusion (方案 A) | DPPO (方案 B) | **SAC-Gaussian (方案 C1)** | **SAC-Diffusion (方案 C2)** |
|------|----------------------|--------------|--------------------------|---------------------------|
| **Actor 类型** | Diffusion Policy | Diffusion Policy | Squashed Gaussian MLP | Diffusion Policy |
| **RL 算法** | TD3 style | PPO | SAC | SAC |
| **On/Off-policy** | Off-policy | On-policy | Off-policy | Off-policy |
| **数据效率** | 高 (replay buffer) | 低 (用完即弃) | **高 (replay buffer)** | **高 (replay buffer)** |
| **探索机制** | 高斯噪声 | 策略随机性 | **最大熵 (自动调节)** | **最大熵 (自动调节)** |
| **熵正则化** | 无 | KL to BC prior | **自动温度 α** | **自动温度 α** |
| **BC 正则化** | λ·L_bc (退火) | KL to frozen ref | 无 (独立 actor) | λ·L_bc (退火) |
| **计算开销** | 中等 | 高 (多步去噪) | **低 (MLP forward)** | 中等 |
| **实现复杂度** | 中等 | 高 | **低** | 中等 |
| **适用场景** | 保留扩散生成能力 | 最细粒度优化 | **快速实验，新 actor** | **保留预训练能力** |

### 为什么选 SAC?

1. **自动探索调节**: 不需要手动设定探索噪声 — α 会自动适应
2. **Off-policy 高效**: 数据复用，比 DPPO 需要少得多的环境交互
3. **训练稳定**: 双 Q 网络 + 熵正则化，比 TD3 更不容易 Q 值爆炸
4. **成熟算法**: SAC 是连续控制领域的标准 baseline，久经考验

---

## 3. 文件结构说明

```
scripts/scripts_sac/
├── __init__.py              # 包标识
├── README.md                # 本文档 — 算法讲解 + 使用指南
├── sac_networks.py          # 网络定义 (Twin Q + Gaussian Actor + α)
├── sac_env_wrapper.py       # 环境包装器 (独立副本，不依赖 scripts_rl)
├── reward.py                # 奖励函数 (独立副本，不依赖 scripts_rl)
├── replay_buffer.py         # 经验回放缓冲区
├── utils.py                 # 工具函数 (seed, soft_update, checkpoint 等)
├── train_sac.py             # 主训练脚本 (支持 gaussian/diffusion 两种模式)
├── eval_sac.py              # 评估脚本 (兼容 eval_fair 输出格式)
└── run_sac_pipeline.sh      # 一键运行全流程（训练 + 评估）
```

### 依赖关系

```
train_sac.py
├── scripts.scripts_sac.sac_env_wrapper  # 独立环境包装器
│   ├── PickPlaceRLEnv                    # 环境接口
│   └── load_pretrained_diffusion_policy  # 加载 BC 预训练权重
├── scripts.scripts_sac.reward           # 独立奖励函数
├── scripts.scripts_sac.sac_networks     # SAC 特有网络
│   ├── SACTwinQNetwork                  # 双 Q 网络
│   ├── SquashedGaussianActor            # 高斯 actor
│   └── AutoEntropyTuning               # 自动 α 调节
├── scripts.scripts_sac.replay_buffer    # 数据存储
└── scripts.scripts_sac.utils            # 工具函数
```

---

## 4. 网络架构详解

### 4.1 Vision Encoder (视觉编码器)

```
Table Image (3, 128, 128) ──→ ResNet18 ──→ AvgPool ──→ FC(512, 512) ──→ table_feat (512)
                                                                            │
Wrist Image (3, 128, 128) ──→ ResNet18 ──→ AvgPool ──→ FC(512, 512) ──→ wrist_feat (512)
                                                                            │
                                                                     cat → FC(1024, 512) → vis_embed (512)
```

- 两个独立 ResNet18 分别编码桌面相机和腕部相机图像
- `DualVisionEncoder` 将两路特征拼接后通过线性层融合为 512 维

### 4.2 Twin Q-Network (Critic)

```
vis_embed (512) ──┐
                  │
state (15)     ───┤── cat ──→ MLP(512+15+128, 256, 256, 1) ──→ Q1 scalar
                  │
action (16×8=128) ┘

vis_embed (512) ──┐
                  │
state (15)     ───┤── cat ──→ MLP(512+15+128, 256, 256, 1) ──→ Q2 scalar
                  │
action (16×8=128) ┘
```

- Q1 和 Q2 **不共享参数**（各自独立的 MLP head）
- 但 **共享** 同一个 `DualVisionEncoder`（减少参数量）
- Action chunk 被 flatten 为 128 维向量 (16 步 × 8 维)

### 4.3 Squashed Gaussian Actor (SAC-Gaussian 模式)

```
vis_embed (512) ──┐
                  │
state (15)     ───┤── cat ──→ MLP(527, 256, 256) ──→ trunk_out (256)
                                                         │
                                                    ┌────┴────┐
                                                    │         │
                                              mean_head   log_std_head
                                              FC(256,128)  FC(256,128)
                                                    │         │
                                                    μ (128)   log σ (128)
                                                    │         │
                                                    └────┬────┘
                                                    Reparameterize:
                                                    u = μ + σ·ε, ε~N(0,I)
                                                    a = tanh(u)
                                                         │
                                                   action (128) → reshape → (16, 8)
```

- `log_std` 被 clamp 在 [-20, 2] 范围内，防止数值问题
- 输出是 action chunk (16×8)，执行时取前 8 步

### 4.4 Automatic Entropy Tuning (α 自动调节)

```
log_α (标量参数, 可学习)
    │
    ├── α = exp(log_α)           # 当前温度
    │
    ├── target_entropy = -dim(A) = -128   # 目标熵 (启发式)
    │
    └── L_α = -α · (log π + H_target)    # α 的损失函数
```

- `log_α` 是唯一的可学习参数
- 如果当前策略的 log π 太大（熵太低），α 会增大 → 更鼓励探索
- 如果当前策略已经足够随机，α 会减小 → 更聚焦于最大化 reward

---

## 5. 训练流程详解

每一轮训练 (iteration) 包含三个阶段：

### Phase 1: 数据收集 (Collect)

```
for _ in range(collect_steps_per_iter):
    if env_step < warmup_steps:
        action = random_action()           # 随机探索 (warmup)
    elif actor_type == "gaussian":
        action, log_prob = actor.sample(obs)  # 从高斯策略采样
    else:
        action = diffusion_policy(obs)     # DDIM 去噪生成
    
    next_obs, reward, done, info = env.step(action)
    replay_buffer.add(obs, action, reward, next_obs, done)
```

**Warmup 阶段** (前 5000 步)：
- 使用 random actions 填充 replay buffer
- 确保训练开始时有足够的数据多样性

### Phase 2: Critic 更新

```
for _ in range(n_critic_updates):
    batch = replay_buffer.sample(batch_size)
    
    with no_grad():
        # 从当前 actor 采样 next action
        next_action, next_log_prob = actor.sample(next_obs)
        
        # Soft Bellman target
        q_target = min(Q1_target(s', a'), Q2_target(s', a')) - α · log_prob
        y = reward + γ · (1 - done) · q_target
    
    # 更新 Q1, Q2
    loss = MSE(Q1(s, a), y) + MSE(Q2(s, a), y)
    loss.backward()
    critic_optimizer.step()
```

### Phase 3: Actor + Alpha 更新

```
for _ in range(n_actor_updates):
    batch = replay_buffer.sample(batch_size)
    
    # Actor loss: α·log_π - min(Q1, Q2)
    new_action, log_prob = actor.sample(obs)
    q_val = min(Q1(s, new_action), Q2(s, new_action))
    actor_loss = (α · log_prob - q_val).mean()
    actor_loss.backward()
    actor_optimizer.step()
    
    # Alpha loss (if auto_entropy)
    alpha_loss = -α · (log_prob + H_target).mean()
    alpha_loss.backward()
    alpha_optimizer.step()

# Target network EMA update
Q_target ← τ · Q + (1 - τ) · Q_target
```

### 完整训练循环示意图

```
┌─────────────────────────────────────────────────────┐
│  Iteration 1, 2, ..., N                             │
│                                                     │
│  ┌──────────────────────────────┐                   │
│  │ Phase 1: Collect 1000 steps  │                   │
│  │   actor → action → env      │                   │
│  │   store to replay buffer    │                   │
│  └──────────────┬───────────────┘                   │
│                 ↓                                   │
│  ┌──────────────────────────────┐                   │
│  │ Phase 2: Critic update ×50   │                   │
│  │   sample batch → TD backup  │                   │
│  │   Q1, Q2 SGD step           │                   │
│  └──────────────┬───────────────┘                   │
│                 ↓                                   │
│  ┌──────────────────────────────┐                   │
│  │ Phase 3: Actor update ×20   │                   │
│  │   α·log π - Q → backprop    │                   │
│  │   actor SGD step            │                   │
│  │   α auto-tune               │                   │
│  └──────────────┬───────────────┘                   │
│                 ↓                                   │
│  ┌──────────────────────────────┐                   │
│  │ EMA: Q_target soft update   │                   │
│  └──────────────┬───────────────┘                   │
│                 ↓                                   │
│  ┌──────────────────────────────┐                   │
│  │ Log metrics / save ckpt     │                   │
│  └──────────────────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

---

## 6. 超参数说明

### 6.1 SAC 核心超参数

| 参数 | 默认值 | 含义 | 调参建议 |
|------|--------|------|----------|
| `--gamma` | 0.99 | 折扣因子 | 短 horizon 任务可以降到 0.95-0.97 |
| `--tau` | 0.005 | Target network EMA 系数 | 越小越稳定，但学习越慢 |
| `--init_alpha` | 0.2 | 初始温度 α | 太大→过度探索，太小→不够探索 |
| `--auto_entropy` | True | 自动调节 α | 强烈建议开启 |
| `--lr_critic` | 3e-4 | Critic 学习率 | Adam 标准默认值 |
| `--lr_actor` | 3e-4 | Actor 学习率 | Gaussian 用 3e-4, Diffusion 用 1e-5 |
| `--lr_alpha` | 3e-4 | α 学习率 | 与 critic 相同通常即可 |
| `--batch_size` | 256 | Mini-batch 大小 | 256 是 SAC 的标准选择 |

### 6.2 数据收集参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--total_env_steps` | 500000 | 总环境交互步数 |
| `--collect_steps_per_iter` | 1000 | 每轮收集的环境步数 |
| `--warmup_steps` | 5000 | 随机探索步数 (训练前) |
| `--replay_buffer_size` | 1000000 | Replay buffer 容量 |
| `--n_critic_updates` | 50 | 每轮 critic 梯度步数 |
| `--n_actor_updates` | 20 | 每轮 actor 梯度步数 |

### 6.3 BC 正则化参数 (仅 diffusion 模式)

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--bc_weight_start` | 2.0 | BC loss 初始权重 |
| `--bc_weight_end` | 0.1 | BC loss 最终权重 |

BC 权重在训练过程中线性退火：开始时强约束到预训练策略，逐渐放松让 RL 主导。

### 6.4 环境参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--num_envs` | 16 | 并行环境数量 |
| `--horizon` | 1500 | 每个 episode 最大步数 |
| `--reward_type` | dense | 奖励类型 (dense/sparse/distance) |
| `--goal_xy` | [0.5, -0.2] | 目标位置 (x, y) |

---

## 7. 使用方法

### 7.1 前提条件

```bash
conda activate rev2fwd_il

# 确认 BC 预训练 checkpoint 存在
ls data/pick_place_isaac_lab_simulation/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model/model.safetensors
```

### 7.2 方式一：一键运行全流程

```bash
# 运行两种模式 + 评估
CUDA_VISIBLE_DEVICES=0 bash scripts/scripts_sac/run_sac_pipeline.sh

# 只运行 Gaussian actor
CUDA_VISIBLE_DEVICES=0 SKIP_DIFFUSION=1 bash scripts/scripts_sac/run_sac_pipeline.sh

# 只运行 Diffusion actor
CUDA_VISIBLE_DEVICES=0 SKIP_GAUSSIAN=1 bash scripts/scripts_sac/run_sac_pipeline.sh

# 自定义参数
CUDA_VISIBLE_DEVICES=0 \
    TOTAL_ENV_STEPS=1000000 \
    NUM_ENVS=32 \
    INIT_ALPHA=0.1 \
    EXP_NAME=sac_more_steps \
    bash scripts/scripts_sac/run_sac_pipeline.sh
```

### 7.3 方式二：分步手动运行

#### Step 1: 训练 SAC-Gaussian

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_sac/train_sac.py \
    --policy_A_ckpt data/pick_place_isaac_lab_simulation/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
    --out runs/sac_gaussian_A \
    --actor_type gaussian \
    --num_envs 16 \
    --total_env_steps 500000 \
    --gamma 0.99 \
    --init_alpha 0.2 \
    --reward_type dense \
    --wandb \
    --headless
```

#### Step 2: 训练 SAC-Diffusion

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_sac/train_sac.py \
    --policy_A_ckpt data/pick_place_isaac_lab_simulation/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
    --out runs/sac_diffusion_A \
    --actor_type diffusion \
    --num_envs 16 \
    --total_env_steps 500000 \
    --lr_actor 1e-5 \
    --bc_weight_start 2.0 \
    --bc_weight_end 0.1 \
    --headless
```

#### Step 3: 评估

```bash
# 评估 SAC-Gaussian
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_sac/eval_sac.py \
    --checkpoint runs/sac_gaussian_A/latest_checkpoint.pt \
    --bc_ckpt data/pick_place_isaac_lab_simulation/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
    --actor_type gaussian \
    --out runs/sac_gaussian_A/eval_fair.json \
    --num_episodes 50 \
    --headless

# 评估 SAC-Diffusion
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_sac/eval_sac.py \
    --checkpoint runs/sac_diffusion_A/latest_checkpoint.pt \
    --bc_ckpt data/pick_place_isaac_lab_simulation/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
    --actor_type diffusion \
    --out runs/sac_diffusion_A/eval_fair.json \
    --num_episodes 50 \
    --headless
```

### 7.4 从 checkpoint 恢复训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_sac/train_sac.py \
    --policy_A_ckpt data/.../pretrained_model \
    --out runs/sac_gaussian_A \
    --actor_type gaussian \
    --resume \
    --headless
```

---

## 8. 核心公式推导

### 8.1 为什么用 min(Q1, Q2)?

双 Q-network 是为了解决 Q-value overestimation 问题。

单一 Q-network 的 TD target 为:
$$y = r + \gamma Q_\theta(s', a')$$

由于 $Q_\theta$ 本身有估计误差，$\max_a Q$ 总是倾向于选出被高估的动作 → 正反馈循环 → Q 值爆炸。

**解决方案**: 训练两个独立的 Q-network，取较小值:
$$y = r + \gamma (1-d) \left( \min(Q_{\bar\theta_1}(s', a'), Q_{\bar\theta_2}(s', a')) - \alpha \log \pi(a'|s') \right)$$

### 8.2 Reparameterization Trick

SAC 需要对 actor 的输出求梯度。如果 action 只是从分布中采样（不可微），就无法反向传播。

**Reparameterization Trick**: 将随机性从参数中分离出来：

$$a = f_\phi(s, \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

具体地:
$$u = \mu_\phi(s) + \sigma_\phi(s) \cdot \epsilon$$
$$a = \tanh(u)$$

现在 $a$ 关于 $\phi$ 是可微的，因为 $\epsilon$ 是常量（采样后固定）。

### 8.3 Tanh Squashing 的 Log Probability

设 $u \sim \mathcal{N}(\mu, \sigma^2)$，$a = \tanh(u)$，则：

$$\log \pi(a|s) = \log \mathcal{N}(u | \mu, \sigma) - \sum_{i=1}^D \log(1 - a_i^2)$$

推导利用变量替换公式:
$$p(a) = p(u) \cdot \left| \det \frac{\partial u}{\partial a} \right| = p(u) \cdot \prod_{i=1}^D \frac{1}{1 - a_i^2}$$

取对数后即得上式。

### 8.4 自动温度调节推导

我们想让策略的平均熵不低于某个目标 $\bar{\mathcal{H}}$:

$$\mathbb{E}_{a \sim \pi}[-\log \pi(a|s)] \geq \bar{\mathcal{H}}$$

这可以写成一个约束优化问题，用对偶变量 $\alpha$ (即温度) 来松弛:

$$\alpha^* = \arg\min_{\alpha \geq 0} \mathbb{E}_{s \sim \mathcal{B}} \left[ -\alpha \left( \mathbb{E}_{a \sim \pi}[\log \pi(a|s)] + \bar{\mathcal{H}} \right) \right]$$

当 $\mathbb{E}[\log \pi] + \bar{\mathcal{H}} < 0$（熵高于目标），$\alpha$ 会减小 (减少探索)。
当 $\mathbb{E}[\log \pi] + \bar{\mathcal{H}} > 0$（熵低于目标），$\alpha$ 会增大 (增加探索)。

论文建议 $\bar{\mathcal{H}} = -\text{dim}(\mathcal{A})$。

---

## 9. 调参建议

### 9.1 按优先级排列

1. **Reward shaping** (最重要):
   - 先用 `dense` reward 验证流程
   - 如果 critic 学得太快导致 Q 值不稳定，考虑 `distance` reward (更平滑)
   - 如果 critic 太保守，打开 `--normalize_reward`

2. **α 初始值**:
   - 太大 (>1.0): actor 过于随机，Q 值学不稳
   - 太小 (<0.01): 几乎退化为 TD3，可能过早收敛
   - 建议: 0.1~0.3，开启 auto_entropy

3. **学习率**:
   - Gaussian actor: `lr_actor=3e-4` (标准值)
   - Diffusion actor: `lr_actor=1e-5` (要小，避免破坏预训练)
   - Critic: `lr_critic=3e-4` (标准值)

4. **n_critic_updates vs n_actor_updates**:
   - 通常 critic 更新次数 > actor (critic 需要先学稳)
   - 默认 50 vs 20 是合理的起点

### 9.2 常见 failure modes

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| Q 值持续增长到 $10^4$+ | Overestimation | 降低 lr_critic, 增加 tau |
| α 迅速降到 ~0 | 策略快速坍缩 | 增大 init_alpha, 检查 reward scale |
| α 持续增长 | 策略始终太确定 | 检查 actor 架构是否正确输出 std |
| Reward 始终为 0 | Exploration 不足 | 增加 warmup_steps, 增大 init_alpha |
| Reward 先升后跌 | Actor 过拟合 Q | 降低 lr_actor, 增加 n_critic_updates |
| 训练很慢 | GPU 利用率低 | 增加 num_envs 和 batch_size |

### 9.3 推荐的实验计划

```
实验 1 (baseline):
    actor_type=gaussian, total_steps=200K, default params
    → 验证 SAC 流程跑通

实验 2 (scale up):
    actor_type=gaussian, total_steps=500K, num_envs=32
    → 看 success rate 能到多少

实验 3 (diffusion):
    actor_type=diffusion, total_steps=500K, lr_actor=1e-5
    → 对比 Gaussian actor

实验 4 (tuning):
    最好的模式 + 不同 reward_type + 不同 gamma
    → 精细调参
```

---

## 10. 常见问题与排查

### Q: SAC vs TD3 的主要区别？

| 特性 | TD3 | SAC |
|------|-----|-----|
| 策略类型 | 确定性 | 随机 (Gaussian) |
| 探索 | 外加噪声 (手动设定) | 熵正则化 (自动调节) |
| Target smoothing | Action noise | 策略采样 + α·log π |
| Actor 更新频率 | 延迟 (每2步更新1次) | 每步都更新 |

### Q: 什么时候用 gaussian 模式 vs diffusion 模式？

- **Gaussian**: 实验速度快，适合快速迭代超参。如果 Gaussian actor 效果就足够好，不需要上 Diffusion。
- **Diffusion**: 保留预训练扩散模型的多模态能力，理论上 ceiling 更高。但训练更慢，调参更困难。

### Q: Replay buffer 需要多大？

- 推荐 100 万 (默认值)。对于 `128×128` 双相机图像:
  - 每条 transition ≈ 2 × (3×128×128) × 2 (当前+下一帧) × 4 bytes ≈ 800 KB
  - 100 万条 ≈ 800 GB (存在 numpy 数组中)
- 如果内存不够，可以降到 50 万 或降低图像分辨率

### Q: 如何判断训练是否收敛？

监控以下指标:
1. `critic_loss`: 应该逐渐下降并稳定
2. `alpha`: 应该从 init_alpha 逐渐调整到合理范围 (0.01~0.5)
3. `avg_log_prob`: 不应该无限制地增大（表示策略坍缩）
4. `episode_reward`: 应该逐渐上升
5. `success_rate`: 最终指标，应该 > 0

### Q: 如何和 scripts_rl 的结果做公平对比？

所有方法的评估脚本输出格式相同 (JSON)，包含 `success_rate`, `avg_reward` 等字段。可以直接对比:

```bash
# 对比所有方法
for f in runs/rl_experiments/*/td_diffusion/eval_fair.json \
         runs/rl_experiments/*/dppo/eval_fair.json \
         runs/sac_experiments/*/sac_gaussian/eval_fair.json \
         runs/sac_experiments/*/sac_diffusion/eval_fair.json; do
    if [ -f "$f" ]; then
        python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
print(f'{d[\"method\"]:20s}: {d[\"success_rate\"]*100:.1f}%  avg_reward={d[\"avg_reward\"]:.1f}')
"
    fi
done
```

---

## 参考文献

1. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML 2018.
2. Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ... & Levine, S. (2019). *Soft Actor-Critic Algorithms and Applications*. arXiv:1812.05905.
3. Fujimoto, S., Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods* (TD3). ICML 2018.
4. Ren, A., Lidard, J., Ankile, L. L., et al. (2025). *Diffusion Policy Policy Optimization* (DPPO). RSS 2025.
5. Wang, Z., Hunt, J. J., & Zhou, M. (2023). *Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning*. ICLR 2023.
