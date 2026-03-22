# RL-Based Rev2Fwd: 实验计划

> **目标**: 用 RL 替代当前的 rollout + DAgger finetune 循环，从 BC 预训练的 Diffusion Policy 出发，直接在线优化。
>
> **核心约束（面向真机部署）**：
> - **不使用模拟器特权信息**: 观测中不包含 obj_pose（物体位姿的 ground truth），reward 不依赖 obj_pose
> - **不使用模拟器 reset**: 用 Task B（反向）policy 执行环境复位，而非 `teleport_object_to_pose()`
> - **利用时间倒放数据**: 正向(A)和反向(B)的数据通过时间反转相互生成，训练的两个 policy 形成 A↔B 闭环
>
> 两条技术路线：
> 1. **方案 A — TD-Learning (Critic-Guided Diffusion Policy)**: 用 TD3/SAC 风格的 Q-function 对 Diffusion Policy 的去噪过程做梯度引导，同时保留 BC 预训练的先验（DDPO / DiffusionQL 思路）。
> 2. **方案 B — SOTA: DPPO (Diffusion Policy Policy Optimization)**: 把整个 DDIM 去噪链条视为一条 multi-step MDP，用 PPO 直接优化每一步去噪（Allen Ren et al., RSS 2025 / CoRL 2024）。

---

## 术语表 / Glossary

> 本节解释文档中出现的所有缩写和关键技术概念，方便快速查阅。

### 基础 RL / IL 概念

| 缩写/术语 | 全称 | 含义 |
|-----------|------|------|
| **RL** | Reinforcement Learning（强化学习） | 智能体通过与环境交互、获取奖励来学习最优策略的框架。与 IL 不同，RL 不需要专家示范，但需要 reward function。 |
| **IL** | Imitation Learning（模仿学习） | 从专家示范数据中学习策略，不需要显式 reward。BC 和 DAgger 是两种常见 IL 方法。 |
| **BC** | Behavioral Cloning（行为克隆） | 最简单的 IL 方法：把专家示范当作监督学习数据，直接让 policy 学习 "看到这个观测 → 输出这个动作"。缺点是分布偏移（compounding error）。 |
| **DAgger** | Dataset Aggregation | 一种迭代式 IL 方法：先用 BC 训练初始 policy → 用 policy 在环境中 rollout → 收集新数据 → 合并到训练集 → 继续训练。不断让 policy 见到自己的错误并纠正。 |
| **MDP** | Markov Decision Process（马尔可夫决策过程） | RL 的数学框架，由 (S, A, P, R, γ) 组成：状态集 S、动作集 A、转移概率 P、奖励函数 R、折扣因子 γ。 |
| **Policy (π)** | 策略 | 从观测（状态）到动作的映射函数。可以是确定性的 `a = π(s)` 或随机的 `a ~ π(·|s)`。神经网络实现时就是一个接受观测输入、输出动作的网络。 |
| **Q-function / Q-value** | Q(s, a)（状态-动作值函数） | 在状态 s 下执行动作 a，之后始终按照最优策略行动所能获得的预期累积回报。Q 值越高，说明这个（状态, 动作）对越好。 |
| **V-function / Value** | V(s)（状态值函数） | 从状态 s 出发，按照当前策略行动所能获得的预期累积回报。`V(s) = E[Q(s, a)]`。 |
| **Reward (r)** | 奖励 | 环境在每一步给智能体的反馈信号。RL 的目标是最大化累积 reward。 |
| **Episode** | 回合 | 从环境重置到终止的完整交互序列。一个 pick-place episode = 从初始状态开始 → 夹取 → 放置 → 结束。 |
| **Rollout** | 展开/试运行 | 用当前策略在环境中运行一个或多个 episode，收集（obs, action, reward）数据。 |
| **On-policy** | 在线策略 | 只用当前最新策略收集的数据来更新参数。优点：数据分布匹配；缺点：数据利用率低（用完即弃）。PPO 属于 on-policy。 |
| **Off-policy** | 离线策略 | 可以用历史旧数据（不同版本 policy 收集的）来更新参数，通过 replay buffer 存储。优点：数据利用率高；缺点：分布偏移问题。TD3/SAC 属于 off-policy。 |
| **Replay Buffer** | 经验回放缓冲区 | Off-policy 方法使用的数据存储结构，存储历史 (s, a, r, s', done) 元组，训练时从中随机采样 mini-batch。 |

### RL 算法

| 缩写/术语 | 全称 | 含义 |
|-----------|------|------|
| **TD Learning** | Temporal Difference Learning（时序差分学习） | 用 `r + γ·V(s')` 来估计 `V(s)` 的方法（bootstrap）。不需要等 episode 结束就能更新，比 Monte Carlo 方差更小。 |
| **TD3** | Twin Delayed DDPG | 一种 off-policy actor-critic 算法。"Twin"指使用双 Q 网络取较小值（防过估计）、"Delayed"指 actor 更新频率低于 critic。 |
| **SAC** | Soft Actor-Critic | 最大化累积 reward + 策略的熵（随机性），鼓励探索。与 TD3 类似但使用随机策略。 |
| **PPO** | Proximal Policy Optimization（近端策略优化） | On-policy RL 算法。核心是 clipped surrogate loss：限制每次更新时 policy 的变化幅度，防止灾难性崩溃。广泛用于 RLHF（ChatGPT）和机器人控制。 |
| **GAE** | Generalized Advantage Estimation（广义优势估计） | 计算 advantage（优势函数）的方法：`A(s,a) = Q(s,a) - V(s)`，衡量 "这个动作比平均水平好多少"。GAE 用 λ 参数在 bias 和 variance 之间权衡。 |
| **EMA** | Exponential Moving Average（指数移动平均） | Target Q-network 的更新方式：`Q̄ ← τ·Q + (1-τ)·Q̄`，让 target 慢慢跟上主网络，提高训练稳定性。τ 越小更新越慢。 |
| **Clipping** | 裁剪 | PPO 的核心机制：把更新比率 (π_new/π_old) 限制在 [1-ε, 1+ε] 范围内，防止单次更新幅度过大。 |
| **Bellman Backup** | 贝尔曼回溯 | TD 更新的核心公式：`Q(s,a) ← r + γ·Q(s',a')`。用未来一步的估计值来更新当前值。 |

### Diffusion Policy 相关

| 缩写/术语 | 全称 | 含义 |
|-----------|------|------|
| **Diffusion Policy** | 扩散策略 | 用扩散模型（类似 Stable Diffusion 生成图像的方式）来生成机器人动作。优点是表达能力强，可以拟合多模态动作分布。 |
| **DDIM** | Denoising Diffusion Implicit Models（去噪扩散隐式模型） | 加速版扩散采样。标准扩散需 50-1000 步去噪，DDIM 只需 10 步，且是确定性的（同一噪声 → 同一动作）。本项目 inference 时用 10 步 DDIM。 |
| **Denoising / 去噪** | — | 扩散模型的推理过程：从纯随机噪声 $x_T$ 出发，逐步"去噪" → $x_{T-1}$ → ... → $x_0$（干净的动作）。每一步由去噪网络 (denoiser) 预测噪声成分并移除。 |
| **Denoiser ($\epsilon_\theta$)** | 去噪网络 | 核心神经网络。输入 = (当前带噪动作 $x_t$, 时间步 $t$, 观测 obs)，输出 = 预测的噪声 $\epsilon$。训练目标是让预测噪声尽量接近真实加入的噪声。 |
| **Score Matching** | 分数匹配 | 扩散模型的训练方法。"Score" 是 $\nabla_x \log p(x)$（数据分布的梯度方向）。训练 denoiser 预测这个梯度方向，等价于让它学会"往干净数据方向走"。 |
| **Action Chunking** | 动作分块 | 一次预测多步动作（本项目：16 步），但只执行前几步（本项目：8 步）。好处：(1) 动作更平滑连贯；(2) 减少决策频率；(3) 解决 BC 中的抖动问题。 |
| **DPPO** | Diffusion Policy Policy Optimization（扩散策略策略优化） | 本文方案 B 的核心方法。把 DDIM 的 10 步去噪过程看作一个 10 步的小 MDP，每一步去噪就是一次"动作"，用 PPO 来优化去噪网络的每一步输出。 |
| **DQL** | Diffusion Q-Learning（扩散 Q 学习） | 用 Q-function 给 Diffusion Policy 生成的动作打分，通过 Q 的梯度来引导 diffusion 输出更好的动作。方案 A 的理论基础。 |
| **IDQL** | Implicit Diffusion Q-Learning | DQL 的改进版：不直接反向传播 Q 梯度到 diffusion chain，而是用 importance weighting 隐式地引导 policy。 |
| **DDPO** | Denoising Diffusion Policy Optimization | 早期工作：用 RL 训练文本到图像的扩散模型。DPPO 将类似思路应用于机器人动作生成。 |

### 网络架构相关

| 缩写/术语 | 全称 | 含义 |
|-----------|------|------|
| **Actor** | 演员网络 | Actor-Critic 架构中负责 "选择动作" 的网络。本项目中 actor = Diffusion Policy（denoiser）。 |
| **Critic** | 评论家网络 | Actor-Critic 架构中负责 "评估动作好坏" 的网络。方案 A 中 critic = Q-network；方案 B 中 critic = Value network。 |
| **Twin Q-Networks** | 双 Q 网络 | 同时训练两个 Q 网络 (Q1, Q2)，取较小值作为 target。目的：防止 Q 值过度乐观（overestimation），这是 TD3 的核心创新。 |
| **Target Network (Q̄)** | 目标网络 | Q-network 的缓慢更新副本，用于计算 TD target `r + γ·Q̄(s',a')`。如果用主网络本身计算 target，会导致自举不稳定。 |
| **ResNet-18** | Residual Network-18 层 | 经典 CNN 图像编码器，18 层深。用于将 128×128 RGB 图像压缩为 512 维向量。 |
| **MLP** | Multi-Layer Perceptron（多层感知机） | 最基本的全连接神经网络。本项目中 Q-network 的 head 是 MLP：输入 obs+action → 输出 Q 值。 |
| **U-Net** | — | 编码器-解码器架构（带跳跃连接），最初用于图像分割。在 Diffusion Policy 中作为 denoiser 的骨干网络，处理时序上的多步动作。 |

### 训练技巧

| 缩写/术语 | 含义 |
|-----------|------|
| **BC Regularization / BC 正则化** | RL 微调时保留一部分 BC loss，防止 policy "忘记" 预训练学到的行为先验。类似于 LLM fine-tune 时的 KL penalty。 |
| **KL Divergence / KL 散度** | 衡量两个概率分布差异的指标。`KL(π_new || π_ref)` 越大，说明新策略偏离原始 BC 策略越多。用作正则化项约束更新幅度。 |
| **学习率退火 (Annealing)** | 某个超参数（如 BC 权重 λ/β）随训练进度逐渐减小。初期重视 BC 先验（大 λ），后期释放 RL 控制（小 λ）。 |
| **梯度裁剪 (Gradient Clipping)** | 限制梯度的最大范数（如 0.5），防止梯度爆炸导致训练不稳定。 |
| **Warmup** | 训练初期用随机策略收集数据填充 replay buffer，只更新 critic、不更新 actor。让 Q 网络先学到合理的估计再指导 policy。 |

### 环境 / 任务相关

| 缩写/术语 | 全称 | 含义 |
|-----------|------|------|
| **Isaac Lab** | — | NVIDIA 的 GPU 加速机器人仿真平台（基于 Isaac Sim）。支持大规模并行仿真（数百个环境同时跑）。 |
| **IK** | Inverse Kinematics（逆运动学） | 给定目标 3D 位置和姿态 → 计算机器人各关节角度。本项目中 policy 输出末端执行器目标位姿，仿真环境通过 IK 转化为关节命令。 |
| **ee_pose** | End-Effector Pose（末端执行器位姿） | 机器人手爪（夹爪）的 3D 位置 (x,y,z) + 姿态四元数 (qw,qx,qy,qz)，共 7 维。 |
| **Quaternion** | 四元数 (qw, qx, qy, qz) | 表示 3D 旋转的数学工具（比欧拉角更稳定，无万向锁问题）。`qw` 是实部，`(qx,qy,qz)` 是虚部。 |
| **Gripper** | 夹爪 | 机器人末端的抓取工具。本项目中简化为 1 维：+1 = 张开（释放物体），-1 = 闭合（夹紧物体）。 |
| **Pick-Place** | 拾取-放置 | 核心任务：从位置 A 拿起物体 → 移动到位置 B → 放下。 |
| **Rev2Fwd** | Reverse-to-Forward | 本项目核心方法：先收集 "反向" 任务 B 的数据（从 goal 到桌面），再时间反转得到 "正向" 任务 A 的数据（从桌面到 goal）。 |
| **Task A (正向)** | Forward Task | 从桌面随机位置 → 拿起物体 → 放到 goal 位置。 |
| **Task B (反向/Reset)** | Reverse Task | 从 goal 位置 → 拿起物体 → 放到桌面随机位置。RL 中同时充当 **reset policy**: 执行 Task B 就等于把环境复位。 |
| **Privileged Info** | 特权信息 | 只在仿真中可获取、真机上不可用的信息（如物体位姿 ground truth）。本项目要求不使用特权信息以确保真机可部署。 |
| **Gym Wrapper** | — | 把仿真环境包装成 OpenAI Gym 标准接口 (`reset()`, `step()`, `obs_space`, `action_space`)，让任何标准 RL 算法都能直接使用。 |
| **Vectorized Env** | 向量化环境 | 同时并行运行多个环境实例（如 64 个），一次 `step()` 调用同时返回 64 组 (obs, reward, done)。大幅加速数据收集。 |

### 数学符号

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 参数为 θ 的策略网络（actor / Diffusion Policy） |
| $\pi_{ref}$ | 冻结的预训练策略副本，用作 KL 正则化的参考基准 |
| $Q_\phi$ | 参数为 φ 的 Q 网络（critic） |
| $\bar{Q}_\phi$ | Q 网络的目标网络（EMA 慢更新版本） |
| $V_\psi$ | 参数为 ψ 的 Value 网络（PPO 的 critic） |
| $\epsilon_\theta$ | Denoiser 网络预测的噪声 |
| $x_t$ | 去噪过程第 t 步的中间带噪动作 |
| $x_T$ | 初始纯噪声（从标准正态分布采样） |
| $x_0$ | 最终去噪完成的干净动作 |
| $\gamma$ | 折扣因子（0.99）：未来 reward 的衰减系数。γ 越接近 1，越关注长期回报。 |
| $\tau$ | Target network EMA 系数（0.005）：值越小，target 网络更新越慢越稳定。 |
| $\lambda$ / $\beta$ | BC 正则化权重：平衡 RL loss 和 BC loss 的相对重要性。 |
| $\epsilon_{ppo}$ | PPO clip 范围（0.2）：限制策略更新步长的阈值。 |

---

## 关键公式详解

> 本节对文档中涉及的所有核心数学公式进行逐一推导和直觉解释。
> 建议结合上方术语表和后续的伪代码一起阅读。

### F1. Diffusion Policy 训练目标（Score Matching / 噪声预测）

BC 预训练阶段，Diffusion Policy 的训练目标是**让去噪网络学会预测"加了多少噪声"**。

**前向加噪过程**: 给专家动作 $a_0$ 加上不同程度的噪声：

$$x_t = \sqrt{\bar{\alpha}_t}\, a_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

- $a_0$: 示范数据中的真实动作（16 步 action chunk, 形状 $16 \times 8$）
- $t$: 噪声时间步（$t=0$ 表示干净, $t=T$ 表示纯噪声）
- $\bar{\alpha}_t$: 噪声调度参数, $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$, 随 $t$ 增大而减小
- 直觉: $t$ 越大, $\sqrt{\bar{\alpha}_t}$ 越小 → 原始信号被压缩越多, $\sqrt{1 - \bar{\alpha}_t}$ 越大 → 噪声成分越大

**训练 Loss (简单版本)**:

$$\mathcal{L}_{BC} = \mathbb{E}_{t, \epsilon, a_0} \left[ \left\| \epsilon_\theta(x_t, t, s) - \epsilon \right\|^2 \right]$$

- $\epsilon_\theta$: 去噪网络（我们要训练的 U-Net），输入带噪动作 $x_t$、时间步 $t$、观测 $s$
- $\epsilon$: 真正加进去的噪声（ground truth）
- 直觉: **让网络猜 "我加了什么噪声"，猜得越准越好**
- 从 score matching 角度: $\epsilon_\theta \approx -\sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log p(x_t|s)$，即学习数据分布的梯度场

### F2. DDIM 去噪采样公式（推理时用）

训练好之后，推理时从纯噪声 $x_T \sim \mathcal{N}(0,I)$ 出发，逐步去噪得到干净动作 $x_0$。

**DDIM 单步更新** (从 $x_t$ 到 $x_{t-1}$):

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1-\bar{\alpha}_t}\, \epsilon_\theta(x_t, t, s)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{predicted } x_0} + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \underbrace{\epsilon_\theta(x_t, t, s)}_{\text{predicted noise direction}}$$

拆解三步理解:

1. **估计干净动作 $\hat{x}_0$**: 先用网络预测噪声 $\epsilon_\theta$，然后"减掉"噪声得到对 $x_0$ 的估计:
$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\, \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$
> 直觉: "如果噪声是这些，那原来的干净数据应该长这样"

2. **重新加噪到 $t$$-1$ 级别**: 把 $\hat{x}_0$ 和噪声方向按 $t-1$ 对应的比例混合，得到 $x_{t-1}$
> 直觉: "不一步跳到干净，而是只去掉一点点噪声，稳步推进"

3. **DDIM 是确定性的**: 没有额外随机噪声项（区别于 DDPM），同一初始噪声 → 同一动作输出

本项目用 $K=10$ 步 DDIM，即在 $[T, \ldots, 1]$ 中均匀选 10 个时间步做去噪。

### F3. Bellman 方程与 TD Target（方案 A 的 Critic 训练）

Q-network 的训练目标：让 $Q(s,a)$ 逼近 **真实的累积折扣回报**。

**Bellman 最优方程**:

$$Q^*(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

> 直觉: "在状态 $s$ 做动作 $a$ 的总价值 = 立刻拿到的 reward $r$ + 未来所有 reward 的折扣总和"

**实际训练用的 TD Target (TD3 风格, Clipped Double-Q)**:

$$y = r + \gamma \cdot \min\left( \bar{Q}_{\phi_1}(s', a'), \; \bar{Q}_{\phi_2}(s', a') \right)$$

其中:
- $a' = \pi_\theta(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$
  → 下一步动作由 policy 生成 + clipped 噪声（Target Policy Smoothing）
- $\bar{Q}_{\phi_1}, \bar{Q}_{\phi_2}$: 两个**目标 Q 网络**（EMA 慢更新版本）
- $\min(\cdot, \cdot)$: 取两个 Q 的较小值

> **为什么取 min?** Q 网络天生倾向于**过度乐观**（高估 Q 值），因为 max 操作会放大估计误差。
> 取两个独立 Q 网络的较小者 → 相当于对乐观估计做了保守修正。这是 TD3 (Twin Delayed DDPG) 的核心贡献。

**Critic Loss**:

$$\mathcal{L}_{critic} = \frac{1}{N}\sum_{i=1}^{N} \left[ \left( Q_{\phi_1}(s_i, a_i) - y_i \right)^2 + \left( Q_{\phi_2}(s_i, a_i) - y_i \right)^2 \right]$$

> 直觉: 让两个 Q 网络的预测都尽量接近 TD target $y$。

### F4. Actor Loss — Critic-Guided Diffusion（方案 A 的 Actor 更新）

用 Q 网络的梯度来引导 Diffusion Policy 生成更好的动作:

$$\mathcal{L}_{actor} = -\underbrace{\frac{1}{N}\sum_{i=1}^{N} Q_{\phi_1}(s_i, \pi_\theta(s_i))}_{\text{RL 项: 最大化 Q 值}} + \lambda \underbrace{\mathcal{L}_{BC}(\theta)}_{\text{BC 正则化: 防遗忘}}$$

- **RL 项** ($-Q$): 让 policy 生成的动作被 Q 网络打更高的分。负号是因为我们做梯度**下降**来**最大化** Q。
  > 梯度流向: $Q_\phi$ 的梯度 → 经过动作 $a = \pi_\theta(s)$ → 反向传播到 policy 参数 $\theta$
  > 关键: 整个 DDIM 去噪链必须保持可微分 (differentiable)

- **BC 正则化** ($\lambda \mathcal{L}_{BC}$): 在示范数据上计算扩散训练 loss（§F1 的公式），防止 policy 偏离预训练太远
  > $\lambda$ 退火策略: $\lambda: 2.0 \to 0.1$，初期以 BC 为主, 后期逐渐释放 RL 控制

### F5. EMA 目标网络更新

$$\bar{Q}_\phi \leftarrow \tau \cdot Q_\phi + (1 - \tau) \cdot \bar{Q}_\phi$$

- $\tau = 0.005$ (非常小)
- 每次只把主网络的 0.5% "混入"目标网络
- 效果: 目标网络**非常缓慢**地追踪主网络，提供稳定的 TD target

> 直觉: 如果直接用主 Q 网络算 TD target，会出现"自己给自己打分"的循环 → 训练不稳定。
> 目标网络相当于一个"滞后版本的考官"，评判标准变化很慢，让训练更平稳。

### F6. PPO Clipped Surrogate Loss（方案 B 核心）

PPO 的核心: 限制每次策略更新的幅度，防止灾难性崩溃。

**概率比 (importance ratio)**:

$$r_t(\theta) = \frac{\pi_\theta(\text{action}_t \mid \text{state}_t)}{\pi_{\theta_{old}}(\text{action}_t \mid \text{state}_t)}$$

- $r_t \approx 1$: 新旧策略对这个动作的概率差不多 → 更新幅度小
- $r_t \gg 1$ 或 $r_t \ll 1$: 新旧策略差异很大 → 需要裁剪

**Clipped Surrogate Loss**:

$$\mathcal{L}_{clip} = -\mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \;\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

- $\hat{A}_t$: advantage 估计（动作比平均好多少），详见 §F7
- $\epsilon = 0.2$: 裁剪范围, 即 $r_t$ 被限制在 $[0.8, 1.2]$

> **直觉解释 (分两种情况)**:
> - **$\hat{A}_t > 0$ (好动作)**: 我们想增加这个动作的概率 (增大 $r_t$)，
>   但 clip 限制 $r_t \leq 1.2$ → 最多增加 20% 的概率，防止过度集中
> - **$\hat{A}_t < 0$ (坏动作)**: 我们想减少这个动作的概率 (减小 $r_t$)，
>   但 clip 限制 $r_t \geq 0.8$ → 最多减少 20% 的概率，防止过度回避
>
> 为什么取 min? 因为 clip 只在"更新有利于目标函数"时起作用。
> 如果 unclipped 已经比 clipped 差了，就用 unclipped (不额外帮倒忙)。

**在去噪 MDP 中的应用**: 这里的 state = $(x_k, s)$ (噪声动作 + 环境观测),
action = $\epsilon_k$ (去噪网络的输出), 概率比通过高斯 log-prob 计算:

$$r_k(\theta) = \exp\left( \log \pi_\theta(\epsilon_k | x_k, k, s) - \log \pi_{\theta_{old}}(\epsilon_k | x_k, k, s) \right)$$

### F7. GAE — 广义优势估计（方案 B 用于计算 Advantage）

**Advantage** 衡量"这个动作比平均水平好多少":

$$A(s, a) = Q(s, a) - V(s)$$

> 直觉: $Q$ 是"做了这个动作能拿多少分", $V$ 是"平均能拿多少分"。
> $A > 0$: 好于平均; $A < 0$: 差于平均。

**TD 残差 (单步 advantage 估计)**:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

> 直觉: "实际拿到的 $r_t$ + 未来估值" vs "之前对当前状态的估值"。$\delta_t > 0$ 说明情况比预期好。

**GAE (多步平滑版)**:

$$\hat{A}_t^{GAE} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

展开:
$$\hat{A}_t = \delta_t + \gamma\lambda\, \delta_{t+1} + (\gamma\lambda)^2 \delta_{t+2} + \cdots$$

- $\lambda = 0.95$: GAE 的平滑参数
  - $\lambda = 0$: 只看 1 步 ($\hat{A}_t = \delta_t$), 偏差大(bias)但方差小(variance)
  - $\lambda = 1$: 看到 episode 结束 (Monte Carlo return), 无偏但方差大
  - $\lambda = 0.95$: 折中，主要看近处几步但也考虑远处

**在去噪 MDP 中**: $t$ 是去噪步数 $(k = K, K-1, \ldots, 1, 0)$，只有 $k=0$ 有 reward。
GAE 把最终的环境 reward 通过 $(\gamma\lambda)^k$ 衰减回传给前面的每一步去噪决策，
让早期去噪步也知道"我这一步去噪做得好不好会影响最终结果"。

### F8. PPO 总 Loss（方案 B 完整训练目标）

$$\mathcal{L}_{total} = \underbrace{\mathcal{L}_{clip}}_{\text{policy 更新}} + c_1 \underbrace{\mathcal{L}_{value}}_{\text{Value 网络}} - c_2 \underbrace{\mathcal{H}[\pi_\theta]}_{\text{熵正则化}} + \beta \underbrace{\mathcal{L}_{KL}}_{\text{BC 正则化}}$$

各项含义:

| 项 | 公式 | 作用 |
|----|------|------|
| $\mathcal{L}_{clip}$ | 见 §F6 | 核心: 优化 policy, 限制更新幅度 |
| $\mathcal{L}_{value}$ | $\frac{1}{N}\sum_i (V_\psi(s_i) - R_i)^2$ | 让 Value 网络的预测接近真实回报 |
| $\mathcal{H}[\pi_\theta]$ | $-\mathbb{E}[\log \pi_\theta]$ | 熵奖励: 鼓励 policy 保持一定随机性, 避免过早收敛到次优解 |
| $\mathcal{L}_{KL}$ | $\|\epsilon_\theta^{new} - \epsilon_\theta^{ref}\|^2$ | 约束去噪输出不偏离 BC 预训练太远 |

系数: $c_1 = 0.5$, $c_2 = 0.01$, $\beta: 1.0 \to 0.01$ (退火)

### F9. KL 正则化 / BC 正则化

两种方案都使用 BC 正则化，但具体形式不同:

**方案 A (显式 BC Loss)**:

$$\mathcal{L}_{actor} = -Q_{\phi}(s, \pi_\theta(s)) + \lambda \cdot \mathbb{E}_{(s,a)\sim D_{demo}} \left[ \| \epsilon_\theta(x_t, t, s) - \epsilon \|^2 \right]$$

> 在示范数据上保持扩散训练 loss，让 policy 不忘记"专家是怎么做的"

**方案 B (MSE KL Penalty)**:

$$\mathcal{L}_{KL} = \mathbb{E}_{x_k, k, s} \left[ \| \epsilon_\theta(x_k, k, s) - \epsilon_{ref}(x_k, k, s) \|^2 \right]$$

> 对于相同输入，比较当前网络和冻结预训练网络的输出差异。差异越大，惩罚越重。
>
> 与真正的 KL 散度 $D_{KL}(\pi_\theta \| \pi_{ref})$ 的关系:
> 如果去噪网络的输出分布是高斯的（均值为 $\epsilon_\theta$, 方差固定），
> 则 MSE $\propto$ KL 散度。这是一种计算上更简单的近似。

**退火策略的直觉**:
```
训练阶段:     初期 ────────────────→ 后期
BC 权重 λ/β:   大 (2.0/1.0)          小 (0.1/0.01)
训练行为:     紧贴 BC 预训练         释放 RL 优化
类比:         学车时握着方向盘       慢慢放手让学员自己开
```

### F10. 折扣累积回报（Return）

在 RL 中，我们真正要最大化的目标:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$$

- $\gamma = 0.99$: 很关注长期
- $\gamma^{100} = 0.99^{100} \approx 0.366$: 100 步后的 reward 仍有 36.6% 的权重
- $\gamma^{500} = 0.99^{500} \approx 0.0066$: 500 步后几乎忽略

> 实际 pick-place episode 约 100-300 步，所以 $\gamma=0.99$ 意味着整个 episode 的 reward 都很重要。

---

## 0. 背景分析

### 整体流程总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Rev2Fwd 整体流程 (面向真机部署)                      │
│                                                                         │
│  Step 1: 数据收集 + 时间反转                                             │
│  ┌──────────┐    时间反转    ┌──────────┐                                │
│  │ Task B   │ ─────────── → │ Task A   │                                │
│  │ 反向数据  │  (B的轨迹倒放)  │ 正向数据  │                                │
│  │ goal→桌面 │               │ 桌面→goal │                                │
│  └──────────┘               └──────────┘                                │
│       ↓                          ↓                                      │
│  Step 2: BC 预训练 (两个 policy)                                         │
│  ┌──────────────────────────┐  ┌──────────────────────────┐             │
│  │ Task A Policy (正向)      │  │ Task B Policy (反向/复位) │             │
│  │ 输入: 图像 + ee_pose(7)   │  │ 输入: 同上              │             │
│  │      + gripper(1) = 8D   │  │ 功能: 把物体放回桌面     │             │
│  │ ⚠ 不含 obj_pose          │  │ = 环境 reset 策略        │             │
│  └──────────────────────────┘  └──────────────────────────┘             │
│       ↓                              ↓                                  │
│  Step 3: RL 在线优化 (A↔B 闭环 rollout)                                  │
│  ┌────────────────────────────────────────────────────────┐             │
│  │    Task A (正向)          Task B (复位)                 │             │
│  │  桌面 ──→ goal           goal ──→ 桌面                 │             │
│  │  ┌──────────┐            ┌──────────┐                  │             │
│  │  │ π_A 执行  │ ──done──→ │ π_B 执行  │ ──done──→ 回到起点│             │
│  │  │ RL 优化中 │            │ 充当reset │                  │             │
│  │  └──────────┘            └──────────┘                  │             │
│  │  ← ─ ─ ─ ─ ─ ─ 循环 ─ ─ ─ ─ ─ ─ ─ →                 │             │
│  │  (无需 sim teleport, 无需 obj_pose)                     │             │
│  └────────────────────────────────────────────────────────┘             │
│       ↓                                                                 │
│  Step 4: 评估 (50 episodes, 统一标准)                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 当前 Pipeline (DAgger) 的工作方式

> **DAgger 通俗解释**: 就像学开车——先看教练示范学一遍(BC)，然后自己上路(rollout)，
> 把开得好的路段记录下来(收集成功数据)，跟教练的示范合在一起再学一遍(finetune)。
> 反复迭代让自己越开越好。

```
BC pretrain (Task A/B)
    ↓                            # 第一步: 用示范数据做行为克隆预训练
for iter in 1..K:                # 然后反复迭代 K 轮:
    cyclic evaluation            #   1. 让当前 policy 在仿真中跑一跑
      → 收集成功 rollout          #   2. 只保留 "成功" 的回合
    aggregate rollout data       #   3. 把成功数据跟原始训练数据合并
      into LeRobot dataset       #      (1:1 权重采样, 新旧各占 50%)
    finetune policy              #   4. 用合并数据继续训练 policy
      with BC loss (5000 steps)  #      (还是用 BC loss, 即监督学习)
```

**局限性**：
- 只从**成功** rollout 中学习——失败的 episode（比如物体掉了、没对准）直接丢弃，浪费了大量信息
- BC loss 本质是监督学习（模仿），无法利用"做错了会有什么后果"这种信息
- 不能对 reward（奖励函数）做梯度优化——BC 只是在学"模仿谁"，不是在"最大化任务成功率"
- 原版 DAgger 需要 oracle expert（全知全能的专家）来纠正错误，但我们没有这种专家；用的是 policy 自身的 rollout，质量有限

### 为什么 RL 适合这个场景
1. **明确的 reward**: pick-place 任务有清晰的成功条件，即使不用 obj_pose 也能用视觉检测 + ee_pose 推断
2. **仿真 → 真机 pipeline**: 先在仿真中验证方案（可高通量并行），确认有效后直接迁移到真机（obs/action 空间完全一致）
3. **BC 预训练提供了好的初始化**: Diffusion Policy 已经通过 BC 学到了"大致正确"的行为，RL 只需微调优化，不需要从零学起
4. **失败 episode 也包含信息**: RL（特别是 Q-learning）天然能从失败中学习——"这样做会失败"也是有价值的信号
5. **A↔B 闭环天然可用**: Task B policy 提供自主 reset 能力，支持长时间无人值守训练

---

## 1. 环境接口设计

> 这一节定义了 RL 环境对外暴露的接口：智能体"看到什么"（Observation）和"能做什么"（Action）。

### 1.1 Observation Space（观测空间 — 智能体看到的信息）
与现有 pipeline 保持一致，**不包含特权信息**（obj_pose 等仿真器独有变量）：
```python
obs = {
    "image":        (3, 128, 128),  # 桌面俯视相机图像 (3 通道 RGB, 128×128 像素)
    "wrist_image":  (3, 128, 128),  # 手腕相机图像 (安装在机器人末端, 提供近距离视角)
    "state":        (8,),           # 数值状态向量：
                                     #   ee_pose(7): 末端执行器位姿 [x,y,z, qw,qx,qy,qz]
                                     #   gripper(1): 夹爪开合状态 (+1=开, -1=闭)
                                     # ⚠ 不含 obj_pose — 物体信息完全由相机图像提供
}
```
**为什么需要两个相机？** 桌面相机提供全局场景理解（物体在哪、目标在哪），手腕相机提供局部精细信息（对准精度、抓取细节）。

> **🔑 设计原则**: 观测空间只包含真机上也能获取的信息。`ee_pose` 可由关节编码器 + FK 计算，
> `gripper` 可由夹爪状态读取，图像来自真实相机。`obj_pose`（物体的 ground truth 位姿）
> 只有仿真器才能直接给出，真机需要额外的感知系统（如标定好的视觉检测 pipeline），
> 因此**不纳入 state**。物体位置信息完全由 Diffusion Policy 的视觉编码器从图像中隐式提取。

### 1.2 Action Space（动作空间 — 智能体的输出）
```python
action = (8,)  # [goal_ee_x, goal_ee_y, goal_ee_z, goal_qw, goal_qx, goal_qy, goal_qz, gripper]
               #  ├─── 目标 3D 位置 ───┤  ├──── 目标姿态(四元数) ────┤  ├夹爪┤
```
- **绝对 IK 目标**: policy 输出的是"机器人末端要去哪个目标位姿"，仿真环境用逆运动学 (IK) 计算对应的关节角度
- **二值夹爪**: 第 8 维控制夹爪的开/闭
- **Action chunking (动作分块)**: policy 一次预测未来 16 步的动作序列，但只执行前 8 步，然后重新观测、重新预测。好处是动作更平滑、连贯，减少高频抖动。

### 1.3 Reward Design（奖励函数设计 — 无特权信息版）

> **核心挑战**: 原版 reward 的所有分量都依赖 `obj_pose`（物体位姿 ground truth），
> 在真机上无法获取。必须设计**仅依赖可观测量**的 reward。
>
> 可观测量：`ee_pose`（关节计算得到）、`gripper`（夹爪状态）、`image`/`wrist_image`（相机图像）。
>
> 以下给出三种可选方案，按实现复杂度递增：

#### 方案 R1: 纯稀疏 Reward（最简单，推荐先尝试）
```python
def sparse_reward_no_privileged(ee_pose, gripper, image, wrist_image, 
                                 goal_xy, success_detector):
    """
    仅在任务成功时给奖励。
    success_detector: 基于视觉的成功检测器（见下方说明）。
    """
    # 成功判定: 用视觉检测器判断物体是否在 goal 位置
    success = success_detector(image, goal_xy)
    return 10.0 if success else 0.0
```
> **Success Detector 选项**:
> - **颜色/模板匹配**: 在桌面相机图像的 goal 区域检测目标物体颜色（简单但鲁棒性差）
> - **预训练视觉分类器**: 小 CNN 二分类 "goal 区域有/无物体"（需要少量标注数据）
> - **人工标注信号**: 真机实验时人工按键标记成功（最可靠，但需要人在旁边）

#### 方案 R2: 阶段性稀疏 Reward（推荐，平衡信号密度和可行性）
```python
def staged_reward_no_privileged(ee_pose, gripper, prev_ee_pose):
    """
    根据 ee_pose + gripper 状态变化推断任务阶段, 给阶段性奖励。
    不依赖 obj_pose, 完全可在真机上部署。
    """
    r = 0.0
    
    # --- ① 抓取阶段: 夹爪从开→闭 (说明正在抓取) ---
    # 条件: 夹爪刚关闭, 且末端执行器在桌面附近 (z 较低)
    just_grasped = (gripper < -0.5) and (prev_gripper > 0.5)
    if just_grasped and ee_pose[2] < 0.15:
        r += 1.0    # 抓取信号 (虽然不确定是否真的抓到了)
    
    # --- ② 抬升阶段: 夹爪闭合 + z 升高 → 可能正在搬运 ---
    if gripper < -0.5 and ee_pose[2] > 0.1:
        r += 0.5    # 搬运信号
    
    # --- ③ 放置阶段: 夹爪打开 + 末端在 goal 附近 ---
    ee_xy = ee_pose[:2]
    dist_to_goal = np.linalg.norm(ee_xy - goal_xy)
    if gripper > 0.5 and dist_to_goal < 0.05:
        r += 2.0    # 在 goal 附近释放 (很可能成功)
    
    # --- ④ 成功奖励: 视觉检测确认 ---
    if success_detector(image, goal_xy):
        r += 10.0
    
    return r
```

#### 方案 R3: 学习的 Reward（最强但最复杂）
```python
# 训练一个 reward 预测网络:
# 输入: (image, wrist_image, ee_pose, gripper)
# 输出: 标量 reward 预测
# 训练数据: 仿真中跑若干 episode, 用仿真 ground truth reward 当标签
#
# 优点: 可以学到接近 dense reward 的效果
# 缺点: 需要额外训练, 且 sim-to-real 存在域差距
class LearnedRewardModel(nn.Module):
    def forward(self, image, wrist_image, ee_pose, gripper):
        # 共享 ResNet 视觉编码器
        vis_feat = self.encoder(image, wrist_image)
        state = torch.cat([vis_feat, ee_pose, gripper], dim=-1)
        return self.mlp(state)  # 预测标量 reward
```

> **推荐路线**: 先用 **R2 (阶段性稀疏)** 跑通 pipeline，验证 RL 框架可行。
> 如果 reward 信号太稀疏导致学习困难，再尝试 R3 (学习的 reward)。

### 1.4 RL Gym Wrapper（真机兼容版）
```python
class PickPlaceRLEnv(gym.Env):
    """
    Wraps env into standard Gym API.
    ⚠ 真机兼容: reset() 不再 teleport, 而是执行 Task B policy 来复位。
    
    obs_space: Dict[image(3,128,128), wrist_image(3,128,128), state(8,)]
    action_space: Box(8,) or chunked Box(n_action_steps, 8)
    
    reset() 流程:
      1. 加载预训练好的 Task B policy (π_B)
      2. 执行 π_B 一个 episode: goal→桌面 (把物体放回桌面)
      3. 仅机器人归位 (move ee to home pose, 不涉及 obj teleport)
      4. 返回新的 obs
    
    step() 流程:
      1. 执行动作 → 获取 obs
      2. 计算 reward (§1.3 中的无特权信息版)
      3. done 判定: timeout 或 success_detector 触发
    """
```

> **关键设计: Task B 作为 Reset Policy**
> - 每个 Task A 的 episode 结束后，不调用 `teleport_object_to_pose()`，
>   而是让 Task B policy 执行一个完整的反向操作（从 goal 把物体放回桌面）
> - 这创造了 A→B→A→B... 的循环执行模式
> - Task B policy 来自 BC 预训练（使用时间反转后的 Task A 数据，或直接用反向收集的数据）
> - 如果 Task B reset 失败（物体没放回桌面），可以：
>   (a) 仿真中：作为异常回退到 teleport reset（仅开发调试用）
>   (b) 真机上：重试 Task B / 人工介入

---

## 2. 方案 A: TD-Learning — Critic-Guided Diffusion Policy

### 2.1 核心思想

> **一句话总结**: 保留 BC 训好的 Diffusion Policy 不变，额外加一个 "评分网络" (Q-network)，
> 让评分网络学会判断 "这个动作在当前状态下能得多少分"，然后用分数的梯度引导 Diffusion Policy 生成更好的动作。

基于 **DQL (Diffusion Q-Learning)** 和 **IDQL (Implicit Diffusion Q-Learning)** 的思路：

1. 保留 Diffusion Policy 的 **actor** 结构不变（DDIM 去噪生成 action chunk）
2. **额外引入一个 Q-network（critic / 评分网络）**: 这个网络输入 (观测, 动作)，输出一个标量 Q 值，代表"这个动作有多好"。通过 TD learning（时序差分学习）训练。
3. RL finetune 阶段交替做两件事：
   - **Critic 更新**：用环境中收集的 (s, a, r, s') 数据，通过 Bellman 公式 `Q(s,a) ≈ r + γ·Q(s',a')` 训练 Q-network
   - **Actor 更新**：把 Diffusion Policy 生成的动作喂给 Q-network 打分，通过**反向传播 Q 的梯度**来微调 Diffusion Policy，使其生成的动作 Q 值更高（同时保留 BC 正则化防止遗忘）
   
> **与直接用 SAC/TD3 的区别**: 传统方法会用高斯分布 (Gaussian) 作为 actor 的输出分布（均值+方差），表达能力有限。
> 这里保留 Diffusion Policy 作为 actor，它通过多步去噪生成动作，可以表达复杂的多模态分布（比如"可以从左边绕过去，也可以从右边"），
> 仅用 Q-network 做"打分+梯度引导"。

### 2.2 技术细节

**Q-Network 架构**:
```python
class QNetwork(nn.Module):
    """
    Twin Q-networks for TD3-style clipped double-Q.
    "Twin" = 两个独立的 Q 网络, 取较小值防止 Q 值过度乐观
    
    数据流:
    图像 ──→ ResNet18 ──→ 512D 视觉嵌入 ──┐
    state (8D: ee_pose+gripper) ───────────┤
                                           ├──→ MLP ──→ Q1 (标量)
    动作 (128D = 16步×8维) ────────────────┘        └──→ Q2 (标量)
    ⚠ 不含 obj_pose: 物体信息完全由图像嵌入隐式提供
    """
    def __init__(self, obs_encoder, action_dim, hidden_dims=[512, 512]):
        # 复用 Diffusion Policy 的 ResNet18 vision encoder（可以冻结参数节省计算，或一起微调）
        # MLP head: concat(obs_embed, action_flat) → Q1, Q2
        # 两个 Q 值取 min, 防止 overestimation (这是 TD3 的关键技巧)
```

**Actor 更新（Critic-Guided Diffusion）— 让 Q 网络的梯度"指挥"扩散模型**:
```python
# 第一步: 用 Diffusion Policy 生成动作 (完整的 DDIM 去噪链)
noisy_action = torch.randn(batch, horizon, action_dim)  # 从纯噪声开始
for t in reversed(ddim_timesteps):                       # 逐步去噪 (10步)
    noisy_action = ddim_step(policy, noisy_action, t, obs)  # 每步移除一点噪声
# 此时 noisy_action 已经是干净的动作 x_0

# 第二步: 用 Q 网络给生成的动作「打分」, 并反向传播梯度
# 关键: 整个去噪链条是可微分的, 所以 Q 的梯度可以一路传回 policy 的参数
q_value = Q_phi(obs, noisy_action)      # Q 网络: "这组动作能得多少分?"
actor_loss = -q_value.mean()            # 最大化 Q 值 = 最小化 -Q
           + α * bc_loss                # + BC 正则化 (别忘了预训练学到的东西)
# α 控制 BC 正则化强度: α 大 → 更保守(接近 BC); α 小 → 更激进(跟着 RL 走)
```

**BC Regularization（BC 正则化 — 防止灾难性遗忘）**:
```python
# 问题: 纯 RL 优化可能让 policy 跑偏, 忘记 BC 预训练学到的合理行为
# 解决: 在 RL loss 里加一项 BC loss, 让 policy 不能偏离太远

# 方案 1: 在示范数据上计算 BC loss (和预训练一样的监督学习)
bc_loss = MSE(policy_output, bc_target_from_demo)

# 方案 2: 保留一部分训练 batch 做纯 BC 更新 (不加 RL loss)
# 比如每 5 个 batch 有 1 个只做 BC, 剩下 4 个做 RL+BC
```

### 2.3 算法伪代码（带详细注释 — 真机兼容版）
```
初始化:
  π_θ ← 加载 BC 预训练的 Diffusion Policy     # 这就是我们的 actor (动作生成器)
  π_B ← 加载 BC 预训练的 Task B Policy         # reset policy (反向放置, 用于复位)
  Q_φ1, Q_φ2 ← 随机初始化两个 Q 网络          # critic (评分网络), 共享 obs encoder
  Q̄_φ1, Q̄_φ2 ← Q 的目标网络 (参数值=Q的副本)  # 用于稳定 TD target 计算
  D_demo ← 原始示范数据 (BC 训练用的 LeRobot 数据集)
  D_online ← 空的在线经验缓冲区               # 存储新收集的 (s,a,r,s') 数据

循环每个迭代:
  # ────── 阶段 1: 数据收集 (A↔B 循环 rollout) ──────
  for env_step in 1..N_collect:
    a = π_θ(obs) + ε       # 用 Diffusion Policy 生成动作 + 加一点噪声鼓励探索
    obs', r, done = env.step(a)   # 执行动作, reward 不依赖 obj_pose (§1.3)
    D_online.add(obs, a, r, obs', done)  # 存入 replay buffer
    if done:
      obs = env.reset()     # → 内部执行 π_B 复位, 不用 teleport

  # ────── 阶段 2: 训练 Critic (Q 网络学会「打分」) ──────
  # 目标: 让 Q(s,a) 逼近真实的累积回报
  for critic_step in 1..N_critic:               # 50 步
    batch ~ D_online ∪ D_demo                   # 从在线+示范数据中采样
    a' = π_θ(obs') + clip(noise)                # 用 policy 生成下一步动作
    target_Q = r + γ * min(Q̄_φ1(obs',a'), Q̄_φ2(obs',a'))  # Bellman target
    # ↑ 用目标网络计算, 取两个 Q 的较小值 (防过估计)
    L_critic = MSE(Q_φ1(obs,a), target_Q) + MSE(Q_φ2(obs,a), target_Q)
    update φ1, φ2   # 让 Q 的预测越来越接近 target

  # ────── 阶段 3: 训练 Actor (用 Q 的梯度引导 Diffusion Policy) ──────
  # 目标: 让 policy 生成 Q 值更高的动作
  for actor_step in 1..N_actor:                 # 20 步 (比 critic 更新少, 防止不稳定)
    batch_online ~ D_online
    batch_demo ~ D_demo
    
    a_gen = run_ddim_chain(π_θ, obs_online)     # 可微分地生成动作
    L_rl = -Q_φ1(obs_online, a_gen).mean()      # 最大化 Q 值
    L_bc = diffusion_loss(π_θ, obs_demo, a_demo) # BC loss: 别忘了示范数据
    L_actor = L_rl + λ * L_bc                   # λ 从 2.0 退火到 0.1
    update θ

  # ────── 阶段 4: 目标网络缓慢更新 ──────
  Q̄ ← τ * Q + (1-τ) * Q̄     # τ=0.005, 目标网络只追踪主网络的 0.5%
```

### 2.4 关键超参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `γ` (discount) | 0.99 | 折扣因子: 0.99 表示很关注长期回报 (100 步后的 reward 仍有 0.99^100≈36.6% 的权重) |
| `τ` (target EMA) | 0.005 | 目标网络更新速率: 每次只更新 0.5%, 让 target 非常平滑稳定 |
| `λ` (BC weight) | 初始 2.0 → 退火到 0.1 | BC 正则化权重: 初期强约束(≈纯 BC), 后期逐渐释放让 RL 主导 |
| `lr_critic` | 3e-4 | Critic 学习率: 标准 Adam 默认值, critic 需要快速学习 |
| `lr_actor` | 1e-5 | Actor 学习率: 远小于 critic, 防止破坏 BC 预训练的行为先验 |
| `batch_size` | 256 | 每个 mini-batch 的大小 |
| `replay_buffer_size` | 1M transitions | replay buffer 最多存 100 万条 (s,a,r,s') 数据 |
| `num_envs` | 64 | 同时并行运行的仿真环境数量 |
| `exploration_noise` | 0.1 (std) | 动作上叠加的高斯噪声标准差, 用于鼓励探索 |
| `n_critic_updates_per_collect` | 50 | 每次数据收集后, 更新 critic 50 次 |
| `n_actor_updates_per_collect` | 20 | 每次数据收集后, 更新 actor 20 次 (比 critic 少, 这是 TD3 的"延迟" 策略) |
| `demo_ratio` | 0.25 | 每个 batch 中 25% 来自示范数据, 75% 来自在线收集的数据 |

### 2.5 文件结构
```
scripts/scripts_rl/
├── README.md                      # 本文件
├── rl_env_wrapper.py             # Gym RL wrapper (obs=8D, reward无obj_pose, reset=π_B)
├── reward.py                      # 无特权信息 reward functions (R1/R2/R3)
├── q_network.py                   # Twin Q-network architecture
├── replay_buffer.py               # Replay buffer + demo buffer
├── train_td_diffusion.py          # 方案 A 主训练脚本
├── train_dppo.py                  # 方案 B 主训练脚本
├── eval_rl.py                     # 公平评估脚本 (同 7_eval_fair.py 接口)
├── success_detector.py            # 视觉成功检测器 (替代 obj_pose 判定)
├── utils.py                       # 共享工具函数
└── run_rl_pipeline.sh             # 自动化 pipeline shell 脚本
```

---

## 3. 方案 B: DPPO (Diffusion Policy Policy Optimization)

### 3.1 核心思想

> **一句话总结**: 把 Diffusion Policy 内部的 10 步去噪过程看作一个「游戏」，每一步去噪就是一次「决策」，
> 用 PPO 算法优化每一步的去噪策略，让最终生成的动作在环境中获得更高的 reward。

**DPPO** (Allen Ren, Justin Lidard et al., 2024) 的创新在于：把 DDIM 去噪过程本身建模为一条小型 MDP：

```
传统 RL 视角:                    DPPO 视角 (去噪也是决策!):

环境状态 s → policy → 动作 a     去噪步骤本身就是一套 MDP:
                                  状态 = (噪声动作 x_t, 环境观测 obs)
                                  动作 = 去噪网络的预测 ε_θ
                                  转移 = DDIM 更新: x_{t-1} = f(x_t, ε_θ)
                                  奖励 = 只在最后一步(完全去噪后)给环境 reward
```

- **State**: 当前去噪步的中间 noisy action $x_t$ + conditioning observation $s$
- **Action**: denoiser 的输出（即 noise prediction $\epsilon_\theta$）
- **Transition**: DDIM update rule $x_{t-1} = f(x_t, \epsilon_\theta)$
- **Reward**: 只在最后一步（$t=0$, 完全去噪后）给 environment reward

这样，整个去噪链 $x_T \to x_{T-1} \to \cdots \to x_0$ 就是一条 "episode"（10 步的小游戏），可以直接用 PPO 优化。

### 3.2 为什么 DPPO 比 TD-Learning 更适合 Diffusion Policy

1. **不需要额外的 Q-network**: 直接用 PPO 的 advantage estimation (GAE)，架构更简单
2. **On-policy (当前数据训当前策略)**: 每次都用最新 policy 生成数据，避免 off-policy 方法中旧数据带来的分布偏移问题
3. **自然处理 action chunking**: 整个 action chunk 的生成过程（10 步去噪）就是一条完整的 MDP trajectory，无需特殊处理
4. **实验验证更强**: DPPO 在 robotic manipulation benchmarks 上一致优于 IDQL、DQL 等 TD 方法
5. **训练更稳定**: PPO 的 clipping 机制天然防止 policy 崩溃——每次更新幅度受限，适合从 BC pretrain 出发的微调场景

### 3.3 技术细节

**去噪 MDP 定义 — 把 10 步去噪看作一个 10 步的小游戏**:
```python
# 一个「去噪 episode」的完整过程 (K=10 步):
#
# 第 10 步 (最初): x_10 = 纯噪声 ~ N(0,I)
# 第  9 步:        x_9  = DDIM_step(x_10, ε_θ(x_10, 10, obs), 10)  → 稍微干净一点
# 第  8 步:        x_8  = DDIM_step(x_9,  ε_θ(x_9,  9,  obs), 9)   → 更干净
#   ...             ...    (以此类推，每一步都去掉一部分噪声)
# 第  1 步:        x_0  = DDIM_step(x_1,  ε_θ(x_1,  1,  obs), 1)   → 干净动作!
#
# 在「去噪 MDP」中:
#   state_k = (x_k, obs)           # 当前带噪动作 + 环境观测
#   action_k = ε_θ(x_k, k, obs)   # 去噪网络的输出 ("我觉得噪声是这些")
#   transition = x_{k-1} = DDIM更新公式(x_k, ε_θ, k)  # 减去噪声, 得到更干净的动作
# 
# Reward 分配 (关键设计!):
#   r_k = 0          (k > 0 的中间步骤不给奖励)
#   r_0 = R_env(x_0) (只在最终干净动作执行后, 拿环境给的 reward)
#   → 这个 reward 需要通过 GAE 回传到前面所有去噪步
```

**Value Network (PPO 的 Critic — 估计「从这一步去噪开始，还能拿多少分」)**:
```python
class DenoisingValueNetwork(nn.Module):
    """
    在去噪 MDP 中充当 critic:
    输入: (当前噪声动作 x_k, 环境观测 obs, 去噪步数 k)
    输出: 标量 V(x_k, obs, k) = "从去噪第 k 步开始, 预计能获得的总 reward"
    
    架构和 denoiser 类似, 但最后一层输出标量而不是动作向量。
    """
    def __init__(self, obs_encoder, denoising_step_embed_dim, hidden_dim):
        # obs_encoder: 处理图像的 ResNet (可以和 policy 共享, 也可以独立)
        # 输入: concat(obs视觉嵌入, 噪声动作x_k展平, 去噪步数的正弦嵌入)
        # 输出: 标量 V 值
```

**PPO 更新（在去噪 MDP 上做 PPO）**:
```python
# ────── 数据收集: 在环境中跑 + 记录去噪过程 ──────
for env_step in range(horizon):   # horizon=1500 步
    # 第一步: 通过 DDIM 去噪链生成一组动作（16 步）
    x_K ~ N(0, I)                 # 从纯噪声开始
    for k in reversed(range(K)):  # K=10 步去噪
        ε_k = policy(x_k, k, obs) # 去噪网络预测噪声
        # ↑ 同时记录 log_prob: log π(ε_k | x_k, k, obs)
        #   PPO 需要 "新旧策略的概率比" 来计算 clipped loss
        x_{k-1} = ddim_step(x_k, ε_k, k)  # 移除噪声
    
    action_chunk = x_0            # 去噪完成 → 16 步干净动作
    
    # 第二步: 在环境中执行前 8 步动作
    for a in action_chunk[:n_execute]:  # n_execute=8
        obs', r, done = env.step(a)     # 环境返回 (新观测, 奖励, 是否结束)
    
    # 第三步: 把环境 reward 分配给去噪链
    # 关键: 只有最后一步去噪 (k=0) 获得 reward, 前面 9 步 reward=0
    # GAE 会自动把 reward 的信用回传给前面的去噪步骤

# ────── PPO 更新: 用收集的数据优化去噪网络 ──────
# 1. 计算 advantage: "这个去噪决策比平均水平好了多少"
advantages = compute_gae(denoising_trajectories, value_network)

# 2. Clipped PPO loss (核心!)
ratio = exp(log_prob_new - log_prob_old)   # 新旧策略的概率比
# ↑ ratio ≈ 1 说明策略没怎么变; ratio >> 1 或 << 1 说明变化大
L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
# ↑ 如果 ratio 超出 [0.8, 1.2], 就「截断」→ 阻止过大更新

L_value = MSE(V(x_k, obs, k), returns)   # Value 网络也要学准
L_entropy = -log_prob.mean()              # 鼓励探索 (增加随机性)

L_total = -L_clip + c1 * L_value - c2 * L_entropy
```

**BC Regularization（KL penalty — 防止偏离 BC 预训练）**:
```python
# 保留 BC 先验: 不让去噪网络的输出偏离预训练版本太远
# 实际操作: 对于相同的输入 (x_k, k, obs), 比较新旧去噪网络的输出差异
ε_new = π_θ_current(x_k, k, obs)       # 当前正在训练的网络输出
ε_ref = π_θ_pretrained(x_k, k, obs)    # 冻结的预训练网络输出
L_kl = MSE(ε_new, ε_ref)               # 差异越大, 惩罚越大

L_total += β * L_kl   # β 从 1.0 退火到 0.01
# ↑ 训练初期: β 大 → 紧贴 BC 先验, 防止崩溃
#   训练后期: β 小 → 释放 RL 控制, 允许更多优化
```

### 3.4 算法伪代码（带详细注释 — 真机兼容版）
```
初始化:
  π_θ ← 加载 BC 预训练的 Diffusion Policy (denoiser 网络)
  π_ref ← π_θ 的冻结副本 (永远不更新, 作为 KL 正则化的参考基准)
  π_B ← 加载 BC 预训练的 Task B Policy (reset policy, 反向放置)
  V_ψ ← Value 网络 (结构类似 denoiser, 但输出标量而非动作)
  K = 10 (DDIM 去噪步数)

循环每个 PPO 迭代:
  # ═══════ 第 1 阶段: A↔B 循环 rollout + 记录去噪过程 ═══════
  buffer = []
  for env_episode in 1..M (64 个环境并行运行):
    obs = env.reset()                     # → 内部执行 π_B 复位 (不用 teleport)
    for t in 0..horizon:                  # horizon=1500 个环境步
      # ── 生成一个 action chunk (记录去噪链的每一步) ──
      x_K ~ N(0, I)                       # 采样纯噪声
      chain = []                           # 记录去噪链
      for k in reversed(range(K)):         # K=10 步去噪
        ε_k = π_θ(x_k, k, obs)           # denoiser 预测噪声
        log_prob_k = gaussian_log_prob(ε_k, predicted_mean, std=1.0)
        # ↑ 记录当前策略下这个输出的概率 (PPO 后续要比较新旧概率)
        x_{k-1} = ddim_step(x_k, ε_k, k) # 执行去噪一步
        chain.append((k, x_k, ε_k, log_prob_k))
      
      action_chunk = x_0                  # (16步, 8维) 的动作序列
      
      # ── 在环境中执行动作 ──
      chunk_reward = 0
      for i, a in enumerate(action_chunk[:n_execute]):  # 执行前 8 步
        obs', r, done, info = env.step(a)
        chunk_reward += r * γ^i           # 折扣累积奖励
        if done: break
        obs = obs'
      
      # ── 把环境 reward 分配给去噪链 ──
      for (k, x_k, ε_k, lp_k) in chain:
        r_k = chunk_reward if k == 0 else 0.0   # 只有最后一步去噪分到 reward
        buffer.append((obs, k, x_k, ε_k, lp_k, r_k, done))

  # ═══════ 第 2 阶段: PPO 更新 (在去噪 MDP 上) ═══════
  # 先计算 GAE advantage (衡量每步去噪比平均水平好多少)
  compute_returns_and_advantages(buffer, V_ψ)

  for ppo_epoch in 1..E:                  # E=5 轮
    for mini_batch in buffer.shuffle().chunks(batch_size):  # batch=256
      # ── Value 网络更新 ──
      L_v = MSE(V_ψ(obs, x_k, k), returns)    # 让 V 的预测接近真实回报
      
      # ── Policy 更新 (Clipped PPO) ──
      ε_new = π_θ(x_k, k, obs)                # 用当前网络重新计算输出
      lp_new = gaussian_log_prob(ε_new, ...)
      ratio = exp(lp_new - lp_old)             # 新旧概率比
      L_clip = -min(ratio * adv, clip(ratio, 1-ε_ppo, 1+ε_ppo) * adv)
      # ↑ PPO 的核心: 限制更新幅度, 防止策略崩溃
      
      # ── KL 正则化 (防止偏离 BC 预训练) ──
      ε_ref = π_ref(x_k, k, obs)              # 冻结的预训练网络输出
      L_kl = MSE(ε_new, ε_ref)                # 当前输出偏离预训练多少?
      
      L = L_clip + c1 * L_v + β * L_kl
      update θ, ψ                              # 同时更新 policy 和 value 网络
```

### 3.5 关键超参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `K` (denoising steps) | 10 | DDIM 去噪步数, 即去噪 MDP 的 episode 长度 |
| `ε_ppo` (clip ratio) | 0.2 | PPO 裁剪范围: 策略更新比率限制在 [0.8, 1.2] 内 |
| `γ` (env discount) | 0.99 | 环境 reward 的折扣因子 |
| `λ_gae` | 0.95 | GAE 的 λ 参数: 在 bias-variance 之间权衡。0.95 偏向低 variance |
| `β` (KL weight) | 初始 1.0 → 退火到 0.01 | KL 正则化权重: 初期紧贴 BC, 后期逐渐放开 |
| `c1` (value coeff) | 0.5 | Value loss 在总 loss 中的权重 |
| `lr_policy` | 1e-5 | Policy 学习率: 很低, 保护 BC 预训练权重 |
| `lr_value` | 3e-4 | Value 网络学习率: 相对较高, 需要快速收敛 |
| `batch_size` | 256 | PPO mini-batch 大小 |
| `ppo_epochs` | 5 | 每次 rollout 后, 用同一批数据更新 5 轮 |
| `num_envs` | 64 | 并行 rollout 的环境数量 |
| `rollout_horizon` | 1500 | 每个环境 episode 最长 1500 步 |
| `n_action_steps_execute` | 8 | 每个 action chunk 实际执行 8 步 |
| `n_action_steps_predict` | 16 | 每个 action chunk 预测 16 步 (只执行前 8 步) |
| `entropy_coeff` | 0.01 | 熵正则化系数: 鼓励去噪空间中的探索 |
| `max_grad_norm` | 0.5 | 梯度裁剪: 梯度范数超过 0.5 就缩放, 防止梯度爆炸 |

---

## 3.6 方案 A vs 方案 B 对比总结

| 对比维度 | 方案 A: TD-Learning | 方案 B: DPPO |
|---------|--------------------|----|
| **核心思路** | 加一个 Q 网络给动作打分，用 Q 的梯度引导 Diffusion Policy | 把去噪过程本身视为 MDP，用 PPO 直接优化每步去噪 |
| **额外网络** | 双 Q 网络 (Twin Q) + 目标网络 | Value 网络 (只需一个) |
| **数据使用** | Off-policy: 有 replay buffer，可复用历史数据 | On-policy: 数据用完即弃，每轮重新收集 |
| **探索方式** | 动作加高斯噪声 | 利用 diffusion 本身的随机性 |
| **优点** | 数据利用率高（replay buffer） | 训练更稳定、架构更简单、实验效果更好 |
| **缺点** | Q 值可能过估计、需要调更多超参数 | 需要更多环境交互（数据不能复用） |
| **推荐场景** | 环境交互成本高时 | 有并行仿真环境时（本项目适用） |

---

## 4. 共享组件

> 以下模块被方案 A 和方案 B 共同使用。

### 4.1 RL Environment Wrapper (`rl_env_wrapper.py`)

两种方案共用同一个 Gym wrapper，封装：
- **环境创建 + camera setup**: 启动仿真环境（或连接真机），配置桌面和手腕两个相机
- **Observation preprocessing**: 图像归一化、状态拼接（**state=8D**: ee_pose 7 + gripper 1，不含 obj_pose）
- **Reward computation**: 调用 §1.3 的**无特权信息** reward 函数（不依赖 obj_pose）
- **Reset via Task B policy**: `reset()` 执行 π_B（Task B policy）将物体从 goal 放回桌面，不使用 `teleport_object_to_pose()`
- **Vectorized batching**: 支持 `num_envs > 1`，一次 `step()` 同时推进 64 个并行环境
- **Action chunk → env step 展开**: 接收 16 步 action chunk，内部循环逐步执行并累积 reward

### 4.2 Reward Function (`reward.py`)

独立模块，方便实验不同 reward designs（所有方案**不依赖** obj_pose）：
- `sparse_reward_no_privileged()`: 纯视觉成功检测 (10.0)
- `staged_reward_no_privileged()`: 基于 ee_pose+gripper 的阶段性奖励（推荐）
- `learned_reward()`: 学习的 reward 模型 (可选)

### 4.3 Evaluation (`eval_rl.py`)

统一评估接口，与现有 `7_eval_fair.py` 可比较：
- 50 episodes 独立评估
- 输出 success rate + stats JSON
- 可选 video 保存

---

## 4.5 A↔B 循环复位机制 与 时间倒放数据

> 本节解释真机部署的核心策略：**用 Task B policy 代替 sim reset**，以及**时间反转数据**如何同时服务于
> 数据增强和 reset policy 训练。

### 4.5.1 为什么需要 A↔B 循环

传统 RL 流程中，每个 episode 结束后通过仿真器 `reset()` 一键复位（teleport 物体、归位机器人）。
但在真机上：
- **不能 teleport 物体**: 没有上帝之手把物体瞬间搬回去
- **不想每次人工摆放**: 效率太低，无法高通量采数据
- **解决方案**: 训练一个 **Task B policy (π_B)** 来执行反向操作——即 "undo" Task A

```
A↔B 循环执行流程 (一轮完整的环境交互):

┌─────────────────────────────────────────────────────────┐
│ Phase 1: Task A (RL 优化目标)                            │
│   π_A: 桌面 → 抓取 → 搬运 → 放到 goal                   │
│   → 收集 (obs, action, reward) 用于 RL 更新              │
│                                                          │
│ Phase 2: Task B (Reset, 不做 RL 优化)                    │
│   π_B: goal → 抓取 → 搬运 → 放回桌面                    │
│   → 作用: 把物体放回接近初始状态, 充当 "reset()"          │
│   → π_B 来自 BC 预训练, 在 RL 过程中冻结                  │
│                                                          │
│ Phase 3: 机器人归位 (ee → home pose)                     │
│   → 简单的关节移动, 不涉及物体操作                        │
│                                                          │
│ → 回到 Phase 1, 开始新的 Task A episode                  │
└─────────────────────────────────────────────────────────┘
```

### 4.5.2 时间倒放数据序列

> **核心洞察**: Task A (正向: 桌面→goal) 和 Task B (反向: goal→桌面) 的轨迹互为时间反转。
> 只需收集一个方向的数据, 就能自动获得另一个方向的训练数据。

**时间反转算法** (已实现于 `scripts/scripts_pick_place/3_make_forward_data.py`):

```python
# 给定 Task B 的一条轨迹 (goal→桌面):
#   frames_B = [f_0, f_1, ..., f_T]     # T+1 帧
#   其中 f_t = {image, wrist_image, ee_pose, gripper}

# 时间反转得到 Task A 的轨迹 (桌面→goal):
frames_A = frames_B[::-1]               # 帧序列倒序: [f_T, f_{T-1}, ..., f_0]

# 动作重新计算:
# 原始 action 定义: action_t = ee_pose_{t+1} (下一帧的 ee_pose 作为当前动作目标)
# 反转后: action_A[i] = frames_A[i+1].ee_pose = frames_B[T-i-1].ee_pose
actions_A = [frames_A[i+1].ee_pose for i in range(len(frames_A)-1)]
# 最后一帧没有下一帧 → 丢弃, 所以动作序列长度 = T

# gripper 动作: 同样取反转后的下一帧 gripper 状态
gripper_A = [frames_A[i+1].gripper for i in range(len(frames_A)-1)]
```

**这种对称性带来的优势**:
1. **数据效率**: 收集 N 条 Task B 轨迹 → 免费得到 N 条 Task A 训练数据（反之亦然）
2. **自然的 reset policy**: Task B policy (从 B 数据 BC 训练) 天然就是 Task A 的 reset 操作
3. **双向监督**: RL 过程中 Task A 的 rollout 数据, 时间反转后还能增强 Task B policy 的训练集

### 4.5.3 Reset 失败处理

Task B policy 不是完美的——可能抓不到物体、放歪了、或者失次等。处理策略：

| 场景 | 仿真中 | 真机上 |
|------|--------|--------|
| Task B 成功 (物体回到桌面区域) | 正常继续 | 正常继续 |
| Task B 部分成功 (物体在桌面附近但不精确) | 接受, 作为 Task A 的 harder init | 接受, 增加多样性 |
| Task B 完全失败 (物体没被抓起) | fallback: 用 teleport 复位 (仅调试用) | 重试 π_B / 人工介入 |
| Task B 超时 | 强制终止 + teleport 复位 | 强制终止 + 人工介入 |

> **注**: Task B reset 的不精确性反而可能是有利的——它让 Task A 的初始状态更加多样化,
> 增强了 policy 的泛化能力（类似 domain randomization）。

---

## 5. 实验设计

> 目标: 公平对比 DAgger (现有方法) 和两种 RL 方案，验证 RL 是否能提升 BC 预训练 policy 的性能。
> **约束**: 所有方案面向真机部署——不使用 obj_pose、不依赖 sim teleport reset。

### 5.1 对比实验矩阵

| 实验 | 方法 | 训练数据 | 备注 |
|------|------|---------|------|
| **Baseline 1** | BC only (纯行为克隆) | 100 demo episodes | 只做预训练, 不做任何后续优化 |
| **Baseline 2** | BC + DAgger (现有 pipeline) | demo + 成功 rollout | 10 次迭代, 每次 5000 步 BC 微调 |
| **Baseline 3** | BC + 纯续训 (无 rollout 数据) | 仅 demo | 只在原始数据上继续训练, 对照 DAgger 的数据聚合是否有效 |
| **Ours A** | BC + TD-Diffusion (方案 A) | demo + online replay | 总训练量对标 Baseline 2; state=8D, 无 obj_pose |
| **Ours B** | BC + DPPO (方案 B) | demo + on-policy rollout | 总训练量对标 Baseline 2; A↔B 循环 reset |

### 5.2 控制变量（确保实验公平性）

- **相同起点**: 所有方法从同一个 BC pretrained checkpoint 出发（相同的随机种子和网络权重）
- **相同观测空间**: state=8D (ee_pose 7 + gripper 1), 无 obj_pose
- **相同 reset 机制**: Task B policy 作为 reset（Baseline 1-3 也改为 Task B reset, 保证公平）
- **相同 reward**: 不依赖 obj_pose 的 reward（§1.3 方案 R1 或 R2）
- **相同评估标准**: 50 个独立 episodes, horizon=1500 步, 统一成功判定（视觉检测）
- **相同总计算量**: 匹配 Baseline 2 的总环境交互步数 (10 iter × 50 cycles)
- **相同硬件**: 2 GPU 训练, 1 GPU 评估

### 5.3 评估指标

1. **Task A 成功率** (主要指标): 物体从桌面随机位置成功放到 goal 位置的比例（视觉检测判定）
2. **Task B 复位成功率**: Task B policy 把物体从 goal 成功放到桌面的比例（衡量 reset 可靠性）
3. **A↔B 循环成功率**: 连续 A→B→A→B... 循环 N 轮成功完成的比率（衡量系统长期稳定性）
4. **样本效率曲线**: 成功率 vs. 环境交互步数（曲线越靠左上越好）
5. **训练稳定性**: reward 曲线是否平稳上升（不崩溃、不震荡）

### 5.4 消融实验（验证各设计选择的影响）

| 消融 | 变量 | 目的 |
|------|------|------|
| Reward 类型 | R1(纯稀疏) vs R2(阶段性) vs R3(学习) | 无特权 reward 的最佳设计 |
| BC 权重 λ/β | {0, 0.1, 1.0, 5.0} | 验证 BC 正则化的重要性 |
| 并行环境数 | {16, 64, 256} | 更多并行是否提高采样效率 |
| 去噪步数 K | {5, 10, 20} | DPPO 去噪 MDP 长度 trade-off |
| Reset 精度 | 精确 reset vs 粗略 reset | Task B reset 不精确是否反而增加泛化性 |

---

## 6. 实施计划

### Phase 1: 基础设施 (共享模块)
1. `rl_env_wrapper.py` — Gym wrapper (obs=8D 无 obj_pose, reset=π_B)
2. `reward.py` — 无特权信息 Reward functions (R1/R2/R3)
3. `success_detector.py` — 视觉成功检测器（替代 obj_pose 判定）
4. `utils.py` — 共享工具
5. `eval_rl.py` — 评估脚本

### Phase 2: 方案 A 实现
6. `q_network.py` — Twin Q-network (输入含视觉嵌入, 不含 obj_pose)
7. `replay_buffer.py` — Replay buffer + demo buffer
8. `train_td_diffusion.py` — TD-Learning 主训练 (A↔B 循环 rollout)

### Phase 3: 方案 B 实现
9. `train_dppo.py` — DPPO 主训练 (A↔B 循环 rollout)

### Phase 4: Pipeline 自动化
10. `run_rl_pipeline.sh` — 自动化实验 shell 脚本

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
