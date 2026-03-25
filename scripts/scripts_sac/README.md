# SAC (Soft Actor-Critic) 微调 Diffusion Policy：完整实现指南

> **目标**: 使用 SAC (Soft Actor-Critic) 算法对 BC 预训练的 Diffusion Policy 进行 RL 微调，提升 pick-place 任务的成功率。
>
> **核心思想**: SAC 是一种 off-policy 的最大熵 RL 算法，在标准 RL 目标（最大化累积回报）的基础上增加了 **策略熵正则化**，鼓励探索的同时保持训练稳定性。
>
> 提供两种 actor 模式：
> 1. **SAC-Gaussian**: 全新的 Squashed Gaussian Actor，从头训练
> 2. **SAC-Diffusion**: 直接微调已有的 Diffusion Policy

---

## 目录

0. [术语表 / Glossary](#术语表--glossary)
1. [关键公式详解](#关键公式详解)
2. [SAC 算法原理详解](#2-sac-算法原理详解)
3. [文件结构说明](#3-文件结构说明)
4. [网络架构详解](#4-网络架构详解)
5. [训练流程详解](#5-训练流程详解)
6. [超参数说明](#6-超参数说明)
7. [使用方法](#7-使用方法)
8. [核心公式推导 (SAC)](#8-核心公式推导-sac)
9. [调参建议](#9-调参建议)
10. [常见问题与排查](#10-常见问题与排查)

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
| **Off-policy** | 离线策略 | 可以用历史旧数据（不同版本 policy 收集的）来更新参数，通过 replay buffer 存储。优点：数据利用率高；缺点：分布偏移问题。SAC 属于 off-policy。 |
| **Replay Buffer** | 经验回放缓冲区 | Off-policy 方法使用的数据存储结构，存储历史 (s, a, r, s', done) 元组，训练时从中随机采样 mini-batch。 |

### RL 算法

| 缩写/术语 | 全称 | 含义 |
|-----------|------|------|
| **TD Learning** | Temporal Difference Learning（时序差分学习） | 用 `r + γ·V(s')` 来估计 `V(s)` 的方法（bootstrap）。不需要等 episode 结束就能更新，比 Monte Carlo 方差更小。 |
| **TD3** | Twin Delayed DDPG | 一种 off-policy actor-critic 算法。"Twin"指使用双 Q 网络取较小值（防过估计）、"Delayed"指 actor 更新频率低于 critic。 |
| **SAC** | Soft Actor-Critic | 最大化累积 reward + 策略的熵（随机性），鼓励探索。与 TD3 类似但使用随机策略，并通过自动温度调节 α 控制探索-利用平衡。 |
| **EMA** | Exponential Moving Average（指数移动平均） | Target Q-network 的更新方式：`Q̄ ← τ·Q + (1-τ)·Q̄`，让 target 慢慢跟上主网络，提高训练稳定性。τ 越小更新越慢。 |
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

### 网络架构相关

| 缩写/术语 | 全称 | 含义 |
|-----------|------|------|
| **Actor** | 演员网络 | Actor-Critic 架构中负责 "选择动作" 的网络。SAC-Gaussian 模式中 actor = Squashed Gaussian MLP；SAC-Diffusion 模式中 actor = Diffusion Policy（denoiser）。 |
| **Critic** | 评论家网络 | Actor-Critic 架构中负责 "评估动作好坏" 的网络。SAC 中 critic = Twin Q-Networks。 |
| **Twin Q-Networks** | 双 Q 网络 | 同时训练两个 Q 网络 (Q1, Q2)，取较小值作为 target。目的：防止 Q 值过度乐观（overestimation），这是 TD3 的核心创新，SAC 同样采用。 |
| **Target Network (Q̄)** | 目标网络 | Q-network 的缓慢更新副本，用于计算 TD target `r + γ·Q̄(s',a')`。如果用主网络本身计算 target，会导致自举不稳定。 |
| **ResNet-18** | Residual Network-18 层 | 经典 CNN 图像编码器，18 层深。用于将 128×128 RGB 图像压缩为 512 维向量。 |
| **MLP** | Multi-Layer Perceptron（多层感知机） | 最基本的全连接神经网络。本项目中 Q-network 的 head 是 MLP：输入 obs+action → 输出 Q 值。 |
| **U-Net** | — | 编码器-解码器架构（带跳跃连接），最初用于图像分割。在 Diffusion Policy 中作为 denoiser 的骨干网络，处理时序上的多步动作。 |

### 训练技巧

| 缩写/术语 | 含义 |
|-----------|------|
| **BC Regularization / BC 正则化** | RL 微调时保留一部分 BC loss，防止 policy "忘记" 预训练学到的行为先验。类似于 LLM fine-tune 时的 KL penalty。 |
| **学习率退火 (Annealing)** | 某个超参数（如 BC 权重 λ）随训练进度逐渐减小。初期重视 BC 先验（大 λ），后期释放 RL 控制（小 λ）。 |
| **梯度裁剪 (Gradient Clipping)** | 限制梯度的最大范数（如 0.5），防止梯度爆炸导致训练不稳定。 |
| **Warmup** | 训练初期用随机策略收集数据填充 replay buffer，只更新 critic、不更新 actor。让 Q 网络先学到合理的估计再指导 policy。 |
| **Entropy Regularization / 熵正则化** | SAC 的核心：在 RL 目标中加入策略熵项 $\alpha \mathcal{H}(\pi)$，鼓励策略保持随机性，防止过早收敛到次优解。 |
| **Automatic Temperature Tuning** | SAC 自动调节温度系数 α：当策略熵低于目标时增大 α（鼓励探索），高于目标时减小 α（聚焦利用）。 |

### 环境 / 任务相关

| 缩写/术语 | 全称 | 含义 |
|-----------|------|------|
| **Isaac Lab** | — | NVIDIA 的 GPU 加速机器人仿真平台（基于 Isaac Sim）。支持大规模并行仿真（数百个环境同时跑）。 |
| **IK** | Inverse Kinematics（逆运动学） | 给定目标 3D 位置和姿态 → 计算机器人各关节角度。本项目中 policy 输出末端执行器目标位姿，仿真环境通过 IK 转化为关节命令。 |
| **ee_pose** | End-Effector Pose（末端执行器位姿） | 机器人手爪（夹爪）的 3D 位置 (x,y,z) + 姿态四元数 (qw,qx,qy,qz)，共 7 维。 |
| **Quaternion** | 四元数 (qw, qx, qy, qz) | 表示 3D 旋转的数学工具（比欧拉角更稳定，无万向锁问题）。`qw` 是实部，`(qx,qy,qz)` 是虚部。 |
| **Gripper** | 夹爪 | 机器人末端的抓取工具。本项目中简化为 1 维：+1 = 张开（释放物体），-1 = 闭合（夹紧物体）。 |
| **Pick-Place** | 拾取-放置 | 核心任务：从位置 A 拿起物体 → 移动到位置 B → 放下。 |
| **Vectorized Env** | 向量化环境 | 同时并行运行多个环境实例（如 16 个），一次 `step()` 调用同时返回多组 (obs, reward, done)。大幅加速数据收集。 |

### 数学符号

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 参数为 θ 的策略网络（actor） |
| $Q_\phi$ | 参数为 φ 的 Q 网络（critic） |
| $\bar{Q}_\phi$ | Q 网络的目标网络（EMA 慢更新版本） |
| $\epsilon_\theta$ | Denoiser 网络预测的噪声（SAC-Diffusion 模式） |
| $x_t$ | 去噪过程第 t 步的中间带噪动作 |
| $x_T$ | 初始纯噪声（从标准正态分布采样） |
| $x_0$ | 最终去噪完成的干净动作 |
| $\gamma$ | 折扣因子（0.99）：未来 reward 的衰减系数。γ 越接近 1，越关注长期回报。 |
| $\tau$ | Target network EMA 系数（0.005）：值越小，target 网络更新越慢越稳定。 |
| $\alpha$ | SAC 温度系数：控制熵正则化强度。可自动调节（auto entropy tuning）。 |
| $\mathcal{H}(\pi)$ | 策略的熵：$-\mathbb{E}[\log \pi(a|s)]$，衡量策略的随机性/探索程度。 |

---

## 关键公式详解

> 本节对文档中涉及的核心数学公式进行逐一推导和直觉解释。
> 建议结合上方术语表和后续的 SAC 原理详解一起阅读。

### F1. Diffusion Policy 训练目标（Score Matching / 噪声预测）

BC 预训练阶段，Diffusion Policy 的训练目标是**让去噪网络学会预测"加了多少噪声"**。

**前向加噪过程**: 给专家动作 $a_0$ 加上不同程度的噪声：

$$x_t = \sqrt{\bar{\alpha}_t}\, a_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

- $a_0$: 示范数据中的真实动作（16 步 action chunk, 形状 $16 \times 8$）
- $t$: 噪声时间步（$t=0$ 表示干净, $t=T$ 表示纯噪声）
- $\bar{\alpha}_t$: 噪声调度参数, $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$, 随 $t$ 增大而减小
- 直觉: $t$ 越大, $\sqrt{\bar{\alpha}_t}$ 越小 → 原始信号被压缩越多, $\sqrt{1 - \bar{\alpha}_t}$ 越大 → 噪声成分越大

**训练 Loss (简单版本)**:

$$\mathcal{L}_{BC} = \mathbb{E}_{t, \epsilon, a_0} \left[ \lVert \epsilon_\theta(x_t, t, s) - \epsilon \rVert^2 \right]$$

- $\epsilon_\theta$: 去噪网络（我们要训练的 U-Net），输入带噪动作 $x_t$、时间步 $t$、观测 $s$
- $\epsilon$: 真正加进去的噪声（ground truth）
- 直觉: **让网络猜 "我加了什么噪声"，猜得越准越好**

### F2. DDIM 去噪采样公式（推理时用）

训练好之后，推理时从纯噪声 $x_T \sim \mathcal{N}(0,I)$ 出发，逐步去噪得到干净动作 $x_0$。

**DDIM 单步更新** (从 $x_t$ 到 $x_{t-1}$):

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1-\bar{\alpha}_t}\, \epsilon_\theta(x_t, t, s)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{predicted } x_0} + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \underbrace{\epsilon_\theta(x_t, t, s)}_{\text{predicted noise direction}}$$

拆解三步理解:

1. **估计干净动作 $\hat{x}_0$**: 先用网络预测噪声 $\epsilon_\theta$，然后"减掉"噪声得到对 $x_0$ 的估计:
$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\, \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$
> 直觉: "如果噪声是这些，那原来的干净数据应该长这样"

2. **重新加噪到 $t{-}1$ 级别**: 把 $\hat{x}_0$ 和噪声方向按 $t-1$ 对应的比例混合，得到 $x_{t-1}$
> 直觉: "不一步跳到干净，而是只去掉一点点噪声，稳步推进"

3. **DDIM 是确定性的**: 没有额外随机噪声项（区别于 DDPM），同一初始噪声 → 同一动作输出

本项目用 $K=10$ 步 DDIM，即在 $[T, \ldots, 1]$ 中均匀选 10 个时间步做去噪。

### F3. Bellman 方程与 TD Target（Critic 训练）

Q-network 的训练目标：让 $Q(s,a)$ 逼近 **真实的累积折扣回报**。

**Bellman 最优方程**:

$$Q^{\ast}(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^{\ast}(s', a') \right]$$

> 直觉: "在状态 $s$ 做动作 $a$ 的总价值 = 立刻拿到的 reward $r$ + 未来所有 reward 的折扣总和"

**SAC 的 Soft TD Target (Clipped Double-Q + 熵正则化)**:

$$y = r + \gamma (1 - d) \left[ \min\left( \bar{Q}_{\phi_1}(s', a'), \; \bar{Q}_{\phi_2}(s', a') \right) - \alpha \log \pi(a'|s') \right]$$

其中:
- $a' \sim \pi(\cdot|s')$: 下一步动作从当前策略**采样**（不是确定性的）
- $\bar{Q}_{\phi_1}$, $\bar{Q}_{\phi_2}$: 两个**目标 Q 网络**（EMA 慢更新版本）
- $\min(\cdot, \cdot)$: 取两个 Q 的较小值（防过估计）
- $-\alpha \log \pi(a'|s')$: 熵项，SAC 特有——鼓励选择高熵（更随机）的动作

> **为什么取 min?** Q 网络天生倾向于**过度乐观**（高估 Q 值），因为 max 操作会放大估计误差。
> 取两个独立 Q 网络的较小者 → 相当于对乐观估计做了保守修正。

**Critic Loss**:

$$\mathcal{L}_{critic} = \frac{1}{N}\sum_{i=1}^{N} \left[ \left( Q_{\phi_1}(s_i, a_i) - y_i \right)^2 + \left( Q_{\phi_2}(s_i, a_i) - y_i \right)^2 \right]$$

> 直觉: 让两个 Q 网络的预测都尽量接近 Soft TD target $y$。

### F4. EMA 目标网络更新

$$\bar{Q}_\phi \leftarrow \tau \cdot Q_\phi + (1 - \tau) \cdot \bar{Q}_\phi$$

- $\tau = 0.005$ (非常小)
- 每次只把主网络的 0.5% "混入"目标网络
- 效果: 目标网络**非常缓慢**地追踪主网络，提供稳定的 TD target

> 直觉: 如果直接用主 Q 网络算 TD target，会出现"自己给自己打分"的循环 → 训练不稳定。
> 目标网络相当于一个"滞后版本的考官"，评判标准变化很慢，让训练更平稳。

### F5. 折扣累积回报（Return）

在 RL 中，我们真正要最大化的目标:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$$

- $\gamma = 0.99$: 很关注长期
- $\gamma^{100} = 0.99^{100} \approx 0.366$: 100 步后的 reward 仍有 36.6% 的权重
- $\gamma^{500} = 0.99^{500} \approx 0.0066$: 500 步后几乎忽略

> 实际 pick-place episode 约 100-300 步，所以 $\gamma=0.99$ 意味着整个 episode 的 reward 都很重要。

### F6. BC 正则化（SAC-Diffusion 模式）

SAC-Diffusion 模式中，RL 微调时保留一部分 BC loss 防止 policy 遗忘预训练行为:

$$\mathcal{L}_{actor} = \underbrace{\alpha \log \pi(a|s) - \min(Q_1, Q_2)}_{\text{SAC RL 项}} + \lambda \underbrace{\mathcal{L}_{BC}(\theta)}_{\text{BC 正则化: 防遗忘}}$$

**退火策略的直觉**:
```
训练阶段:     初期 ────────────────→ 后期
BC 权重 λ:     大 (2.0)               小 (0.1)
训练行为:     紧贴 BC 预训练         释放 RL 优化
类比:         学车时握着方向盘       慢慢放手让学员自己开
```

---

## 2. SAC 算法原理详解

### 2.1 最大熵 RL 框架

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

### 2.2 Soft Bellman 方程

在最大熵框架下，Q-function 和 V-function 的定义相应修改：

**Soft Q-function:**

$$Q^{\text{soft}}(s, a) = r(s, a) + \gamma \mathbb{E}_{s'} \left[ V^{\text{soft}}(s') \right]$$

**Soft V-function:**

$$V^{\text{soft}}(s) = \mathbb{E}_{a \sim \pi} \left[ Q^{\text{soft}}(s, a) - \alpha \log \pi(a|s) \right]$$

将两者结合，得到 **Soft Bellman Backup**:

$$y = r + \gamma (1 - d) \left[ \min(Q_1^{\bar{\theta}}(s', a'), Q_2^{\bar{\theta}}(s', a')) - \alpha \log \pi(a'|s') \right]$$

其中 $a' \sim \pi(\cdot|s')$（从当前策略采样），$Q^{\bar{\theta}}$ 是 target Q-network。

### 2.3 SAC 的三个核心组件

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

### 2.4 Squashed Gaussian Policy

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

## 3. 文件结构说明

```
scripts/scripts_sac/
├── __init__.py              # 包标识
├── README.md                # 本文档 — 算法讲解 + 使用指南
├── sac_networks.py          # 网络定义 (Twin Q + Gaussian Actor + α)
├── sac_env_wrapper.py       # 环境包装器
├── reward.py                # 奖励函数
├── replay_buffer.py         # 经验回放缓冲区
├── utils.py                 # 工具函数 (seed, soft_update, checkpoint 等)
├── train_sac.py             # 主训练脚本 (支持 gaussian/diffusion 两种模式)
├── eval_sac.py              # 评估脚本 (兼容 eval_fair 输出格式)
└── run_sac_pipeline.sh      # 一键运行全流程（训练 + 评估）
```

### 依赖关系

```
train_sac.py
├── scripts.scripts_sac.sac_env_wrapper  # 环境包装器
│   ├── PickPlaceRLEnv                    # 环境接口
│   └── load_pretrained_diffusion_policy  # 加载 BC 预训练权重
├── scripts.scripts_sac.reward           # 奖励函数
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

## 8. 核心公式推导 (SAC)

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

---

## 参考文献

1. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML 2018.
2. Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ... & Levine, S. (2019). *Soft Actor-Critic Algorithms and Applications*. arXiv:1812.05905.
3. Fujimoto, S., Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods* (TD3). ICML 2018.
4. Wang, Z., Hunt, J. J., & Zhou, M. (2023). *Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning*. ICLR 2023.
