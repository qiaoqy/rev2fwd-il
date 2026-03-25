# RECAP 中的 Advantage 函数详解及 Diffusion Policy 强化学习实现方案

> 基于论文：**π0.6∗: a VLA That Learns From Experience**（Physical Intelligence, 2026）  
> 方法全称：**RECAP — RL with Experience and Corrections via Advantage-conditioned Policies**

---

## 1. 总体框架回顾

RECAP 的目标是让大型视觉-语言-动作（VLA）模型通过真实环境中的经验（自主轨迹 + 人工纠错干预）进行强化学习。整个方法分为三步，可以迭代进行：

```
1. 数据收集  →  自主执行任务，收集轨迹；记录末端成功/失败标签；可选人工介入纠错
2. 价值函数训练  →  用全部历史数据训练多任务分布式价值函数 V^π
3. Advantage 条件化策略训练  →  从 V^π 推导每帧 advantage，生成二值改进指标 I_t，
                                以此作为额外输入重新训练策略
```

---

## 2. 稀疏奖励下的奖励函数设计

论文使用一个适用于"只知道每条轨迹末端成功/失败"场景的通用稀疏奖励函数：

$$
r_t = \begin{cases}
0            & \text{if } t = T \text{ 且任务成功} \\
-C_{\text{fail}} & \text{if } t = T \text{ 且任务失败} \\
-1           & \text{其他时间步}
\end{cases}
$$

其中 $C_{\text{fail}}$ 是一个较大的常数（如最大轨迹长度），确保失败轨迹的总回报远低于成功轨迹。

**直觉解释**：
- 成功轨迹的从时间步 $t$ 出发的回报 $R_t = -(T - t)$，即剩余步数的负值（值越接近 0 越好）
- 失败轨迹的回报 $R_t = -(T - t) - C_{\text{fail}}$，远比成功轨迹低
- 这样价值函数 $V^\pi(o_t)$ 学到的是"预计剩余成功步数的负值"，论文会将其归一化到 $(-1, 0)$

---

## 3. 价值函数的建模与训练

### 3.1 分布式价值函数

论文采用**分布式（distributional）价值函数**：

$$
p_\phi(V | o_t, \ell) \in \Delta^B
$$

- 输入：当前观测 $o_t$（图像 + 机器人状态）和语言指令 $\ell$
- 输出：对 $B$ 个离散价值区间（value bins）的概率分布，而不是单一标量
- 连续价值函数用期望恢复：$V^{\pi_\text{ref}}(o_t, \ell) = \sum_{b} p_\phi(V=b | o_t) \cdot v(b)$

采用分布式表示的原因：对训练目标更加鲁棒，能够捕捉价值的不确定性，在稀疏奖励场景下训练更稳定。

### 3.2 训练目标（Monte Carlo 估计）

**训练使用 Monte Carlo 回报估计**（而非 TD/Q-learning）：

$$
\min_\phi \mathbb{E}_{\tau \in \mathcal{D}} \left[ \sum_{o_t \in \tau} H\!\left(R_t(\tau),\, p_\phi(V | o_t, \ell)\right) \right]
$$

其中：
- $R_t(\tau) = \sum_{t'=t}^{T} r_{t'}$ 是从时间步 $t$ 到轨迹末尾的实际累积回报
- $H(\cdot, \cdot)$ 是交叉熵损失（先将 $R_t(\tau)$ 映射到最近的离散 bin）
- $\mathcal{D}$ 是目前收集的所有数据（演示数据 + 自主轨迹数据）

**关键优势**：无需贝尔曼迭代（Bellman backup），不存在 Q-learning 的 bootstrapping 不稳定问题，对 off-policy 数据也天然适用。

### 3.3 价值函数网络结构

论文用一个**较小的 VLM 主干**（670M 参数，Gemma 3 初始化，与策略网络架构相同但更小）来实现价值函数。之所以采用 VLM 级别的模型而不是简单 CNN，是因为任务是多任务且语言条件化的，需要语义理解能力。为防止过拟合，还混合了少量多模态网页数据做联合训练。

---

## 4. Advantage 的计算

### 4.1 数学定义

n 步 advantage 的定义为：

$$
A^{\pi}(o_t, a_t) = \mathbb{E}_{\rho^\pi(\tau)} \left[\sum_{t'=t}^{t+N-1} r_{t'} + V^\pi(o_{t+N})\right] - V^\pi(o_t)
$$

在实际 offline 数据中，论文采用 **Monte Carlo 版本**（即 $N = T - t$，走到轨迹末尾）：

$$
A^{\pi_\text{ref}}(o_t, a_t, \ell) \approx R_t(\tau) - V^{\pi_\text{ref}}(o_t, \ell)
$$

**具体含义**：
- $R_t(\tau)$：该轨迹从时间步 $t$ 开始的**实际回报**（已知，来自轨迹末端的成功/失败标签）
- $V^{\pi_\text{ref}}(o_t, \ell)$：价值函数对当前状态的**预测期望回报**
- 二者之差就是 advantage：实际回报高于预期 → 正 advantage（动作比平均水平好）；低于预期 → 负 advantage

### 4.2 在只有末端成功/失败标签的情况下如何计算每帧 advantage

这是问题的核心。以下是完整流程：

**Step 1：用末端标签计算每帧的实际回报 $R_t(\tau)$**

给定一条轨迹 $\tau = (o_0, a_0, \ldots, o_T)$，末端标签为 `success` 或 `failure`，则：

```
# 成功轨迹，共 T 步
R_t = -(T - t)             # 例如 T=300（demo）：R_0=-300, R_150=-150, R_299=-1, R_300=0

# 失败轨迹
R_t = -(T - t) - C_fail    # C_fail=1200：R_0=-1500（300步失败），归一化后 = -1500/2400 ≈ -0.625
```

注意：**所有中间 reward 都是 -1，不需要逐帧打标签**，这正是稀疏奖励的关键优势。

**Step 2：训练价值函数 $V^{\pi_\text{ref}}(o_t, \ell)$**

对所有数据中的每一帧 $(o_t, \ell, R_t(\tau))$，用交叉熵损失（见 Eq.1）训练分布式价值函数。

**Step 3：计算每帧 advantage**

对数据集中每一帧：
```python
A_t = R_t(tau) - V_ref(o_t, ell)
# R_t(tau) 来自 step 1（由末端标签直接计算）
# V_ref(o_t, ell) 来自训练好的价值函数前向推断
```

**Step 4：计算每个任务的改进阈值 $\epsilon_\ell$**

```python
# 取价值函数输出在该任务下的第 30 百分位数作为阈值
epsilon_ell = np.percentile(V_ref_values_for_task_ell, 30)
```

**Step 5：生成二值改进指标 $I_t$**

```python
I_t = int(A_t > epsilon_ell)  # 1: 正（该帧动作高于平均水平），0: 负
```

---

## 5. 策略提取：Advantage 条件化训练

### 5.1 理论基础

论文基于正则化 RL 的最优策略闭合形式：

$$
\hat{\pi}(a|o, \ell) \propto \pi_\text{ref}(a|o, \ell) \cdot p(I | A^{\pi_\text{ref}}(o, a, \ell))^\beta
$$

其中 $p(I | A) = \mathbb{1}[A > \epsilon_\ell]$（大于阈值则改进指标为正）。

当 $\beta=1$ 时简化为 $\hat{\pi}(a|o,\ell) = \pi_\text{ref}(a|I,o,\ell)$，即：**训练策略在条件化改进指标后就等于最优策略**。

### 5.2 训练目标

$$
\min_\theta \mathbb{E}_{\mathcal{D}^{\pi_\text{ref}}} \left[ -\log \pi_\theta(a_t | o_t, \ell) - \alpha \log \pi_\theta(a_t | I_t, o_t, \ell) \right]
$$

- 第一项：无条件生成（标准行为克隆）
- 第二项：条件化改进指标的生成（乘以超参数 $\alpha$ 调节权重）
- 两项一起类比于 **classifier-free guidance（CFG）**：同时训练有/无条件的分布

**在推断时**：固定 $I_t = \text{True}$（正），相当于要求模型输出"高 advantage"的动作。

### 5.3 如何将 advantage 注入模型

在 RECAP 中，改进指标通过**文本 token** 注入 VLA：
- `"Advantage: positive"` → $I_t = 1$
- `"Advantage: negative"` → $I_t = 0$

这个文本 token 放在语言指令之后、动作之前，只影响动作的 log-likelihood，不影响视觉/语言表示。

---

## 6. 用于 Diffusion Policy 的 RL 实现方案

Diffusion Policy（和 Flow Matching Policy）天然**不提供可微分的 log-likelihood**，这使得 PPO/policy gradient 等方法难以直接应用。RECAP 中的 advantage conditioning 恰好绕过了这一问题，以下是为 Diffusion Policy 实现该方法的具体方案。

### 6.1 为什么 Advantage Conditioning 适合 Diffusion Policy

| 方法 | 需要 log π(a|o) | 是否适用于 Diffusion Policy |
|------|:--------------:|:---------------------------:|
| PPO / REINFORCE | ✅ 是 | ❌ 困难（无解析 likelihood） |
| AWR（加权回归） | ❌ 否 | ✅ 可用（但丢弃负样本） |
| Advantage Conditioning（RECAP）| ❌ 否 | ✅ **完全适用** |

Advantage conditioning 只是**给模型加一个条件输入并用加权监督学习目标**，不需要对动作分布求梯度。这与 diffusion/flow matching 完全兼容。

### 6.2 实现步骤

#### Step 1：数据格式

每条轨迹保存以下信息：
```python
trajectory = {
    "observations": [o_0, o_1, ..., o_T],   # 图像 + 机器人状态
    "actions":      [a_0, a_1, ..., a_{T-1}],
    "success":      True / False,            # 末端标签（稀疏奖励来源）
    "task_id":      "pick_and_place",        # 任务标识符
}
```

#### Step 2：价值函数设计（共享 Diffusion Policy 编码器 + Value Head）

由于你的 Diffusion Policy 没有语言输入，直接**复用它的观测编码器**接 Value Head 是最自然的方案，无需另建独立的编码器。整体结构如下：

```
observations (images + proprioception)
        │
        ▼
┌──────────────────────┐
│  Diffusion Policy    │   ← 共享或 EMA copy，参数可冻结
│  Observation Encoder │
│  (CNN + state MLP)   │
└──────────┬───────────┘
           │ z_obs  (e.g. 512-d feature)
           ▼
┌──────────────────────┐
│    Value Head        │   ← 新增，轻量
│   (MLP, 2-3 层)      │
└──────────┬───────────┘
           │
           ▼
     p(V|o_t) ∈ ℝ^B (分布式)
     → 取期望得 V(o_t) ∈ ℝ
```

**选定方案：冻结编码器**（`encoder.requires_grad_(False)`，只训练 head）。

理由：数据量有限时，避免 value head 的训练信号破坏策略编码器已经学好的视觉表示；同时 value head 参数量极小，训练速度快。

---

#### 分布式价值函数输出是什么意思？

**分布式输出（distributional）**：网络不输出单个数，而是输出一个**概率分布** $p(V | o_t)$，描述"这个状态的回报有多大可能落在哪个区间"。

##### 具体例子

假设将回报范围 $(-1, 0)$ 均匀切成 **5 个 bin**（实际会用 32~64 个）：

```
bin 0: (-1.0, -0.8)  中心 = -0.9
bin 1: (-0.8, -0.6)  中心 = -0.7
bin 2: (-0.6, -0.4)  中心 = -0.5
bin 3: (-0.4, -0.2)  中心 = -0.3
bin 4: (-0.2,  0.0)  中心 = -0.1
```

对于同一个观测 $o_t$，假设数据集中来自该状态的轨迹有成功也有失败：

```
# 成功轨迹（50步）在 t=25 时：R_t = -(50-25)/550 ≈ -0.045  → 落在 bin 4
# 失败轨迹（50步）在 t=25 时：R_t = -(50-25-500)/550 ≈ -0.954 → 落在 bin 0
# （C_fail=500, max_len=50, norm=550）
```

网络输出 softmax 分布，比如：
```
p(V | o_t) = [0.45, 0.05, 0.05, 0.05, 0.40]
              bin0   bin1  bin2  bin3  bin4
```
表示：模型认为这个状态 **45% 概率对应失败轨迹，40% 概率对应成功轨迹**，中间状态各 5%。

从分布中取期望得到标量价值：
$$V(o_t) = \sum_b p_b \cdot c_b = 0.45 \times (-0.9) + \ldots + 0.40 \times (-0.1) \approx -0.45$$

训练时，对每条轨迹的真实 $R_t$ 做**交叉熵**（把 $R_t$ 映射到最近的 bin，当作分类标签）：
```python
# 真实 R_t = -0.045 → 最近 bin = 4
target = torch.tensor([4])
loss = F.cross_entropy(logits, target)   # logits = 网络输出，未过 softmax
```

##### 优点（相比标量 MSE 回归）

1. **对稀疏回报的双峰分布建模**：如上例，成功/失败轨迹都经过状态 $o_t$，回报是**双峰分布**（一堆在 -0.9，一堆在 -0.1）。标量回归只能预测均值 -0.45（两头都不对），而分布式输出可以完整保留这个双峰形状。

2. **交叉熵比 MSE 训练更稳定**：MC 回报有高方差，极端值（如很长的失败轨迹）对分类损失不敏感，不会产生平方级的巨大梯度。

3. **advantage 计算更精准**：期望值 $\mathbb{E}[V]$ 由完整分布算出，比直接回归的标量更稳健。

##### 缺点

1. **bin 数量是额外超参**：推荐 $B=32$，原因见下方框注。

2. **实现比直接回归稍复杂**：需要把连续 $R_t$ 离散化为 bin 索引，推断时取期望（已在下方代码中封装好）。

> **为什么推荐 $B=32$？**
>
> 回报范围归一化后固定为 $(-1, 0)$，$B=32$ 时每个 bin 宽度为 $1/32 = 0.031$。
>
> - **分辨率够用**：Pick-and-place 通常是短 episode，成功/失败两个峰天然离得很远（成功聚集在 $-0.3\sim0$，失败聚集在 $-0.7\sim-1.0$），0.031 的 bin 宽足以分离两峰，区分有意义的 advantage 差异。
> - **数据量较小时更稳定**：每个 bin 分配到的样本数是 $B=64$ 的两倍，cross-entropy 的 one-hot 目标不会过于稀疏，训练更稳定。
> - **太少（如 8）** 会把双峰压缩到相邻 bin，无法区分成功/失败；**太多（如 512）** 每个 bin 样本极少，反而难以训练。
> - 数据量充足（>1000 条轨迹）或需要在成功集合内部做更细的 advantage 区分时，可升级到 $B=64$。

---

**Value Head 代码（冻结编码器版）**：

```python
class ValueHead(nn.Module):
    """
    分布式价值函数头，接在冻结的 Diffusion Policy 编码器之后。
    输出 B 个 bin 的概率分布，取期望得到标量价值。
    """
    def __init__(self, feat_dim: int, num_bins: int = 32, dropout: float = 0.1):
        """
        feat_dim : 编码器输出维度（e.g. 512）
        num_bins : 离散化 bin 数量，推荐 32（bin 宽 0.031，足以分离成功/失败双峰；
                   数据量有限时比 64 更稳定；数据充足时可升至 64）
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),          # 比 BatchNorm 更适合小 batch / 序列
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_bins),
        )
        # 回报范围归一化到 (-1, 0)，均匀分 num_bins 个 bin 中心
        centers = torch.linspace(-1.0 + 0.5/num_bins, -0.5/num_bins, num_bins)
        self.register_buffer("bin_centers", centers)       # 不参与梯度

    def forward(self, z_obs: torch.Tensor) -> torch.Tensor:
        """z_obs: [B, feat_dim] → 期望价值标量 [B]"""
        logits = self.net(z_obs)                            # [B, num_bins]
        probs = F.softmax(logits, dim=-1)                   # [B, num_bins]
        return (probs * self.bin_centers).sum(-1)           # 期望，[B]

    def loss(self, z_obs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        z_obs  : [B, feat_dim]
        returns: [B]，归一化后的 R_t ∈ (-1, 0)
        """
        logits = self.net(z_obs)                            # [B, num_bins]
        # 将连续 R_t 映射到最近的 bin 索引作为分类标签
        dists = (returns.unsqueeze(1) - self.bin_centers.unsqueeze(0)).abs()
        target_bins = dists.argmin(dim=1)                   # [B]
        return F.cross_entropy(logits, target_bins)


class DiffusionPolicyWithValueHead(nn.Module):
    """
    冻结 Diffusion Policy 的观测编码器，只训练 ValueHead。
    policy_encoder 是你已有模型中提取观测特征的那一部分。
    """
    def __init__(self, policy_encoder, feat_dim: int, num_bins: int = 32):
        super().__init__()
        import copy
        self.encoder = copy.deepcopy(policy_encoder)
        self.encoder.requires_grad_(False)   # 冻结：不更新编码器参数
        self.value_head = ValueHead(feat_dim, num_bins=num_bins)

    def predict_value(self, obs):
        with torch.no_grad():                # 冻结时无需记录编码器梯度
            z = self.encoder(obs)
        return self.value_head(z)            # [B]，标量期望值

    def compute_loss(self, obs, returns):
        with torch.no_grad():
            z = self.encoder(obs)
        return self.value_head.loss(z, returns)
```

**为什么推荐 LayerNorm + Dropout 而不是 BatchNorm**：
- 价值函数训练时 batch 内混合不同轨迹/时间步，BN 的统计量不稳定
- LayerNorm 对任意 batch size 均有效，Dropout 缓解 MC 回报的高方差过拟合

**是否需要 LSTM/GRU？**

| 情况 | 建议 |
|------|------|
| 观测帧已包含足够时序信息（如堆叠多帧图像，或 proprioception 包含速度）| **不需要 LSTM**，单帧 MLP head 足矣 |
| 观测只有当前单帧图像，任务状态本质上不是 Markovian（如需要感知"放了多少个零件"）| **加一层 GRU**，在时序特征上聚合后再接 MLP |

如果 Diffusion Policy 已经在输入端做了时序堆叠（observation horizon > 1），则 Value Head 接堆叠后的特征即可，无需再加 RNN：

```python
# 若 obs_horizon=2，编码器输出 z.shape = [B, obs_horizon * feat_dim]
# 可先做 mean pooling 或直接 flatten 后接 MLP
z_pooled = z.mean(dim=1)   # [B, feat_dim]
value = self.value_head(z_pooled)
```

**价值函数训练**（完整流程）：

```python
import torch, numpy as np

C_FAIL = 1200         # = MAX_EP_LEN，使成功/失败归一化回报在 -0.5 处完美分离
MAX_EP_LEN = 1200     # rollout 最大帧数（demo ~300帧，rollout 上限 1200帧）

def compute_normalized_returns(trajectory: dict, c_fail=C_FAIL, max_len=MAX_EP_LEN):
    """
    由末端成功/失败标签直接计算每帧归一化回报，无需逐帧 reward。
    返回 List[float]，值域 (-1, 0)。
    """
    T = len(trajectory["actions"])
    norm = max_len + c_fail          # 使 R ∈ (-1, 0)
    returns = []
    for t in range(T):
        base = -(T - t)              # 时间惩罚（每步 -1）
        penalty = 0 if trajectory["success"] else -c_fail
        returns.append((base + penalty) / norm)
    return returns


def vf_train_step(vf_model, obs_batch, return_batch, optimizer):
    """
    obs_batch   : [B, C, H, W] 或 [B, obs_dim]，原始观测
    return_batch: [B]，归一化后的 R_t（float，范围 (-1, 0)）
    """
    optimizer.zero_grad()
    loss = vf_model.compute_loss(obs_batch, return_batch)   # 交叉熵，分布式
    loss.backward()
    optimizer.step()
    return loss.item()
```

#### Step 3：计算每帧 advantage 并生成二值指标

```python
def compute_advantages_and_indicators(dataset, vf_model, percentile=30):
    """
    dataset: List[dict]，每条轨迹包含 observations, actions, success
    返回: 每条轨迹、每帧的 I_t (0 或 1) 写入 trajectory["indicators"]
    """
    all_advantages = []

    # Pass 1：收集全部 advantage 值
    for traj in dataset:
        returns = compute_normalized_returns(traj)
        for t, (obs, R_t) in enumerate(zip(traj["observations"], returns)):
            obs_tensor = torch.tensor(obs).unsqueeze(0)
            V_t = vf_model.predict_value(obs_tensor).item()
            A_t = R_t - V_t
            all_advantages.append((traj, t, A_t))

    # Pass 2：按整体分位数确定阈值（单任务场景；多任务则按 task_id 分组）
    A_values = [a for _, _, a in all_advantages]
    threshold = np.percentile(A_values, percentile)   # 成功率~50%时取第30百分位，约70%帧为positive

    # Pass 3：写入二值指标
    for traj in dataset:
        traj["indicators"] = [0] * len(traj["actions"])
    for traj, t, A_t in all_advantages:
        traj["indicators"][t] = int(A_t > threshold)

    print(f"Threshold: {threshold:.4f}  |  "
          f"Positive ratio: {np.mean([a > threshold for _,_,a in all_advantages]):.1%}")
    return threshold
```

#### Step 4：Advantage-conditioned Diffusion Policy 训练

核心思路：给 Diffusion Policy 的去噪网络额外输入一个表示 advantage 正负的 token/embedding：

```python
class AdvantageConditionedDiffusionPolicy(nn.Module):
    def __init__(self, base_diffusion_policy):
        super().__init__()
        self.base = base_diffusion_policy
        # advantage 条件 embedding：positive/negative 各一个
        self.adv_embed = nn.Embedding(2, self.base.cond_dim)
    
    def forward(self, noisy_action, timestep, obs, lang, advantage_indicator=None):
        """
        advantage_indicator: None（无条件）/ 0（负）/ 1（正）
        """
        cond = self.base.encode_condition(obs, lang)
        
        if advantage_indicator is not None:
            adv_emb = self.adv_embed(advantage_indicator)   # [B, cond_dim]
            cond = cond + adv_emb                           # 简单加法融合，也可拼接
        
        return self.base.denoise(noisy_action, timestep, cond)

def train_step(policy, batch, alpha=1.0):
    """
    batch 包含: obs, lang, actions, I_t (0 或 1), 以及部分样本的 I_t=None（随机丢弃）
    """
    obs, lang, actions, indicators = batch
    
    # 1. 无条件生成损失（CFG 训练：随机 drop advantage indicator）
    loss_unconditional = diffusion_loss(policy, obs, lang, actions, indicator=None)
    
    # 2. 条件化损失
    loss_conditional = diffusion_loss(policy, obs, lang, actions, indicator=indicators)
    
    total_loss = loss_unconditional + alpha * loss_conditional
    return total_loss

# 推断时：固定 indicator = 1 (positive)，等价于要求输出高 advantage 动作
def inference(policy, obs, lang):
    return policy.sample(obs, lang, advantage_indicator=torch.ones(1, dtype=torch.long))
```

**训练时建议以一定概率随机置空 `advantage_indicator=None`**（如 20% 的概率），这正是 CFG 训练范式的核心——训练后推断时可以用 CFG guidance 放大改进效果：

$$
\epsilon_\theta^*(x, c) = \epsilon_\theta(x, \emptyset) + \beta \cdot (\epsilon_\theta(x, c) - \epsilon_\theta(x, \emptyset))
$$

其中 $c$ 是正 advantage 指标，$\beta > 1$ 可以进一步放大 advantage 条件的引导强度。

---

## 7. RECAP 引用的关键参考方法

### 7.1 核心参考文献 [4]：CFGRL

> **Frans et al., "Diffusion Guidance is a Controllable Policy Improvement Operator"**（arXiv:2505.23458, 2025）

这篇论文从理论上证明了：对一个条件扩散模型应用 classifier-free guidance（CFG），等价于在某个 KL 正则化 RL 目标下对策略进行改进。

正则化 RL 最优策略：$\hat{\pi}(a|o) \propto \pi_\text{ref}(a|o) \exp(A^{\pi_\text{ref}}(o,a)/\beta)$

当 advantage 被二值化为改进指标 $I = \mathbb{1}[A > \epsilon]$ 时，CFG 的 guidance 方向天然对应"向高 advantage 区域移动"，**不需要任何策略梯度**。

### 7.2 参考文献 [48]：Advantage Conditioning

> **Decision Transformer / ESPER / OPAL** 等方法（条件化回报/advantage 的序列建模）

这一系列工作提出：直接将回报/advantage 作为条件输入策略（如 "Return-conditioned" 策略），来自先前经验中高回报轨迹上的 BC 就能模拟出RL效果。

### 7.3 参考文献 [23]：DPPO（Diffusion Policy Policy Optimization）

> **Ren et al., "Diffusion Policy Policy Optimization"（ICLR 2025）**

使用单步扩散目标近似 log-likelihood，对 Diffusion Policy 应用 PPO。但实验表明 RECAP 的 advantage conditioning 方法显著优于 DPPO。

### 7.4 参考文献 [72]：分布式 RL（Distributional RL）

由此衍生了 C51, QR-DQN, IQN 等方法。RECAP 中用分布式价值函数建模不确定性，比标量 VF 更稳定。

---

## 8. 与其他方法的对比

| 方法 | 策略提取方式 | 适用于 Diffusion Policy | 利用负样本数据 | Off-policy 数据 |
|------|------------|:---:|:---:|:---:|
| PPO/DPPO | 策略梯度 | 困难 | ✅ | ❌ |
| AWR（Advantage-Weighted Regression）| 加权 BC | ✅ | ❌（负样本权重趋零） | ✅ |
| **RECAP（Advantage Conditioning）** | 条件化 BC | **✅** | **✅（负样本作为负指标）** | **✅** |

RECAP 相比 AWR 的核心优势是：AWR 会丢弃（或极低权重）advantage 为负的样本，而 RECAP 保留了所有样本，并让模型同时学习"什么是好动作"和"什么是差动作"，对数据效率更高。

---

## 9. 实现建议

### 9.1 最简版（适合小规模实验）

1. **价值函数**：复用 Diffusion Policy 的观测编码器（deepcopy 后**冻结**），接 2-3 层 MLP Head
   - 分布式输出，`num_bins=32`，交叉熵损失（成功/失败回报天然双峰，比 MSE 更稳定；数据充足时可升至 64）
   - LayerNorm + Dropout(0.1) 防止 MC 高方差过拟合
   - 若观测已多帧堆叠则无需 RNN；否则可在 head 前加一层 GRU 聚合时序

2. **Advantage 计算**：$A_t = R_t - V(o_t)$，按任务取 30 百分位作为阈值，生成 $I_t \in \{0, 1\}$

3. **策略训练**：在 Diffusion Policy 的 unet/transformer 条件化输入中加一个 `adv_token` embedding，
   随机 dropout（20%），其余按原有 diffusion BC 损失训练。

### 9.2 关键超参

| 超参 | **选定值** | 说明 |
|------|:----------:|------|
| `C_fail` | **1200** | = `MAX_EP_LEN`，成功回报 ∈ $(-0.5, 0)$，失败回报 ∈ $(-1, -0.5)$，在 $-0.5$ 处完美分离 |
| `MAX_EP_LEN` | **1200** | rollout 最大帧数（demo ~300帧，rollout 上限 1200帧） |
| `percentile` | **30** | 成功率 ~50%，约 70% 的帧为 positive；成功率变化时参考：20%成功→20，80%成功→45 |
| `num_bins` | **32** | 成功/失败各占 16 个 bin，数据量有限时比 64 更稳定 |
| `alpha` | 1.0 | 条件化损失权重（与无条件损失等权） |
| `null_prob` | 0.2 | CFG 训练中随机丢弃 advantage 指标的概率 |
| `beta`（推断时 CFG） | 1.0 | 先用 1.0（不做 guidance），效果不佳时尝试 1.5 |
| VF 更新频率 | 每轮数据收集后重新训练 | 迭代优化策略和价值函数 |

### 9.3 需要注意的问题

1. **价值函数过拟合**：MC 回报估计方差较大，需做好正则化（L2、dropout、混合数据）
2. **advantage 阈值的任务差异**：不同任务长度差异大，需 per-task 归一化
3. **人工纠错数据**：直接强制 $I_t = 1$（正），不经过 advantage 计算
4. **分布式 VF vs 标量 VF**：标量 VF 实现更简单，但分布式 VF 在稀疏奖励下更稳定
5. **固定 $I_t = 1$ 做 SFT**：在 RL 之前先用演示数据做一轮 SFT（固定 $I_t = \text{True}$），作为更好的起点

---

## 10. 总结

RECAP 中的 advantage 函数实现本质上是：

$$
\boxed{
A_t = R_t(\tau) - V^\pi(o_t, \ell) \quad \Rightarrow \quad I_t = \mathbb{1}[A_t > \epsilon_\ell]
}
$$

其中 $R_t(\tau)$ 由末端成功/失败标签直接计算（不需要逐帧 reward），$V^\pi$ 由 Monte Carlo 监督训练，$\epsilon_\ell$ 为任务级百分位阈值。这个二值指标 $I_t$ 作为额外条件注入策略，利用 classifier-free guidance 原理让模型在推断时"向高 advantage 区域对齐"。

该方法对 Diffusion Policy 友好，因为它完全不依赖策略 log-likelihood，只需要在条件化输入中加一个 advantage token，并用标准 diffusion 训练目标做加权监督学习。与 PPO 和 AWR 相比，RECAP 在 off-policy 数据的利用率和负样本信息的保留上具有明显优势。
