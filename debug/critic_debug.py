from  src.rev2fwd_il.models.critic_model import DiffusionConditionalUnet1d, DiffusionConfig

import torch

config = DiffusionConfig(
    input_features={
        "obs_state": {
            "type": "STATE",
            "shape": [10],
        },
        "action": {
            "type": "ACTION",
            "shape": [7],
        }
    }
)
# config.action_feature = [10, 7]

net = DiffusionConditionalUnet1d(
    config=config,
    global_cond_dim=0,
)

action_x = torch.randn(4, 16, 7)  # (B, T, action_dim)
timestep = torch.tensor([4,])  # (1,)
out_y = net(action_x, timestep=timestep)
print(out_y.shape)

