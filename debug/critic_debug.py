import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from src.rev2fwd_il.models.critic_config import CriticConfig
from src.rev2fwd_il.models.critic_model import DiffusionConditionalUnet1d

# use the virtual environment
# conda activate rev2fwd_il

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

B, T = 4, 16

# ========== Test 1: UNet 维度验证（无图像） ==========
config_simple = CriticConfig(
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),
    },
    action_dim=7,
    horizon=16,
)
net = DiffusionConditionalUnet1d(config=config_simple, global_cond_dim=0).to(device)

action_x = torch.randn(B, T, 7, device=device)  # (B, T, action_dim)
timestep = torch.tensor([4], device=device)
out_y = net(action_x, timestep=timestep)
assert out_y.shape == (B, T, 1), f"Expected ({B}, {T}, 1), got {out_y.shape}"
print(f"Test 1 (UNet only) passed ✓  output shape: {out_y.shape}")

