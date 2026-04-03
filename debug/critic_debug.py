import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from rev2fwd_il.models.critic_config import CriticConfig
from rev2fwd_il.models.critic_model import CriticModel, DiffusionConditionalUnet1d

# use the virtual environment
# conda activate rev2fwd_il

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

B, T = 4, 16

# ========== Test 1: UNet 维度验证（无图像） ==========
print("\n========== Test 1: UNet only ==========")
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

# ========== Test 2: CriticModel 带双相机图像输入 ==========
print("\n========== Test 2: CriticModel + dual cameras ==========")
config_full = CriticConfig(
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    },
    action_dim=7,
    horizon=16,
    n_obs_steps=2,
    crop_shape=(84, 84),
    action_model_checkpoint=None,  # no pretrained weights for debug
)
critic = CriticModel(config_full).to(device)

batch = {
    "action": torch.randn(B, T, 7, device=device),
    "observation.state": torch.randn(B, 2, 7, device=device),
    "observation.images": torch.rand(B, 2, 2, 3, 96, 96, device=device),
    "bellman_value": torch.rand(B, T, device=device),
}

pred = critic(batch)
assert pred.shape == (B, T, 1), f"Expected ({B}, {T}, 1), got {pred.shape}"
print(f"Test 2 (CriticModel + images) passed ✓  output shape: {pred.shape}")

# ========== Test 3: compute_loss + backward ==========
print("\n========== Test 3: compute_loss + backward ==========")
loss = critic.compute_loss(batch)
loss.backward()
print(f"Test 3 (loss backward) passed ✓  loss={loss.item():.4f}")

# ========== Test 4: All parameters have requires_grad == True ==========
print("\n========== Test 4: requires_grad check ==========")
all_grad = True
for name, param in critic.named_parameters():
    if not param.requires_grad:
        print(f"  WARNING: {name} has requires_grad=False")
        all_grad = False
assert all_grad, "Some parameters have requires_grad=False!"
total_params = sum(p.numel() for p in critic.parameters())
trainable_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
print(f"Test 4 passed ✓  all parameters have requires_grad=True")
print(f"  total params: {total_params:,}  trainable: {trainable_params:,}")

# ========== Test 5: Gradient flows to both rgb_encoder and UNet ==========
print("\n========== Test 5: gradient flow check ==========")
has_rgb_grad = False
has_unet_grad = False
for name, param in critic.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        if "rgb_encoder" in name:
            has_rgb_grad = True
        if "unet" in name:
            has_unet_grad = True
assert has_rgb_grad, "rgb_encoder has no gradient!"
assert has_unet_grad, "unet has no gradient!"
print(f"Test 5 passed ✓  gradients flow to both rgb_encoder and unet")

print("\n========== All tests passed! ==========")

