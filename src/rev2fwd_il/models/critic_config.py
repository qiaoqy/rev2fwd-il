from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PolicyFeature


@dataclass
class CriticConfig:
    """Critic model configuration.

    Uses input_features dict to describe observation inputs (same pattern as DiffusionConfig/PreTrainedConfig).
    Properties robot_state_feature / image_features are derived automatically from input_features.
    """

    # === Input features (aligned with DiffusionConfig, can copy action_config.input_features directly) ===
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    action_dim: int = 7  # action dimension, UNet input channel count

    # === Vision backbone params (aligned with DiffusionConfig for weight reuse) ===
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # === Vision encoder weight initialization ===
    action_model_checkpoint: str | None = None  # Load rgb_encoder weights from action model as initialization;
                                                 # not frozen, trained end-to-end with UNet

    # === UNet structure params (aligned with DiffusionConfig) ===
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # === Trajectory params ===
    horizon: int = 16
    n_obs_steps: int = 2

    # === Critic-specific params ===
    gamma: float = 0.995
    value_loss_type: str = "mse"  # "mse" | "huber"

    # === Training params ===
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-6

    # --- Derived properties (consistent with PreTrainedConfig behavior) ---

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == "observation.state":
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {k: ft for k, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    def __post_init__(self):
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                f"horizon ({self.horizon}) must be divisible by "
                f"downsampling_factor ({downsampling_factor}, from len(down_dims)={len(self.down_dims)})"
            )
        if self.value_loss_type not in ("mse", "huber"):
            raise ValueError(f"value_loss_type must be 'mse' or 'huber', got '{self.value_loss_type}'")
