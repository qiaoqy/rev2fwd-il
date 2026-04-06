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
                                                 # not frozen, trained end-to-end with MLP value head

    # === MLP value head params ===
    mlp_hidden_dims: tuple[int, ...] = (512, 512, 256, 256)

    # === Observation params ===
    n_obs_steps: int = 2

    # === Critic-specific params ===
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
        if self.value_loss_type not in ("mse", "huber"):
            raise ValueError(f"value_loss_type must be 'mse' or 'huber', got '{self.value_loss_type}'")
