"""Model & Runtime Manager: Unified abstraction for Topaz Video AI models and hardware runtimes."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("topaz_cluster.model_manager")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    description: str
    supports_scale_2: bool
    recommended_scale: int
    default_blur: float
    default_blend: float
    max_instances_nvidia: int
    max_instances_apple: int


# Catalog of supported Topaz Video AI models and hardware execution profiles
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "iris-3": ModelSpec(
        name="iris-3",
        display_name="Iris MQ v3",
        description="Face restoration, low-light denoise, and progressive detail enhancement.",
        supports_scale_2=True,
        recommended_scale=2,  # scale=2 engages ANE on Apple Silicon and gives 2.6x on NVIDIA
        default_blur=-0.10,
        default_blend=0.20,
        max_instances_nvidia=2,
        max_instances_apple=1,
    ),
    "rhea-1": ModelSpec(
        name="rhea-1",
        display_name="Rhea v1",
        description="High-fidelity 4x generative detail reconstruction for SD/HD inputs.",
        supports_scale_2=False,
        recommended_scale=0,  # scale=0 auto-fits target resolution
        default_blur=0.0,
        default_blend=0.0,
        max_instances_nvidia=1,
        max_instances_apple=1,
    ),
    "prob-4": ModelSpec(
        name="prob-4",
        display_name="Proteus v4",
        description="Fine-grained parameter tuning for compression, noise, and halos.",
        supports_scale_2=False,
        recommended_scale=0,
        default_blur=0.0,
        default_blend=0.20,
        max_instances_nvidia=1,
        max_instances_apple=1,
    ),
    "nyx-3": ModelSpec(
        name="nyx-3",
        display_name="Nyx v3",
        description="Fast temporal denoise and detail enhancement for noisy high-ISO footage.",
        supports_scale_2=False,
        recommended_scale=0,
        default_blur=0.0,
        default_blend=0.20,
        max_instances_nvidia=1,
        max_instances_apple=1,
    ),
}


class ModelManager:
    """Central abstraction for Topaz neural models, hardware profiles, and filter graphs."""

    def __init__(self, default_model: str = "iris-3"):
        if default_model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{default_model}'. Supported: {list(MODEL_REGISTRY.keys())}")
        self.default_model = default_model

    def get_model_spec(self, model_name: Optional[str] = None) -> ModelSpec:
        name = model_name or self.default_model
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' not found in registry.")
        return MODEL_REGISTRY[name]

    def get_optimal_instances(self, accelerator_type: str, model_name: Optional[str] = None) -> int:
        """Return the empirically verified maximum safe instances for the target accelerator."""
        spec = self.get_model_spec(model_name)
        accel = accelerator_type.lower()
        if "ane" in accel or "apple" in accel or "metal" in accel:
            return spec.max_instances_apple  # ANE hardware queue strictly serializes; instances must be 1
        elif "cuda" in accel or "nvidia" in accel or "volta" in accel or "blackwell" in accel:
            return spec.max_instances_nvidia
        return 1

    @staticmethod
    def compute_native_4k_dims(
        src_w: int = 0,
        src_h: int = 0,
        target_w: int = 3840,
        target_h: int = 2160,
    ) -> Tuple[int, int]:
        """Compute native aspect-ratio 4K target dimensions (even integers, no stretching or black bars).
        
        Fits the source aspect ratio inside a 3840x2160 (landscape) or 2160x3840 (portrait/square)
        bounding box so vertical 9:16 becomes 2160x3840, 720x1190 becomes 2160x3570, and 1:1 becomes 2160x2160.
        """
        if src_w <= 0 or src_h <= 0:
            return target_w, target_h
        max_long = max(target_w, target_h)
        max_short = min(target_w, target_h)
        if src_w >= src_h:
            box_w, box_h = max_long, max_short
        else:
            box_w, box_h = max_short, max_long
        scale_factor = min(box_w / float(src_w), box_h / float(src_h))
        out_w = max(2, int(round(src_w * scale_factor / 2.0)) * 2)
        out_h = max(2, int(round(src_h * scale_factor / 2.0)) * 2)
        return out_w, out_h

    def build_filter(
        self,
        model_name: Optional[str] = None,
        scale: Optional[int] = None,
        target_w: int = 3840,
        target_h: int = 2160,
        instances: int = 1,
        device_id: int = 0,
        blur: Optional[float] = None,
        blend: Optional[float] = None,
        scaling_filter: str = "lanczos",
        allow_crop: bool = False,
        src_w: int = 0,
        src_h: int = 0,
    ) -> str:
        """Generate canonical Topaz filter graph strictly enforcing native aspect ratio and uncropped output."""
        if allow_crop:
            raise ValueError("Safety invariant violated: allow_crop must remain False to prevent framing loss.")

        spec = self.get_model_spec(model_name)
        active_scale = scale if scale is not None else spec.recommended_scale
        active_blur = blur if blur is not None else spec.default_blur
        active_blend = blend if blend is not None else spec.default_blend

        # Clamp instances to legal Topaz range [0, 3]
        if not (0 <= instances <= 3):
            raise ValueError(f"Topaz instances={instances} out of range [0, 3].")

        out_w, out_h = self.compute_native_4k_dims(src_w, src_h, target_w, target_h)

        model_params = (
            f"model={spec.name}:scale={active_scale}:preblur=0:noise=0:details=0:halo=0:"
            f"blur={active_blur:.2f}:compression=0:estimate=8:blend={active_blend:.2f}:"
            f"device={device_id}:vram=1:instances={instances}"
        )

        scaler_params = f"scale=w={out_w}:h={out_h}:flags={scaling_filter}:threads=0"

        # Immutable native aspect ratio preservation: 1:1 square pixel SAR, DAR matching out_w/out_h
        if src_w > 0 and src_h > 0:
            return f"tvai_up={model_params},{scaler_params},setsar=1:1"
        return f"tvai_up={model_params},{scaler_params},setsar=1:1,setdar=16/9"

    def get_runtime_environment(self, platform: str) -> Dict[str, str]:
        """Return platform-specific environment variables for model paths and scratch folders."""
        env_vars = {}
        plat = platform.lower()
        if "windows" in plat or "wsl" in plat:
            env_vars["TVAI_MODEL_DIR"] = r"C:\ProgramData\Topaz Labs LLC\Topaz Video\models"
            env_vars["TVAI_MODEL_DATA_DIR"] = r"C:\ProgramData\Topaz Labs LLC\Topaz Video\models"
        elif "darwin" in plat or "mac" in plat or "apple" in plat:
            env_vars["TVAI_MODEL_DIR"] = "/Applications/Topaz Video.app/Contents/Resources/models"
            env_vars["TVAI_MODEL_DATA_DIR"] = "/Applications/Topaz Video.app/Contents/Resources/models"
        return env_vars
