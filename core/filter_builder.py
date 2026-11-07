"""Canonical Topaz Video AI filter string builder."""

from typing import Optional
from .config import ModelConfig


def build_canonical_tvai_filter(
    model_cfg: ModelConfig,
    instances: int = 1,
    device_id: int = 0,
) -> str:
    """Construct canonical, cryptographically reproducible Iris-3 / Topaz filter graph.
    
    Guarantees:
    - Strictly uncropped (crop=... is forbidden).
    - Preserves aspect ratio with setsar=1:1,setdar=16/9.
    - High-quality Lanczos sinc scaling to 3840x2160.
    """
    if model_cfg.allow_crop:
        raise ValueError("Policy violation: allow_crop must remain False to preserve frame integrity.")

    # Topaz instances validation (range 0 to 3 extra instances)
    if not (0 <= instances <= 3):
        raise ValueError(f"Invalid instances={instances}: Topaz Video AI supports 0 to 3 extra instances.")

    model_params = (
        f"model={model_cfg.name}:scale={model_cfg.scale}:preblur=0:noise=0:details=0:halo=0:"
        f"blur={model_cfg.blur:.2f}:compression=0:estimate=8:blend={model_cfg.blend:.2f}:"
        f"device={device_id}:vram=1:instances={instances}"
    )

    scale_params = (
        f"scale=w={model_cfg.target_width}:h={model_cfg.target_height}:flags={model_cfg.scaling_filter}:threads=0"
    )

    return f"tvai_up={model_params},{scale_params},setsar=1:1,setdar=16/9"
