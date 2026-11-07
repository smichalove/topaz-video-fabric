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
    is_starlight: bool = False
    is_neuroserver: bool = False
    supports_linux_deb: bool = True
    supports_enhancement_strength: bool = False


# Friendly CLI aliases mapped to canonical Topaz model IDs
MODEL_ALIASES: Dict[str, str] = {
    "iris": "iris-3",
    "iris3": "iris-3",
    "rhea": "rhea-1",
    "rhea1": "rhea-1",
    "proteus": "prob-4",
    "prob4": "prob-4",
    "nyx": "nyx-3",
    "nyx3": "nyx-3",
    # Project Starlight / NeuroStream local models (macOS & Windows Topaz Video 7.x+)
    "starlight": "slp-2.5",
    "starlight-precise": "slp-2.5",
    "slp": "slp-2.5",
    "slp25": "slp-2.5",
    "slp26": "slp-2.5",
    "starlight-mini": "slm-1",
    "slm": "slm-1",
    "slm1": "slm-1",
    "starlight-sharp": "astrasharp",
    "astra-sharp": "astrasharp",
    "starlight-fast": "astrafast",
    "astra-fast": "astrafast",
    "starlight-hq": "astrahq",
    "astra-hq": "astrahq",
    "starlight-creative": "astra",
}


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
    # === Project Starlight / NeuroStream Local Temporal Diffusion Models (Windows & macOS 7.x+) ===
    "slp-2.5": ModelSpec(
        name="slp-2.5",
        display_name="Starlight Precise v2.5 (slp26)",
        description="Multi-frame temporal latent diffusion preserving geometry/identity while reconstructing micro-textures (Windows/macOS neuroserver).",
        supports_scale_2=False,
        recommended_scale=1,
        default_blur=0.0,
        default_blend=0.0,
        max_instances_nvidia=0,
        max_instances_apple=0,
        is_starlight=True,
        is_neuroserver=True,
        supports_linux_deb=False,
        supports_enhancement_strength=False,
    ),
    "astrasharp": ModelSpec(
        name="astrasharp",
        display_name="Starlight Sharp (Astra Sharp)",
        description="Pro temporal diffusion tuned for crisp edge recovery, small faces, and reduced lattice/flicker on low-res inputs (Windows/macOS neuroserver, >=16GB VRAM).",
        supports_scale_2=True,
        recommended_scale=2,
        default_blur=0.0,
        default_blend=0.0,
        max_instances_nvidia=0,
        max_instances_apple=0,
        is_starlight=True,
        is_neuroserver=True,
        supports_linux_deb=False,
        supports_enhancement_strength=True,
    ),
    "astrafast": ModelSpec(
        name="astrafast",
        display_name="Starlight Fast (Astra Fast)",
        description="Step-distilled temporal latent diffusion for 2x-3x faster diffusion throughput on low-res clips (Windows/macOS neuroserver).",
        supports_scale_2=True,
        recommended_scale=2,
        default_blur=0.0,
        default_blend=0.0,
        max_instances_nvidia=0,
        max_instances_apple=0,
        is_starlight=True,
        is_neuroserver=True,
        supports_linux_deb=False,
        supports_enhancement_strength=True,
    ),
    "astrahq": ModelSpec(
        name="astrahq",
        display_name="Starlight HQ (Astra HQ)",
        description="High-quality temporal diffusion enhancement for degraded archival or generative video (Windows/macOS neuroserver).",
        supports_scale_2=True,
        recommended_scale=2,
        default_blur=0.0,
        default_blend=0.0,
        max_instances_nvidia=0,
        max_instances_apple=0,
        is_starlight=True,
        is_neuroserver=True,
        supports_linux_deb=False,
        supports_enhancement_strength=True,
    ),
    "astra": ModelSpec(
        name="astra",
        display_name="Astra / Starlight Creative",
        description="Generative creative diffusion upscaler that synthesizes natural surface texture and skin detail on AI/CGI video (Windows/macOS neuroserver).",
        supports_scale_2=True,
        recommended_scale=2,
        default_blur=0.0,
        default_blend=0.0,
        max_instances_nvidia=0,
        max_instances_apple=0,
        is_starlight=True,
        is_neuroserver=True,
        supports_linux_deb=False,
        supports_enhancement_strength=True,
    ),
    "slm-1": ModelSpec(
        name="slm-1",
        display_name="Starlight Mini v1",
        description="Consumer local diffusion model for low-quality inputs (bundled ONNX/CoreML/DirectML).",
        supports_scale_2=True,
        recommended_scale=2,
        default_blur=0.0,
        default_blend=0.0,
        max_instances_nvidia=0,
        max_instances_apple=0,
        is_starlight=True,
        is_neuroserver=False,
        supports_linux_deb=False,
        supports_enhancement_strength=False,
    ),
}


class ModelManager:
    """Central abstraction for Topaz neural models, hardware profiles, and filter graphs."""

    @staticmethod
    def resolve_model_name(model_name: str) -> str:
        """Resolve friendly CLI aliases (e.g. 'starlight-sharp' -> 'astrasharp', 'starlight' -> 'slp-2.5')."""
        clean = model_name.strip().lower()
        return MODEL_ALIASES.get(clean, clean)

    def __init__(self, default_model: str = "iris-3"):
        resolved = self.resolve_model_name(default_model)
        if resolved not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{default_model}'. Supported: {list(MODEL_REGISTRY.keys())} (or aliases: {list(MODEL_ALIASES.keys())})")
        self.default_model = resolved

    def get_model_spec(self, model_name: Optional[str] = None) -> ModelSpec:
        name = self.resolve_model_name(model_name) if model_name else self.default_model
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' not found in registry.")
        return MODEL_REGISTRY[name]

    @staticmethod
    def get_output_tag(model_name: str) -> str:
        """Return compact filename tag for deliverable suffix (e.g. 'iris-3' -> 'iris3', 'slp-2.5' -> 'slp25')."""
        resolved = ModelManager.resolve_model_name(model_name)
        return resolved.replace("-", "").replace(".", "")

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

    @classmethod
    def compute_smart_4k_framing(
        cls,
        src_w: int = 0,
        src_h: int = 0,
        target_w: int = 3840,
        target_h: int = 2160,
        allow_crop: bool = False,
        smart_crop: bool = True,
        auto_crop_tolerance: float = 0.15,
    ) -> Tuple[int, int, bool]:
        """Determine optimal 4K output dimensions and whether center-crop fill should engage.

        Returns (out_w, out_h, use_crop_filter):
        - When allow_crop=True: forces full-frame 16:9 (3840x2160) or 9:16 (2160x3840) center-crop fill.
        - When smart_crop=True (default):
          1. Near-16:9 or Near-9:16 sources (within auto_crop_tolerance=15%, e.g. 404x720, 720x1198, 832x480)
             automatically snap and center-crop to exact 3840x2160 or 2160x3840 4K UHD.
          2. Near-1:1 square sources (within 8%, e.g. 720x720, 720x750) snap to 2160x2160 4K square.
          3. Wide-delta sources (e.g. 4:3 640x480/640x464 or 21:9 cinema where cropping to 16:9 would chop >15%
             of the subject) preserve their exact uncropped native aspect ratio inside the 4K bounding box.
        """
        if src_w <= 0 or src_h <= 0:
            return target_w, target_h, allow_crop

        max_long = max(target_w, target_h)
        max_short = min(target_w, target_h)
        is_landscape = src_w >= src_h
        std_w, std_h = (max_long, max_short) if is_landscape else (max_short, max_long)

        if allow_crop:
            return std_w, std_h, True

        if smart_crop:
            std_ratio = max_long / float(max_short)  # 1.7778 (16:9)
            src_ratio = max(src_w, src_h) / float(min(src_w, src_h))
            rel_diff_16_9 = abs(src_ratio - std_ratio) / std_ratio
            rel_diff_1_1 = abs(src_ratio - 1.0)

            if rel_diff_16_9 <= 0.002:
                return std_w, std_h, False
            elif rel_diff_16_9 <= auto_crop_tolerance:
                return std_w, std_h, True
            elif rel_diff_1_1 <= 0.002:
                return max_short, max_short, False
            elif rel_diff_1_1 <= 0.08:
                return max_short, max_short, True

        out_w, out_h = cls.compute_native_4k_dims(src_w, src_h, target_w, target_h)
        return out_w, out_h, False

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
        smart_crop: bool = True,
        enhancement_strength: float = 1.0,
    ) -> str:
        """Generate canonical Topaz filter graph with smart aspect-ratio & auto-crop optimization."""
        spec = self.get_model_spec(model_name)
        active_scale = scale if scale is not None else spec.recommended_scale
        active_blur = blur if blur is not None else spec.default_blur
        active_blend = blend if blend is not None else spec.default_blend

        # Clamp instances to legal Topaz range [0, 3]
        if not (0 <= instances <= 3):
            raise ValueError(f"Topaz instances={instances} out of range [0, 3].")

        if spec.is_starlight:
            # Project Starlight / NeuroStream models are standalone diffusion models (no GAN preblur/noise/estimate params)
            sl_instances = 0
            model_params = f"model={spec.name}:scale={active_scale}:device={device_id}:vram=1:instances={sl_instances}"
            if spec.supports_enhancement_strength:
                # Allowed values in Astra/Starlight JSON: 0.7, 1.0, 1.3
                valid_strength = min((0.7, 1.0, 1.3), key=lambda v: abs(v - float(enhancement_strength)))
                model_params += f":parameters=enhancement_strength={valid_strength}"
        else:
            model_params = (
                f"model={spec.name}:scale={active_scale}:preblur=0:noise=0:details=0:halo=0:"
                f"blur={active_blur:.2f}:compression=0:estimate=8:blend={active_blend:.2f}:"
                f"device={device_id}:vram=1:instances={instances}"
            )

        out_w, out_h, use_crop = self.compute_smart_4k_framing(
            src_w=src_w,
            src_h=src_h,
            target_w=target_w,
            target_h=target_h,
            allow_crop=allow_crop,
            smart_crop=smart_crop,
        )

        if use_crop:
            scaler_params = (
                f"scale=w={out_w}:h={out_h}:flags={scaling_filter}:threads=0:"
                f"force_original_aspect_ratio=increase,crop={out_w}:{out_h}"
            )
            return f"tvai_up={model_params},{scaler_params},setsar=1:1"

        scaler_params = f"scale=w={out_w}:h={out_h}:flags={scaling_filter}:threads=0"

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
