"""Unit tests for ModelManager and filter builder."""

import pytest

from topaz_cluster.core.config import ModelConfig
from topaz_cluster.core.filter_builder import build_canonical_tvai_filter
from topaz_cluster.core.model_manager import MODEL_REGISTRY, ModelManager


def test_model_registry_entries():
    mgr = ModelManager(default_model="iris-3")
    assert "iris-3" in MODEL_REGISTRY
    assert "prob-4" in MODEL_REGISTRY
    assert "rhea-1" in MODEL_REGISTRY
    assert "nyx-3" in MODEL_REGISTRY

    spec = mgr.get_model_spec("iris-3")
    assert spec.supports_scale_2 is True
    assert spec.recommended_scale == 2
    assert spec.max_instances_apple == 1
    assert spec.max_instances_nvidia == 2


def test_unknown_model_raises_error():
    with pytest.raises(ValueError, match="Unknown model 'non_existent_model'"):
        ModelManager(default_model="non_existent_model")


def test_hardware_instances_invariants():
    mgr = ModelManager()
    # Apple Silicon Neural Engine MUST strictly be 1 instance
    assert mgr.get_optimal_instances(accelerator_type="ane") == 1
    assert mgr.get_optimal_instances(accelerator_type="apple_m4") == 1
    assert mgr.get_optimal_instances(accelerator_type="metal") == 1

    # NVIDIA CUDA supports 2 instances for Iris-3
    assert mgr.get_optimal_instances(accelerator_type="cuda") == 2
    assert mgr.get_optimal_instances(accelerator_type="nvidia_rtx5080") == 2
    assert mgr.get_optimal_instances(accelerator_type="volta_v100") == 2


def test_canonical_filter_string_construction():
    mgr = ModelManager()
    filt = mgr.build_filter(
        model_name="iris-3",
        scale=2,
        target_w=3840,
        target_h=2160,
        instances=2,
        device_id=0,
    )
    # Check essential filter components
    assert "tvai_up=model=iris-3:scale=2" in filt
    assert "blur=-0.10" in filt
    assert "blend=0.20" in filt
    assert "instances=2" in filt
    assert "scale=w=3840:h=2160:flags=lanczos:threads=0" in filt
    assert "setsar=1:1,setdar=16/9" in filt

    # STRICT DEFAULT INVARIANT: Default allow_crop=False must NEVER include crop
    assert "crop=" not in filt

    # Optional center-crop fill mode when allow_crop=True is explicitly passed
    filt_crop = mgr.build_filter(
        model_name="iris-3",
        scale=2,
        target_w=3840,
        target_h=2160,
        instances=2,
        device_id=0,
        allow_crop=True,
    )
    assert "force_original_aspect_ratio=increase,crop=3840:2160" in filt_crop


def test_filter_builder_rejects_crop():
    model_cfg = ModelConfig(allow_crop=True)
    with pytest.raises(ValueError, match="allow_crop must remain False"):
        build_canonical_tvai_filter(model_cfg)


def test_filter_builder_instance_bounds():
    model_cfg = ModelConfig(allow_crop=False)
    with pytest.raises(ValueError, match="Invalid instances="):
        build_canonical_tvai_filter(model_cfg, instances=4)
    with pytest.raises(ValueError, match="Invalid instances="):
        build_canonical_tvai_filter(model_cfg, instances=-1)


def test_smart_4k_framing_auto_crop_optimization():
    mgr = ModelManager(default_model="iris-3")

    # 1. Near-vertical 9:16 (404x720 and 720x1198) -> Auto-crops to exact 2160x3840 UHD
    assert mgr.compute_smart_4k_framing(404, 720) == (2160, 3840, True)
    assert mgr.compute_smart_4k_framing(720, 1198) == (2160, 3840, True)
    filt_vert = mgr.build_filter("iris-3", scale=2, src_w=720, src_h=1198)
    assert "force_original_aspect_ratio=increase,crop=2160:3840" in filt_vert

    # 2. Near-landscape 16:9 (832x480) -> Auto-crops to exact 3840x2160 UHD
    assert mgr.compute_smart_4k_framing(832, 480) == (3840, 2160, True)

    # 3. Exact 9:16 (720x1280) and 1:1 Square (720x720) -> Exact 4K with no crop filter needed
    assert mgr.compute_smart_4k_framing(720, 1280) == (2160, 3840, False)
    assert mgr.compute_smart_4k_framing(720, 720) == (2160, 2160, False)

    # 4. Classic 4:3 / 640x464 (>15% aspect delta from 16:9) -> Preserves native uncropped aspect ratio
    assert mgr.compute_smart_4k_framing(640, 464) == (2980, 2160, False)
    filt_4_3 = mgr.build_filter("iris-3", scale=2, src_w=640, src_h=464)
    assert "crop=" not in filt_4_3
    assert "scale=w=2980:h=2160" in filt_4_3


def test_starlight_models_and_aliases():
    mgr = ModelManager(default_model="starlight-sharp")
    spec_sharp = mgr.get_model_spec()
    assert spec_sharp.name == "astrasharp"
    assert spec_sharp.is_starlight is True
    assert spec_sharp.is_neuroserver is True
    assert spec_sharp.supports_linux_deb is False
    assert spec_sharp.supports_enhancement_strength is True

    # Verify all Starlight aliases resolve cleanly
    assert ModelManager.resolve_model_name("starlight") == "slp-2.5"
    assert ModelManager.resolve_model_name("slp26") == "slp-2.5"
    assert ModelManager.resolve_model_name("starlight-mini") == "slm-1"
    assert ModelManager.resolve_model_name("starlight-fast") == "astrafast"
    assert ModelManager.resolve_model_name("starlight-hq") == "astrahq"
    assert ModelManager.resolve_model_name("starlight-creative") == "astra"

    # Starlight filter must omit GAN params (preblur/noise/estimate) and include enhancement_strength for Astra/Starlight Sharp
    filt_sharp = mgr.build_filter("starlight-sharp", scale=2, src_w=1280, src_h=720, enhancement_strength=1.3)
    assert "tvai_up=model=astrasharp:scale=2:device=0:vram=1:instances=0:parameters=enhancement_strength=1.3" in filt_sharp
    assert "preblur=" not in filt_sharp
    assert "estimate=" not in filt_sharp

    # Starlight Precise (slp-2.5) filter & neuroserver JSON
    filt_slp = mgr.build_filter("slp-2.5", src_w=1280, src_h=720)
    assert "tvai_up=model=slp-2.5:scale=1:device=0:vram=1:instances=0" in filt_slp
    assert "parameters=" not in filt_slp
    assert mgr.build_neuroserver_filters_json("slp-2.5") == '[{"model": "slp-26", "softness": 1.0}]'
    assert mgr.build_neuroserver_filters_json("astrasharp", enhancement_strength=1.3) == '[{"model": "astrasharp", "enhancement_strength": 1.3}]'
