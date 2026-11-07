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
