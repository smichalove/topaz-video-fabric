"""Configuration loader for Topaz Cluster Orchestrator with YAML and .env support."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from dotenv import dotenv_values

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "cluster_config.yaml"
LOCAL_CONFIG_PATH = CONFIG_DIR / "cluster_config.local.yaml"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


@dataclass
class StorageConfig:
    archive_host: str = "user@archive-host"
    destination_dir: str = "/path/to/destination"
    source_dir: str = "/path/to/source"
    min_free_gb: float = 25.0
    check_interval_seconds: int = 15


@dataclass
class ModelConfig:
    name: str = "iris-3"
    scale: int = 2
    target_width: int = 3840
    target_height: int = 2160
    scaling_filter: str = "lanczos"
    blur: float = -0.10
    blend: float = 0.20
    allow_crop: bool = False


@dataclass
class NodeConfig:
    name: str
    ssh_host: str
    accelerator: str
    device_id: int = 0
    instances: int = 1
    enabled: bool = True
    priority_tiers: List[str] = field(default_factory=list)
    wsl_shm_dir: Optional[str] = None
    win_unc_shm: Optional[str] = None
    vm_temp_dir: Optional[str] = None
    local_hb_dir: Optional[str] = None
    ramdisk_mount: Optional[str] = None
    ramdisk_size_mb: int = 3500
    ffmpeg_path: Optional[str] = None
    model_dir: Optional[str] = None


@dataclass
class ClusterConfig:
    storage: StorageConfig
    model: ModelConfig
    nodes: Dict[str, NodeConfig]
    staging_base_dir: Path = Path("/dev/shm/topaz_stage")
    claims_dir: Path = Path("/dev/shm/topaz_stage/claimed_stems")
    log_dir: Path = Path("/dev/shm/topaz_stage")


def load_cluster_config(config_path: Optional[Path] = None, env_path: Optional[Union[Path, bool]] = None) -> ClusterConfig:
    """Load configuration from YAML and apply .env environment variable overrides."""
    env_vars: Dict[str, Any] = {}
    if env_path is not False:
        target_env = env_path if isinstance(env_path, Path) else (ENV_PATH if ENV_PATH.is_file() else None)
        if target_env and Path(target_env).is_file():
            env_vars = dict(dotenv_values(target_env))

    def get_val(key: str, fallback: Any) -> Any:
        if key in os.environ:
            return os.environ[key]
        if key in env_vars and env_vars[key] is not None:
            return env_vars[key]
        return fallback

    if config_path is not None:
        cfg_file = config_path
    elif LOCAL_CONFIG_PATH.is_file():
        cfg_file = LOCAL_CONFIG_PATH
    else:
        cfg_file = DEFAULT_CONFIG_PATH
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Cluster config file not found: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    storage_data = data.get("storage", {})
    storage_cfg = StorageConfig(
        archive_host=str(get_val("TOPAZ_ARCHIVE_HOST", storage_data.get("archive_host", "user@archive-host"))),
        destination_dir=str(get_val("TOPAZ_DEST_DIR", storage_data.get("destination_dir", "/path/to/destination"))),
        source_dir=str(get_val("TOPAZ_SOURCE_DIR", storage_data.get("source_dir", "/path/to/source"))),
        min_free_gb=float(get_val("TOPAZ_MIN_FREE_GB", storage_data.get("min_free_gb", 25.0))),
        check_interval_seconds=int(storage_data.get("check_interval_seconds", 15)),
    )

    model_data = data.get("model", {})
    model_cfg = ModelConfig(
        name=str(get_val("TOPAZ_MODEL", model_data.get("name", "iris-3"))),
        scale=int(model_data.get("scale", 2)),
        target_width=int(model_data.get("target_width", 3840)),
        target_height=int(model_data.get("target_height", 2160)),
        scaling_filter=str(model_data.get("scaling_filter", "lanczos")),
        blur=float(model_data.get("blur", -0.10)),
        blend=float(model_data.get("blend", 0.20)),
        allow_crop=bool(model_data.get("allow_crop", False)),
    )

    cluster_data = data.get("cluster", {})
    staging_base = Path(cluster_data.get("staging_base_dir", "/dev/shm/topaz_stage"))
    claims_dir = Path(cluster_data.get("claims_dir", "/dev/shm/topaz_stage/claimed_stems"))
    log_dir = Path(cluster_data.get("log_dir", "/dev/shm/topaz_stage"))

    nodes: Dict[str, NodeConfig] = {}
    nodes_data = data.get("nodes", {})
    for node_id, node_dict in nodes_data.items():
        nodes[node_id] = NodeConfig(
            name=node_dict.get("name", node_id),
            ssh_host=node_dict.get("ssh_host", ""),
            accelerator=node_dict.get("accelerator", "cuda"),
            device_id=int(node_dict.get("device_id", 0)),
            instances=int(node_dict.get("instances", 1)),
            enabled=bool(node_dict.get("enabled", True)),
            priority_tiers=list(node_dict.get("priority_tiers", [])),
            wsl_shm_dir=node_dict.get("wsl_shm_dir"),
            win_unc_shm=node_dict.get("win_unc_shm"),
            vm_temp_dir=node_dict.get("vm_temp_dir"),
            local_hb_dir=node_dict.get("local_hb_dir"),
            ramdisk_mount=node_dict.get("ramdisk_mount"),
            ramdisk_size_mb=int(node_dict.get("ramdisk_size_mb", 3500)),
            ffmpeg_path=node_dict.get("ffmpeg_path"),
            model_dir=node_dict.get("model_dir"),
        )

    return ClusterConfig(
        storage=storage_cfg,
        model=model_cfg,
        nodes=nodes,
        staging_base_dir=staging_base,
        claims_dir=claims_dir,
        log_dir=log_dir,
    )
