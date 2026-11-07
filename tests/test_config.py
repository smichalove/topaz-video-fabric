"""Unit tests for ClusterConfig and .env loader."""

import os
import pytest
from pathlib import Path

from topaz_cluster.core.config import (
    ClusterConfig,
    ModelConfig,
    NodeConfig,
    StorageConfig,
    load_cluster_config,
)


def test_default_config_loading():
    cfg = load_cluster_config()
    assert isinstance(cfg, ClusterConfig)
    assert isinstance(cfg.storage, StorageConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert cfg.storage.min_free_gb >= 25.0
    assert cfg.model.name == "iris-3"
    assert cfg.model.allow_crop is False
    assert len(cfg.nodes) >= 2


def test_env_overrides(tmp_path, monkeypatch):
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text("""
storage:
  archive_host: "test@127.0.0.1"
  destination_dir: "/tmp/dest"
  source_dir: "/tmp/src"
  min_free_gb: 10.0
model:
  name: "prob-4"
  scale: 2
cluster:
  staging_base_dir: "/tmp/staging"
nodes:
  test_node:
    name: "Test Node"
    ssh_host: "test_host"
    accelerator: "cuda"
""")

    monkeypatch.setenv("TOPAZ_SOURCE_DIR", "/override/source")
    monkeypatch.setenv("TOPAZ_MIN_FREE_GB", "50.0")

    cfg = load_cluster_config(config_path=custom_yaml, env_path=False)
    assert cfg.storage.source_dir == "/override/source"
    assert cfg.storage.min_free_gb == 50.0
    assert cfg.storage.destination_dir == "/tmp/dest"
    assert "test_node" in cfg.nodes
