"""Unit tests for FabricDispatcher concurrent.futures execution and dynamic reload."""

import threading
import time
import pytest
from pathlib import Path

from topaz_cluster.core.config import ClusterConfig, ModelConfig, NodeConfig, StorageConfig
from topaz_cluster.core.fabric_dispatcher import FabricDispatcher


def test_fabric_dispatcher_reads_fresh_config(tmp_path):
    cfg_file = tmp_path / "test_cluster.yaml"
    cfg_file.write_text("""
storage:
  archive_host: "test_host"
  destination_dir: "/tmp/dest"
  source_dir: "/tmp/src"
  min_free_gb: 20.0
model:
  name: "iris-3"
  scale: 2
cluster:
  staging_base_dir: "/tmp/staging"
nodes:
  mock_node:
    name: "Mock Node"
    ssh_host: "localhost"
    accelerator: "cuda"
""")

    dispatcher = FabricDispatcher(config_path=cfg_file, env_path=False)
    c1 = dispatcher.read_fresh_config()
    assert c1.storage.min_free_gb == 20.0

    # Modify YAML on disk dynamically
    cfg_file.write_text("""
storage:
  archive_host: "test_host"
  destination_dir: "/tmp/dest"
  source_dir: "/tmp/src"
  min_free_gb: 35.0
model:
  name: "iris-3"
  scale: 2
cluster:
  staging_base_dir: "/tmp/staging"
nodes:
  mock_node:
    name: "Mock Node"
    ssh_host: "localhost"
    accelerator: "cuda"
""")

    # Fabric re-reads fresh config prior to cycle
    c2 = dispatcher.read_fresh_config()
    assert c2.storage.min_free_gb == 35.0


def test_fabric_dispatcher_storage_gate_blocks(monkeypatch, tmp_path):
    cfg_file = tmp_path / "gated_cluster.yaml"
    cfg_file.write_text("""
storage:
  archive_host: "test_host"
  destination_dir: "/tmp/dest"
  source_dir: "/tmp/src"
  min_free_gb: 50.0
model:
  name: "iris-3"
  scale: 2
nodes:
  mock_node:
    name: "Mock Node"
    ssh_host: "localhost"
    accelerator: "cuda"
""")

    dispatcher = FabricDispatcher(config_path=cfg_file, env_path=False)

    # Mock storage guardian reporting insufficient space (10 GB < 50 GB threshold)
    monkeypatch.setattr("topaz_cluster.core.storage_guardian.StorageGuardian.is_safe", lambda self: (False, 10.0))

    started = dispatcher.start(run_preflight=False)
    # Must block startup when storage is unsafe!
    assert started is False


def test_fabric_dispatcher_threadpool_lifecycle(monkeypatch, tmp_path):
    cfg_file = tmp_path / "lifecycle_cluster.yaml"
    cfg_file.write_text("""
storage:
  archive_host: "test_host"
  destination_dir: "/tmp/dest"
  source_dir: "/tmp/src"
  min_free_gb: 10.0
model:
  name: "iris-3"
  scale: 2
nodes:
  node_a:
    name: "Node A"
    ssh_host: "localhost"
    accelerator: "cuda"
  node_b:
    name: "Node B"
    ssh_host: "localhost"
    accelerator: "ane"
""")

    dispatcher = FabricDispatcher(config_path=cfg_file, env_path=False)

    # Mock storage safe
    monkeypatch.setattr("topaz_cluster.core.storage_guardian.StorageGuardian.is_safe", lambda self: (True, 100.0))

    run_calls = []

    def mock_run_fabric_worker(node_id):
        run_calls.append(node_id)
        while not dispatcher._stop_event.is_set():
            time.sleep(0.05)
        return 1

    monkeypatch.setattr(dispatcher, "_run_fabric_worker", mock_run_fabric_worker)

    started = dispatcher.start(run_preflight=False)
    assert started is True
    assert len(dispatcher.futures) == 2
    assert "node_a" in dispatcher.futures
    assert "node_b" in dispatcher.futures

    # Allow threads to begin
    time.sleep(0.15)
    assert "node_a" in run_calls
    assert "node_b" in run_calls

    # Stop gracefully
    dispatcher.stop()
    assert dispatcher._stop_event.is_set()
    assert all(fut.done() for fut in dispatcher.futures.values())


def test_fabric_dispatcher_dynamic_hot_join(monkeypatch, tmp_path):
    cfg_file = tmp_path / "hot_join_cluster.yaml"
    cfg_file.write_text("""
storage:
  archive_host: "test_host"
  destination_dir: "/tmp/dest"
  source_dir: "/tmp/src"
  min_free_gb: 10.0
model:
  name: "iris-3"
  scale: 2
nodes:
  node_a:
    enabled: true
    name: "Node A"
    ssh_host: "localhost"
    accelerator: "cuda"
  node_b:
    enabled: false
    name: "Node B"
    ssh_host: "localhost"
    accelerator: "ane"
""")

    dispatcher = FabricDispatcher(config_path=cfg_file, env_path=False)
    monkeypatch.setattr("topaz_cluster.core.storage_guardian.StorageGuardian.is_safe", lambda self: (True, 100.0))

    run_calls = []

    def mock_run_fabric_worker(node_id):
        run_calls.append(node_id)
        while not dispatcher._stop_event.is_set():
            time.sleep(0.05)
        return 1

    monkeypatch.setattr(dispatcher, "_run_fabric_worker", mock_run_fabric_worker)

    started = dispatcher.start(run_preflight=False)
    assert started is True
    assert list(dispatcher.futures.keys()) == ["node_a"]

    # Dynamically enable node_b AND add a brand-new node_c in YAML while fabric is running
    cfg_file.write_text("""
storage:
  archive_host: "test_host"
  destination_dir: "/tmp/dest"
  source_dir: "/tmp/src"
  min_free_gb: 10.0
model:
  name: "iris-3"
  scale: 2
nodes:
  node_a:
    enabled: true
    name: "Node A"
    ssh_host: "localhost"
    accelerator: "cuda"
  node_b:
    enabled: true
    name: "Node B"
    ssh_host: "localhost"
    accelerator: "ane"
  node_c:
    enabled: true
    name: "Node C"
    ssh_host: "localhost"
    accelerator: "linux_cuda"
""")

    joined = dispatcher.sync_dynamic_nodes()
    assert sorted(joined) == ["node_b", "node_c"]
    assert "node_b" in dispatcher.futures
    assert "node_c" in dispatcher.futures

    time.sleep(0.15)
    assert "node_b" in run_calls
    assert "node_c" in run_calls

    dispatcher.stop()
