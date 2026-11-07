"""Unit tests for SystemChecker host assurance audit."""

import subprocess
import pytest
from pathlib import Path

from topaz_cluster.core.config import ClusterConfig, NodeConfig, StorageConfig, ModelConfig
from topaz_cluster.core.syschk import SystemChecker, NodeCheckResult


def test_syschk_node_pass_mocked(monkeypatch):
    cfg = ClusterConfig(
        storage=StorageConfig(archive_host="localhost", destination_dir="/tmp", min_free_gb=10.0),
        model=ModelConfig(name="iris-3", scale=2),
        nodes={
            "mock_mac": NodeConfig(
                name="Mock Mac",
                ssh_host="mock_mac_host",
                accelerator="ane",
                ffmpeg_path="/Applications/Topaz Video.app/Contents/MacOS/ffmpeg",
                model_dir="/Applications/Topaz Video.app/Contents/Resources/models",
                ramdisk_mount="/Volumes/TopazRAM",
            )
        },
    )

    chk = SystemChecker(cfg)

    def mock_ssh(host, cmd, timeout=10):
        if "echo ok" in cmd:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n")
        if "test -x" in cmd:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n")
        if "test -d" in cmd:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n")
        if "hdiutil" in cmd:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n")
        if "sysctl" in cmd:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="Apple M4\n")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n")

    monkeypatch.setattr(chk, "run_remote_ssh", mock_ssh)

    res = chk.check_node("mock_mac", cfg.nodes["mock_mac"])
    assert res.status == "PASS"
    assert res.ssh_ok is True
    assert res.ffmpeg_ok is True
    assert res.model_ok is True
    assert res.staging_ok is True
    assert "Apple Silicon" in res.accelerator_info


def test_syschk_node_fail_when_binary_missing(monkeypatch):
    cfg = ClusterConfig(
        storage=StorageConfig(archive_host="localhost", destination_dir="/tmp", min_free_gb=10.0),
        model=ModelConfig(name="iris-3", scale=2),
        nodes={
            "broken_win": NodeConfig(
                name="Broken Windows Node",
                ssh_host="broken_host",
                accelerator="cuda",
                wsl_shm_dir="/dev/shm/ram",
                win_unc_shm="\\\\wsl\\ram",
            )
        },
    )

    chk = SystemChecker(cfg)

    def mock_ssh(host, cmd, timeout=10):
        if "echo ok" in cmd:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n")
        if "Test-Path" in cmd:
            # Binary missing
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="False\n")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="")

    monkeypatch.setattr(chk, "run_remote_ssh", mock_ssh)

    res = chk.check_node("broken_win", cfg.nodes["broken_win"])
    assert res.status == "FAIL"
    assert res.ffmpeg_ok is False


def test_syschk_format_report():
    cfg = ClusterConfig(
        storage=StorageConfig(archive_host="localhost", destination_dir="/tmp", min_free_gb=10.0),
        model=ModelConfig(name="iris-3", scale=2),
        nodes={},
    )
    chk = SystemChecker(cfg)
    results = {
        "node_1": NodeCheckResult(
            node_id="node_1",
            name="Node 1",
            ssh_host="host_1",
            status="PASS",
            ssh_ok=True,
            ssh_latency_ms=15.2,
            ffmpeg_ok=True,
            model_ok=True,
            staging_ok=True,
            accelerator_info="NVIDIA RTX 5080",
            details=["Everything optimal"],
        )
    }
    report = chk.format_report(results)
    assert "TOPAZ CLUSTER PREFLIGHT & HOST ASSURANCE AUDIT" in report
    assert "node_1" in report
    assert "PASS" in report
    assert "Everything optimal" in report
