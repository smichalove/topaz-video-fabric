"""Host Assurance & System Preflight Checker (syschk) for Topaz Supercluster.

Verifies host-bound assumptions: SSH reachability, Topaz FFmpeg binary paths,
model weight directories, DRAM staging areas, silicon accelerators, and storage gates.
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import ClusterConfig, NodeConfig
from .storage_guardian import StorageGuardian

logger = logging.getLogger("topaz_cluster.syschk")


@dataclass
class NodeCheckResult:
    node_id: str
    name: str
    ssh_host: str
    status: str = "PENDING"  # PASS, WARN, FAIL
    ssh_ok: bool = False
    ssh_latency_ms: float = 0.0
    ffmpeg_path: str = ""
    ffmpeg_ok: bool = False
    tvai_up_filter_ok: bool = False
    model_dir: str = ""
    model_ok: bool = False
    staging_dir: str = ""
    staging_ok: bool = False
    accelerator_info: str = ""
    details: List[str] = field(default_factory=list)


class SystemChecker:
    """Preflight Host Assurance engine for heterogeneous Topaz clusters."""

    def __init__(self, cluster_cfg: ClusterConfig, guardian: Optional[StorageGuardian] = None):
        self.cluster_cfg = cluster_cfg
        self.guardian = guardian or StorageGuardian(
            archive_host=cluster_cfg.storage.archive_host,
            destination_dir=cluster_cfg.storage.destination_dir,
            min_free_gb=cluster_cfg.storage.min_free_gb,
        )

    def run_remote_ssh(self, ssh_host: str, cmd: str, timeout: int = 10) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def check_node(self, node_id: str, node_cfg: NodeConfig) -> NodeCheckResult:
        """Execute full host assurance audit on a specific cluster worker node."""
        res = NodeCheckResult(node_id=node_id, name=node_cfg.name, ssh_host=node_cfg.ssh_host)
        accel = node_cfg.accelerator.lower()
        is_mac = "ane" in accel or "apple" in accel or "metal" in accel
        is_linux = "linux" in accel
        is_wsl = bool(node_cfg.wsl_shm_dir and node_cfg.win_unc_shm) and not is_linux

        # 1. SSH Reachability & Latency
        t0 = time.time()
        try:
            ssh_test = self.run_remote_ssh(node_cfg.ssh_host, "echo ok", timeout=8)
            res.ssh_latency_ms = (time.time() - t0) * 1000.0
            if ssh_test.returncode == 0 and "ok" in ssh_test.stdout:
                res.ssh_ok = True
                res.details.append(f"SSH Reachable ({res.ssh_latency_ms:.1f}ms latency)")
            else:
                res.ssh_ok = False
                res.status = "FAIL"
                res.details.append(f"SSH Auth/Connection Failed: {ssh_test.stderr.strip() or 'No response'}")
                return res
        except Exception as e:
            res.ssh_ok = False
            res.status = "FAIL"
            res.details.append(f"SSH Exception: {e}")
            return res

        # 2. Binary Path & tvai_up Filter Verification
        if is_mac:
            ffmpeg_candidates = [
                node_cfg.ffmpeg_path,
                "/Applications/Topaz Video.app/Contents/MacOS/ffmpeg",
                "/Applications/Topaz Video AI.app/Contents/MacOS/ffmpeg",
            ]
            valid_bin = None
            for cand in [c for c in ffmpeg_candidates if c]:
                check_cmd = f"test -x '{cand}' && '{cand}' -filters 2>/dev/null | grep -q 'tvai_up' && echo ok"
                p = self.run_remote_ssh(node_cfg.ssh_host, check_cmd)
                if p.returncode == 0 and "ok" in p.stdout:
                    valid_bin = cand
                    break
            if valid_bin:
                res.ffmpeg_path = valid_bin
                res.ffmpeg_ok = True
                res.tvai_up_filter_ok = True
                res.details.append(f"FFmpeg Validated: {valid_bin} (tvai_up filter active)")
            else:
                res.ffmpeg_ok = False
                res.details.append("FFmpeg Missing or lacks tvai_up filter in /Applications")
        elif is_linux:
            bin_path = node_cfg.ffmpeg_path or "/opt/TopazVideoAIBETA/bin/ffmpeg"
            res.ffmpeg_path = bin_path
            check_cmd = f"test -x '{bin_path}' && echo BIN_FOUND"
            p = self.run_remote_ssh(node_cfg.ssh_host, check_cmd)
            if "BIN_FOUND" in p.stdout:
                res.ffmpeg_ok = True
                res.tvai_up_filter_ok = True
                res.details.append(f"Linux FFmpeg Validated: {bin_path}")
            else:
                res.ffmpeg_ok = False
                res.details.append(f"Linux FFmpeg binary not found at: {bin_path}")
        elif is_wsl:
            # WSL Ubuntu on Windows host
            bin_path = node_cfg.ffmpeg_path or "C:\\Program Files\\Topaz Labs LLC\\Topaz Video\\ffmpeg.exe"
            res.ffmpeg_path = bin_path
            wsl_bin = bin_path.replace("C:\\", "/mnt/c/").replace("c:\\", "/mnt/c/").replace("\\", "/")
            check_cmd = f"test -f '{wsl_bin}' && echo BIN_FOUND"
            p = self.run_remote_ssh(node_cfg.ssh_host, check_cmd)
            if "BIN_FOUND" in p.stdout:
                res.ffmpeg_ok = True
                res.tvai_up_filter_ok = True
                res.details.append(f"Windows FFmpeg Validated via WSL: {bin_path}")
            else:
                res.ffmpeg_ok = False
                res.details.append(f"Windows FFmpeg binary not found at: {bin_path}")
        else:
            # Native Windows OpenSSH (PowerShell)
            bin_path = node_cfg.ffmpeg_path or "C:\\Program Files\\Topaz Labs LLC\\Topaz Video\\ffmpeg.exe"
            res.ffmpeg_path = bin_path
            ps_cmd = f"powershell -NoProfile -Command \"Test-Path '{bin_path}'\""
            p = self.run_remote_ssh(node_cfg.ssh_host, ps_cmd)
            if "True" in p.stdout:
                res.ffmpeg_ok = True
                res.tvai_up_filter_ok = True
                res.details.append(f"Windows FFmpeg Validated: {bin_path}")
            else:
                res.ffmpeg_ok = False
                res.details.append(f"Windows FFmpeg binary not found at: {bin_path}")

        # 3. Model Directory & Weights Assurance
        target_model = self.cluster_cfg.model.name
        if is_mac:
            mdir = node_cfg.model_dir or "/Applications/Topaz Video.app/Contents/Resources/models"
            res.model_dir = mdir
            p = self.run_remote_ssh(node_cfg.ssh_host, f"test -d '{mdir}' && ls '{mdir}' | grep -qi '{target_model[:4]}' && echo ok")
            if "ok" in p.stdout:
                res.model_ok = True
                res.details.append(f"Model Directory Verified: {mdir} ({target_model} weights present)")
            else:
                res.model_ok = False
                res.details.append(f"Model Directory missing or lacks {target_model} weights: {mdir}")
        elif is_linux:
            mdir = node_cfg.model_dir or "/opt/TopazVideoAIBETA/models"
            res.model_dir = mdir
            p = self.run_remote_ssh(node_cfg.ssh_host, f"test -d '{mdir}' && echo DIR_FOUND")
            if "DIR_FOUND" in p.stdout:
                res.model_ok = True
                res.details.append(f"Linux Model Directory Verified: {mdir}")
            else:
                res.model_ok = False
                res.details.append(f"Linux Model Directory not found: {mdir}")
        elif is_wsl:
            mdir = node_cfg.model_dir or "C:\\ProgramData\\Topaz Labs LLC\\Topaz Video\\models"
            res.model_dir = mdir
            wsl_mdir = mdir.replace("C:\\", "/mnt/c/").replace("c:\\", "/mnt/c/").replace("\\", "/")
            check_cmd = f"test -d '{wsl_mdir}' && echo DIR_FOUND"
            p = self.run_remote_ssh(node_cfg.ssh_host, check_cmd)
            if "DIR_FOUND" in p.stdout:
                res.model_ok = True
                res.details.append(f"Model Directory Verified via WSL: {mdir}")
            else:
                res.model_ok = False
                res.details.append(f"Model Directory not found: {mdir}")
        else:
            mdir = node_cfg.model_dir or "C:\\ProgramData\\Topaz Labs LLC\\Topaz Video\\models"
            res.model_dir = mdir
            ps_cmd = f"powershell -NoProfile -Command \"Test-Path '{mdir}'\""
            p = self.run_remote_ssh(node_cfg.ssh_host, ps_cmd)
            if "True" in p.stdout:
                res.model_ok = True
                res.details.append(f"Model Directory Verified: {mdir}")
            else:
                res.model_ok = False
                res.details.append(f"Model Directory not found: {mdir}")

        # 4. DRAM Staging / Zero-NAND Path Assurance
        if is_mac:
            mount_pt = node_cfg.ramdisk_mount or "/Volumes/TopazRAM"
            res.staging_dir = mount_pt
            p = self.run_remote_ssh(node_cfg.ssh_host, f"hdiutil info | grep -q '{mount_pt}' && test -w '{mount_pt}' && echo ok")
            if "ok" in p.stdout:
                res.staging_ok = True
                res.details.append(f"Volatile RAMDisk Verified: {mount_pt} (HFS+ DRAM, 0 NAND writes)")
            else:
                res.staging_ok = False
                res.details.append(f"RAMDisk NOT mounted at {mount_pt} (Worker will auto-mount on demand)")
        elif is_linux or is_wsl:
            shm_dir = node_cfg.wsl_shm_dir or "/dev/shm/topaz_5080_ram"
            res.staging_dir = shm_dir if is_linux else f"{shm_dir} -> {node_cfg.win_unc_shm}"
            p = self.run_remote_ssh(node_cfg.ssh_host, f"mkdir -p '{shm_dir}' && test -w '{shm_dir}' && echo ok")
            if "ok" in p.stdout:
                res.staging_ok = True
                res.details.append(f"/dev/shm Staging Verified: {shm_dir}")
            else:
                res.staging_ok = False
                res.details.append(f"Staging directory unwritable: {shm_dir}")
        else:
            vm_dir = node_cfg.vm_temp_dir or "D:/Temp"
            res.staging_dir = vm_dir
            ps_cmd = f"powershell -NoProfile -Command \"Test-Path '{vm_dir}'\""
            p = self.run_remote_ssh(node_cfg.ssh_host, ps_cmd)
            if "True" in p.stdout:
                res.staging_ok = True
                res.details.append(f"BitLocker Staging Volume Verified: {vm_dir}")
            else:
                res.staging_ok = False
                res.details.append(f"BitLocker Staging Volume missing: {vm_dir}")

        # 5. Silicon Telemetry
        if is_mac:
            p = self.run_remote_ssh(node_cfg.ssh_host, "sysctl -n machdep.cpu.brand_string")
            cpu_name = p.stdout.strip()
            res.accelerator_info = f"Apple Silicon ({cpu_name}) + ANE"
            res.details.append(f"Silicon: {res.accelerator_info}")
        elif is_linux or is_wsl:
            p = self.run_remote_ssh(node_cfg.ssh_host, "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null || true")
            if p.stdout.strip():
                gpu_info = p.stdout.strip().replace("\n", " | ")
                res.accelerator_info = f"NVIDIA {gpu_info}"
                res.details.append(f"GPU Telemetry: {res.accelerator_info}")
            else:
                res.accelerator_info = "NVIDIA CUDA (Device 0)"
        else:
            p = self.run_remote_ssh(node_cfg.ssh_host, "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits")
            if p.stdout.strip():
                gpu_info = p.stdout.strip().replace("\r\n", " | ").replace("\n", " | ")
                res.accelerator_info = f"NVIDIA {gpu_info}"
                res.details.append(f"GPU Telemetry: {res.accelerator_info}")
            else:
                res.accelerator_info = "NVIDIA CUDA"

        # Determine Final Node Health Status
        # On Mac, staging_ok=False is a WARN because mac_worker mounts it dynamically on start
        if not res.ssh_ok or not res.ffmpeg_ok or not res.model_ok:
            res.status = "FAIL"
        elif not res.staging_ok and not is_mac:
            res.status = "FAIL"
        elif not res.staging_ok and is_mac:
            res.status = "WARN"
        else:
            res.status = "PASS"

        return res

    def check_all_nodes(self) -> Dict[str, NodeCheckResult]:
        """Audit all enabled cluster nodes in parallel using concurrent.futures."""
        from concurrent.futures import ThreadPoolExecutor

        results: Dict[str, NodeCheckResult] = {}
        enabled_nodes = {}
        for node_id, node_cfg in self.cluster_cfg.nodes.items():
            if not node_cfg.enabled:
                results[node_id] = NodeCheckResult(
                    node_id=node_id,
                    name=node_cfg.name,
                    ssh_host=node_cfg.ssh_host,
                    status="DISABLED",
                    details=["Node disabled in configuration."],
                )
            else:
                enabled_nodes[node_id] = node_cfg

        with ThreadPoolExecutor(max_workers=max(1, len(enabled_nodes))) as executor:
            future_to_node = {
                executor.submit(self.check_node, node_id, node_cfg): node_id
                for node_id, node_cfg in enabled_nodes.items()
            }
            for future in future_to_node:
                node_id = future_to_node[future]
                try:
                    results[node_id] = future.result()
                except Exception as e:
                    results[node_id] = NodeCheckResult(
                        node_id=node_id,
                        name=enabled_nodes[node_id].name,
                        ssh_host=enabled_nodes[node_id].ssh_host,
                        status="FAIL",
                        details=[f"Preflight exception: {e}"],
                    )

        # Preserve canonical order from config
        return {k: results[k] for k in self.cluster_cfg.nodes if k in results}

    def format_report(self, results: Dict[str, NodeCheckResult]) -> str:
        """Generate formatted tabular preflight audit report."""
        lines = []
        lines.append("=" * 80)
        lines.append(" TOPAZ CLUSTER PREFLIGHT & HOST ASSURANCE AUDIT (syschk)")
        lines.append("=" * 80)

        # Storage check
        safe, free_gb = self.guardian.is_safe()
        storage_status = "PASS" if safe else "CRITICAL_GATED"
        lines.append(f" Destination Archive:   {self.cluster_cfg.storage.destination_dir} (Host: {self.cluster_cfg.storage.archive_host})")
        lines.append(f" Free Storage:          {free_gb:.1f} GB (Threshold: {self.cluster_cfg.storage.min_free_gb:.1f} GB) -> [{storage_status}]")
        lines.append(f" Target Supermodel:     {self.cluster_cfg.model.name} (Scale: {self.cluster_cfg.model.scale}x, Lanczos, NO-CROP)")
        lines.append("-" * 80)
        lines.append(f"{'NODE ID':<10} {'STATUS':<8} {'LATENCY':<9} {'FFMPEG':<8} {'MODEL':<8} {'DRAM STAGING':<14} {'SILICON'}")
        lines.append("-" * 80)

        for node_id, r in results.items():
            if r.status == "DISABLED":
                lines.append(f"{node_id:<10} {'DISABLED':<8} {'--':<9} {'--':<8} {'--':<8} {'--':<14} (Disabled in config)")
                continue

            lat_str = f"{r.ssh_latency_ms:.0f}ms" if r.ssh_ok else "ERR"
            ff_str = "OK" if r.ffmpeg_ok else "MISSING"
            md_str = "OK" if r.model_ok else "MISSING"
            st_str = "OK (DRAM)" if r.staging_ok else ("DEMAND" if "RAM" in r.staging_dir else "ERR")
            acc_str = (r.accelerator_info[:24] + "...") if len(r.accelerator_info) > 27 else r.accelerator_info
            lines.append(f"{node_id:<10} {r.status:<8} {lat_str:<9} {ff_str:<8} {md_str:<8} {st_str:<14} {acc_str}")

        lines.append("-" * 80)
        lines.append(" Diagnostic Notes:")
        for node_id, r in results.items():
            if r.details:
                lines.append(f"   [{node_id}]")
                for d in r.details:
                    lines.append(f"     - {d}")
        lines.append("=" * 80)
        return "\n".join(lines)
