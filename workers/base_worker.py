"""Base Worker class defining the unified cluster worker lifecycle."""

import abc
import logging
import subprocess
import threading
import time
from typing import Optional, Tuple, Union
from pathlib import Path

from ..core.config import ClusterConfig, NodeConfig
from ..core.model_manager import ModelManager
from ..core.queue_manager import JobItem, QueueManager
from ..core.storage_guardian import StorageGuardian


class BaseWorker(abc.ABC, threading.Thread):
    def __init__(
        self,
        node_id: str,
        node_cfg: NodeConfig,
        cluster_cfg: ClusterConfig,
        queue_mgr: QueueManager,
        guardian: StorageGuardian,
        config_path: Optional[Path] = None,
        env_path: Optional[Union[Path, bool]] = None,
    ):
        threading.Thread.__init__(self, name=f"Worker-{node_id}", daemon=True)
        self.node_id = node_id
        self.node_cfg = node_cfg
        self.cluster_cfg = cluster_cfg
        self.queue_mgr = queue_mgr
        self.guardian = guardian
        self.config_path = config_path
        self.env_path = env_path
        self.logger = logging.getLogger(f"topaz_cluster.{node_id}")
        self._stop_event = threading.Event()
        self.completed_count = 0

    def reload_config(self) -> None:
        """Reload configuration from disk (.config and .env) dynamically prior to each fabric cycle."""
        try:
            import importlib
            from ..core import config as cfg_mod
            from ..core import queue_manager as qm_mod
            importlib.reload(qm_mod)
            fresh_cfg = cfg_mod.load_cluster_config(config_path=self.config_path, env_path=self.env_path)
            self.cluster_cfg = fresh_cfg
            if self.node_id in fresh_cfg.nodes:
                self.node_cfg = fresh_cfg.nodes[self.node_id]
            self.guardian.min_free_gb = fresh_cfg.storage.min_free_gb
            self.guardian.destination_dir = fresh_cfg.storage.destination_dir
            self.guardian.archive_host = fresh_cfg.storage.archive_host
            self.queue_mgr.__class__ = qm_mod.QueueManager
            self.queue_mgr.source_dir = fresh_cfg.storage.source_dir
            self.queue_mgr.destination_dir = fresh_cfg.storage.destination_dir
            self.queue_mgr.archive_host = fresh_cfg.storage.archive_host
            self.queue_mgr.claims_dir = fresh_cfg.claims_dir
            if not hasattr(self.queue_mgr, "_last_source_sync"):
                self.queue_mgr._last_source_sync = 0.0
        except Exception as e:
            self.logger.warning("[%s] Failed to reload config dynamically: %s", self.node_id, e)

    def stop(self) -> None:
        """Signal worker to gracefully stop after current job finishes."""
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        """Worker lifecycle loop."""
        self.logger.info("Worker %s started on %s.", self.node_cfg.name, self.node_cfg.ssh_host)
        processed_stems = set()
        idx = 0

        while not self.is_stopped():
            # Reload config & manifest dynamically prior to each job cycle
            self.reload_config()
            all_jobs = self.queue_mgr.load_all_jobs()
            node_queue = self.queue_mgr.build_node_queue(self.node_cfg.priority_tiers, all_jobs)
            total_jobs = len(node_queue)

            if not self.node_cfg.enabled:
                self.logger.info("[%s] Node marked disabled in dynamic config. Pausing 10s...", self.node_id)
                time.sleep(10)
                continue

            model_spec = ModelManager(self.cluster_cfg.model.name).get_model_spec()
            if self.node_cfg.accelerator.lower() == "linux_cuda" and not model_spec.supports_linux_deb:
                self.logger.warning(
                    "[%s] Local Starlight model '%s' is not supported on native Linux .deb distros (Windows Virtualization with PCIe GPU passthrough via TopazWindowsWorker or macOS TopazMacWorker is required). Skipping Linux worker.",
                    self.node_id,
                    model_spec.name,
                )
                break

            # Find next unclaimed, uncompleted job from the dynamically reloaded queue
            done_stems = self.queue_mgr.fetch_completed_stems()
            next_job: Optional[JobItem] = None
            for cand_idx, cand in enumerate(node_queue, start=1):
                if cand.stem in done_stems or cand.stem in processed_stems:
                    continue
                if self.queue_mgr.try_claim(cand.stem, worker_tag=self.node_id):
                    next_job = cand
                    idx = cand_idx
                    break

            if next_job is None:
                break

            job = next_job
            processed_stems.add(job.stem)

            # 1. Check destination disk guardian
            self.guardian.wait_until_safe(worker_name=self.node_id, stop_event=self._stop_event)
            if self.is_stopped():
                self.queue_mgr.release_claim(job.stem)
                break

            self.logger.info(
                "[%s #%d | %d/%d] Claimed job '%s' (%dx%d, %s). Processing...",
                self.node_id,
                self.completed_count + 1,
                idx,
                total_jobs,
                job.filename,
                job.width,
                job.height,
                job.tier,
            )

            # 4. Execute node-specific processing (stage -> render -> deliver)
            try:
                success, out_bytes, elapsed, fps = self.process_job(job)
                if success and out_bytes > 1048576:
                    self.completed_count += 1
                    model_tag = ModelManager.get_output_tag(self.cluster_cfg.model.name)
                    self.logger.info(
                        "[%s #%d] VERIFIED DELIVERABLE: %s_%s.mp4 (%.1f MiB | %.2f FPS | %dm%02ds)",
                        self.node_id,
                        self.completed_count,
                        job.stem,
                        model_tag,
                        out_bytes / (1024 * 1024),
                        fps,
                        int(elapsed) // 60,
                        int(elapsed) % 60,
                    )
                    if self.cluster_cfg.storage.delete_source_after_render:
                        src_file = f"{self.cluster_cfg.storage.source_dir}/{job.filename}"
                        subprocess.run(
                            ["ssh", "-n", "-o", "BatchMode=yes", self.cluster_cfg.storage.archive_host, f'rm -f "{src_file}"'],
                            check=False,
                        )
                        self.logger.info("[%s] Removed verified source file: %s", self.node_id, src_file)
                else:
                    self.logger.warning(
                        "[%s] Job '%s' failed or byte count invalid (%dB). Releasing lock for retry.",
                        self.node_id,
                        job.filename,
                        out_bytes,
                    )
                    self.cleanup_job(job)
                    self.queue_mgr.release_claim(job.stem)
            except Exception as e:
                self.logger.error("[%s] Exception processing '%s': %s", self.node_id, job.filename, e, exc_info=True)
                self.cleanup_job(job)
                self.queue_mgr.release_claim(job.stem)

        self.logger.info("[%s] Worker finished queue. Completed %d jobs.", self.node_id, self.completed_count)

    @abc.abstractmethod
    def process_job(self, job: JobItem) -> Tuple[bool, int, float, float]:
        """Node-specific staging, rendering, delivery, and verification.
        
        Returns (success: bool, delivered_bytes: int, elapsed_sec: float, fps: float).
        """
        pass

    @abc.abstractmethod
    def cleanup_job(self, job: JobItem) -> None:
        """Clean up any partial or temporary files on failure."""
        pass
