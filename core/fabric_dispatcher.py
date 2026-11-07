"""Compute Fabric Dispatcher for Topaz Supercluster.

Distributes work across heterogeneous worker nodes using concurrent.futures,
reading configuration dynamically prior to each fabric thread and job cycle.
"""

import logging
import signal
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

from .config import ClusterConfig, NodeConfig, load_cluster_config
from .model_manager import ModelManager
from .queue_manager import QueueManager
from .storage_guardian import StorageGuardian
from .syschk import SystemChecker
from ..workers.base_worker import BaseWorker
from ..workers.linux_worker import TopazLinuxWorker
from ..workers.mac_worker import TopazMacWorker
from ..workers.windows_worker import TopazWindowsWorker

logger = logging.getLogger("topaz_cluster.fabric_dispatcher")


class FabricDispatcher:
    """Central orchestrator engine managing worker futures across the compute fabric."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        env_path: Optional[Union[Path, bool]] = None,
        selected_nodes: Optional[List[str]] = None,
    ):
        self.config_path = config_path
        self.env_path = env_path
        self.selected_nodes = selected_nodes
        self._stop_event = threading.Event()
        self.executor: Optional[ThreadPoolExecutor] = None
        self.futures: Dict[str, Future] = {}
        self.workers: Dict[str, BaseWorker] = {}

    def read_fresh_config(self) -> ClusterConfig:
        """Read and validate configuration prior to thread dispatch or cycle."""
        return load_cluster_config(config_path=self.config_path, env_path=self.env_path)

    def instantiate_worker(
        self,
        node_id: str,
        node_cfg: NodeConfig,
        cluster_cfg: ClusterConfig,
        queue_mgr: QueueManager,
        guardian: StorageGuardian,
        model_mgr: ModelManager,
    ) -> BaseWorker:
        """Instantiate the platform-specific worker for the target host."""
        accel = node_cfg.accelerator.lower()
        if "ane" in accel or "apple" in accel or "metal" in accel or "mac" in node_id:
            return TopazMacWorker(
                node_id,
                node_cfg,
                cluster_cfg,
                queue_mgr,
                guardian,
                model_mgr,
                config_path=self.config_path,
                env_path=self.env_path,
            )
        elif "linux" in accel:
            return TopazLinuxWorker(
                node_id,
                node_cfg,
                cluster_cfg,
                queue_mgr,
                guardian,
                model_mgr,
                config_path=self.config_path,
                env_path=self.env_path,
            )
        else:
            return TopazWindowsWorker(
                node_id,
                node_cfg,
                cluster_cfg,
                queue_mgr,
                guardian,
                model_mgr,
                config_path=self.config_path,
                env_path=self.env_path,
            )

    def _run_fabric_worker(self, node_id: str) -> int:
        """Target function executed inside each fabric thread future."""
        # 1. Read config prior to thread execution
        cluster_cfg = self.read_fresh_config()
        if node_id not in cluster_cfg.nodes:
            logger.error("[%s] Node not found in configuration.", node_id)
            return 0

        node_cfg = cluster_cfg.nodes[node_id]
        if not node_cfg.enabled:
            logger.info("[%s] Node is disabled in configuration. Exiting fabric thread.", node_id)
            return 0

        guardian = StorageGuardian(
            archive_host=cluster_cfg.storage.archive_host,
            destination_dir=cluster_cfg.storage.destination_dir,
            min_free_gb=cluster_cfg.storage.min_free_gb,
        )
        model_mgr = ModelManager(default_model=cluster_cfg.model.name)
        queue_mgr = QueueManager(
            archive_host=cluster_cfg.storage.archive_host,
            source_dir=cluster_cfg.storage.source_dir,
            destination_dir=cluster_cfg.storage.destination_dir,
            claims_dir=cluster_cfg.claims_dir,
        )

        worker = self.instantiate_worker(node_id, node_cfg, cluster_cfg, queue_mgr, guardian, model_mgr)
        self.workers[node_id] = worker

        logger.info("[%s] Fabric thread started for %s (%s).", node_id, node_cfg.name, node_cfg.accelerator)

        # Connect stop event
        def check_stop():
            return self._stop_event.is_set()

        try:
            worker.run()
            return worker.completed_count
        except Exception as e:
            logger.error("[%s] Unhandled exception in fabric thread: %s", node_id, e, exc_info=True)
            return worker.completed_count

    def start(self, run_preflight: bool = True) -> bool:
        """Start worker threads along the compute fabric using concurrent.futures."""
        # Read config fresh prior to fabric startup
        cluster_cfg = self.read_fresh_config()

        guardian = StorageGuardian(
            archive_host=cluster_cfg.storage.archive_host,
            destination_dir=cluster_cfg.storage.destination_dir,
            min_free_gb=cluster_cfg.storage.min_free_gb,
        )

        # Enforce storage gate before starting any worker
        safe, free_gb = guardian.is_safe()
        if not safe:
            logger.critical(
                "FABRIC START BLOCKED: Storage guardian reports %.1f GB free (< %.1f GB threshold) on %s!",
                free_gb,
                cluster_cfg.storage.min_free_gb,
                cluster_cfg.storage.destination_dir,
            )
            return False

        # Run preflight host assurance (syschk)
        failed_nodes = set()
        if run_preflight:
            logger.info("Executing preflight syschk host assurance before launching fabric threads...")
            chk = SystemChecker(cluster_cfg, guardian)
            results = chk.check_all_nodes()
            has_failures = False
            for nid, r in results.items():
                if self.selected_nodes and nid not in self.selected_nodes:
                    continue
                if r.status == "FAIL":
                    logger.error("[%s] Host preflight failed: %s", nid, "; ".join(r.details))
                    has_failures = True
                    failed_nodes.add(nid)
                elif r.status == "PASS" or r.status == "WARN":
                    logger.info("[%s] Host preflight PASSED (%s, %.0fms latency)", nid, r.name, r.ssh_latency_ms)

            if has_failures:
                logger.warning("One or more worker nodes failed preflight checks (%s). Non-failing nodes will proceed.", sorted(failed_nodes))

        # Determine target nodes
        target_nodes = []
        for node_id, node_cfg in cluster_cfg.nodes.items():
            if self.selected_nodes and node_id not in self.selected_nodes:
                continue
            if not node_cfg.enabled:
                logger.info("[%s] Node disabled in config. Skipping.", node_id)
                continue
            if node_id in failed_nodes:
                logger.warning("[%s] Skipping node due to preflight failure.", node_id)
                continue
            target_nodes.append(node_id)

        if not target_nodes:
            logger.error("No enabled nodes available to launch in compute fabric.")
            return False

        logger.info("Deploying compute fabric across %d nodes: %s", len(target_nodes), target_nodes)
        max_pool = max(16, len(cluster_cfg.nodes) + 8)
        self.executor = ThreadPoolExecutor(max_workers=max_pool, thread_name_prefix="FabricWorker")

        for node_id in target_nodes:
            # Read config prior to each fabric thread dispatch
            fut = self.executor.submit(self._run_fabric_worker, node_id)
            self.futures[node_id] = fut

        return True

    def sync_dynamic_nodes(self) -> List[str]:
        """Re-read cluster_config.yaml and hot-join any newly added or newly enabled nodes on the fly."""
        if not self.executor or self._stop_event.is_set():
            return []
        try:
            fresh_cfg = self.read_fresh_config()
        except Exception as e:
            logger.warning("FabricDispatcher: failed to read config during dynamic node sync: %s", e)
            return []

        joined: List[str] = []
        for node_id, node_cfg in fresh_cfg.nodes.items():
            if self.selected_nodes and node_id not in self.selected_nodes:
                continue
            if node_cfg.enabled and node_id not in self.futures:
                logger.info(
                    "[%s] Dynamically hot-joining newly enabled node '%s' (%s) into active compute fabric!",
                    node_id,
                    node_cfg.name,
                    node_cfg.accelerator,
                )
                self.futures[node_id] = self.executor.submit(self._run_fabric_worker, node_id)
                joined.append(node_id)
        return joined

    def stop(self) -> None:
        """Signal all fabric workers to gracefully stop and wait for completions."""
        logger.warning("FabricDispatcher: Graceful shutdown initiated. Signaling all workers...")
        self._stop_event.set()
        for node_id, worker in self.workers.items():
            worker.stop()

        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("FabricDispatcher: All fabric threads cleanly joined and terminated.")

    def monitor(self, poll_interval_sec: int = 15) -> None:
        """Monitor running futures, hot-join newly enabled nodes, and audit cluster health until completion."""
        logged_done = set()
        while any(not fut.done() for fut in self.futures.values()):
            if self._stop_event.is_set():
                break
            time.sleep(poll_interval_sec)

            # Hot-join any newly added or newly enabled nodes from live config
            self.sync_dynamic_nodes()

            # Check futures for exceptions or completions
            for node_id, fut in list(self.futures.items()):
                if fut.done() and node_id not in logged_done:
                    logged_done.add(node_id)
                    exc = fut.exception()
                    if exc:
                        logger.error("[%s] Future exited with exception: %s", node_id, exc)
                    else:
                        res = fut.result()
                        logger.info("[%s] Future completed (%d jobs finished).", node_id, res)
