#!/usr/bin/env python3
"""Topaz Video AI Supercluster Master Orchestrator.

Centralized coordinator for heterogeneous upscaling across NVIDIA and Apple Silicon nodes
using concurrent.futures, dynamic per-thread configuration reloading, and preflight host assurance.
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List

from .core.config import ClusterConfig, load_cluster_config
from .core.fabric_dispatcher import FabricDispatcher
from .core.model_manager import ModelManager
from .core.queue_manager import QueueManager
from .core.storage_guardian import StorageGuardian
from .core.syschk import SystemChecker


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cluster_orchestrator.log"

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    return logging.getLogger("topaz_cluster.orchestrator")


def print_status(cluster_cfg: ClusterConfig, queue_mgr: QueueManager, guardian: StorageGuardian) -> None:
    safe, free_gb = guardian.is_safe()
    done_stems = queue_mgr.fetch_completed_stems()
    all_jobs = queue_mgr.load_all_jobs()
    total_jobs = len(all_jobs)
    completed_count = sum(1 for j in all_jobs if j.stem in done_stems)
    remaining_count = max(0, total_jobs - completed_count)

    print("\n" + "=" * 70)
    print(" TOPAZ 4K CLUSTER ORCHESTRATOR STATUS")
    print("=" * 70)
    print(f" Source Directory:      {cluster_cfg.storage.source_dir}")
    print(f" Destination Directory: {cluster_cfg.storage.destination_dir} (on {cluster_cfg.storage.archive_host})")
    print(f" Destination Free Space: {free_gb:.1f} GB ({'SAFE' if safe else 'GATED (< 25GB)'})")
    print(f" Total Queue Items:     {total_jobs}")
    print(f" Completed Masters:     {completed_count} ({completed_count / max(1, total_jobs) * 100:.1f}%)")
    print(f" Remaining Jobs:        {remaining_count}")
    print("-" * 70)
    print(" Active Worker Configurations:")
    for node_id, node_cfg in cluster_cfg.nodes.items():
        state = "ENABLED" if node_cfg.enabled else "DISABLED"
        print(f"   [{node_id}] {node_cfg.name} -> {state} (Instances: {node_cfg.instances}, Accel: {node_cfg.accelerator})")
    print("=" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Topaz Video AI Supercluster Master Orchestrator")
    parser.add_argument("--config", type=Path, default=None, help="Path to cluster_config.yaml")
    parser.add_argument("--source", type=str, default=None, help="Override source directory")
    parser.add_argument("--dest", type=str, default=None, help="Override destination directory")
    parser.add_argument("--min-free-gb", type=float, default=None, help="Minimum destination free space (GB)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override Topaz model: iris-3, rhea-1, prob-4, nyx-3, or local Project Starlight models (Windows/macOS only): slp-2.5 (starlight), astrasharp (starlight-sharp), astrafast (starlight-fast), astrahq (starlight-hq), astra (starlight-creative), slm-1 (starlight-mini)",
    )
    parser.add_argument("--scale", type=int, default=None, help="Override neural scale factor (e.g. 1, 2, 4, 0)")
    parser.add_argument(
        "--starlight-strength",
        type=float,
        choices=[0.7, 1.0, 1.3],
        default=None,
        help="Enhancement strength for Starlight Sharp / Fast / HQ / Astra models (0.7, 1.0, or 1.3; default: 1.0)",
    )
    parser.add_argument("--width", type=int, default=None, help="Override target bounding width (default: 3840)")
    parser.add_argument("--height", type=int, default=None, help="Override target bounding height (default: 2160)")
    parser.add_argument("--allow-crop", action="store_true", help="Enable center-crop fill (force_original_aspect_ratio=increase,crop=W:H)")
    parser.add_argument("--force-condition", action="store_true", help="Force Phase-1 in-DRAM bitstream/timestamp repair on all inputs")
    parser.add_argument("--rm-source", action="store_true", help="Remove source video file after its 4K deliverable is verified")
    parser.add_argument("--workers", nargs="+", default=None, help="Specific worker nodes to run (e.g. rtx5080 v100)")
    parser.add_argument("--status", action="store_true", help="Print live cluster queue and storage status and exit")
    parser.add_argument("--dry-run", action="store_true", help="Preview prioritized queues per worker without launching")
    parser.add_argument("--syschk", action="store_true", help="Run full preflight host assurance and silicon audit")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip host preflight check before launching")
    args = parser.parse_args()

    # Propagate CLI overrides into process environment so worker hot-reloads preserve them
    if args.source:
        os.environ["TOPAZ_SOURCE_DIR"] = args.source
    if args.dest:
        os.environ["TOPAZ_DEST_DIR"] = args.dest
    if args.min_free_gb is not None:
        os.environ["TOPAZ_MIN_FREE_GB"] = str(args.min_free_gb)
    if args.model:
        os.environ["TOPAZ_MODEL"] = args.model
    if args.scale is not None:
        os.environ["TOPAZ_SCALE"] = str(args.scale)
    if args.starlight_strength is not None:
        os.environ["TOPAZ_STARLIGHT_STRENGTH"] = str(args.starlight_strength)
    if args.width is not None:
        os.environ["TOPAZ_TARGET_WIDTH"] = str(args.width)
    if args.height is not None:
        os.environ["TOPAZ_TARGET_HEIGHT"] = str(args.height)
    if args.allow_crop:
        os.environ["TOPAZ_ALLOW_CROP"] = "true"
    if args.force_condition:
        os.environ["TOPAZ_FORCE_CONDITION"] = "true"
    if args.rm_source:
        os.environ["TOPAZ_DELETE_SOURCE"] = "true"

    # Strip bundled Topaz lib paths from parent process LD_LIBRARY_PATH so system ssh/ffprobe/ffmpeg are unaffected
    if "LD_LIBRARY_PATH" in os.environ:
        clean_ld = [p for p in os.environ["LD_LIBRARY_PATH"].split(":") if p and "TopazVideo" not in p]
        os.environ["LD_LIBRARY_PATH"] = ":".join(clean_ld)

    cluster_cfg = load_cluster_config(config_path=args.config)

    logger = setup_logging(cluster_cfg.log_dir)
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

    if args.syschk:
        chk = SystemChecker(cluster_cfg, guardian)
        results = chk.check_all_nodes()
        print(chk.format_report(results))
        return 0

    if args.status:
        print_status(cluster_cfg, queue_mgr, guardian)
        return 0

    if args.dry_run:
        print_status(cluster_cfg, queue_mgr, guardian)
        all_jobs = queue_mgr.load_all_jobs()
        for node_id, node_cfg in cluster_cfg.nodes.items():
            if args.workers and node_id not in args.workers:
                continue
            if not node_cfg.enabled:
                continue
            node_q = queue_mgr.build_node_queue(node_cfg.priority_tiers, all_jobs)
            print(f"[{node_id}] Priority Queue ({len(node_q)} jobs):")
            for j in node_q[:5]:
                print(f"   -> {j.filename} ({j.width}x{j.height}, {j.tier})")
            if len(node_q) > 5:
                print(f"   ... and {len(node_q) - 5} more.")
        return 0

    # Ensure storage is safe before starting
    logger.info("Verifying destination storage before launching workers...")
    guardian.wait_until_safe(worker_name="Orchestrator")

    # Launch via FabricDispatcher (concurrent.futures)
    dispatcher = FabricDispatcher(
        config_path=args.config,
        selected_nodes=args.workers,
    )

    def handle_shutdown(signum, frame):
        logger.warning("Shutdown signal received (%d). Requesting graceful stop for fabric dispatcher...", signum)
        dispatcher.stop()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    launched = dispatcher.start(run_preflight=not args.skip_preflight)
    if not launched:
        logger.error("Failed to launch compute fabric. Exiting.")
        return 1

    dispatcher.monitor(poll_interval_sec=20)
    logger.info("All fabric threads completed. Topaz Cluster Orchestrator shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
