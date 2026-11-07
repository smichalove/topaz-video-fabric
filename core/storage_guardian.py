"""Storage Guardian: Centralized disk capacity gating and automated health checks."""

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger("topaz_cluster.storage_guardian")


class StorageGuardian:
    def __init__(self, archive_host: str, destination_dir: str, min_free_gb: float = 25.0, cache_ttl_sec: float = 5.0):
        self.archive_host = archive_host
        self.destination_dir = destination_dir
        self.min_free_gb = min_free_gb
        self.cache_ttl_sec = cache_ttl_sec
        self._last_check_epoch: float = 0.0
        self._cached_free_gb: float = 0.0

    def query_free_space_gb(self) -> float:
        """Query actual free gigabytes on archive host destination directory."""
        now = time.time()
        if now - self._last_check_epoch < self.cache_ttl_sec and self._last_check_epoch > 0:
            return self._cached_free_gb

        host = self.archive_host.split("@")[-1] if "@" in self.archive_host else self.archive_host
        if host in ("localhost", "127.0.0.1", ""):
            try:
                dest = self.destination_dir if os.path.exists(self.destination_dir) else "/"
                usage = shutil.disk_usage(dest)
                self._cached_free_gb = usage.free / (1024.0 ** 3)
                self._last_check_epoch = now
                return self._cached_free_gb
            except Exception as e:
                logger.warning("StorageGuardian: failed to query local disk space: %s", e)
                return self._cached_free_gb

        try:
            res = subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.archive_host, f"df -m '{self.destination_dir}' | tail -n 1"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            parts = res.stdout.strip().split()
            if len(parts) >= 4 and parts[3].isdigit():
                free_mb = int(parts[3])
                self._cached_free_gb = free_mb / 1024.0
                self._last_check_epoch = now
                return self._cached_free_gb
        except Exception as e:
            logger.warning("StorageGuardian: failed to query remote disk space: %s", e)

        return self._cached_free_gb

    def is_safe(self, min_gb: Optional[float] = None) -> Tuple[bool, float]:
        """Return (True, free_gb) if space is >= threshold, else (False, free_gb)."""
        threshold = min_gb if min_gb is not None else self.min_free_gb
        free_gb = self.query_free_space_gb()
        return (free_gb >= threshold, free_gb)

    def wait_until_safe(
        self,
        worker_name: str = "Worker",
        poll_interval_sec: int = 30,
        stop_event: Optional[threading.Event] = None,
    ) -> float:
        """Block execution in a sleep loop until free space meets or exceeds threshold."""
        while True:
            if stop_event and stop_event.is_set():
                return 0.0
            safe, free_gb = self.is_safe()
            if safe:
                return free_gb
            logger.critical(
                "[%s] STORAGE GUARDIAN ALERT: Destination %s has only %.1f GB free (< %.1f GB threshold)! Pausing %ds...",
                worker_name,
                self.destination_dir,
                free_gb,
                self.min_free_gb,
                poll_interval_sec,
            )
            if stop_event:
                for _ in range(poll_interval_sec):
                    if stop_event.is_set():
                        return 0.0
                    time.sleep(1)
            else:
                time.sleep(poll_interval_sec)
