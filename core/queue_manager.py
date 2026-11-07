"""Centralized queue and atomic lock manager for Topaz Supercluster."""

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("topaz_cluster.queue_manager")


@dataclass
class JobItem:
    filename: str
    stem: str
    width: int
    height: int
    tier: str = ""

    def __post_init__(self):
        if not self.tier:
            self.tier = QueueManager.classify_tier(self.height)


class QueueManager:
    def __init__(
        self,
        archive_host: str,
        source_dir: str,
        destination_dir: str,
        claims_dir: Path = Path("/dev/shm/topaz_stage/claimed_stems"),
        manifest_path: Optional[Path] = Path("/dev/shm/topaz_stage/sorted_manifest.json"),
    ):
        self.archive_host = archive_host
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        self.claims_dir = claims_dir
        self.manifest_path = manifest_path
        self.claims_dir.mkdir(parents=True, exist_ok=True)
        self._cached_completed: Set[str] = set()
        self._last_completed_fetch: float = 0.0
        self._last_source_sync: float = 0.0

    def sync_manifest_with_source(self, min_interval_sec: float = 30.0) -> None:
        """Scan source_dir on archive_host for newly added files, deduplicate by SHA-256 (first 1MiB + size), skip >=4K, and append to manifest."""
        if not self.manifest_path:
            return
        now = time.time()
        if (now - self._last_source_sync) < min_interval_sec and self.manifest_path.is_file():
            return
        self._last_source_sync = now
        if self.manifest_path.is_file() and (now - self.manifest_path.stat().st_mtime) < 60.0:
            return

        remote_py = f"""python3 -c '
import os, json, subprocess, hashlib
src = {json.dumps(self.source_dir)}
if not os.path.isdir(src):
    print("[]")
    raise SystemExit(0)
files = sorted(os.listdir(src))
out = []
seen = set()
for f in files:
    p = os.path.join(src, f)
    if not os.path.isfile(p): continue
    ext = os.path.splitext(f)[1].lower()
    if ext not in (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".ts"): continue
    sz = os.path.getsize(p)
    if sz < 32768: continue
    h_sha = hashlib.sha256(open(p, "rb").read(1048576)).hexdigest() + f"_{{sz}}"
    if h_sha in seen: continue
    seen.add(h_sha)
    out.append([f, sz, h_sha])
print(json.dumps(out))
'"""
        try:
            res = subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.archive_host, remote_py],
                capture_output=True,
                text=True,
                timeout=25,
                check=False,
            )
            if res.returncode != 0 or not res.stdout.strip():
                return
            remote_items = json.loads(res.stdout.strip().splitlines()[-1])
        except Exception as e:
            logger.debug("QueueManager: sync_manifest_with_source scan skipped: %s", e)
            return

        existing_manifest = []
        existing_files = set()
        if self.manifest_path.is_file():
            try:
                existing_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                for item in existing_manifest:
                    existing_files.add(item[0])
            except Exception:
                existing_manifest = []

        # Map SHA-256 of existing manifest files + completed destination stems so alternate filenames of already-completed hashes are skipped
        done_stems = self.fetch_completed_stems()
        remote_by_name = {it[0]: it[2] for it in remote_items}
        covered_hashes = set()
        for fn, _, h_sha in remote_items:
            st = re.sub(r"[^A-Za-z0-9._-]", "_", Path(fn).stem)
            if st in done_stems or fn in existing_files:
                covered_hashes.add(h_sha)

        new_candidates = [it for it in remote_items if it[0] not in existing_files and it[2] not in covered_hashes]
        if not new_candidates:
            return

        # Probe dimensions for newly discovered candidates only
        added = 0
        for fn, sz, _ in new_candidates:
            if self.is_4k_or_rendered(fn):
                existing_files.add(fn)
                continue
            p_cmd = (
                f"export PATH=\"/opt/homebrew/bin:/usr/local/bin:/Applications/Topaz Video.app/Contents/MacOS:$PATH\"; "
                f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 \"{self.source_dir}/{fn}\" 2>/dev/null || "
                f"(w=$(mdls -raw -name kMDItemPixelWidth \"{self.source_dir}/{fn}\" 2>/dev/null); "
                f"h=$(mdls -raw -name kMDItemPixelHeight \"{self.source_dir}/{fn}\" 2>/dev/null); "
                f"echo \"${{w}},${{h}}\") || true"
            )
            pr = subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.archive_host, p_cmd],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            parts = [x.strip() for x in pr.stdout.strip().replace("\n", ",").split(",") if x.strip().isdigit()]
            w = int(parts[0]) if len(parts) >= 2 else 640
            h = int(parts[1]) if len(parts) >= 2 else 464
            existing_manifest.append([fn, w, h])
            existing_files.add(fn)
            if not (max(w, h) >= 2140 or min(w, h) >= 2100 or self.is_4k_or_rendered(fn, w, h)):
                added += 1

        if new_candidates:
            tmp_path = self.manifest_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(existing_manifest, indent=2), encoding="utf-8")
            tmp_path.replace(self.manifest_path)
            if added > 0:
                logger.info("QueueManager: dynamically discovered and added %d new unique non-4K video(s) from %s", added, self.source_dir)

    @staticmethod
    def sanitize_stem(raw_name: str) -> str:
        stem = Path(raw_name).stem
        return re.sub(r"[^A-Za-z0-9._-]", "_", stem)

    @staticmethod
    def classify_tier(height: int) -> str:
        if 400 <= height <= 464:
            return "464p"
        elif 464 < height <= 480:
            return "480p"
        elif 480 < height <= 720:
            return "720p"
        elif 720 < height <= 1080:
            return "1080p"
        return "other"

    @staticmethod
    def is_4k_or_rendered(filename: str, width: int = 0, height: int = 0) -> bool:
        """Return True if filename is an already-rendered Topaz deliverable or if video resolution is >= 4K."""
        lower_fn = filename.lower()
        for tag in ("_iris3", "_rhea1", "_prob4", "_nyx3", "_ahq12", "_M4Mini_", "_V100_", "_RTX5080_", "_M1Max_", "_repaired"):
            if tag.lower() in lower_fn:
                return True
        if max(width, height) >= 2140 or min(width, height) >= 2100:
            return True
        return False

    def fetch_completed_stems(self, max_age_sec: float = 10.0) -> Set[str]:
        """Query destination directory and return set of completed base stems (cached)."""
        now = time.time()
        if self._cached_completed and (now - self._last_completed_fetch) < max_age_sec:
            return set(self._cached_completed)

        try:
            cmd = f"ls -1 '{self.destination_dir}' 2>/dev/null || true"
            res = subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.archive_host, cmd],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            done_stems: Set[str] = set()
            for ln in res.stdout.splitlines():
                fn = ln.strip()
                if not fn:
                    continue
                for sep in ("_M4Mini_", "_V100_", "_RTX5080_", "_M1Max_", "_iris3", "_rhea1", "_prob4", "_nyx3", "_ahq12"):
                    if sep in fn:
                        raw_stem = fn.split(sep)[0]
                        done_stems.add(re.sub(r"[^A-Za-z0-9._-]", "_", raw_stem))
                        break
            self._cached_completed = done_stems
            self._last_completed_fetch = now
            return done_stems
        except Exception as e:
            logger.error("QueueManager: error querying completed deliverables: %s", e)
            return self._cached_completed if self._cached_completed else set()

    def load_all_jobs(self) -> List[JobItem]:
        """Load jobs from pre-probed manifest if available (auto-syncing new files from source_dir), otherwise probe source directory."""
        self.sync_manifest_with_source()
        jobs: List[JobItem] = []
        if self.manifest_path and self.manifest_path.is_file():
            try:
                manifest_data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                for item in manifest_data:
                    fn, w, h = item[0], int(item[1]), int(item[2])
                    if self.is_4k_or_rendered(fn, w, h):
                        continue
                    stem = self.sanitize_stem(Path(fn).name)
                    tier = self.classify_tier(min(w, h) if w < h else h)
                    jobs.append(JobItem(filename=fn, stem=stem, width=w, height=h, tier=tier))
                return jobs
            except Exception as e:
                logger.warning("QueueManager: failed to parse manifest %s: %s", self.manifest_path, e)

        # Fallback: list source dir via SSH ls -1
        logger.info("QueueManager: probing source directory %s on %s...", self.source_dir, self.archive_host)
        probe_cmd = f"ls -1 '{self.source_dir}' 2>/dev/null || true"
        res = subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.archive_host, probe_cmd],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        for fn in sorted(res.stdout.splitlines()):
            fn = fn.strip()
            ext = Path(fn).suffix.lower()
            if not fn or ext not in (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".ts") or self.is_4k_or_rendered(fn):
                continue
            jobs.append(JobItem(filename=fn, stem=self.sanitize_stem(fn), width=640, height=464, tier="464p"))
        return jobs

    def build_node_queue(self, priority_tiers: List[str], all_jobs: List[JobItem]) -> List[JobItem]:
        """Sort and filter all_jobs according to a worker node's configured tier priorities."""
        by_tier: Dict[str, List[JobItem]] = {"464p": [], "480p": [], "720p": [], "1080p": [], "other": []}
        for j in all_jobs:
            by_tier.get(j.tier, by_tier["other"]).append(j)

        node_queue: List[JobItem] = []
        for rule in priority_tiers:
            is_rev = rule.endswith("_reverse")
            tier_name = rule.replace("_reverse", "")
            items = list(by_tier.get(tier_name, []))
            if is_rev:
                items.reverse()
            node_queue.extend(items)

        # Append any remaining tiers not explicitly configured
        for tier_name, items in by_tier.items():
            if tier_name not in [r.replace("_reverse", "") for r in priority_tiers]:
                node_queue.extend(items)

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for j in node_queue:
            if j.stem not in seen:
                seen.add(j.stem)
                deduped.append(j)
        return deduped

    def try_claim(self, stem: str, worker_tag: str) -> bool:
        """Atomically claim a job stem across all cluster workers using O_EXCL file lock."""
        lock_file = self.claims_dir / f"{stem}.lock"
        raw_file = self.claims_dir / stem

        if lock_file.exists() or raw_file.exists():
            return False

        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{worker_tag}:{time.time()}\n".encode("utf-8"))
            os.close(fd)
        except FileExistsError:
            return False

        try:
            raw_file.touch(exist_ok=False)
        except FileExistsError:
            pass

        return True

    def fetch_active_claims(self) -> Set[str]:
        """Return set of currently claimed stems."""
        claims = set()
        if not self.claims_dir.is_dir():
            return claims
        for f in self.claims_dir.iterdir():
            if f.name.endswith(".lock"):
                claims.add(f.stem)
            elif not f.name.endswith((".tmp", ".log")):
                claims.add(f.name)
        return claims

    def release_claim(self, stem: str) -> None:
        """Release lock files for stem so it can be retried or picked up by another worker."""
        (self.claims_dir / f"{stem}.lock").unlink(missing_ok=True)
        (self.claims_dir / stem).unlink(missing_ok=True)
