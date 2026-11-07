"""Topaz macOS Runtime Worker.

Unified runtime for macOS Apple Silicon execution (M4, M1 Max, ANE, Metal, VideoToolbox).
Strict Zero-NAND policy: stages exclusively in volatile /Volumes/TopazRAM HFS+ RAMDisk.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, Union

from .base_worker import BaseWorker
from ..core.config import ClusterConfig, NodeConfig
from ..core.model_manager import ModelManager
from ..core.queue_manager import JobItem, QueueManager
from ..core.storage_guardian import StorageGuardian


class TopazMacWorker(BaseWorker):
    def __init__(
        self,
        node_id: str,
        node_cfg: NodeConfig,
        cluster_cfg: ClusterConfig,
        queue_mgr: QueueManager,
        guardian: StorageGuardian,
        model_mgr: ModelManager,
        config_path: Optional[Path] = None,
        env_path: Optional[Union[Path, bool]] = None,
    ):
        super().__init__(node_id, node_cfg, cluster_cfg, queue_mgr, guardian, config_path=config_path, env_path=env_path)
        self.model_mgr = model_mgr
        self.mac_ssh = node_cfg.ssh_host
        self.ramdisk_mount = node_cfg.ramdisk_mount or "/Volumes/TopazRAM"
        self.local_hb_dir = Path(node_cfg.local_hb_dir or f"/dev/shm/topaz_{node_id}_ram")
        self.local_hb_dir.mkdir(parents=True, exist_ok=True)
        self.mac_ram_in = f"{self.ramdisk_mount}/in"
        self.mac_ram_out = f"{self.ramdisk_mount}/out"
        self.mac_ram_tmp = f"{self.ramdisk_mount}/tmp"

    def _run_mac_ssh(self, script: str, timeout: int = 600) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.mac_ssh, script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _ensure_ramdisk(self) -> bool:
        check = self._run_mac_ssh(f"mount | grep -q '{self.ramdisk_mount}' && echo MOUNTED || echo NOT_MOUNTED", timeout=15)
        if "MOUNTED" in check.stdout:
            self._run_mac_ssh(f'mkdir -p "{self.mac_ram_in}" "{self.mac_ram_out}" "{self.mac_ram_tmp}"', timeout=15)
            return True

        ram_sectors = int((self.node_cfg.ramdisk_size_mb * 1024 * 1024) / 512)
        mount_script = f"""
export PATH="/usr/sbin:/sbin:$PATH"
DEV=$(hdiutil attach -nomount ram://{ram_sectors} | awk '{{print $1}}')
diskutil eraseVolume HFS+ TopazRAM "$DEV" >/dev/null 2>&1
mkdir -p "{self.mac_ram_in}" "{self.mac_ram_out}" "{self.mac_ram_tmp}"
chmod -R 777 "{self.ramdisk_mount}"
mount | grep -q '{self.ramdisk_mount}' && echo MOUNTED || echo NOT_MOUNTED
"""
        res = self._run_mac_ssh(mount_script, timeout=60)
        return "MOUNTED" in res.stdout

    def _condition_in_dram(self, filename: str, orig_h: int) -> Tuple[Optional[Path], int, int, int]:
        ext = Path(filename).suffix.lower()
        src_path = f"{self.cluster_cfg.storage.source_dir}/{filename}"
        raw_shm = self.local_hb_dir / f"mac_raw{ext}"
        repaired_shm = self.local_hb_dir / "mac_repaired.mp4"

        for p in (raw_shm, repaired_shm):
            p.unlink(missing_ok=True)

        cp_res = subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", self.cluster_cfg.storage.archive_host, f"cat -- \"{src_path}\""],
            stdout=raw_shm.open("wb"),
            check=False,
        )
        if cp_res.returncode != 0 or not raw_shm.is_file() or raw_shm.stat().st_size < 32768:
            raw_shm.unlink(missing_ok=True)
            return None, 0, 0, 0

        ac_res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(raw_shm)],
            capture_output=True,
            text=True,
            check=False,
        )
        a_codec = ac_res.stdout.strip()
        need_hb = ext not in (".mp4", ".mov", ".m4v") or a_codec in ("pcm_alaw", "pcm_mulaw", "adpcm_ima_wav")

        if not need_hb:
            raw_shm.rename(repaired_shm)
        else:
            hb_env = os.environ.copy()
            hb_env["TMPDIR"] = str(self.local_hb_dir)
            if ext in (".avi", ".wmv", ".ts") and shutil.which("HandBrakeCLI"):
                hb_cmd = [
                    "nice", "-n", "15", "HandBrakeCLI",
                    "-i", str(raw_shm),
                    "-o", str(repaired_shm),
                    "-f", "av_mp4", "-e", "x264", "-q", "16", "--cfr",
                    "-E", "av_aac", "-B", "192",
                ]
            else:
                hb_cmd = [
                    "nice", "-n", "15", "ffmpeg", "-hide_banner", "-nostdin", "-y",
                    "-i", str(raw_shm),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    str(repaired_shm),
                ]
            hb_res = subprocess.run(
                hb_cmd,
                env=hb_env,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            raw_shm.unlink(missing_ok=True)
            if hb_res.returncode != 0 or not repaired_shm.is_file() or repaired_shm.stat().st_size < 65536:
                repaired_shm.unlink(missing_ok=True)
                return None, 0, 0, 0

        w, h, nbf = 640, orig_h, 0
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames", "-of", "csv=p=0", str(repaired_shm)],
            capture_output=True,
            text=True,
            check=False,
        )
        parts = [x.strip() for x in pr.stdout.strip().split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            w, h = int(parts[0]), int(parts[1])
        if len(parts) >= 3 and parts[2].isdigit():
            nbf = int(parts[2])

        return repaired_shm, w, h, nbf

    def process_job(self, job: JobItem) -> Tuple[bool, int, float, float]:
        if not self._ensure_ramdisk():
            self.logger.error("TopazMacWorker (%s): RAMDisk setup failed.", self.node_id)
            return False, 0, 0.0, 0.0

        repaired_shm, w, h, nb_frames = self._condition_in_dram(job.filename, job.height)
        if repaired_shm is None:
            return False, 0, 0.0, 0.0

        m_in = f"{self.mac_ram_in}/active_in.mp4"
        m_out = f"{self.mac_ram_out}/active_out.mp4"
        m_stderr = f"{self.mac_ram_out}/active_stderr.log"

        self._run_mac_ssh(f'rm -P -f "{self.mac_ram_in}"/* "{self.mac_ram_out}"/* "{self.mac_ram_tmp}"/* 2>/dev/null || true', timeout=20)

        with repaired_shm.open("rb") as in_f:
            up_res = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", self.mac_ssh, f'cat > "{m_in}"'],
                stdin=in_f,
                capture_output=True,
                check=False,
            )
        repaired_shm.unlink(missing_ok=True)
        if up_res.returncode != 0:
            return False, 0, 0.0, 0.0

        # Build filter from ModelManager (instances strictly 1 for Apple Silicon)
        out_w, out_h = self.model_mgr.compute_native_4k_dims(
            w, h, self.cluster_cfg.model.target_width, self.cluster_cfg.model.target_height
        )
        tvai_filter = self.model_mgr.build_filter(
            model_name=self.cluster_cfg.model.name,
            scale=self.cluster_cfg.model.scale,
            target_w=self.cluster_cfg.model.target_width,
            target_h=self.cluster_cfg.model.target_height,
            instances=1,
            device_id=0,
            blur=self.cluster_cfg.model.blur,
            blend=self.cluster_cfg.model.blend,
            scaling_filter=self.cluster_cfg.model.scaling_filter,
            allow_crop=self.cluster_cfg.model.allow_crop,
            src_w=w,
            src_h=h,
        )

        meta_comment = f"Topaz Video 4K | Node: {self.node_cfg.name} | model={self.cluster_cfg.model.name} src={w}x{h} out={out_w}x{out_h}"
        ffmpeg_bin = self.node_cfg.ffmpeg_path or "/Applications/Topaz Video.app/Contents/MacOS/ffmpeg"
        model_dir = self.node_cfg.model_dir or "/Applications/Topaz Video.app/Contents/Resources/models"

        mac_render_script = f"""
export TVAI_MODEL_DIR="{model_dir}"
export TVAI_MODEL_DATA_DIR="{model_dir}"
export TMPDIR="{self.mac_ram_tmp}"

"{ffmpeg_bin}" -hide_banner -nostdin -y \\
  -i "{m_in}" -sws_flags spline+accurate_rnd+full_chroma_int \\
  -filter_complex "{tvai_filter}" \\
  -c:v h264_videotoolbox -b:v 35M -profile:v high -pix_fmt yuv420p \\
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \\
  -movflags frag_keyframe+empty_moov+delay_moov+use_metadata_tags+write_colr \\
  -map_metadata 0 -map_metadata:s:v 0:s:v -metadata comment="{meta_comment}" \\
  -map "0:a:0?" -c:a aac -ar 48000 -ac 2 -b:a 192k \\
  "{m_out}" 2> "{m_stderr}"

stat -f %z "{m_out}" 2>/dev/null || echo 0
"""
        t_render = time.time()
        ren_res = self._run_mac_ssh(mac_render_script, timeout=14400)
        elapsed = max(0.1, time.time() - t_render)

        sz_lines = [ln.strip() for ln in ren_res.stdout.splitlines() if ln.strip().isdigit()]
        out_bytes = int(sz_lines[-1]) if sz_lines else 0

        if ren_res.returncode != 0 or out_bytes < 1048576:
            err_tail = self._run_mac_ssh(f"tail -n 12 '{m_stderr}' 2>/dev/null || true", timeout=15).stdout
            self.logger.error("TopazMacWorker (%s): render failed for '%s' (rc=%d, size=%dB):\n%s", self.node_id, job.filename, ren_res.returncode, out_bytes, err_tail)
            self._run_mac_ssh(f'rm -P -f "{self.mac_ram_in}"/* "{self.mac_ram_out}"/* "{self.mac_ram_tmp}"/* 2>/dev/null || true', timeout=20)
            return False, 0, elapsed, 0.0

        fps = (nb_frames / elapsed) if nb_frames > 0 else 0.0
        final_name = f"{job.stem}_iris3.mp4"
        remote_dest = f"{self.cluster_cfg.storage.destination_dir}/{final_name}"

        # Stream directly from macOS volatile RAMDisk to destination archive
        deliv_cmd = f"ssh -o BatchMode=yes {self.mac_ssh} 'cat \"{m_out}\"' | ssh -o BatchMode=yes {self.cluster_cfg.storage.archive_host} 'cat > \"{remote_dest}\"'"
        subprocess.run(deliv_cmd, shell=True, check=False)

        st_res = subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.cluster_cfg.storage.archive_host, f"stat -c %s '{remote_dest}' 2>/dev/null || echo 0"],
            capture_output=True,
            text=True,
            check=False,
        )
        st_lines = [ln.strip() for ln in st_res.stdout.splitlines() if ln.strip().isdigit()]
        dest_bytes = int(st_lines[-1]) if st_lines else 0

        # Secure wipe in RAMDisk
        self._run_mac_ssh(f'rm -P -f "{self.mac_ram_in}"/* "{self.mac_ram_out}"/* "{self.mac_ram_tmp}"/* 2>/dev/null || true', timeout=30)

        if dest_bytes == out_bytes and dest_bytes > 0:
            return True, dest_bytes, elapsed, fps
        else:
            self.logger.warning("TopazMacWorker (%s): delivery byte mismatch for '%s' (%dB vs %dB)", self.node_id, final_name, out_bytes, dest_bytes)
            subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.cluster_cfg.storage.archive_host, f"rm -f '{remote_dest}' 2>/dev/null || true"], check=False)
            return False, 0, elapsed, 0.0

    def cleanup_job(self, job: JobItem) -> None:
        self._run_mac_ssh(f'rm -P -f "{self.mac_ram_in}"/* "{self.mac_ram_out}"/* "{self.mac_ram_tmp}"/* 2>/dev/null || true', timeout=20)
