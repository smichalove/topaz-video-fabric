"""Linux Worker implementation for Native Linux CUDA / TensorRT Topaz Video AI nodes."""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, Union

from ..core.config import ClusterConfig, NodeConfig
from ..core.model_manager import ModelManager
from ..core.queue_manager import JobItem, QueueManager
from ..core.storage_guardian import StorageGuardian
from .base_worker import BaseWorker


class TopazLinuxWorker(BaseWorker):
    """Executes Zero-NAND DRAM-staged 4K upscaling on native Linux NVIDIA GPUs."""

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
        self.shm_dir = Path(node_cfg.wsl_shm_dir or f"/dev/shm/topaz_{node_id}_ram")
        self.hb_dir = self.shm_dir / "hb_tmp"
        self.shm_dir.mkdir(parents=True, exist_ok=True)
        self.hb_dir.mkdir(parents=True, exist_ok=True)

    def _stage_and_condition(self, filename: str, orig_h: int) -> Tuple[Optional[Path], int, int, int, str]:
        ext = Path(filename).suffix.lower()
        src_path = f"{self.cluster_cfg.storage.source_dir}/{filename}"
        raw_shm = self.shm_dir / f"raw_in{ext}"
        active_in = self.shm_dir / "active_in.mp4"

        for p in (raw_shm, active_in):
            p.unlink(missing_ok=True)

        with raw_shm.open("wb") as f_out:
            cp_res = subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", self.cluster_cfg.storage.archive_host, f"cat -- \"{src_path}\""],
                stdout=f_out,
                check=False,
            )
        if cp_res.returncode != 0 or not raw_shm.is_file() or raw_shm.stat().st_size < 32768:
            raw_shm.unlink(missing_ok=True)
            return None, 0, 0, 0, ""

        ac_res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(raw_shm)],
            capture_output=True,
            text=True,
            check=False,
        )
        a_codec = ac_res.stdout.strip()
        need_hb = (
            self.cluster_cfg.model.force_condition
            or ext not in (".mp4", ".mov", ".m4v")
            or a_codec in ("pcm_alaw", "pcm_mulaw", "adpcm_ima_wav")
        )

        if not need_hb:
            raw_shm.rename(active_in)
        else:
            hb_env = os.environ.copy()
            hb_env["TMPDIR"] = str(self.hb_dir)
            if ext in (".avi", ".wmv", ".ts") and shutil.which("HandBrakeCLI"):
                hb_cmd = [
                    "nice", "-n", "15", "HandBrakeCLI",
                    "-i", str(raw_shm),
                    "-o", str(active_in),
                    "-f", "av_mp4", "-e", "x264", "-q", "16", "--cfr",
                    "-E", "av_aac", "-B", "192",
                ]
            else:
                hb_cmd = [
                    "nice", "-n", "15", "ffmpeg", "-hide_banner", "-nostdin", "-y",
                    "-fflags", "+genpts+igndts", "-err_detect", "ignore_err",
                    "-i", str(raw_shm),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    str(active_in),
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
            if hb_res.returncode != 0 or not active_in.is_file() or active_in.stat().st_size < 65536:
                active_in.unlink(missing_ok=True)
                return None, 0, 0, 0, ""

        w, h, nbf, avg_fps = 640, orig_h, 0, ""
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,avg_frame_rate", "-of", "csv=p=0", str(active_in)],
            capture_output=True,
            text=True,
            check=False,
        )
        ln0 = pr.stdout.strip().splitlines()[-1] if pr.stdout.strip() else ""
        parts = [x.strip() for x in ln0.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            w, h = int(parts[0]), int(parts[1])
        if len(parts) >= 3 and parts[2] and parts[2] not in ("0/0", "N/A"):
            avg_fps = parts[2]
        if len(parts) >= 4 and parts[3].isdigit():
            nbf = int(parts[3])

        return active_in, w, h, nbf, avg_fps

    def process_job(self, job: JobItem) -> Tuple[bool, int, float, float]:
        active_in, w, h, nb_frames, avg_fps = self._stage_and_condition(job.filename, job.height)
        if active_in is None:
            return False, 0, 0.0, 0.0

        active_out = self.shm_dir / "active_out.mp4"
        active_err = self.shm_dir / "active_stderr.log"
        active_out.unlink(missing_ok=True)
        active_err.unlink(missing_ok=True)

        instances = self.node_cfg.instances if self.node_cfg.instances > 0 else 2
        out_w, out_h, use_crop = self.model_mgr.compute_smart_4k_framing(
            w, h, self.cluster_cfg.model.target_width, self.cluster_cfg.model.target_height, allow_crop=self.cluster_cfg.model.allow_crop
        )
        tvai_filter = self.model_mgr.build_filter(
            model_name=self.cluster_cfg.model.name,
            scale=self.cluster_cfg.model.scale,
            target_w=self.cluster_cfg.model.target_width,
            target_h=self.cluster_cfg.model.target_height,
            instances=instances,
            device_id=self.node_cfg.device_id,
            blur=self.cluster_cfg.model.blur,
            blend=self.cluster_cfg.model.blend,
            scaling_filter=self.cluster_cfg.model.scaling_filter,
            allow_crop=self.cluster_cfg.model.allow_crop,
            src_w=w,
            src_h=h,
        )

        ffmpeg_bin = self.node_cfg.ffmpeg_path or "/opt/TopazVideoAIBETA/bin/ffmpeg"
        model_dir = self.node_cfg.model_dir or "/opt/TopazVideoAIBETA/models"
        lib_dir = str(Path(ffmpeg_bin).parent.parent / "lib")
        egl_json = str(Path(ffmpeg_bin).parent.parent / "10_nvidia.json")
        safe_node_name = self.node_cfg.name.replace("(", "").replace(")", "").replace("|", "-")
        meta_comment = f"Topaz Video 4K - Node: {safe_node_name} - model={self.cluster_cfg.model.name} src={w}x{h} out={out_w}x{out_h}"

        is_remote = self.node_cfg.ssh_host and "127.0.0.1" not in self.node_cfg.ssh_host and "localhost" not in self.node_cfg.ssh_host

        if is_remote:
            if "wsl" in self.node_id.lower() or "wsl" in (self.node_cfg.wsl_shm_dir or "").lower():
                tvai_filter = tvai_filter.replace(":estimate=8", "")
            r_shm = str(self.shm_dir)
            subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", self.node_cfg.ssh_host, f"mkdir -p '{r_shm}/hb_tmp' && rm -f '{r_shm}/active_in.mp4' '{r_shm}/active_out.mp4'"],
                check=False,
            )
            with active_in.open("rb") as f_in:
                cp_r = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", self.node_cfg.ssh_host, f"cat > '{r_shm}/active_in.mp4'"],
                    stdin=f_in,
                    check=False,
                )
            active_in.unlink(missing_ok=True)
            if cp_r.returncode != 0:
                self.cleanup_job(job)
                return False, 0, 0.0, 0.0

            fps_arg = f"-r {avg_fps} " if avg_fps else ""
            r_script = f"""
export CUDA_FORCE_PTX_JIT=1
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:{lib_dir}"
export TVAI_MODEL_DIR="{model_dir}"
export TVAI_MODEL_DATA_DIR="{model_dir}"
export VK_ICD_FILENAMES="${{HOME}}/.config/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="{egl_json}"
export TMPDIR="{r_shm}/hb_tmp"
"{ffmpeg_bin}" -hide_banner -nostdin -y -i "{r_shm}/active_in.mp4" {fps_arg}-sws_flags spline+accurate_rnd+full_chroma_int \
  -filter_complex "{tvai_filter}" \
  -c:v h264_nvenc -profile:v high -pix_fmt yuv420p -preset p7 -tune hq -rc vbr -cq 19 -b:v 0 -spatial_aq 1 -rc-lookahead 20 \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv -brand mp42 \
  -movflags +faststart+use_metadata_tags+write_colr \
  -map_metadata 0 -map_metadata:s:v 0:s:v -metadata comment="{meta_comment}" \
  -map "0:a:0?" -c:a aac -ar 48000 -ac 2 -b:a 192k \
  "{r_shm}/active_out.mp4"
"""
            t_render = time.time()
            with active_err.open("w") as err_f:
                ren_res = subprocess.run(
                    ["ssh", "-n", "-o", "BatchMode=yes", self.node_cfg.ssh_host, r_script],
                    stdout=subprocess.DEVNULL,
                    stderr=err_f,
                    timeout=14400,
                    check=False,
                )
            elapsed = max(0.1, time.time() - t_render)
            with active_out.open("wb") as f_out:
                subprocess.run(
                    ["ssh", "-n", "-o", "BatchMode=yes", self.node_cfg.ssh_host, f"cat '{r_shm}/active_out.mp4' 2>/dev/null"],
                    stdout=f_out,
                    check=False,
                )
            subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", self.node_cfg.ssh_host, f"rm -f '{r_shm}/active_in.mp4' '{r_shm}/active_out.mp4'"],
                check=False,
            )
        else:
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
            env["TVAI_MODEL_DIR"] = model_dir
            env["TVAI_MODEL_DATA_DIR"] = model_dir
            env["TMPDIR"] = str(self.hb_dir)

            cmd = [
                ffmpeg_bin,
                "-hide_banner",
                "-nostdin",
                "-y",
                "-i",
                str(active_in),
            ]
            if avg_fps:
                cmd.extend(["-r", avg_fps])
            cmd.extend([
                "-sws_flags",
                "spline+accurate_rnd+full_chroma_int",
                "-filter_complex",
                tvai_filter,
                "-c:v",
                "h264_nvenc",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "p7",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "19",
                "-b:v",
                "0",
                "-spatial_aq",
                "1",
                "-rc-lookahead",
                "20",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-color_range",
                "tv",
                "-brand",
                "mp42",
                "-movflags",
                "+faststart+use_metadata_tags+write_colr",
                "-map_metadata",
                "0",
                "-map_metadata:s:v",
                "0:s:v",
                "-metadata",
                f"comment={meta_comment}",
                "-map",
                "0:a:0?",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-b:a",
                "192k",
                str(active_out),
            ])

            t_render = time.time()
            with active_err.open("w") as err_f:
                ren_res = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=err_f, timeout=14400, check=False)
            elapsed = max(0.1, time.time() - t_render)
            active_in.unlink(missing_ok=True)

        out_bytes = active_out.stat().st_size if active_out.is_file() else 0
        if ren_res.returncode != 0 or out_bytes < 1048576:
            err_tail = subprocess.run(["tail", "-n", "12", str(active_err)], capture_output=True, text=True, errors="replace", check=False).stdout
            self.logger.error(
                "TopazLinuxWorker (%s): render failed for '%s' (rc=%d, size=%dB):\n%s",
                self.node_id,
                job.filename,
                ren_res.returncode,
                out_bytes,
                err_tail,
            )
            self.cleanup_job(job)
            return False, 0, elapsed, 0.0

        fps = (nb_frames / elapsed) if nb_frames > 0 else 0.0
        final_name = f"{job.stem}_iris3.mp4"
        remote_dest = f"{self.cluster_cfg.storage.destination_dir}/{final_name}"
        remote_part = f"{remote_dest}.part"

        with active_out.open("rb") as f_in:
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", self.cluster_cfg.storage.archive_host, f'cat > "{remote_part}"'],
                stdin=f_in,
                check=False,
            )

        st_res = subprocess.run(
            [
                "ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                self.cluster_cfg.storage.archive_host,
                f"stat -c %s \"{remote_part}\" 2>/dev/null || stat -f %z \"{remote_part}\" 2>/dev/null || echo 0",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        st_lines = [ln.strip() for ln in st_res.stdout.splitlines() if ln.strip().isdigit()]
        dest_bytes = int(st_lines[-1]) if st_lines else 0

        self.cleanup_job(job)

        if dest_bytes == out_bytes and dest_bytes > 0:
            subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.cluster_cfg.storage.archive_host, f"mv -f \"{remote_part}\" \"{remote_dest}\""],
                check=False,
            )
            return True, dest_bytes, elapsed, fps
        else:
            self.logger.warning("TopazLinuxWorker (%s): delivery byte mismatch for '%s' (%dB vs %dB)", self.node_id, final_name, out_bytes, dest_bytes)
            subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.cluster_cfg.storage.archive_host, f"rm -f \"{remote_part}\" 2>/dev/null || true"],
                check=False,
            )
            return False, 0, elapsed, 0.0

    def cleanup_job(self, job: JobItem) -> None:
        for p in self.shm_dir.glob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)
