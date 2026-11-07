"""Topaz Windows Runtime Worker.

Unified runtime for Windows-based Topaz Video AI execution (Blackwell, Volta, TensorRT, NVENC).
Supports both local WSL-shared DRAM (UNC paths) and remote Windows Server VMs (D:\\Temp).
"""

import base64
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


class TopazWindowsWorker(BaseWorker):
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
        self.is_wsl = bool(node_cfg.wsl_shm_dir and node_cfg.win_unc_shm)
        self.wsl_shm_dir = node_cfg.wsl_shm_dir or f"/dev/shm/topaz_{node_id}_ram"
        self.win_unc_shm = node_cfg.win_unc_shm
        self.vm_temp_dir = node_cfg.vm_temp_dir or "D:/Temp"
        self.local_hb_dir = Path(node_cfg.local_hb_dir or f"/dev/shm/topaz_{node_id}_hb")
        self.local_hb_dir.mkdir(parents=True, exist_ok=True)

    def _run_node_cmd(self, cmd: str, timeout: int = 14400) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.node_cfg.ssh_host, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _condition_audio_and_containers(self, raw_path: Path, repaired_path: Path, orig_h: int) -> Tuple[bool, int, int, int]:
        ext = raw_path.suffix.lower()
        ac_res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(raw_path)],
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
            raw_path.rename(repaired_path)
        else:
            hb_env = os.environ.copy()
            hb_env["TMPDIR"] = str(self.local_hb_dir)
            if ext in (".avi", ".wmv", ".ts") and shutil.which("HandBrakeCLI"):
                hb_cmd = [
                    "nice", "-n", "15", "HandBrakeCLI",
                    "-i", str(raw_path),
                    "-o", str(repaired_path),
                    "-f", "av_mp4", "-e", "x264", "-q", "16", "--cfr",
                    "-E", "av_aac", "-B", "192",
                ]
            else:
                hb_cmd = [
                    "nice", "-n", "15", "ffmpeg", "-hide_banner", "-nostdin", "-y",
                    "-fflags", "+genpts+igndts", "-err_detect", "ignore_err",
                    "-i", str(raw_path),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    str(repaired_path),
                ]
            hb_res = subprocess.run(
                hb_cmd,
                env=hb_env,
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            raw_path.unlink(missing_ok=True)
            if hb_res.returncode != 0 or not repaired_path.is_file() or repaired_path.stat().st_size < 65536:
                repaired_path.unlink(missing_ok=True)
                return False, 0, 0, 0

        # Probe dimensions
        w, h, nbf = 1280, orig_h, 0
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames", "-of", "csv=p=0", str(repaired_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        parts = [x.strip() for x in pr.stdout.strip().split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            w, h = int(parts[0]), int(parts[1])
        if len(parts) >= 3 and parts[2].isdigit():
            nbf = int(parts[2])
        return True, w, h, nbf

    def process_job(self, job: JobItem) -> Tuple[bool, int, float, float]:
        ext = Path(job.filename).suffix.lower()
        src_path = f"{self.cluster_cfg.storage.source_dir}/{job.filename}"

        # 1. Build canonical model filter from ModelManager
        instances = self.model_mgr.get_optimal_instances(self.node_cfg.accelerator, self.cluster_cfg.model.name)
        if self.node_cfg.instances > 0:
            instances = self.node_cfg.instances

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
        )

        final_name = f"{job.stem}_iris3.mp4"
        remote_dest = f"{self.cluster_cfg.storage.destination_dir}/{final_name}"

        if self.is_wsl:
            # === WSL DRAM Staging Branch (e.g. RTX 5080 local, RTX 4070 remote) ===
            is_local = (self.node_cfg.ssh_host == self.cluster_cfg.storage.archive_host)
            force_hb_flag = "1" if self.cluster_cfg.model.force_condition else "0"
            if is_local:
                stage_script = f"""
set -e
mkdir -p "{self.wsl_shm_dir}/hb_tmp"
rm -f "{self.wsl_shm_dir}"/*.mp4 "{self.wsl_shm_dir}"/*.avi "{self.wsl_shm_dir}"/*.mov "{self.wsl_shm_dir}"/*.log 2>/dev/null || true
cp "{src_path}" "{self.wsl_shm_dir}/raw_in{ext}"

A_CODEC=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "{self.wsl_shm_dir}/raw_in{ext}" 2>/dev/null || echo "")
NEED_HB={force_hb_flag}
if [[ "{ext}" != ".mp4" && "{ext}" != ".mov" && "{ext}" != ".m4v" ]]; then
    NEED_HB=1
elif [[ "$A_CODEC" == "pcm_alaw" || "$A_CODEC" == "pcm_mulaw" || "$A_CODEC" == "adpcm_ima_wav" ]]; then
    NEED_HB=1
fi

if [[ "$NEED_HB" == "1" ]]; then
    export TMPDIR="{self.wsl_shm_dir}/hb_tmp" TEMP="{self.wsl_shm_dir}/hb_tmp" TMP="{self.wsl_shm_dir}/hb_tmp"
    if [[ ("{ext}" == ".avi" || "{ext}" == ".wmv" || "{ext}" == ".ts") ]] && command -v HandBrakeCLI >/dev/null 2>&1; then
        nice -n 10 HandBrakeCLI -i "{self.wsl_shm_dir}/raw_in{ext}" -o "{self.wsl_shm_dir}/active_in.mp4" -f av_mp4 -e x264 -q 16 --cfr -E av_aac -B 192 >/dev/null 2>&1
    else
        nice -n 10 ffmpeg -hide_banner -nostdin -y -fflags +genpts+igndts -err_detect ignore_err -i "{self.wsl_shm_dir}/raw_in{ext}" -c:v libx264 -preset veryfast -crf 16 -pix_fmt yuv420p -c:a aac -b:a 192k "{self.wsl_shm_dir}/active_in.mp4" >/dev/null 2>&1
    fi
    rm -f "{self.wsl_shm_dir}/raw_in{ext}"
else
    mv "{self.wsl_shm_dir}/raw_in{ext}" "{self.wsl_shm_dir}/active_in.mp4"
fi

ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 "{self.wsl_shm_dir}/active_in.mp4" | tail -n 1
"""
                st_res = self._run_node_cmd(stage_script, timeout=600)
                if st_res.returncode != 0:
                    return False, 0, 0.0, 0.0

                w, h, nb_frames = job.width, job.height, 0
                ln0 = st_res.stdout.strip().splitlines()[-1] if st_res.stdout.strip() else ""
                parts = [x.strip() for x in ln0.split(",")]
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    w, h = int(parts[0]), int(parts[1])
                if len(parts) >= 3 and parts[2].isdigit():
                    nb_frames = int(parts[2])
            else:
                # Remote WSL host: condition in local orchestrator DRAM and upload to remote /dev/shm
                raw_shm = self.local_hb_dir / f"{self.node_id}_raw{ext}"
                repaired_shm = self.local_hb_dir / f"{self.node_id}_repaired.mp4"
                for p in (raw_shm, repaired_shm):
                    p.unlink(missing_ok=True)

                with raw_shm.open("wb") as f_out:
                    cp_res = subprocess.run(
                        ["ssh", "-n", "-o", "BatchMode=yes", self.cluster_cfg.storage.archive_host, f"cat \"{src_path}\""],
                        stdout=f_out,
                        check=False,
                    )
                if cp_res.returncode != 0 or not raw_shm.is_file() or raw_shm.stat().st_size < 32768:
                    raw_shm.unlink(missing_ok=True)
                    return False, 0, 0.0, 0.0

                ok, w, h, nb_frames = self._condition_audio_and_containers(raw_shm, repaired_shm, job.height)
                if not ok:
                    return False, 0, 0.0, 0.0

                self._run_node_cmd(f'rm -rf "{self.wsl_shm_dir}"/* 2>/dev/null || true && mkdir -p "{self.wsl_shm_dir}/hb_tmp"')
                scp_up = subprocess.run(
                    ["scp", "-q", str(repaired_shm), f"{self.node_cfg.ssh_host}:{self.wsl_shm_dir}/active_in.mp4"],
                    check=False,
                )
                repaired_shm.unlink(missing_ok=True)
                if scp_up.returncode != 0:
                    return False, 0, 0.0, 0.0

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
            win_in = f"{self.win_unc_shm}\\active_in.mp4"
            win_out = f"{self.win_unc_shm}\\active_out.mp4"
            win_err = f"{self.win_unc_shm}\\active_stderr.log"
            safe_node_name = self.node_cfg.name.replace("(", "").replace(")", "").replace("|", "-")
            meta_comment = f"Topaz Video 4K - Node: {safe_node_name} - model={self.cluster_cfg.model.name} src={w}x{h} out={out_w}x{out_h}"
            ffmpeg_bin = self.node_cfg.ffmpeg_path or "C:\\Program Files\\Topaz Labs LLC\\Topaz Video\\ffmpeg.exe"
            model_dir = self.node_cfg.model_dir or "C:\\ProgramData\\Topaz Labs LLC\\Topaz Video\\models"

            ps1_content = (
                "$ProgressPreference = 'SilentlyContinue'\n"
                "$ErrorActionPreference = 'SilentlyContinue'\n"
                f'$env:TVAI_MODEL_DIR = "{model_dir}"\n'
                f'$env:TVAI_MODEL_DATA_DIR = "{model_dir}"\n'
                f'$env:TMP = "{self.win_unc_shm}\\hb_tmp"\n'
                f'$env:TEMP = "{self.win_unc_shm}\\hb_tmp"\n'
                f'& "{ffmpeg_bin}" -hide_banner -nostdin -y '
                f'-i "{win_in}" -sws_flags spline+accurate_rnd+full_chroma_int '
                f'-filter_complex "{tvai_filter}" '
                f"-c:v h264_nvenc -profile:v high -pix_fmt yuv420p -preset p7 -tune hq -rc vbr -cq 19 -b:v 0 -spatial_aq 1 -rc-lookahead 20 "
                f"-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv "
                f"-movflags frag_keyframe+empty_moov+delay_moov+use_metadata_tags+write_colr "
                f'-map_metadata 0 -map_metadata:s:v 0:s:v -metadata comment="{meta_comment}" '
                f'-map "0:a:0?" -c:a aac -ar 48000 -ac 2 -b:a 192k '
                f'"{win_out}" 2> "{win_err}"\n'
                "exit $LASTEXITCODE\n"
            )
            encoded_ps1 = base64.b64encode(ps1_content.encode("utf-16le")).decode("ascii")

            t_render = time.time()
            encode_script = f"""
/init /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -EncodedCommand "{encoded_ps1}" >/dev/null 2>&1
stat -c %s "{self.wsl_shm_dir}/active_out.mp4" 2>/dev/null || echo 0
"""
            enc_res = self._run_node_cmd(encode_script, timeout=14400)
            elapsed = max(0.1, time.time() - t_render)
            sz_lines = [ln.strip() for ln in enc_res.stdout.splitlines() if ln.strip().isdigit()]
            out_bytes = int(sz_lines[-1]) if sz_lines else 0

            if out_bytes < 1048576:
                return False, 0, elapsed, 0.0

            fps = (nb_frames / elapsed) if nb_frames > 0 else 0.0

            if is_local:
                deliver_script = f"""
mv "{self.wsl_shm_dir}/active_out.mp4" "{remote_dest}"
rm -rf "{self.wsl_shm_dir}"/* 2>/dev/null || true
stat -c %s "{remote_dest}" 2>/dev/null || echo 0
"""
                deliv_res = self._run_node_cmd(deliver_script, timeout=600)
                d_lines = [ln.strip() for ln in deliv_res.stdout.splitlines() if ln.strip().isdigit()]
                dest_bytes = int(d_lines[-1]) if d_lines else 0
            else:
                subprocess.run(
                    ["scp", "-3", "-q", f"{self.node_cfg.ssh_host}:{self.wsl_shm_dir}/active_out.mp4", f"{self.cluster_cfg.storage.archive_host}:{remote_dest}"],
                    check=False,
                )
                self._run_node_cmd(f'rm -rf "{self.wsl_shm_dir}"/* 2>/dev/null || true')
                st_res = subprocess.run(
                    ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.cluster_cfg.storage.archive_host, f"stat -c %s '{remote_dest}' 2>/dev/null || echo 0"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                d_lines = [ln.strip() for ln in st_res.stdout.splitlines() if ln.strip().isdigit()]
                dest_bytes = int(d_lines[-1]) if d_lines else 0

            if dest_bytes == out_bytes and dest_bytes > 0:
                return True, dest_bytes, elapsed, fps
            else:
                self._run_node_cmd(f"rm -f '{remote_dest}' 2>/dev/null || true", timeout=15)
                return False, 0, elapsed, 0.0

        else:
            # === Remote VM / D:\Temp Staging Branch (e.g. Tesla V100) ===
            raw_shm = self.local_hb_dir / f"v100_raw{ext}"
            repaired_shm = self.local_hb_dir / "v100_repaired.mp4"
            for p in (raw_shm, repaired_shm):
                p.unlink(missing_ok=True)

            cp_res = subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", self.cluster_cfg.storage.archive_host, f"cat '{src_path}'"],
                stdout=raw_shm.open("wb"),
                check=False,
            )
            if cp_res.returncode != 0 or not raw_shm.is_file() or raw_shm.stat().st_size < 1048576:
                raw_shm.unlink(missing_ok=True)
                return False, 0, 0.0, 0.0

            ok, w, h, nb_frames = self._condition_audio_and_containers(raw_shm, repaired_shm, job.height)
            if not ok:
                return False, 0, 0.0, 0.0

            remote_in = f"{self.vm_temp_dir}/active_{self.node_id}_in.mp4"
            remote_out = f"{self.vm_temp_dir}/active_{self.node_id}_out.mp4"
            remote_bat = f"{self.vm_temp_dir}/active_{self.node_id}_run.bat"
            remote_stdout = f"{self.vm_temp_dir}/active_{self.node_id}_stdout.log"
            remote_stderr = f"{self.vm_temp_dir}/active_{self.node_id}_stderr.log"

            win_in = remote_in.replace("/", "\\")
            win_out = remote_out.replace("/", "\\")
            win_bat = remote_bat.replace("/", "\\")
            win_stdout = remote_stdout.replace("/", "\\")
            win_stderr = remote_stderr.replace("/", "\\")

            # Upload to VM
            scp_res = subprocess.run(["scp", "-q", str(repaired_shm), f"{self.node_cfg.ssh_host}:{remote_in}"], check=False)
            repaired_shm.unlink(missing_ok=True)
            if scp_res.returncode != 0:
                return False, 0, 0.0, 0.0

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
            safe_node_name = self.node_cfg.name.replace("(", "").replace(")", "").replace("|", "-")
            meta_comment = f"Topaz Video 4K - Node: {safe_node_name} - model={self.cluster_cfg.model.name} src={w}x{h} out={out_w}x{out_h}"
            ffmpeg_bin = self.node_cfg.ffmpeg_path or "C:\\Program Files\\Topaz Labs LLC\\Topaz Video\\ffmpeg.exe"
            model_dir = self.node_cfg.model_dir or "C:\\ProgramData\\Topaz Labs LLC\\Topaz Video\\models"

            bat_content = (
                "@echo off\r\n"
                f'set "TVAI_MODEL_DIR={model_dir}"\r\n'
                f'set "TVAI_MODEL_DATA_DIR={model_dir}"\r\n'
                f'set "TMP={self.vm_temp_dir.replace("/", chr(92))}"\r\n'
                f'set "TEMP={self.vm_temp_dir.replace("/", chr(92))}"\r\n'
                f'"{ffmpeg_bin}" -hide_banner -nostdin -y '
                f'-i "{win_in}" -sws_flags spline+accurate_rnd+full_chroma_int '
                f'-filter_complex "{tvai_filter}" '
                f"-c:v h264_nvenc -profile:v high -pix_fmt yuv420p -preset p7 -tune hq -rc vbr -cq 19 -b:v 0 -spatial_aq 1 -rc-lookahead 20 "
                f"-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv "
                f"-movflags frag_keyframe+empty_moov+delay_moov+use_metadata_tags+write_colr "
                f'-map_metadata 0 -map_metadata:s:v 0:s:v -metadata comment="{meta_comment}" '
                f'-map "0:a:0?" -c:a aac -ar 48000 -ac 2 -b:a 192k '
                f'"{win_out}" > "{win_stdout}" 2> "{win_stderr}"\r\n'
                "exit /b %ERRORLEVEL%\r\n"
            )

            local_bat = self.local_hb_dir / f"active_{self.node_id}_run.bat"
            local_bat.write_text(bat_content, encoding="utf-8")
            up_bat = subprocess.run(["scp", "-q", str(local_bat), f"{self.node_cfg.ssh_host}:{remote_bat}"], check=False)
            local_bat.unlink(missing_ok=True)
            if up_bat.returncode != 0:
                return False, 0, 0.0, 0.0

            t_render = time.time()
            enc_proc = self._run_node_cmd(f'cmd /c "{win_bat}"', timeout=14400)
            elapsed = max(0.1, time.time() - t_render)

            sz_res = self._run_node_cmd(f'powershell -NoProfile -Command "(Get-Item -Path \'{win_out}\' -ErrorAction SilentlyContinue).Length"')
            sz_lines = [ln.strip() for ln in sz_res.stdout.splitlines() if ln.strip().isdigit()]
            out_bytes = int(sz_lines[-1]) if sz_lines else 0

            if enc_proc.returncode != 0 or out_bytes < 1048576:
                self._run_node_cmd(f'cmd /c del /f /q "{win_in}" "{win_out}" "{win_bat}" "{win_stdout}" "{win_stderr}"')
                return False, 0, elapsed, 0.0

            fps = (nb_frames / elapsed) if nb_frames > 0 else 0.0
            subprocess.run(["scp", "-3", "-q", f"{self.node_cfg.ssh_host}:{remote_out}", f"{self.cluster_cfg.storage.archive_host}:{remote_dest}"], check=False)

            st_res = subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.cluster_cfg.storage.archive_host, f"stat -c %s '{remote_dest}' 2>/dev/null || echo 0"],
                capture_output=True,
                text=True,
                check=False,
            )
            st_lines = [ln.strip() for ln in st_res.stdout.splitlines() if ln.strip().isdigit()]
            dest_bytes = int(st_lines[-1]) if st_lines else 0

            # Wipe remote scratch files
            self._run_node_cmd(f'cmd /c del /f /q "{win_in}" "{win_out}" "{win_bat}" "{win_stdout}" "{win_stderr}"')

            if dest_bytes == out_bytes and dest_bytes > 0:
                return True, dest_bytes, elapsed, fps
            else:
                subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.cluster_cfg.storage.archive_host, f"rm -f '{remote_dest}' 2>/dev/null || true"], check=False)
                return False, 0, elapsed, 0.0

    def cleanup_job(self, job: JobItem) -> None:
        if self.is_wsl:
            self._run_node_cmd(f"rm -rf '{self.wsl_shm_dir}'/* 2>/dev/null || true", timeout=15)
        else:
            self._run_node_cmd(f"cmd /c del /f /q {self.vm_temp_dir.replace('/', chr(92))}\\{job.stem}*")
