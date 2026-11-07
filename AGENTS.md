# AGENTS.md — AI Pair-Configuration Playbook & Architectural Invariants

This document is designed for AI coding assistants (**Google Antigravity**, **Claude Code**, **Cursor**, **Windsurf**, **GitHub Copilot**) and maintainers working with `topaz-video-fabric`.

If a user asks you to **"help install, configure, or run `topaz-video-fabric`"** on their machine(s), follow the **AI Pair-Configuration Playbook** in Section 1 step-by-step.

---

## 1. AI Pair-Configuration Playbook (Step-by-Step)

When pair-configuring this repository for a user, follow these steps:

### Step 1: Interview the User & Identify Target Hardware Topology
Ask the user which deployment path matches their hardware:
1. **Path 1 — Pure Apple Silicon macOS (`ane` / `ane_gpu`):** One or more Macs (`M1` / `M2` / `M3` / `M4`) running macOS with Topaz Video AI installed in `/Applications/Topaz Video.app` (or `/Applications/Topaz Video AI.app`).
2. **Path 2 — Heterogeneous Mixed Fabric (`linux_cuda` + `cuda` + `ane`):** Any combination of Ubuntu/Debian Linux NVIDIA servers (`.deb` package installed in `/opt/TopazVideoAIBETA` or extracted in `~/topaz_linux/opt/TopazVideoAIBETA`), Windows NVIDIA workstations (bridged via WSL2 `/dev/shm` UNC paths `\\wsl.localhost\<Distro>\dev\shm\...`), and Apple Silicon Macs.

Collect from the user:
- **Source video directory** (`source_dir`) and **4K output directory** (`destination_dir`) on the storage archive host (`archive_host`, which can be `127.0.0.1` for local storage or `<USER>@<HOST>` over SSH).
- **Target worker nodes** (SSH `user@host`, OS type, GPU / Apple Silicon model).

### Step 2: Generate `config/cluster_config.local.yaml` (Never Edit Tracked `cluster_config.yaml` with Private IPs)
- Copy `config/cluster_config.example.yaml` to `config/cluster_config.local.yaml` (which is gitignored and automatically prioritized by `load_cluster_config()`).
- Populate `storage` and `nodes` in `config/cluster_config.local.yaml`:
  - For **macOS Apple Silicon nodes**: set `accelerator: "ane"` (or `"ane_gpu"`), `instances: 1` (strictly `1` due to ANE hardware queue serialization), `ramdisk_mount: "/Volumes/TopazRAM"`, and `ramdisk_size_mb: 3500`.
  - For **Native Linux / WSL2 CUDA nodes**: set `accelerator: "linux_cuda"`, `instances: 2` or `3` (for 16GB-32GB VRAM), `wsl_shm_dir: "/dev/shm/topaz_<node>_ram"`, and verify `ffmpeg_path` / `model_dir`.
  - For **Windows WSL2 CUDA nodes**: set `accelerator: "cuda"`, `wsl_shm_dir: "/dev/shm/topaz_<node>_ram"`, and `win_unc_shm: "\\\\wsl.localhost\\<DistroName>\\dev\\shm\\topaz_<node>_ram"`.

### Step 3: Verify Prerequisites & OAuth Activation (`auth.tpz`)
- Ensure `ffmpeg` and `handbrake-cli` (`brew install ffmpeg handbrake` on macOS, `sudo apt install -y ffmpeg handbrake-cli` on Ubuntu/WSL2) are installed for Phase-1 in-DRAM container/bitstream repair.
- For Linux `.deb` / WSL2 nodes, verify that `/opt/TopazVideoAIBETA/models/auth.tpz` (or `~/topaz_linux/opt/TopazVideoAIBETA/models/auth.tpz`) is present (either via GUI login, cloning an already-activated `topaz_linux/` tree, or copying `auth.tpz` from the user's licensed macOS/Windows GUI `models/` directory).

### Step 4: Run Preflight Host Assurance (`--syschk`) & Dry-Run Queue Audit
Execute the built-in preflight checker and dry-run preview:
```bash
python3 -m topaz_cluster.orchestrator --syschk
python3 -m topaz_cluster.orchestrator --dry-run
```
Confirm every enabled node reports `PASS` (or `WARN (DEMAND)` for unmounted macOS `/Volumes/TopazRAM`, which `TopazMacWorker` mounts automatically on start).

### Step 5: Launch & Dynamically Manage the Fabric
- Launch the orchestrator:
  ```bash
  python3 -m topaz_cluster.orchestrator
  ```
- If the user has corrupted web/cloud clips, add `--force-condition`. If the user wants center-cropped 16:9/9:16 fill instead of uncropped native aspect ratio, add `--allow-crop`.
- To add, enable, or disable worker nodes while a batch is running, edit `config/cluster_config.local.yaml` on the fly—`FabricDispatcher.sync_dynamic_nodes()` and `BaseWorker.reload_config()` apply changes dynamically without restarting the process.

---

## 2. Core Architectural & Safety Invariants

1. **Zero-Plaintext-on-NAND (Volatile DRAM Staging Only):**
   - Workers must **never** stage raw inputs, intermediate conditioned videos, or unverified output streams on unencrypted internal SSDs/HDDs (`/tmp`, `/private/tmp`, `C:\Temp`).
   - **macOS (`TopazMacWorker`):** Must stage exclusively on a volatile HFS+ RAMDisk mounted at `/Volumes/TopazRAM` (`hdiutil attach -nomount ram://...`) and wipe files with `rm -P -f` after each job.
   - **Linux / WSL2 (`TopazLinuxWorker`):** Must stage exclusively in `/dev/shm/topaz_*`.
   - **Windows (`TopazWindowsWorker`):** Must stage in WSL2 `/dev/shm` accessed via `\\wsl.localhost\<Distro>\dev\shm\...` UNC paths (or an explicitly configured encrypted scratch volume).

2. **Phase-1 In-DRAM Corruption Protection (`hb_tmp`) & Automatic Repair Retry:**
   - Legacy containers (`.avi`, `.wmv`, `.mkv`, `.ts`), non-standard audio (`pcm_alaw`, `pcm_mulaw`, `adpcm_ima_wav`), or streams flagged via `--force-condition` are repaired in DRAM (`hb_tmp`) using `HandBrakeCLI --cfr` or `ffmpeg -fflags +genpts+igndts -err_detect ignore_err` before invoking `tvai_up`.
   - If a raw `.mp4`/`.mov` fails inside Topaz `ffmpeg` due to hidden bitstream/slice corruption, workers automatically retry the job once with Phase-1 in-DRAM corruption repair enabled (`force_repair=True`).

3. **Smart Aspect-Threshold 4K Framing Policy (`ModelManager.compute_smart_4k_framing`):**
   - Sources within 15% of `16:9` or `9:16` (e.g., `832x480`, `404x720`, `720x1198`) automatically snap and center-crop (`force_original_aspect_ratio=increase,crop=W:H,setsar=1:1`) to exact `3840x2160` or `2160x3840` 4K UHD.
   - Near-square (`1:1`) sources snap to `2160x2160`, and wider-delta sources (such as `4:3` `640x480` / `640x464`) preserve their uncropped native aspect ratio inside the 4K bounding box (`ModelManager.compute_native_4k_dims`).
   - Passing `--allow-crop` (`allow_crop=True`) forces center-crop fill across all sources, and `--rm-source` (`delete_source_after_render=True`) deletes each source video after its 4K deliverable is byte-verified.

4. **Hardware Instance Bounds, Iris-3 `scale=2` Rule & Local Project Starlight Support:**
   - **Apple Silicon (`ane`, `ane_gpu`, `metal`):** `instances` must remain `1` for GAN models (`0` for Starlight `neuroserver` models) because the Apple Neural Engine hardware queue strictly serializes neural graph execution.
   - **NVIDIA (`cuda`, `linux_cuda`):** `instances` may be configured between `1` and `3` depending on VRAM capacity (`16 GB - 32 GB`) for TensorRT GAN models (`iris-3`, `rhea-1`, `prob-4`, `nyx-3`).
   - **Iris-3 `scale=2`:** Default `iris-3` 4K upscaling to `scale=2` followed by `scale=w=<out_w>:h=<out_h>:flags=lanczos:threads=0,setsar=1:1` (unlocks native `-2x-13-ane-mlp` ANE weights on macOS and boosts NVIDIA TensorRT throughput by `2.6x`).
   - **Local Project Starlight Models (`slp-2.5` / `starlight`, `astrasharp` / `starlight-sharp`, `astrafast` / `starlight-fast`, `astrahq` / `starlight-hq`, `astra` / `starlight-creative`, `slm-1` / `starlight-mini`):**
     - Local Starlight (`neuroserver`) models are **not** supported on native Linux `.deb` distributions (`TopazVideoAIBETA`).
     - To run local Starlight models on a Linux GPU server, **Windows Virtualization (KVM/QEMU with PCIe VFIO GPU passthrough)** is required and orchestrated via `TopazWindowsWorker` (`vm_temp_dir`), or run on macOS Apple Silicon (`TopazMacWorker`) / native Windows (`TopazWindowsWorker`). Native `linux_cuda` workers automatically detect `supports_linux_deb == False` and skip Starlight jobs cleanly.

---

## 3. Configuration & Public Sanitization Policy

- **Private Local Configs (`config/cluster_config.local.yaml` & `.env`):**
  - `load_cluster_config()` automatically loads `config/cluster_config.local.yaml` when present.
  - Never commit `cluster_config.local.yaml`, `.env`, `*.tpz`, or `*.tz` to git.
- **Public Sanitization Policy:**
  - All tracked files (`config/cluster_config.yaml`, `README.md`, `AGENTS.md`, Python source files, and unit tests) must use generic placeholders (`<USER>@<HOST>`, `/path/to/source`, `/path/to/destination`).
  - Never hardcode private LAN/Tailscale IPv4 addresses, personal hostnames, or personal usernames in tracked files.
  - Always run `pytest -v` before committing changes.
