# AGENTS.md — AI Coding Assistant & Maintainer Guidelines

This document provides architectural context, safety invariants, and operational rules for AI coding assistants (Antigravity, Claude Code, Cursor, GitHub Copilot) and contributors working on `topaz-cluster`.

---

## 1. Core Architectural Invariants

1. **Zero-Plaintext-on-NAND (Volatile DRAM Staging Only):**
   - Workers must **never** stage raw inputs, intermediate conditioned videos, or unverified output streams on unencrypted internal SSDs/HDDs (`/tmp`, `/private/tmp`, `C:\Temp`).
   - **macOS (`TopazMacWorker`):** Must stage exclusively on a volatile HFS+ RAMDisk mounted at `/Volumes/TopazRAM` (`hdiutil attach -nomount ram://...`) and wipe files with `rm -P -f` after each job.
   - **Linux / WSL2 (`TopazLinuxWorker`):** Must stage exclusively in `/dev/shm/topaz_*`.
   - **Windows (`TopazWindowsWorker`):** Must stage in WSL2 `/dev/shm` accessed via `\\wsl.localhost\<Distro>\dev\shm\...` UNC paths (or an explicitly configured encrypted scratch volume).

2. **Immutable Aspect Ratio & Zero-Cropping Policy (`allow_crop = False`):**
   - Never inject `crop=` or `force_original_aspect_ratio=increase` into any FFmpeg filter graph.
   - Always compute native even-integer 4K bounding dimensions via `ModelManager.compute_native_4k_dims(src_w, src_h, target_w, target_h)` and append `setsar=1:1` with `flags=lanczos`.

3. **Hardware Instance Bounds:**
   - **Apple Silicon (`ane`, `ane_gpu`, `metal`):** `instances` must remain `1` because the Apple Neural Engine hardware queue strictly serializes neural graph execution.
   - **NVIDIA (`cuda`, `linux_cuda`):** `instances` may be configured between `1` and `3` depending on VRAM capacity (`16 GB - 32 GB`).

4. **Iris-3 `scale=2` Rule:**
   - For `iris-3` 4K upscaling, always default to `scale=2` followed by `scale=w=<out_w>:h=<out_h>:flags=lanczos:threads=0,setsar=1:1`.

---

## 2. Configuration & Sanitization Rules

- **Private Local Configs (`config/cluster_config.local.yaml` & `.env`):**
  - `load_cluster_config()` automatically loads `config/cluster_config.local.yaml` when present.
  - Never commit `cluster_config.local.yaml` or `.env` to git.
- **Public Sanitization Policy:**
  - All tracked files (`config/cluster_config.yaml`, `README.md`, `AGENTS.md`, Python source files, and unit tests) must use generic placeholders (`<USER>@<HOST>`, `/path/to/source`, `/path/to/destination`).
  - Never hardcode private LAN/Tailscale IPv4 addresses, personal hostnames, or personal usernames in tracked files.
  - Before committing, verify zero private identifiers exist in tracked files using `git grep`.
