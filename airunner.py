#!/usr/bin/env python3
"""
Airunner — local model runner manager.

Zero-dependency Python stdlib backend. Manages llama.cpp (llama-server) and
DwarfStar (ds4-server) model processes: launch, stop, watch/restart, remember
setups, autostart at boot (systemd user unit), and stream output to a web UI.
"""

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import shutil
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

APP_NAME = "airunner"
STATE_DIR = os.path.expanduser("~/.local/share/airunner")
STATE_FILE = os.path.join(STATE_DIR, "state.json")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
SYSTEMD_UNIT = "airunner.service"
SYSTEMD_DIR = os.path.expanduser("~/.config/systemd/user")

DWARFSTAR_GGUF_DIR = "/home/fred/dwarfstar/gguf"
# Official Qwen3.8 chat template (carries the reasoning_effort / preserve_thinking
# logic the model's GGUFs lack). Needed for the reasoning_effort option to work.
QWEN38_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "templates", "qwen3.8-chat-template.jinja")

DEFAULT_CONFIG = {
    "llamacpp_bin": "/home/fred/ai/llama.cpp/build/bin/llama-server",
    "dwarfstar_bin": "/home/fred/dwarfstar/ds4-server",
    "strix_bin": "/home/fred/ai/strix-halo-llamacpp/build-vk/bin/llama-server",
    "rocmfpx_bin": "/home/fred/ai/rocmfpx-llamacpp/build-strix-rocmfp4/bin/llama-server",
    "model_dirs": ["/home/fred/ai/models", DWARFSTAR_GGUF_DIR],
    "host": "127.0.0.1",
    "port": 8090,
    "watchdog_interval": 5,
}

RUNNER_LABEL = {
    "llamacpp": "llama.cpp (llama-server)",
    "dwarfstar": "DwarfStar (ds4-server)",
    "strix": "StrixHalo llama.cpp (Vulkan)",
    "rocmfpx": "ROCmFPX llama.cpp (Vulkan/ROCm)",
}

# ---------------------------------------------------------------------------
# Launch option knowledge base.
#   Each option: {label, type, default, desc, suggested}
#   type: int | float | str | flag | choice
# ---------------------------------------------------------------------------

LLAMACPP_OPTS = [
    {"label": "-m, --model", "key": "model", "type": "str", "default": "",
     "desc": "GGUF model path to load", "suggested": ""},
    {"label": "-lm, --load-mode", "key": "load_mode", "type": "choice", "default": "mmap",
     "choices": ["mmap", "none", "mlock", "mmap+mlock", "dio"],
     "desc": "Model loading mode (dio = DirectIO disk reads; replaces the deprecated -dio/--direct-io flag)", "suggested": "dio"},
    {"label": "--mmproj", "key": "mmproj", "type": "str", "default": "",
     "desc": "Multimodal projector GGUF path for vision support", "suggested": ""},
    {"label": "-c, --ctx-size", "key": "ctx_size", "type": "int", "default": "131072",
     "desc": "Prompt context size (tokens)", "suggested": "131072"},
    {"label": "-t, --threads", "key": "threads", "type": "int", "default": "-1",
     "desc": "CPU threads for generation (-1 = auto)", "suggested": str(min(32, os.cpu_count() or 4))},
    {"label": "-ngl, --n-gpu-layers", "key": "gpu_layers", "type": "int", "default": "-1",
     "desc": "Layers to keep in VRAM (-1 = all, 0 = CPU only)", "suggested": "-1"},
    {"label": "--host", "key": "host", "type": "str", "default": "0.0.0.0",
     "desc": "Bind address", "suggested": "0.0.0.0"},
    {"label": "--port", "key": "port", "type": "int", "default": "8080",
     "desc": "HTTP API port", "suggested": ""},
    {"label": "--temp", "key": "temp", "type": "float", "default": "0.8",
     "desc": "Sampling temperature", "suggested": "0.8"},
    {"label": "--top-k", "key": "top_k", "type": "int", "default": "40",
     "desc": "Top-k sampling (0 = disabled)", "suggested": "40"},
    {"label": "--top-p", "key": "top_p", "type": "float", "default": "0.95",
     "desc": "Top-p sampling", "suggested": "0.95"},
    {"label": "--min-p", "key": "min_p", "type": "float", "default": "0.05",
     "desc": "Min-p sampling", "suggested": "0.05"},
    {"label": "-rea, --reasoning", "key": "reasoning", "type": "choice", "default": "auto",
     "choices": ["on", "off", "auto"], "desc": "Use reasoning/thinking in the chat", "suggested": "auto"},
    {"label": "--reasoning-format", "key": "reasoning_format", "type": "choice", "default": "auto",
     "choices": ["auto", "none", "deepseek", "deepseek-legacy"], "desc": "Reasoning content format", "suggested": "auto"},
    {"label": "--reasoning-budget", "key": "reasoning_budget", "type": "int", "default": "-1",
     "desc": "Token budget for thinking (-1 = unrestricted, 0 = immediate end)", "suggested": "-1"},
    {"label": "--reasoning-preserve", "key": "reasoning_preserve", "type": "flag", "default": "off",
     "desc": "Preserve reasoning trace in full history", "suggested": "off"},
    {"label": "--reasoning-effort", "key": "reasoning_effort", "type": "choice", "default": "auto",
     "choices": ["auto", "xhigh", "medium", "low"], "no_cli": True,
     "desc": "Qwen3.8 reasoning depth (auto = model default: xhigh); sent as a chat template kwarg, needs the qwen3.8 chat template", "suggested": "medium"},
    {"label": "-s, --seed", "key": "seed", "type": "int", "default": "-1",
     "desc": "RNG seed (-1 = random)", "suggested": "-1"},
    {"label": "-fa, --flash-attn", "key": "flash_attn", "type": "choice", "default": "auto",
     "choices": ["auto", "on", "off"], "desc": "Use Flash Attention", "suggested": "auto"},
    {"label": "-np, --parallel", "key": "parallel", "type": "int", "default": "1",
     "desc": "Number of parallel slots", "suggested": "1"},
    {"label": "-a, --alias", "key": "alias", "type": "str", "default": "",
     "desc": "Model alias used by the API", "suggested": ""},
    {"label": "--chat-template-file", "key": "chat_template", "type": "str", "default": "",
     "desc": "Path to a chat template .jinja file", "suggested": ""},
    {"label": "--api-key", "key": "api_key", "type": "str", "default": "",
     "desc": "API key for auth (optional)", "suggested": ""},
    {"label": "--no-webui", "key": "no_webui", "type": "flag", "default": "off",
     "desc": "Disable the built-in web UI", "suggested": "on"},
    {"label": "--no-cache-prompt", "key": "no_cache_prompt", "type": "flag", "default": "off",
     "desc": "Disable prompt caching", "suggested": "off"},
    {"label": "--cache-disk", "key": "cache_disk", "type": "str", "default": "",
     "desc": "Directory for persistent prompt/KV disk cache (empty = disabled; dir is auto-created)", "suggested": "/var/cache/llama-server/prompt-cache"},
    {"label": "--cache-disk-max", "key": "cache_disk_max", "type": "int", "default": "32768",
     "desc": "Maximum persistent cache size in MiB (0 = unlimited)", "suggested": "20480"},
    {"label": "--cache-disk-block", "key": "cache_disk_block", "type": "int", "default": "256",
     "desc": "Token block size for persistent cache lookup", "suggested": "256"},
    {"label": "--no-jinja", "key": "no_jinja", "type": "flag", "default": "off",
     "desc": "Disable jinja chat template engine", "suggested": "off"},
    {"label": "--metrics", "key": "metrics", "type": "flag", "default": "on",
     "desc": "Enable prometheus-compatible /metrics endpoint", "suggested": "on"},
]

DWARFSTAR_OPTS = [
    {"label": "-m, --model", "key": "model", "type": "str", "default": "",
     "desc": "GGUF model path to load", "suggested": ""},
    {"label": "--backend", "key": "backend", "type": "choice", "default": "rocm",
     "choices": ["metal", "rocm", "cpu"], "desc": "Compute backend", "suggested": "rocm"},
    {"label": "-c, --ctx", "key": "ctx", "type": "int", "default": "131072",
     "desc": "Allocated context tokens", "suggested": "131072"},
    {"label": "-n, --tokens", "key": "tokens", "type": "int", "default": "",
     "desc": "Default max output tokens when clients omit a limit (blank = server default, capped by ctx)", "suggested": ""},
    {"label": "-t, --threads", "key": "threads", "type": "int", "default": str(min(32, os.cpu_count() or 4)),
     "desc": "CPU helper threads for host-side work", "suggested": str(min(32, os.cpu_count() or 4))},
    {"label": "--power", "key": "power", "type": "int", "default": "100",
     "desc": "GPU duty-cycle target 1..100", "suggested": "100"},
    {"label": "--ssd-streaming", "key": "ssd_streaming", "type": "flag", "default": "off",
     "desc": "SSD-backed model streaming (avoid full RAM residency)", "suggested": "off"},
    {"label": "--host", "key": "host", "type": "str", "default": "0.0.0.0",
     "desc": "Bind address", "suggested": "0.0.0.0"},
    {"label": "--port", "key": "port", "type": "int", "default": "8000",
     "desc": "HTTP API port", "suggested": ""},
    {"label": "--cors", "key": "cors", "type": "flag", "default": "off",
     "desc": "Add CORS headers for browser clients", "suggested": "off"},
    {"label": "--kv-disk-dir", "key": "kv_disk_dir", "type": "str", "default": "~/.ds4/server-kv",
     "desc": "Disk KV checkpoint directory (empty = disabled)", "suggested": "~/.ds4/server-kv"},
    {"label": "--kv-disk-space-mb", "key": "kv_disk_space_mb", "type": "int", "default": "8192",
     "desc": "Disk KV budget in MiB", "suggested": "8192"},
    {"label": "--batched-session", "key": "batched_session", "type": "int", "default": "",
     "desc": "Resident batched sessions (blank = server auto; forcing a high value multiplies KV-cache RAM and can OOM)", "suggested": ""},
    {"label": "--mtp", "key": "mtp", "type": "str", "default": "",
     "desc": "MTP/draft support GGUF (e.g. DSpark draft)", "suggested": ""},
    {"label": "--mtp-draft", "key": "mtp_draft", "type": "int", "default": "1",
     "desc": "Max autoregressive MTP draft tokens", "suggested": "1"},
    {"label": "--mtp-margin", "key": "mtp_margin", "type": "float", "default": "3",
     "desc": "Verifier confidence margin for fast MTP acceptance", "suggested": "3"},
    {"label": "--dspark", "key": "dspark", "type": "flag", "default": "off",
     "desc": "Enable DSpark speculation with --mtp", "suggested": "off"},
    {"label": "--dspark-confidence", "key": "dspark_confidence", "type": "float", "default": "",
     "desc": "DSpark confidence pruning threshold 0..1 (blank = server default; only applies with --dspark)", "suggested": ""},
    {"label": "--dspark-strict", "key": "dspark_strict", "type": "flag", "default": "off",
     "desc": "Load DSpark support but keep target-only decode", "suggested": "off"},
    {"label": "--glm-mtp", "key": "glm_mtp", "type": "flag", "default": "off",
     "desc": "Enable integrated greedy GLM MTP speculation", "suggested": "off"},
    {"label": "--glm-mtp-timing", "key": "glm_mtp_timing", "type": "flag", "default": "off",
     "desc": "Enable GLM MTP and print acceptance/timing counters", "suggested": "off"},
    {"label": "--warm-weights", "key": "warm_weights", "type": "flag", "default": "off",
     "desc": "Touch mapped tensor pages at startup to reduce first-use stalls", "suggested": "off"},
    {"label": "--quality", "key": "quality", "type": "flag", "default": "off",
     "desc": "Prefer exact kernels where faster approximate paths exist", "suggested": "off"},
    {"label": "--think", "key": "think", "type": "flag", "default": "off",
     "desc": "Enable normal thinking mode", "suggested": "off"},
    {"label": "--think-budget", "key": "think_budget", "type": "int", "default": "",
     "desc": "Max thinking tokens (blank = server default)", "suggested": "4096"},
]

STRIX_OPTS = [
    {"label": "-m, --model", "key": "model", "type": "str", "default": "",
     "desc": "GGUF model path to load", "suggested": ""},
    {"label": "-lm, --load-mode", "key": "load_mode", "type": "choice", "default": "mmap",
     "choices": ["mmap", "none", "mlock", "mmap+mlock", "dio"],
     "desc": "Model loading mode (dio = DirectIO disk reads; replaces the deprecated -dio/--direct-io flag)", "suggested": "dio"},
    {"label": "--mmproj", "key": "mmproj", "type": "str", "default": "",
     "desc": "Multimodal projector GGUF path for vision support", "suggested": ""},
    {"label": "-md, --model-draft", "key": "model_draft", "type": "str", "default": "",
     "desc": "Draft model for speculative decoding (e.g. DSpark drafter)", "suggested": ""},
    {"label": "-ngl, --n-gpu-layers", "key": "gpu_layers", "type": "str", "default": "all",
     "desc": "Layers to keep in VRAM (number, 'auto' or 'all')", "suggested": "all"},
    {"label": "-ngld, --n-gpu-layers-draft", "key": "draft_gpu_layers", "type": "str", "default": "all",
     "desc": "Layers of the draft model to keep in VRAM (number, 'auto' or 'all')", "suggested": "all"},
    {"label": "-fa, --flash-attn", "key": "flash_attn", "type": "choice", "default": "on",
     "choices": ["on", "off", "auto"], "desc": "Use Flash Attention", "suggested": "on"},
    {"label": "-ctk, --cache-type-k", "key": "cache_type_k", "type": "choice", "default": "q8_0",
     "choices": ["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"], "desc": "KV cache data type for K", "suggested": "q8_0"},
    {"label": "-ctv, --cache-type-v", "key": "cache_type_v", "type": "choice", "default": "q8_0",
     "choices": ["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"], "desc": "KV cache data type for V", "suggested": "q8_0"},
    {"label": "--cache-disk", "key": "cache_disk", "type": "str", "default": "",
     "desc": "Directory for persistent prompt/KV disk cache (empty = disabled; dir is auto-created)", "suggested": "/var/cache/llama-server/prompt-cache"},
    {"label": "--cache-disk-max", "key": "cache_disk_max", "type": "int", "default": "32768",
     "desc": "Maximum persistent cache size in MiB (0 = unlimited)", "suggested": "20480"},
    {"label": "--cache-disk-block", "key": "cache_disk_block", "type": "int", "default": "256",
     "desc": "Token block size for persistent cache lookup", "suggested": "256"},
    {"label": "-c, --ctx-size", "key": "ctx_size", "type": "int", "default": "131072",
     "desc": "Prompt context size (tokens)", "suggested": "131072"},
    {"label": "-np, --parallel", "key": "parallel", "type": "int", "default": "1",
     "desc": "Number of parallel slots", "suggested": "1"},
    {"label": "-a, --alias", "key": "alias", "type": "str", "default": "",
     "desc": "Model alias used by the API (comma-separated); the name Open WebUI etc. see", "suggested": ""},
    {"label": "-b, --batch-size", "key": "batch_size", "type": "int", "default": "2048",
     "desc": "Logical batch size (prompt processing)", "suggested": "2048"},
    {"label": "-ub, --ubatch-size", "key": "ubatch_size", "type": "int", "default": "2048",
     "desc": "Micro batch size (physical batch)", "suggested": "2048"},
    {"label": "--spec-type", "key": "spec_type", "type": "str", "default": "draft-dspark",
     "desc": "Speculative decoding types (comma-separated, e.g. draft-dspark)", "suggested": "draft-dspark"},
    {"label": "--spec-draft-n-max", "key": "spec_draft_n_max", "type": "int", "default": "64",
     "desc": "Max draft tokens per speculative pass", "suggested": "64"},
    {"label": "-t, --threads", "key": "threads", "type": "int", "default": "-1",
     "desc": "CPU threads for generation (-1 = auto)", "suggested": str(min(32, os.cpu_count() or 4))},
    {"label": "--temp", "key": "temp", "type": "float", "default": "0.8",
     "desc": "Sampling temperature", "suggested": "0.6"},
    {"label": "--top-k", "key": "top_k", "type": "int", "default": "40",
     "desc": "Top-k sampling (0 = disabled)", "suggested": "40"},
    {"label": "--top-p", "key": "top_p", "type": "float", "default": "0.95",
     "desc": "Top-p sampling", "suggested": "0.95"},
    {"label": "--min-p", "key": "min_p", "type": "float", "default": "0.05",
     "desc": "Min-p sampling", "suggested": "0.05"},
    {"label": "-rea, --reasoning", "key": "reasoning", "type": "choice", "default": "auto",
     "choices": ["on", "off", "auto"], "desc": "Use reasoning/thinking in the chat", "suggested": "auto"},
    {"label": "--reasoning-format", "key": "reasoning_format", "type": "choice", "default": "auto",
     "choices": ["auto", "none", "deepseek", "deepseek-legacy"], "desc": "Reasoning content format", "suggested": "auto"},
    {"label": "--reasoning-budget", "key": "reasoning_budget", "type": "int", "default": "-1",
     "desc": "Token budget for thinking (-1 = unrestricted, 0 = immediate end)", "suggested": "-1"},
    {"label": "--reasoning-preserve", "key": "reasoning_preserve", "type": "flag", "default": "off",
     "desc": "Preserve reasoning trace in full history", "suggested": "off"},
    {"label": "--reasoning-effort", "key": "reasoning_effort", "type": "choice", "default": "auto",
     "choices": ["auto", "xhigh", "medium", "low"], "no_cli": True,
     "desc": "Qwen3.8 reasoning depth (auto = model default: xhigh); sent as a chat template kwarg, needs the qwen3.8 chat template", "suggested": "medium"},
    {"label": "--chat-template-file", "key": "chat_template_file", "type": "str", "default": "",
     "desc": "Chat template .jinja file (Qwen3.8 needs the official one for reasoning_effort)", "suggested": ""},
    {"label": "-s, --seed", "key": "seed", "type": "int", "default": "-1",
     "desc": "RNG seed (-1 = random)", "suggested": "-1"},
    {"label": "--jinja", "key": "jinja", "type": "flag", "default": "on",
     "desc": "Enable jinja chat template engine", "suggested": "on"},
    {"label": "--host", "key": "host", "type": "str", "default": "127.0.0.1",
     "desc": "Bind address", "suggested": "127.0.0.1"},
    {"label": "--port", "key": "port", "type": "int", "default": "8080",
     "desc": "HTTP API port", "suggested": ""},
    {"label": "--no-webui", "key": "no_webui", "type": "flag", "default": "off",
     "desc": "Disable the built-in web UI", "suggested": "on"},
    {"label": "--metrics", "key": "metrics", "type": "flag", "default": "on",
     "desc": "Enable prometheus-compatible /metrics endpoint", "suggested": "on"},
]

# MTP speed presets for the ROCmFPX runner (measured on Strix Halo / Qwen3.8-27B
# ROCmFP4-FAST, Vulkan0, ctk q8_0 / ctv turbo4): code/JSON prompts see full-draft
# acceptance at n7/p0.35 (~44 t/s), while free prose is best at n4 (~23-33 t/s).
ROCMFPX_PRESETS = {
    "default": {
        "label": "Mixed chat (n4 / p0.55 / ub 512)",
        "overrides": {"spec_draft_n_max": "4", "spec_draft_p_min": "0.55", "batch_size": "512", "ubatch_size": "512"},
    },
    "code": {
        "label": "Code / JSON deep-spec (n7 / p0.35 / ub 2048)",
        "overrides": {"spec_draft_n_max": "7", "spec_draft_p_min": "0.35", "batch_size": "2048", "ubatch_size": "2048"},
    },
}

ROCMFPX_OPTS = [
    {"label": "-m, --model", "key": "model", "type": "str", "default": "",
     "desc": "GGUF model path to load", "suggested": ""},
    {"label": "--mmproj", "key": "mmproj", "type": "str", "default": "",
      "desc": "Multimodal projector GGUF path for vision support", "suggested": ""},
    {"label": "-dio, --direct-io", "key": "direct_io", "type": "flag", "default": "off",
     "desc": "Use DirectIO if available (this build predates --load-mode, so the legacy flag is used)", "suggested": "off"},
    {"label": "-dev, --device", "key": "device", "type": "str", "default": "Vulkan0",
     "desc": "Backend device to offload to (Vulkan0 for decode speed, ROCm0 for prefill/TTFT)", "suggested": "Vulkan0"},
    {"label": "-ngl, --n-gpu-layers", "key": "gpu_layers", "type": "str", "default": "999",
     "desc": "Layers to keep on device (999 = all)", "suggested": "999"},
    {"label": "-fa, --flash-attn", "key": "flash_attn", "type": "choice", "default": "on",
     "choices": ["on", "off", "auto"], "desc": "Use Flash Attention", "suggested": "on"},
    {"label": "-ctk, --cache-type-k", "key": "cache_type_k", "type": "str", "default": "q8_0",
     "desc": "KV cache data type for K (q8_0 keeps attention quality; turbo4 compresses)", "suggested": "q8_0"},
    {"label": "-ctv, --cache-type-v", "key": "cache_type_v", "type": "str", "default": "q8_0",
     "desc": "KV cache data type for V (turbo4 for max compression)", "suggested": "q8_0"},
    {"label": "--mtp-preset", "key": "mtp_preset", "type": "preset", "no_cli": True,
     "choices": list(ROCMFPX_PRESETS.keys()), "default": "default",
     "desc": "MTP speed preset (sets draft depth/temperature + batch sizes)", "suggested": "default"},
    {"label": "--spec-type", "key": "spec_type", "type": "choice", "default": "draft-mtp",
     "choices": ["none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash", "draft-dspark"],
     "desc": "Speculative decoding type (draft-mtp for embedded MTP heads)", "suggested": "draft-mtp"},
    {"label": "--spec-draft-n-max", "key": "spec_draft_n_max", "type": "int", "default": "4",
     "desc": "Max draft tokens per speculative pass", "suggested": "4"},
    {"label": "--spec-draft-n-min", "key": "spec_draft_n_min", "type": "int", "default": "0",
     "desc": "Minimum draft tokens to use", "suggested": "0"},
    {"label": "--spec-draft-p-min", "key": "spec_draft_p_min", "type": "float", "default": "0.55",
     "desc": "Minimum speculative decoding probability (greedy)", "suggested": "0.55"},
    {"label": "--spec-draft-p-split", "key": "spec_draft_p_split", "type": "float", "default": "0.10",
     "desc": "Draft probability split", "suggested": "0.10"},
    {"label": "--spec-draft-type-k", "key": "spec_draft_type_k", "type": "str", "default": "q4_0",
     "desc": "KV cache type for draft model K", "suggested": "q4_0"},
    {"label": "--spec-draft-type-v", "key": "spec_draft_type_v", "type": "str", "default": "q4_0",
     "desc": "KV cache type for draft model V", "suggested": "q4_0"},
    {"label": "-c, --ctx-size", "key": "ctx_size", "type": "int", "default": "131072",
     "desc": "Prompt context size (tokens)", "suggested": "131072"},
    {"label": "-b, --batch-size", "key": "batch_size", "type": "int", "default": "512",
     "desc": "Logical batch size (prompt processing)", "suggested": "512"},
    {"label": "-ub, --ubatch-size", "key": "ubatch_size", "type": "int", "default": "512",
     "desc": "Micro batch size (physical batch)", "suggested": "512"},
    {"label": "-np, --parallel", "key": "parallel", "type": "int", "default": "1",
     "desc": "Number of parallel slots", "suggested": "1"},
    {"label": "-a, --alias", "key": "alias", "type": "str", "default": "",
     "desc": "Model alias used by the API (comma-separated); the name Open WebUI etc. see", "suggested": ""},
    {"label": "-t, --threads", "key": "threads", "type": "int", "default": "-1",
     "desc": "CPU threads for generation (-1 = auto)", "suggested": str(min(32, os.cpu_count() or 4))},
    {"label": "-rea, --reasoning", "key": "reasoning", "type": "choice", "default": "auto",
     "choices": ["on", "off", "auto"], "desc": "Use reasoning/thinking in the chat", "suggested": "auto"},
    {"label": "--reasoning-format", "key": "reasoning_format", "type": "choice", "default": "auto",
     "choices": ["auto", "none", "deepseek", "deepseek-legacy"], "desc": "Reasoning content format", "suggested": "auto"},
    {"label": "--reasoning-budget", "key": "reasoning_budget", "type": "int", "default": "-1",
     "desc": "Token budget for thinking (-1 = unrestricted)", "suggested": "-1"},
    {"label": "--reasoning-preserve", "key": "reasoning_preserve", "type": "flag", "default": "off",
     "desc": "Preserve reasoning trace in full history", "suggested": "off"},
    {"label": "--reasoning-effort", "key": "reasoning_effort", "type": "choice", "default": "auto",
     "choices": ["auto", "xhigh", "medium", "low"], "no_cli": True,
     "desc": "Qwen3.8 reasoning depth (auto = model default: xhigh); sent as a chat template kwarg, needs the qwen3.8 chat template", "suggested": "medium"},
    {"label": "--chat-template-file", "key": "chat_template_file", "type": "str", "default": "",
     "desc": "Chat template .jinja file (Qwen3.8 needs the official one for reasoning_effort)", "suggested": ""},
    {"label": "--jinja", "key": "jinja", "type": "flag", "default": "on",
     "desc": "Enable jinja chat template engine", "suggested": "on"},
    {"label": "--temp", "key": "temp", "type": "float", "default": "0.8",
     "desc": "Sampling temperature (0 = greedy)", "suggested": "0"},
    {"label": "--top-k", "key": "top_k", "type": "int", "default": "40",
     "desc": "Top-k sampling (0 = disabled)", "suggested": "40"},
    {"label": "--top-p", "key": "top_p", "type": "float", "default": "0.95",
     "desc": "Top-p sampling", "suggested": "0.95"},
    {"label": "--min-p", "key": "min_p", "type": "float", "default": "0.05",
     "desc": "Min-p sampling", "suggested": "0.05"},
    {"label": "-s, --seed", "key": "seed", "type": "int", "default": "-1",
     "desc": "RNG seed (-1 = random)", "suggested": "-1"},
    {"label": "--host", "key": "host", "type": "str", "default": "127.0.0.1",
     "desc": "Bind address", "suggested": "127.0.0.1"},
    {"label": "--port", "key": "port", "type": "int", "default": "8080",
     "desc": "HTTP API port", "suggested": ""},
    {"label": "--no-webui", "key": "no_webui", "type": "flag", "default": "off",
     "desc": "Disable the built-in web UI", "suggested": "on"},
    {"label": "--metrics", "key": "metrics", "type": "flag", "default": "on",
     "desc": "Enable prometheus-compatible /metrics endpoint", "suggested": "on"},
]

RUNNER_OPTS = {"llamacpp": LLAMACPP_OPTS, "dwarfstar": DWARFSTAR_OPTS, "strix": STRIX_OPTS, "rocmfpx": ROCMFPX_OPTS}


# ---------------------------------------------------------------------------
# Per-model default sampling/launch settings.
#   Keyed by a substring of the model filename; values are option-key overrides
#   per runner. The local GGUF files are future/fictional releases, so each is
#   mapped to its closest real published sibling and that sibling's officially
#   recommended defaults (temperature, top-k/top-p/min-p, ctx).
#     qwen3.5/qwen3.6 -> Qwen3   (temp 0.6, top-k 20, top-p 0.95, min-p 0, ctx 40K)
#     deepseek        -> DeepSeek-R1/V3 (temp 0.6, top-p 0.95)
#     gemma           -> Gemma 3 instruct (temp 0.9, top-k 40, top-p 0.95)
#     minimax         -> MiniMax M1 reasoning (temp 0.5, top-p 0.95)
# ---------------------------------------------------------------------------

MODEL_FAMILY_DEFAULTS = {
    "qwen3.5": {
        "llamacpp": {"temp": "0.6", "top_k": "20", "top_p": "0.95", "min_p": "0", "ctx_size": "40960"},
        "strix": {"temp": "0.6", "top_k": "20", "top_p": "0.95", "min_p": "0", "ctx_size": "40960"},
        "dwarfstar": {"ctx": "40960"},
    },
    "qwen3.6": {
        "llamacpp": {"temp": "0.6", "top_k": "20", "top_p": "0.95", "min_p": "0", "ctx_size": "40960"},
        "strix": {"temp": "0.6", "top_k": "20", "top_p": "0.95", "min_p": "0", "ctx_size": "40960"},
        "dwarfstar": {"ctx": "40960"},
    },
    "qwen3.8": {
        # Official Qwen3.8 sampling (thinking mode) + the official chat template
        # (the GGUFs carry no template) so reasoning_effort can take effect.
        # reasoning_effort: medium by default — the model's own default is xhigh
        # (very deep reasoning); pick low/medium/xhigh in the UI to taste.
        "llamacpp": {"temp": "1.0", "top_k": "20", "top_p": "0.95", "min_p": "0", "ctx_size": "131072",
                     "chat_template": QWEN38_TEMPLATE, "reasoning_effort": "medium",
                     "mmproj": "/home/fred/ai/models/mmproj-Qwen3.8-27B-BF16.gguf"},
        "strix": {"temp": "1.0", "top_k": "20", "top_p": "0.95", "min_p": "0", "ctx_size": "131072",
                  "chat_template_file": QWEN38_TEMPLATE, "reasoning_effort": "medium",
                  "mmproj": "/home/fred/ai/models/mmproj-Qwen3.8-27B-BF16.gguf"},
        "rocmfpx": {"temp": "1.0", "top_k": "20", "top_p": "0.95", "min_p": "0", "ctx_size": "131072",
                    "chat_template_file": QWEN38_TEMPLATE, "reasoning_effort": "medium",
                    "mmproj": "/home/fred/ai/models/mmproj-Qwen3.8-27B-BF16.gguf"},
    },
    "deepseek": {
        "llamacpp": {"temp": "0.6", "top_p": "0.95", "min_p": "0"},
        "strix": {"temp": "0.6", "top_p": "0.95", "min_p": "0", "ctx_size": "131072", "cache_type_k": "q8_0", "cache_type_v": "q8_0", "spec_type": "draft-dspark", "spec_draft_n_max": "64"},
        "dwarfstar": {},
    },
    "gemma": {
        "llamacpp": {"temp": "0.9", "top_k": "40", "top_p": "0.95", "min_p": "0"},
        "strix": {"temp": "0.9", "top_k": "40", "top_p": "0.95", "min_p": "0"},
        "dwarfstar": {},
    },
    "minimax": {
        "llamacpp": {"temp": "0.5", "top_p": "0.95", "min_p": "0"},
        "strix": {"temp": "0.5", "top_p": "0.95", "min_p": "0"},
        "dwarfstar": {},
    },
}


def model_family_of(model_path):
    name = os.path.basename(model_path).lower()
    if any(tok in name for tok in ("mtp", "draft", "dspark")):
        return None
    for fam in MODEL_FAMILY_DEFAULTS:
        if fam in name:
            return fam
    return None


def model_defaults_for(models):
    """Return {model_path: {runner: {key: value}}} for each discovered model."""
    out = {}
    for m in models:
        fam = model_family_of(m)
        if fam:
            out[m] = MODEL_FAMILY_DEFAULTS[fam]
    return out


# ---------------------------------------------------------------------------
# Per-runner model compatibility.
#   Only models physically located in the DwarfStar gguf dir are DwarfStar
#   compatible; everything else (incl. the Unsloth DeepSeek files in
#   /home/fred/ai/models) is a llama.cpp model.
# ---------------------------------------------------------------------------

def model_runner_of(model_path):
    if os.path.dirname(os.path.abspath(model_path)) == os.path.abspath(DWARFSTAR_GGUF_DIR):
        return ["dwarfstar"]
    # everything else is a llama.cpp-family model: usable by mainline, the
    # StrixHalo Vulkan fork, and the ROCmFPX fork
    return ["llamacpp", "strix", "rocmfpx"]


def option_cli_args(runner, opts):
    """Convert {key: value} option dict to CLI argument list."""
    args = []
    for o in RUNNER_OPTS[runner]:
        key = o["key"]
        if key not in opts:
            continue
        if o.get("no_cli"):
            continue
        v = opts[key]
        if v is None or v == "":
            continue
        if o["type"] == "flag":
            if str(v).lower() in ("on", "true", "1", "yes"):
                args.append(o["label"].split(",")[-1].strip())
            continue
        args.append(o["label"].split(",")[-1].strip())
        args.append(str(v))
    return args


def port_opt_key(runner):
    for o in RUNNER_OPTS[runner]:
        if o["label"].split(",")[-1].strip() == "--port":
            return o["key"]
    return None


def find_free_port(requested, tries=100):
    """Return the first free port starting at requested (or next free if taken)."""
    port = requested
    for _ in range(tries):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return port
        except OSError:
            s.close()
            port += 1
    return None


# ---------------------------------------------------------------------------
# State store
# ---------------------------------------------------------------------------

class StateStore:
    def __init__(self, path=STATE_FILE):
        self.path = path
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"config": DEFAULT_CONFIG, "setups": []}

    def save(self):
        with self.lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)

    def get(self, key, default=None):
        with self.lock:
            return self.data.get(key, default)

    def set(self, key, value):
        with self.lock:
            self.data[key] = value
        self.save()


# ---------------------------------------------------------------------------
# Process manager
# ---------------------------------------------------------------------------

class Process:
    def __init__(self, pid, setup):
        self.id = uuid.uuid4().hex[:12]
        self.pid = pid
        self.stdout = None
        self.reader = None
        self.dead = False
        self.setup = setup
        self.start_time = time.time()
        self.status = "running"  # running | crashed | stopped | restarting
        self.exit_code = None
        self.log = deque(maxlen=2000)
        self.restarts = 0
        self.started_by_watchdog = False

    def to_dict(self):
        return {
            "id": self.id,
            "pid": self.pid,
            "name": self.setup.get("name") or os.path.basename(self.setup.get("model", "")),
            "runner": self.setup.get("runner"),
            "model": self.setup.get("model"),
            "port": self.setup.get("port") or "",
            "status": self.status,
            "exit_code": self.exit_code,
            "start_time": self.start_time,
            "uptime": round(time.time() - self.start_time, 1),
            "restarts": self.restarts,
            "setup_id": self.setup.get("id", ""),
            "log": list(self.log)[-400:],
        }


class ProcessManager:
    def __init__(self, store, cfg):
        self.store = store
        self.cfg = cfg
        self.lock = threading.Lock()
        self.procs = {}  # id -> Process
        self.readers = {}  # pid -> reader thread
        self.stop_flags = set()  # pids intentionally stopped
        self.watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)

    def _bin_for(self, runner):
        return {"llamacpp": self.cfg["llamacpp_bin"], "dwarfstar": self.cfg["dwarfstar_bin"], "strix": self.cfg["strix_bin"], "rocmfpx": self.cfg["rocmfpx_bin"]}[runner]

    def _port_holder(self, port):
        """What is holding a port right now, or None if free.
        Only live processes count: another airunner model still up, or anything
        else bound to the port. Stopped/removed airunner entries never count."""
        with self.lock:
            for p in self.procs.values():
                if p.status in ("running", "restarting") and p.pid is not None \
                        and (p.setup.get("port") or "") == str(port):
                    return f"airunner model '{p.setup.get('name') or os.path.basename(p.setup.get('model', ''))}' (pid {p.pid})"
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
            return None
        except OSError:
            return "another process on this machine"
        finally:
            s.close()

    def _reader(self, pid, proc):
        try:
            for line in proc.stdout:
                proc.log.append(line.rstrip("\n"))
        except Exception:
            pass
        # EOF => process exited; reap it so the watchdog can detect the exit
        try:
            proc.wait(timeout=1)
        except Exception:
            pass
        proc.dead = True

    def start(self, setup):
        runner = setup["runner"]
        binary = self._bin_for(runner)
        model = setup.get("model", "")
        args = [binary]
        if model:
            args.append("-m")
            args.append(model)
        args += option_cli_args(runner, setup.get("options", {}))
        # llama.cpp has no --reasoning-effort flag; Qwen3.8-style jinja templates
        # take it as a template kwarg, so translate it to --chat-template-kwargs.
        eff = setup.get("options", {}).get("reasoning_effort", "")
        if eff in ("xhigh", "medium", "low"):
            args += ["--chat-template-kwargs", json.dumps({"reasoning_effort": eff})]
        # --- port policy: use the configured port exactly; never reassign.
        # Refuse to start if the port is taken (by another airunner model or any
        # other live process). Stopped/removed airunner entries never count ---
        # --- as holders, only processes that are actually up do. ---
        pk = port_opt_key(runner)
        req = setup.get("options", {}).get(pk, "") if pk is not None else ""
        if req and req.isdigit():
            holder = self._port_holder(int(req))
            if holder:
                proc = Process(None, setup)
                proc.setup["port"] = req
                proc.status = "stopped"
                proc.log.append("$ " + " ".join(args))
                proc.log.append(
                    f"[airunner] REFUSED to start: port {req} is in use ({holder}). "
                    f"Stop what is holding it, or change this setup's port."
                )
                with self.lock:
                    self.procs[proc.id] = proc
                return proc
        # --- persistent disk prompt cache: expand ~ and create the dir (the server requires it to exist) ---
        cache_dir = setup.get("options", {}).get("cache_disk", "")
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            try:
                i = args.index("--cache-disk")
                args[i + 1] = cache_dir
            except (ValueError, IndexError):
                pass
        # ensure port option applied for bookkeeping
        port = ""
        for o in RUNNER_OPTS[runner]:
            if o["label"].split(",")[-1].strip() == "--port":
                port = setup.get("options", {}).get(o["key"], "")
                break
        proc = None
        with self.lock:
            proc = Process(None, setup)
            proc.setup["port"] = port or ""
            self.procs[proc.id] = proc
        env = dict(os.environ)
        env["LLAMA_ARG_CTX_SIZE"] = str(setup.get("options", {}).get("ctx_size", ""))
        if runner == "rocmfpx":
            # ROCmFPX fork: needed for the ROCm/HIP backend on gfx1151 (Strix Halo).
            # Harmless for the Vulkan path but required if the user selects ROCm0.
            env.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
            env.setdefault("GGML_HIP_ENABLE_UNIFIED_MEMORY", "1")
        try:
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, env=env)
            proc.pid = p.pid
            proc.stdout = p.stdout
            proc.log.append("$ " + " ".join(args))
            r = threading.Thread(target=self._reader, args=(p.pid, proc), daemon=True)
            proc.reader = r
            self.readers[p.pid] = r
            r.start()
        except Exception as e:
            proc.status = "crashed"
            proc.log.append(f"[airunner] failed to launch: {e}")
        return proc

    def stop(self, pid):
        with self.lock:
            proc = self.procs.get(pid)
            if not proc:
                return False
            self.stop_flags.add(proc.pid)
            pidv = proc.pid
        if pidv is None:
            proc.status = "stopped"
            return True
        # Kill in the background so the HTTP request returns immediately.
        # Send SIGINT (same as Ctrl+C, which DwarfStar handles instantly),
        # then escalate to SIGKILL if it does not exit.
        def _kill():
            def _forget():
                # drop the record once the process is really gone, so stopped
                # entries never linger (and never look like port holders)
                proc.status = "stopped"
                with self.lock:
                    self.procs.pop(proc.id, None)
            try:
                os.kill(pidv, 2)  # SIGINT
            except Exception:
                pass
            for _ in range(50):  # up to ~5s grace for e.g. KV cache flush to disk
                try:
                    os.kill(pidv, 0)
                except OSError:
                    _forget()
                    return
                except Exception:
                    _forget()
                    return
                time.sleep(0.1)
            try:
                os.kill(pidv, 9)  # SIGKILL fallback
            except Exception:
                pass
            _forget()
        threading.Thread(target=_kill, daemon=True).start()
        return True

    def _watchdog_loop(self):
        while True:
            time.sleep(max(1, int(self.cfg.get("watchdog_interval", 5))))
            self._watchdog_tick()

    def _watchdog_tick(self):
        # reap stale refused-launch records (no pid, parked as "stopped") after 2 min
        stale = []
        with self.lock:
            for qid, q in self.procs.items():
                if q.pid is None and q.status == "stopped" and time.time() - q.start_time > 120:
                    stale.append(qid)
            for qid in stale:
                self.procs.pop(qid, None)
        snapshot = []
        with self.lock:
            snapshot = [(pid, p) for pid, p in self.procs.items()]
        for pid, proc in snapshot:
            alive = True
            if proc.pid is not None:
                if proc.reader is not None:
                    alive = proc.reader.is_alive()
                else:
                    alive = self._alive(proc.pid)
            if alive:
                continue
            if proc.pid in self.stop_flags:
                continue
            # crashed unexpectedly
            rc = self._exit_code(proc.pid) if proc.pid else None
            proc.status = "crashed"
            proc.exit_code = rc
            if proc.setup.get("restart_on_crash", False) and not proc.setup.get("autostart_only_once", False):
                proc.restarts += 1
                proc.status = "restarting"
                proc.log.append(f"[airunner] process died (rc={rc}); restarting ({proc.restarts})")
                new = self.start(proc.setup)
                if new and new.pid is not None:
                    # carry restart count across the fresh Process record
                    new.restarts = proc.restarts
                    with self.lock:
                        self.procs.pop(pid, None)
                        self.procs[new.id] = new
                else:
                    # refused (e.g. port busy): keep the crashed record visible,
                    # drop the parked refused record, and don't loop on it
                    if new is not None:
                        with self.lock:
                            self.procs.pop(new.id, None)
                    proc.status = "crashed"
                    proc.log.append("[airunner] restart refused (port in use); left in crashed state")
                continue

    @staticmethod
    def _alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    @staticmethod
    def _exit_code(pid):
        try:
            w = os.waitpid(pid, os.WNOHANG)
            if w[0] == 0:
                return None
            return (w[1] & 0xFF) >> 8
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Autostart / systemd
# ---------------------------------------------------------------------------

def systemd_unit_path():
    return os.path.join(SYSTEMD_DIR, SYSTEMD_UNIT)


def systemd_installed():
    return os.path.exists(systemd_unit_path())


def install_systemd(app_path):
    os.makedirs(SYSTEMD_DIR, exist_ok=True)
    unit = f"""[Unit]
Description=Airunner - local model runner manager
After=network.target

[Service]
Type=simple
ExecStart={sys.executable} {app_path}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""
    with open(systemd_unit_path(), "w") as f:
        f.write(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    subprocess.run(["systemctl", "--user", "enable", SYSTEMD_UNIT], capture_output=True)
    subprocess.run(["systemctl", "--user", "restart", SYSTEMD_UNIT], capture_output=True)


def remove_systemd():
    subprocess.run(["systemctl", "--user", "disable", SYSTEMD_UNIT], capture_output=True)
    if os.path.exists(systemd_unit_path()):
        os.remove(systemd_unit_path())
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)


def service_status():
    try:
        out = subprocess.run(["systemctl", "--user", "is-active", SYSTEMD_UNIT],
                             capture_output=True, text=True).stdout.strip()
        return out
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models(dirs):
    found = []
    for d in dirs or []:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(".gguf"):
                continue
            m = re.search(r"-(\d+)-of-(\d+)\.gguf$", name)
            if m:
                this, total = int(m.group(1)), int(m.group(2))
                if total > 1 and this > 1:
                    continue
            found.append(os.path.join(d, name))
    return found


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server = None  # set on server instance

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if n == 0:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_OPTIONS(self):
        self._send(200, b"", "text/plain")

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/":
            return self._serve_index()
        api = self.server.api
        if route == "/api/state":
            return self._send(200, api.state())
        if route == "/api/models":
            return self._send(200, {"models": discover_models(api.cfg.get("model_dirs", []))})
        if route == "/api/config":
            return self._send(200, api.cfg)
        if route == "/api/setups":
            return self._send(200, {"setups": api.store.get("setups", [])})
        if route == "/api/launch-options":
            runner = self._query("runner") or "llamacpp"
            return self._send(200, {"options": RUNNER_OPTS.get(runner, []), "runner": runner})
        if route == "/api/systemd":
            return self._send(200, {"installed": systemd_installed(),
                                    "active": service_status()})
        self._send(404, {"error": "not found"})

    def _query(self, key):
        from urllib.parse import parse_qs
        qs = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
        return qs.get(key, [""])[0]

    def _serve_index(self):
        try:
            with open(os.path.join(WEB_DIR, "index.html")) as f:
                html = f.read()
            self._send(200, html.encode(), "text/html")
        except Exception as e:
            self._send(500, {"error": str(e)}, "application/json")

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        api = self.server.api
        if route == "/api/remove":
            body = self._read_json()
            api.remove_proc(body.get("id", ""))
            return self._send(200, {"ok": True})
        if route == "/api/run":
            body = self._read_json()
            setup = body.get("setup", {})
            runner = setup.get("runner", "llamacpp")
            if runner not in RUNNER_OPTS:
                return self._send(400, {"error": f"unknown runner {runner}"})
            model = setup.get("model", "")
            if model and not os.path.exists(model):
                return self._send(400, {"error": f"model not found: {model}"})
            proc = api.pm.start(setup)
            if proc.pid is None:
                return self._send(200, {"ok": False, "error": proc.log[-1] if proc.log else "failed to start"})
            return self._send(200, {"proc": proc.to_dict()})
        if route == "/api/stop":
            body = self._read_json()
            ok = api.pm.stop(body.get("id", ""))
            return self._send(200, {"ok": ok})
        if route == "/api/restart":
            body = self._read_json()
            return self._send(200, api.restart_proc(body.get("id", "")))
        if route == "/api/save-setup":
            body = self._read_json()
            return self._send(200, api.save_setup(body))
        if route == "/api/apply-setup":
            body = self._read_json()
            return self._send(200, api.apply_setup(body))
        if route == "/api/delete-setup":
            body = self._read_json()
            return self._send(200, api.delete_setup(body.get("id", "")))
        if route == "/api/setup-autostart":
            body = self._read_json()
            return self._send(200, api.set_setup_autostart(body))
        if route == "/api/config":
            body = self._read_json()
            api.cfg.update({k: v for k, v in body.items() if k in DEFAULT_CONFIG})
            api.store.set("config", api.cfg)
            return self._send(200, api.cfg)
        if route == "/api/systemd/install":
            return self._send(200, api.install_systemd())
        if route == "/api/systemd/remove":
            return self._send(200, api.remove_systemd())
        if route == "/api/autostart/start-all":
            return self._send(200, api.start_autostart())
        self._send(404, {"error": "not found"})

    def do_PUT(self):
        self.do_POST()

    def do_DELETE(self):
        route = self.path.split("?", 1)[0]
        if route == "/api/delete-setup":
            api = self.server.api
            sid = self._query("id")
            return self._send(200, api.delete_setup(sid))
        self._send(404, {"error": "not found"})


class API:
    def __init__(self, cfg):
        self.cfg = cfg
        self.store = StateStore()
        self.pm = ProcessManager(self.store, cfg)
        self.pm.watchdog.start()
        self._boot_autostart()

    def state(self):
        procs = []
        with self.pm.lock:
            procs = [p.to_dict() for p in self.pm.procs.values()]
        models = discover_models(self.cfg.get("model_dirs", []))
        return {
            "config": self.cfg,
            "setups": self.store.get("setups", []),
            "procs": procs,
            "systemd": {"installed": systemd_installed(), "active": service_status()},
            "models": models,
            "runners": RUNNER_LABEL,
            "options": RUNNER_OPTS,
            "rocmfpx_presets": ROCMFPX_PRESETS,
            "model_defaults": model_defaults_for(models),
            "model_runners": {m: model_runner_of(m) for m in models},
        }

    def save_setup(self, body):
        setups = self.store.get("setups", [])
        sid = body.get("id")
        if sid:
            # update an existing setup in place (preserve list position + created time)
            for i, s in enumerate(setups):
                if s.get("id") == sid:
                    body.setdefault("created", s.get("created", time.time()))
                    setups[i] = body
                    self.store.set("setups", setups)
                    return {"ok": True, "setup": body}
        if not body.get("id"):
            body["id"] = uuid.uuid4().hex[:12]
        body.setdefault("created", time.time())
        setups.append(body)
        self.store.set("setups", setups)
        return {"ok": True, "setup": body}

    def delete_setup(self, sid):
        setups = self.store.get("setups", [])
        setups = [s for s in setups if s.get("id") != sid]
        self.store.set("setups", setups)
        return {"ok": True}

    def set_setup_autostart(self, body):
        setups = self.store.get("setups", [])
        sid = body.get("id")
        setup = next((s for s in setups if s.get("id") == sid), None)
        if not setup:
            return {"error": "setup not found"}
        setup["autostart"] = bool(body.get("autostart"))
        self.store.set("setups", setups)
        return {"ok": True, "setup": setup}

    def remove_proc(self, pid):
        with self.pm.lock:
            self.pm.procs.pop(pid, None)
        return {"ok": True}

    def restart_proc(self, pid):
        with self.pm.lock:
            proc = self.pm.procs.get(pid)
        if not proc:
            return {"error": "not found"}
        new = self.pm.start(proc.setup)
        if new.pid is None:
            self.pm.procs.pop(new.id, None)
            proc.status = "crashed"
            proc.log.append("[airunner] restart refused (port in use); left in crashed state")
            return {"error": proc.log[-1] if proc.log else "failed to start"}
        with self.pm.lock:
            self.pm.procs.pop(pid, None)
            self.pm.procs[new.id] = new
        return {"proc": new.to_dict()}

    def apply_setup(self, body):
        setups = self.store.get("setups", [])
        sid = body.get("id")
        setup = next((s for s in setups if s.get("id") == sid), None)
        if not setup:
            return {"error": "setup not found"}
        proc = self.pm.start(setup)
        if proc.pid is None:
            return {"ok": False, "error": proc.log[-1] if proc.log else "failed to start"}
        return {"ok": True, "proc": proc.to_dict()}

    def install_systemd(self):
        app_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "airunner.py"))
        install_systemd(app_path)
        return {"installed": True, "active": service_status()}

    def remove_systemd(self):
        remove_systemd()
        return {"installed": False}

    def start_autostart(self):
        started = []
        for s in self.store.get("setups", []):
            if s.get("autostart"):
                proc = self.pm.start(s)
                started.append(proc.to_dict())
        return {"ok": True, "started": started}

    def _boot_autostart(self):
        for s in self.store.get("setups", []):
            if s.get("autostart"):
                try:
                    self.pm.start(s)
                except Exception:
                    pass


def main():
    cfg = dict(DEFAULT_CONFIG)
    # merge saved config
    store = StateStore()
    saved = store.get("config", {})
    cfg.update({k: v for k, v in saved.items() if k in DEFAULT_CONFIG})
    # always scan the DwarfStar gguf dir so its models show up
    if DWARFSTAR_GGUF_DIR not in cfg["model_dirs"]:
        cfg["model_dirs"].append(DWARFSTAR_GGUF_DIR)
    # CLI overrides
    import argparse
    ap = argparse.ArgumentParser(description="Airunner local model manager")
    ap.add_argument("--host", help="listen host")
    ap.add_argument("--port", type=int, help="listen port")
    ap.add_argument("--llamacpp-bin", help="llama-server binary")
    ap.add_argument("--dwarfstar-bin", help="ds4-server binary")
    ap.add_argument("--strix-bin", help="StrixHalo llama.cpp (Vulkan) llama-server binary")
    ap.add_argument("--rocmfpx-bin", help="ROCmFPX llama.cpp (Vulkan/ROCm) llama-server binary")
    args = ap.parse_args()
    if args.host:
        cfg["host"] = args.host
    if args.port:
        cfg["port"] = args.port
    if args.llamacpp_bin:
        cfg["llamacpp_bin"] = args.llamacpp_bin
    if args.dwarfstar_bin:
        cfg["dwarfstar_bin"] = args.dwarfstar_bin
    if args.strix_bin:
        cfg["strix_bin"] = args.strix_bin
    if args.rocmfpx_bin:
        cfg["rocmfpx_bin"] = args.rocmfpx_bin

    api = API(cfg)
    server = ThreadingHTTPServer((cfg["host"], int(cfg["port"])), Handler)
    server.api = api
    print(f"airunner: web UI at http://{cfg['host']}:{cfg['port']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
