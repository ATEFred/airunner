# Airunner — Project Overview

> Read this first if you are a new session taking over this project. It captures
> the whole context so you can continue without re-discovering everything.

## What it is

Airunner is a **local model runner manager** for this machine (`halo-server`,
CachyOS/Arch Linux). It is a small web app that lets the user launch, manage,
watch, and auto-restart the GGUF models stored locally. It supports two runners:

- **llama.cpp** → `llama-server` (`/home/fred/ai/llama.cpp/build/bin/llama-server`)
- **DwarfStar** → `ds4-server` (`/home/fred/dwarfstar/ds4-server`)

It is intentionally **zero-dependency**: the backend is Python 3 stdlib (no pip
packages are installed on this box) and the frontend is a single vanilla-JS
HTML file. There is no `pip`, so keep it stdlib-only.

## What it provides (feature list)

- Launch models from either runner with any of that runner's launch options.
- Per-runner **launch-option table**: every option, its default, and a suggested
  value, editable before launch. For both runners.
- **Parallel models**: many models running at once; each has its own process,
  output console, Stop button.
- **Live output console** per running model (streams stdout/stderr).
- **Remembers setups**: save any configuration as a named "setup"; apply/delete
  later. Persisted in `~/.local/share/airunner/state.json`.
- **Autostart at machine boot**: a setup can be marked autostart; Airunner
  installs a systemd *user* service, and on boot auto-launches autostart setups.
- **Restart on crash** (watchdog): a setup marked "restart on crash" is
  relaunched whenever its process dies; restart count shown per model.
- **Config panel** to point at the llama.cpp / DwarfStar binaries and the model
  directories scanned for `.gguf` files.
- **Port policy**: the configured port is used exactly as set. If it is taken
  (by another airunner model or any other live process), the launch is refused and
  the reason is shown. Stopped/removed entries never reserve a port.
- **MTP/DSpark support** for DwarfStar: `--mtp` renders as a dropdown of
  candidate support/draft GGUF files found in the model dirs.

## How it works (architecture)

Two files:

- `airunner.py` — backend + REST API + process manager
- `web/index.html` — the single-page UI

### Backend (`airunner.py`)

- `ThreadingHTTPServer` serving the REST API and the static `web/index.html`.
  Default bind `127.0.0.1:8090` (configurable).
- **`RUNNER_OPTS`**: the launch-option knowledge base. Two lists
  (`LLAMACPP_OPTS`, `DWARFSTAR_OPTS`), each entry `{label, key, type, default,
  desc, suggested}`. `type` is `int|float|str|flag|choice`.
  - `option_cli_args(runner, opts)` converts a `{key: value}` dict into CLI
    args. **Gotcha:** `flag` options are passed as bare flags; value options
    pass `--flag value`. `--flash-attn` was a bug here — it is a `choice`
    (`on|off|auto`), NOT a bare flag.
  - `port_opt_key(runner)` + `find_free_port(requested)` implement port
    conflict resolution inside `ProcessManager.start()`.
- **`StateStore`**: JSON persistence at `~/.local/share/airunner/state.json`
  (`{config, setups}`). Thread-safe via a lock; writes via tmp+rename.
- **`ProcessManager`**: holds `self.procs` (dict id→`Process`), starts/stops
  processes, and runs a daemon **watchdog** thread.
  - `Process`: id, pid, setup, status (`running|stopped|crashed|restarting`),
    a `deque(maxlen=2000)` log, restart counter, and `stdout`/`reader`.
  - `start(setup)`: builds `[binary, -m, model] + option_cli_args(...)`, handles
    port conflicts, `Popen`s with stdout+stderr merged, spawns a **reader
    thread** that appends lines to the log.
  - **Watchdog detection**: the reader thread reaps the child on EOF and sets
    `proc.dead`; the watchdog checks `proc.reader.is_alive()` (not `kill 0`,
    which lies for zombies). If a non-stopped process is dead and
    `restart_on_crash` is set, it restarts and **carries the restart count
    across** onto the fresh `Process` record.
  - `stop(id)`: closes stdout, SIGTERM, waits 5s, then SIGKILL fallback. Adds
    the pid to `stop_flags` so the watchdog won't resurrect it.
- **Autostart**: `_boot_autostart()` in `API.__init__` launches every setup with
  `autostart:true` on app start. systemd integration:
  - `install_systemd()` writes `~/.config/systemd/user/airunner.service`
    (`Restart=always`) and `systemctl --user enable + restart`.
  - `remove_systemd()` disables + deletes the unit.
- **`API`**: the handler glue. `state()` returns config, setups, procs,
  systemd status, discovered models, and the option tables. `discover_models`
  scans `model_dirs` for `*.gguf`.

### REST API (the UI talks to this)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/state` | full state (config, setups, procs, models, options) |
| GET | `/api/models` | discovered `.gguf` models |
| GET/PUT | `/api/config` | runner binaries, model dirs, host/port, watchdog interval |
| GET | `/api/launch-options?runner=` | option table for `llamacpp` / `dwarfstar` |
| POST | `/api/run` | start a model (`{setup}`) |
| POST | `/api/stop` | stop a model (`{id}`) |
| POST | `/api/remove` | drop a stopped/crashed record from the list (`{id}`) |
| POST | `/api/save-setup` | remember a setup |
| POST | `/api/apply-setup` | launch a saved setup (`{id}`) |
| POST | `/api/delete-setup` | forget a setup |
| POST | `/api/systemd/install` `/api/systemd/remove` | autostart service |
| POST | `/api/autostart/start-all` | launch all autostart setups now |

### Frontend (`web/index.html`)

Single page, vanilla JS. Left column: Config panel, launch-a-model (runner
selector, model chips, editable option grid, port, autostart + restart-on-crash
checkboxes, Run/Save), saved setups. Right column: running models with live
consoles, Stop buttons, and **Remove** for stopped/crashed models. The `mtp`
option is rendered specially as a dropdown of candidate support/draft GGUF
files (names matching `mtp|draft|dspark`). Polls `/api/state` every 2 s.

## How to run

```bash
cd /home/fred/ai/airunner
python3 airunner.py
# web UI: http://127.0.0.1:8090
```

Optional CLI args: `--host`, `--port`, `--llamacpp-bin`, `--dwarfstar-bin`.

## Machine facts / environment

- CachyOS (Arch) Linux, 32 cores, 124 GB RAM, no NVIDIA (AMD — DwarfStar has
  `rocm`). `nvidia-smi` absent.
- `open-webui` (Open WebUI) container serves on `:3000`; a `ds4-server --rocm`
  model serves Open WebUI on `127.0.0.1:8080` — **that port is occupied**, so
  new models need other ports (auto-assign handles this).
- Models live in `/home/fred/ai/models/` (16 `.gguf` files, several split into
  `-0000X-of-0000Y` parts). Notable: `Gemma4-26B-A4B-QAT-...-Q4_K_M.gguf`,
  `Qwen3.5-122B-A10B-UD-Q4_K_XL-*.gguf` (3 parts), `DeepSeek-V4-Flash-...`,
  `MiniMax-M2.7-...` (4 parts), plus support/draft files:
  `DSV4-Flash-DSpark-draft-bf16.gguf`, `mtp-gemma-4-26B-A4B-it-Q8_0.gguf`,
  `mtp-gemma-4-26B-A4B-it-Q8_0.gguf` (Gemma MTP), `templates/qwen35-...jinja`.

## Known issues / gotchas (important for future work)

1. **`--flash-attn` must be a `choice`** (`on|off|auto`), never a bare `flag` —
   llama-server consumes the next arg as its value (fixed, but keep it).
2. **Watchdog uses the reader-thread lifecycle**, not `os.kill(pid, 0)` —
   zombies make `kill 0` return true and hide deaths. Keep this approach.
3. **Restart count is carried onto the fresh `Process`** on watchdog restart,
   otherwise it resets to 0.
4. **systemd *user* services start only after login.** For pre-login autostart
   you'd need a root-level unit (not implemented).
5. **Split models**: multi-part `.gguf` files each appear as a separate model
   entry; there's no grouping yet.
6. **Stopped/crashed processes linger** in the list until Remove is clicked —
   by design, with a Remove button.

## Potential improvements (ideas to continue)

- **Group multi-part `.gguf` files** (`-00001-of-00003`) into one logical model
  entry (the runner loads part 1 and mmaps the rest).
- **Detect model VRAM/RAM fit** and suggest `--n-gpu-layers` / `--ctx` / SSD
  streaming automatically from model size vs free memory.
- **Chat UI** (a prompt box per model) instead of just the console, using each
  runner's OpenAI-compatible `/v1` API.
- **Persist running processes** across restarts (currently only autostart
  setups relaunch on boot).
- **Health-check** the HTTP port (e.g. `/v1/models`) to mark a model "ready"
  once it finishes loading (currently 503 while loading).
- **Per-model resource limits** (CPU affinity, cgroup, GPU power).
- **Template/alias helpers** for the Qwen chat template and model aliases.
- **Better watchdog**: restart with backoff to avoid hot-looping a crash-looping
  model.
- **Remove stale `stop_flags`** / cleanup reader threads for removed processes.
