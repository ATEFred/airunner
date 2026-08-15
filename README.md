# Airunner — local model runner manager

Manage and launch the GGUF models on this machine. Supports both runners:

- **llama.cpp** (`llama-server`) — `/home/fred/ai/llama.cpp/build/bin/llama-server`
- **DwarfStar** (`ds4-server`) — `/home/fred/dwarfstar/ds4-server`

Zero dependencies: Python 3 stdlib backend + a single-page vanilla JS frontend.

## Run

```bash
cd /home/fred/ai/airunner
python3 airunner.py
# web UI: http://127.0.0.1:8090
```

Optional CLI args: `--host`, `--port`, `--llamacpp-bin`, `--dwarfstar-bin`.

## What it does

- **Launch models** from either runner, with any of the runner's launch options
  (context, threads, GPU layers, backend, KV cache, SSD streaming, DSpark, …).
- **Launch-option table** per runner: shows every option with its default, a
  suggested value, and lets you edit before launching. For both `llama-server`
  and `ds4-server`.
- **Parallel models**: run multiple models at once — each gets its own process,
  console, and Stop button.
- **Output console**: a live textbox per running model showing the runner's
  stdout/stderr.
- **Remembers past setups**: save any configuration as a "setup"; apply or
  delete them later. Setups are persisted in
  `~/.local/share/airunner/state.json`.
- **Autostart at machine boot**: mark a setup "autostart". Airunner installs a
  systemd *user* service (via the UI button) so it comes back on reboot, then
  auto-launches every autostart setup.
- **Restart if it crashes**: mark a setup "restart on crash" and a watchdog
  (default every 5 s) relaunches the model whenever its process dies. Restart
  count is shown per model.
- **Point at binaries & models**: the Config panel sets the llama.cpp /
  DwarfStar binary paths and the model directories scanned for `.gguf` files.

## Autostart at boot

In the UI: open **Config** → **Install autostart service**. This writes
`~/.config/systemd/user/airunner.service` and enables it, so after a reboot the
service starts Airunner, which then launches every setup marked **Autostart**.
Use **Remove service** to disable.

> Note: systemd *user* services start after your desktop login. If you want the
> models up before login, you'd need a system-level unit (requires root) — not
> installed here by default.

## REST API (same as the UI)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/state` | full state (config, setups, procs, models, options) |
| GET | `/api/models` | discovered `.gguf` models |
| GET/PUT | `/api/config` | runner binaries, model dirs, host/port, watchdog interval |
| GET | `/api/launch-options?runner=` | option table for `llamacpp` / `dwarfstar` |
| POST | `/api/run` | start a model (`{setup}`) |
| POST | `/api/stop` | stop a model (`{id}`) |
| POST | `/api/save-setup` | remember a setup |
| POST | `/api/apply-setup` | launch a saved setup (`{id}`) |
| POST | `/api/delete-setup` | forget a setup |
| POST | `/api/systemd/install` `/api/systemd/remove` | autostart service |
| POST | `/api/autostart/start-all` | launch all autostart setups now |

## Files

- `airunner.py` — backend: process manager, watchdog, autostart, REST API
- `web/index.html` — the single-page UI
