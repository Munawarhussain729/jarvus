# Jarvus

Local-first, security-minded personal assistant — JARVIS-style voice control, 100% local inference. Optimized for Apple Silicon, portable to a private server.

## Stack (all local)

| Layer | Default | Fallback / options |
|---|---|---|
| **LLM** | Ollama, `qwen3:8b` (native API, thinking off) | any Ollama model; any OpenAI-compatible endpoint (llama.cpp, LM Studio, vLLM); OpenAI via `--openai` |
| **STT** | mlx-whisper `large-v3-turbo` (GPU on Apple Silicon) | faster-whisper (`JARVUS_STT_ENGINE=faster-whisper`), smaller models via `JARVUS_STT_MODEL` |
| **TTS** | Kokoro-82M via mlx-audio (`af_heart`) | macOS `say` (`JARVUS_TTS_ENGINE=say`) |
| **Memory** | SQLite (`~/.jarvus/jarvus.db`): chats, audit, long-term facts | — |
| **Remote** | built-in HTTP server (`POST /chat`) | client mode via `--remote-url` |

## Architecture

```
mic ──> VoiceListener ──> STT (mlx-whisper) ──> JarvusAgent ──> LocalProvider (Ollama)
                                                  │
                                   ToolRouter ────┴── policy + confirm + audit
                                                  │
                            allowlisted tools (fs, shell, macOS, web, memory…)
                                                  │
reply ──> clean_for_speech ──> TTS (Kokoro/say) ──> speaker
```

- The model only proposes structured `tool_calls`; the router validates, applies risk policy, asks for confirmation on HIGH-risk tools (typed `yes` in text mode, spoken `yes` in voice mode), executes, and writes an audit row.
- A deterministic fast path (`app/assistant/intents.py`) answers the most common commands — timers, clock, notifications, screenshot, music, volume, weather, open-app, **daily brief** — without an LLM round-trip. Everything else goes to the model; if the model promises an action without calling a tool, the agent nudges it once to actually call it.
- A conservative keyword prefilter (`JARVUS_TOOL_PREFILTER`) shrinks the ~40 tool schemas sent to the local model; turns expand to the full set after the first tool round so nothing is ever unrecoverable.
- **Daily tasks**: `list_reminders` / `calendar_read` read macOS Reminders & Calendar; `daily_brief` merges them with the persisted todo list (`todo_write`) and latest notes into one spoken digest. Say "brief me on my day", or let the morning auto-brief speak it on the first session of the day (`JARVUS_AUTO_BRIEF=auto|on|off`).
- While Jarvus speaks, the mic loop drains and discards audio — it can't hear itself.
- Long-term facts (`remember`/`recall`/`forget` tools) persist in SQLite and are injected into the system prompt. Todos and plans persist too; internal state (e.g. last brief date) lives in a separate `meta` table that never reaches the prompt.
- `read_notifications` needs Full Disk Access granted to your terminal app (System Settings → Privacy & Security → Full Disk Access), then restart Jarvus. `list_reminders`/`calendar_read`/`daily_brief` trigger a one-time macOS Automation permission prompt on first use.

## Setup (macOS)

```bash
cd /path/to/Jarvus
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[voice,dev]"

brew install espeak-ng        # phonemizer for Kokoro
brew services start ollama    # local LLM server, auto-starts at login
ollama pull qwen3:8b          # or qwen3:4b for lower latency
```

## Run

```bash
./jarvus.sh                    # text chat, local model
./jarvus_voice.sh              # continuous voice control
./jarvus_voice_wake.sh         # wake-word mode: "hey jarvis, ..."
./jarvus_fast.sh               # text chat with qwen3:4b (faster)
# or: python -m app.main [--voice] [--wake-word] [--no-speak] [--model X]
```

Voice tips:
- Say **"Jarvis"** alone → it answers "Yes?", then just speak the command.
- HIGH-risk tools ask for confirmation — answer **"yes"** out loud.
- Say **"go to sleep"** to stop listening.

## Remote access / future server

```bash
python -m app.main --serve --host 0.0.0.0 --port 8765   # on the server
JARVUS_SERVER_TOKEN=secret                              # required off-localhost
python -m app.main --voice --remote-url http://server:8765   # Mac client
```

`POST /chat {"text": "...", "confirm": true}` — `confirm` pre-approves HIGH-risk tools for that single request. `GET /health` for monitoring. When you move the brain to your private server, set `JARVUS_LLM_BASE_URL` on it (Ollama/llama.cpp/vLLM all work) and keep the Mac as the voice + macOS-control client.

## Config

Everything is env-driven (`JARVUS_*`, see `.env.example`): LLM URL/model/ctx/think, STT engine+model, TTS engine+voice, wake words, server host/port/token, `JARVUS_FS_ROOTS` for extra file-sandbox roots, `JARVUS_AUTO_BRIEF` for the morning digest, `JARVUS_TOOL_PREFILTER` for the schema prefilter.

## Security model

- Tools are an explicit allowlist; model text is never executed.
- HIGH risk (run_command, python_exec, write_file, open_app, message, email, web_fetch, safe_shell) requires confirmation — typed or spoken.
- File tools are sandboxed to `$HOME`, `/tmp`, and `JARVUS_FS_ROOTS`.
- Every tool call is audit-logged with args and outcome.

## Tests

```bash
pytest    # 30+ tests incl. sandbox boundaries, memory, voice confirmer
```
