# Jarvus

Local-first, security-minded personal assistant for macOS (MVP). Text-first CLI; no web server, no always-on microphone, no execution of raw model text as shell.

## Architecture (high level)

| Layer | Role |
|--------|------|
| **CLI (`app/main.py`)** | REPL only; wires dependencies; no business logic. |
| **Assistant (`app/assistant/core.py`)** | Conversation loop; calls the model, then the router for tool calls. |
| **Router (`app/assistant/router.py`)** | Only path from model structured output to tools; policy + confirm + audit. |
| **Model (`app/models/*`)** | Reasoning only — `ModelProvider` abstraction; OpenAI today, local later. |
| **Tools (`app/tools/*`)** | Allowlisted capabilities; args are JSON schemas, not shell strings. |
| **Security (`app/security/*`)** | Sanitization, risk levels, interactive confirm for high-risk tools. |
| **Memory (`app/memory/store.py`)** | SQLite for chat turns and audit rows (single DB file). |
| **Audit (`app/audit/logger.py`)** | One log row per tool attempt with outcome. |
| **Config (`app/config.py`)** | Env + optional `.env`; secrets from Keychain or `JARVUS_OPENAI_API_KEY`. |
| **Voice (`app/voice/*`)** | Stubs only until explicitly implemented. |

**Reasoning vs execution:** The model proposes structured `tool_calls`; the router validates names, parses JSON, applies policy, optionally asks for confirmation, runs the tool, and records audit. Model output is never passed to `subprocess` or `shell=True`.

## Phased implementation plan

1. **Phase 0 (this repo):** CLI, OpenAI provider, registry + echo + allowlisted `safe_shell`, SQLite memory/audit, config + Keychain, tests for safety boundaries.
2. **Phase 1:** Additional read-only tools (filesystem browse with path allowlist), structured config UI, richer tests, optional SQLCipher / encrypted store.
3. **Phase 2:** `ModelProvider` implementation for a local engine (e.g. MLX / llama.cpp) behind the same interface.
4. **Phase 3:** Voice I/O behind explicit toggles and permissions; still no always-on mic by default.
5. **Phase 4:** Menu bar or desktop shell reusing the same session/router stack.

## macOS setup

**Requirements:** Python 3.11+, internet only for OpenAI API calls (no inbound listeners).

```bash
cd /path/to/Jarvus
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**API key (pick one):**

- **Environment:** `export JARVUS_OPENAI_API_KEY='...'` or add to `.env` (copy from `.env.example`).
- **Keychain (recommended):**

```bash
python -c "import keyring; keyring.set_password('jarvus', 'openai_api_key', 'YOUR_KEY')"
```

**Run:**

```bash
pip install -e .   # re-run after pulling changes so CLI scripts update
typhoon            # same app as jarvus
# or: jarvus
# or: python -m app.main
```

Data defaults to `~/.jarvus/jarvus.db`. Override with `JARVUS_DATA_DIR` or `JARVUS_DB_PATH`.

**Assistant name vs API model:** The CLI and system prompt present the assistant as **Typhoon**. The actual LLM is whatever you set with **`JARVUS_MODEL`** (default **`gpt-4o-mini`** on OpenAI). You can call the app `typhoon` while using any supported model id. For OpenTyphoon or other OpenAI-compatible hosts, set the appropriate model string (and base URL when supported).

**Tests:**

```bash
pytest
```

## Encrypted SQLite (later)

For sensitive deployments, consider **SQLCipher** (encrypted SQLite) or **OS-level** encryption (FileVault) plus least-privilege file permissions. The store API is intentionally small so the backend can be swapped.

## License

Add your own; none bundled in the MVP scaffold.
