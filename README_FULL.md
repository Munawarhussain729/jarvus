# Jarvus - Iron Man Style AI Agent

Fully local-first, security-hardened autonomous agent. Like JARVIS - can control your system, build projects, manage tasks, with 200% protection.

## Quick Start

```bash
cd /Users/mac/Projects/Jarvus
source .venv/bin/activate

# Run full agent (uses OpenAI - needs API key)
python -m app.main_full

# Run with local Ollama model
python -m app.main_full --local --model llama3.1:8b
```

## Capabilities

- **File System**: Read, write, list, glob files (sandboxed to user dirs)
- **Planning**: Create plans, track todos
- **Code Execution**: Run Python code safely
- **Shell**: Read-only shell + controlled commands (with confirmation)
- **System**: Process management, system info
- **Memory**: Persistent SQLite with facts & context
- **Security**: All HIGH risk actions require confirmation (type 'yes')

## Protection

- Sandboxed file access (home, /tmp, Projects only)
- Allowlisted tools only - model cannot exec arbitrary shell
- Interactive confirmation for high-risk operations
- Input sanitization, audit logging
- No always-on listeners, local-first by design
