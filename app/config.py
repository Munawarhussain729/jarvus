"""Application configuration from environment (fail-closed defaults)."""
from __future__ import annotations

import os
from pathlib import Path

import keyring
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


KEYRING_SERVICE = "jarvus"
KEYRING_OPENAI_USER = "openai_api_key"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JARVUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = Field(default="gpt-4o-mini")
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".jarvus")
    db_path: Path | None = Field(default=None)
    log_level: str = Field(default="INFO")
    openai_api_key: str | None = Field(default=None, repr=False)
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=512)  # smaller for speed

    # Local LLM (Ollama native API preferred; any OpenAI-compatible base URL works)
    llm_base_url: str = Field(default="http://localhost:11434")
    # gpt-oss:20b MoE (~3.6B active params, ~13GB MXFP4) is the largest model that
    # runs fluidly on a 16GB M1 alongside STT+TTS. Lighter fallback:
    # JARVUS_LLM_MODEL=qwen3:14b or qwen3:4b-instruct-2507-q4_K_M
    llm_model: str = Field(default="gpt-oss:20b")
    llm_num_ctx: int = Field(default=8192)  # fits 16GB Macs; raise via JARVUS_PROFILE=coding
    llm_think: bool = Field(default=False)  # qwen3-style thinking; off for voice latency
    llm_keep_alive: str = Field(default="30m")  # Ollama model residency; lower frees RAM sooner

    # Voice pipeline
    stt_engine: str = Field(default="auto")  # auto | mlx | faster-whisper
    stt_model: str = Field(default="small")  # small|turbo|medium|tiny|base|large or mlx repo id
    # Hallucination gates: raise for noisy rooms (drop more), lower for soft mics
    stt_no_speech_threshold: float = Field(default=0.6)  # drop segments above this no_speech_prob
    stt_min_logprob: float = Field(default=-1.0)  # drop segments below this avg_logprob
    stt_silence_rms: float = Field(default=0.003)  # skip whisper entirely below this audio RMS
    tts_engine: str = Field(default="auto")  # auto | kokoro | say
    tts_voice: str = Field(default="af_heart")  # kokoro voice id or macOS voice name
    tts_rate: int = Field(default=200)  # 'say' words-per-minute
    wake_words: str = Field(default="jarvis,jarvus,hey jarvis,hey jarvus")
    # Exact-match phrases that interrupt TTS playback mid-speech (barge-in)
    interrupt_phrases: str = Field(
        default="stop,stop it,cancel,be quiet,quiet,enough,shut up,never mind,shush")

    # Daily briefing: auto = voice sessions only, on = text too, off = never
    auto_brief: str = Field(default="auto")
    # Heuristic tool-payload prefilter (off = always send all schemas)
    tool_prefilter: bool = Field(default=True)
    # Background fact extraction after each turn - Jarvus personalises over time
    auto_learn: bool = Field(default=True)
    # Devin project registry seed: "name=/abs/path,other=/abs/path2"
    # (memory-stored registrations override entries with the same name)
    devin_projects: str = Field(default="")
    # Named preset for context/generation/memory budgets: lean | balanced | coding.
    # Individual JARVUS_* vars always win over the profile.
    profile: str = Field(default="balanced")

    # Remote access server (POST /chat)
    server_host: str = Field(default="127.0.0.1")
    server_port: int = Field(default=8765)
    server_token: str | None = Field(default=None, repr=False)

    @property
    def wake_word_list(self) -> list[str]:
        return [w.strip().lower() for w in self.wake_words.split(",") if w.strip()]

    @property
    def interrupt_phrase_list(self) -> list[str]:
        return [p.strip().lower() for p in self.interrupt_phrases.split(",") if p.strip()]

    @property
    def devin_project_map(self) -> dict[str, str]:
        """Parse 'name=/path,name2=/path2' into {name: path}."""
        out: dict[str, str] = {}
        for pair in self.devin_projects.split(","):
            if "=" in pair:
                name, path = pair.split("=", 1)
                if name.strip() and path.strip():
                    out[name.strip().lower().replace(" ", "-")] = path.strip()
        return out

    @field_validator("data_dir", mode="before")
    @classmethod
    def expand_data_dir(cls, v: object) -> Path:
        if v is None:
            return Path.home() / ".jarvus"
        return Path(os.path.expanduser(str(v))).resolve()

    @field_validator("db_path", mode="before")
    @classmethod
    def expand_db_path(cls, v: object) -> Path | None:
        if v is None or v == "":
            return None
        return Path(os.path.expanduser(str(v))).resolve()

    def resolved_db_path(self) -> Path:
        if self.db_path is not None:
            return self.db_path
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "jarvus.db"

    def get_openai_api_key(self) -> str:
        if self.openai_api_key and self.openai_api_key.strip():
            return self.openai_api_key.strip()
        secret = keyring.get_password(KEYRING_SERVICE, KEYRING_OPENAI_USER)
        if secret and secret.strip():
            return secret.strip()
        raise RuntimeError(
            "Missing OpenAI API key. Set JARVUS_OPENAI_API_KEY or store with: "
            f'python -c "import keyring; keyring.set_password(\'{KEYRING_SERVICE}\', '
            f"'{KEYRING_OPENAI_USER}', 'YOUR_KEY')\""
        )


# Budget presets - JARVUS_PROFILE selects one; any explicit JARVUS_* var wins.
_PROFILES: dict[str, dict[str, object]] = {
    "lean":     {"llm_num_ctx": 8192,  "max_tokens": 256,  "stt_model": "small", "llm_keep_alive": "10m"},
    "balanced": {"llm_num_ctx": 8192,  "max_tokens": 384,  "stt_model": "small", "llm_keep_alive": "30m"},
    "coding":   {"llm_num_ctx": 16384, "max_tokens": 1024, "stt_model": "small", "llm_keep_alive": "60m"},
}


def _explicit_env_keys() -> set[str]:
    """JARVUS_* names the user set explicitly - in the process env or .env."""
    keys = {k for k in os.environ if k.startswith("JARVUS_")}
    try:
        for line in Path(".env").read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                keys.add(line.split("=", 1)[0].strip())
    except OSError:
        pass
    return keys


def load_settings() -> Settings:
    s = Settings()
    prof = str(s.profile).lower()
    cfg = _PROFILES.get(prof)
    if cfg:
        explicit = _explicit_env_keys()
        for field, val in cfg.items():
            if f"JARVUS_{field.upper()}" not in explicit:
                setattr(s, field, val)
    return s
