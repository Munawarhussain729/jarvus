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
    llm_model: str = Field(default="qwen3:8b")
    llm_num_ctx: int = Field(default=16384)
    llm_think: bool = Field(default=False)  # qwen3-style thinking; off for voice latency

    # Voice pipeline
    stt_engine: str = Field(default="auto")  # auto | mlx | faster-whisper
    stt_model: str = Field(default="turbo")  # turbo|small|medium|tiny|base|large or mlx repo id
    tts_engine: str = Field(default="auto")  # auto | kokoro | say
    tts_voice: str = Field(default="af_heart")  # kokoro voice id or macOS voice name
    tts_rate: int = Field(default=200)  # 'say' words-per-minute
    wake_words: str = Field(default="jarvis,jarvus,hey jarvis,hey jarvus")

    # Remote access server (POST /chat)
    server_host: str = Field(default="127.0.0.1")
    server_port: int = Field(default=8765)
    server_token: str | None = Field(default=None, repr=False)

    @property
    def wake_word_list(self) -> list[str]:
        return [w.strip().lower() for w in self.wake_words.split(",") if w.strip()]

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


def load_settings() -> Settings:
    return Settings()
