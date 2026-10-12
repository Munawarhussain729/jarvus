"""Jarvus with voice control - continuous listening mode with full Siri-like capabilities."""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time

from app.assistant.agent import JarvusAgent
from app.assistant.router import ToolRouter
from app.audit.logger import AuditLogger
from app.config import load_settings
from app.memory.store import MemoryStore
from app.models.openai_provider import OpenAIProvider
from app.tools.registry import ToolRegistry, EchoTool
from app.tools.filesystem import ReadFileTool, WriteFileTool, ListDirTool, GlobTool
from app.tools.shell_extended import ShellReadOnlyTool
from app.tools.system import SystemInfoTool, PsTool
from app.tools.plan import PlanTool, TodoWriteTool
from app.tools.code import PythonExecTool
from app.tools.terminal import RunCommandTool
from app.tools.safe_shell import SafeShellTool
from app.tools.macos import OpenAppTool, SayTool
from app.tools.docx import CreateDocxTool
from app.tools.weather import WeatherTool
from app.tools.timer import TimerTool, AlarmTool
from app.tools.reminders import ReminderTool, CalendarTool
from app.tools.music import MusicTool
from app.tools.messaging import MessageTool, EmailTool
from app.tools.search import WebSearchTool
from app.tools.web import WebFetchTool

try:
    from app.voice.listener import VoiceListener
    from app.voice.output import speak_text, is_available as tts_available
except ImportError:
    VoiceListener = None
    speak_text = None
    tts_available = lambda: False


def build_full_registry():
    return ToolRegistry([
        EchoTool(),
        SafeShellTool(),
        ReadFileTool(),
        WriteFileTool(),
        ListDirTool(),
        GlobTool(),
        ShellReadOnlyTool(),
        SystemInfoTool(),
        PsTool(),
        PlanTool(),
        TodoWriteTool(),
        PythonExecTool(),
        RunCommandTool(),
        OpenAppTool(),
        SayTool(),
        CreateDocxTool(),
        WeatherTool(),
        TimerTool(),
        AlarmTool(),
        ReminderTool(),
        CalendarTool(),
        MusicTool(),
        MessageTool(),
        EmailTool(),
        WebSearchTool(),
        WebFetchTool(),
    ])


def clean_for_speech(text: str) -> str:
    text = re.sub(r'\[tool.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[SECURITY.*?\].*?(?=\n\n|\n$|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)
    text = re.sub(r'Jarvus>', '', text)
    text = re.sub(r'Jarvus:', '', text)
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:400]


class JarvusVoice:
    def __init__(self, agent: JarvusAgent, wake_word: bool = False, always_speak: bool = True):
        self.agent = agent
        self.wake_word = wake_word
        self.always_speak = always_speak
        self.listener = VoiceListener(callback=self.handle_voice, use_wake_word=wake_word) if VoiceListener else None
        self.processing = False

    def speak(self, text: str):
        clean = clean_for_speech(text)
        if not clean:
            return
        if speak_text:
            speak_text(clean)
        else:
            try:
                subprocess.run(['say', clean], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    def handle_voice(self, text: str):
        if self.processing:
            return
        self.processing = True
        try:
            print(f"\n🎤 You said: {text}")
            print("🤔 Thinking...")
            res = self.agent.process_user_text(text)
            print(f"🤖 Jarvus: {res}\n")
            if self.always_speak:
                self.speak(res)
        except Exception as e:
            print(f"Error: {e}")
            if self.always_speak:
                self.speak("An error occurred")
        finally:
            self.processing = False

    def start(self):
        if self.listener:
            if self.wake_word:
                print("\n🔊 Wake-word mode")
            else:
                print("\n🔊 Continuous listening mode")
            print("Say 'ok go to sleep' to stop")
            self.speak("Jarvus is online and ready")
            self.listener.start()
        else:
            print("Voice input not available. Install speech_recognition and pyaudio.")


def main():
    parser = argparse.ArgumentParser(description="Jarvus - Voice-controlled AI Assistant (100% local)")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI instead of local LLM")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--fast", action="store_true", help="Use faster smaller model")
    parser.add_argument("--voice", action="store_true", help="Enable voice input/output")
    parser.add_argument("--wake-word", action="store_true", help="Use wake word activation")
    parser.add_argument("--no-speak", action="store_true", help="Disable voice output")
    parser.add_argument("--text", action="store_true", help="Force text-only mode")
    args = parser.parse_args()

    settings = load_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    registry = build_full_registry()
    memory = MemoryStore(settings.resolved_db_path())
    audit = AuditLogger(memory)
    router = ToolRouter(registry, audit)

    use_local = not args.openai

    if use_local:
        try:
            from app.models.local_provider import LocalProvider
            if args.model:
                model = args.model
            elif args.fast:
                model = "qwen2.5:3b"
            else:
                model = "qwen2.5:3b"
            provider = LocalProvider(
                base_url=args.base_url,
                model=model,
                temperature=0.1,
                max_tokens=128,
            )
            print(f"Using local model: {model}")
        except Exception as e:
            print(f"[warn] local provider failed: {e}, falling back to OpenAI", file=sys.stderr)
            api_key = settings.get_openai_api_key()
            provider = OpenAIProvider(api_key=api_key, model=args.model or settings.model)
    else:
        api_key = settings.get_openai_api_key()
        provider = OpenAIProvider(api_key=api_key, model=args.model or settings.model)

    agent = JarvusAgent(provider=provider, registry=registry, router=router, memory=memory)

    use_voice = args.voice and not args.text

    if use_voice:
        voice = JarvusVoice(
            agent=agent,
            wake_word=args.wake_word,
            always_speak=not args.no_speak,
        )
        voice.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            if voice.listener:
                voice.listener.stop()
    else:
        print("\n🤖 Jarvus (Typhoon) - Full Agent Ready")
        print("Type 'exit' to quit.")
        print(f"DB: {settings.resolved_db_path()}\n")

        try:
            while True:
                try:
                    line = input("You> ").strip()
                except EOFError:
                    break
                if not line:
                    continue
                if line.lower() in ("exit", "quit"):
                    break
                try:
                    res = agent.process_user_text(line)
                    print(f"\nJarvus> {res}\n")
                except Exception as e:
                    logging.exception("error")
                    print(f"[error] {e}", file=sys.stderr)
        finally:
            memory.close()


if __name__ == "__main__":
    main()
