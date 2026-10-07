"""Jarvus with voice control - continuous listening mode."""
from __future__ import annotations

import argparse
import logging
import sys
import time
import re
import subprocess

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

try:
    from app.voice.listener import VoiceListener
except ImportError:
    VoiceListener = None


def build_full_registry():
    return ToolRegistry([
        EchoTool(), SafeShellTool(), ReadFileTool(), WriteFileTool(),
        ListDirTool(), GlobTool(), ShellReadOnlyTool(), SystemInfoTool(),
        PsTool(), PlanTool(), TodoWriteTool(), PythonExecTool(),
        RunCommandTool(), OpenAppTool(), SayTool(), CreateDocxTool(),
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
