"""Voice input listener with wake word and continuous mode."""
from __future__ import annotations

import threading
import time
from typing import Optional, Callable

try:
    import speech_recognition as sr
except ImportError:
    sr = None


class VoiceListener:
    def __init__(self, wake_words: list[str] = None, callback: Callable[[str], None] = None, use_wake_word: bool = False):
        self.wake_words = wake_words or ["jarvis", "jarvus", "hey jarvis", "hey jarvus"]
        self.callback = callback
        self.listening = False
        self.should_stop = False
        self._thread: Optional[threading.Thread] = None
        self.recognizer = sr.Recognizer() if sr else None
        self.microphone = sr.Microphone() if sr else None
        self.use_wake_word = use_wake_word

    def _adjust_for_noise(self):
        if self.microphone and self.recognizer:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

    def _contains_wake_word(self, text: str) -> bool:
        text_low = text.lower()
        for w in self.wake_words:
            if w in text_low:
                return True
        return False

    def _listen_continuous(self):
        print("👂 Jarvus is listening... (say 'ok go to sleep' to stop listening)")
        self._adjust_for_noise()
        active = not self.use_wake_word  # if no wake word, always active
        while not self.should_stop:
            try:
                if not self.recognizer or not self.microphone:
                    time.sleep(1)
                    continue
                with self.microphone as source:
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=8)
                    except sr.WaitTimeoutError:
                        continue
                try:
                    text = self.recognizer.recognize_google(audio)
                except:
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                    except:
                        continue
                if not text:
                    continue
                text_low = text.lower()
                print(f"🎤 Heard: {text}")
                if "ok go to sleep" in text_low or "go to sleep" in text_low or "stop listening" in text_low:
                    print("😴 Going to sleep...")
                    self.stop()
                    break
                if self.use_wake_word:
                    if self._contains_wake_word(text):
                        # extract command after wake word
                        for w in self.wake_words:
                            if w in text_low:
                                cmd = text_low.replace(w, "").strip()
                                if cmd:
                                    if self.callback:
                                        self.callback(cmd)
                                break
                    continue
                else:
                    if self.callback:
                        self.callback(text)
            except Exception:
                time.sleep(0.2)
        self.listening = False

    def start(self):
        if self.listening:
            return
        self.should_stop = False
        self._thread = threading.Thread(target=self._listen_continuous, daemon=True)
        self._thread.start()
        self.listening = True

    def stop(self):
        self.should_stop = True
        self.listening = False
