"""Voice I/O: local STT input and TTS output (100% local engines)."""
from app.voice.input import listen_and_transcribe_local, transcribe_audio_data, is_available as stt_available
from app.voice.output import speak_text, get_available_voices, is_available as tts_available
from app.voice.listener import VoiceListener
from app.voice.session import JarvusVoice
from app.voice.stt import get_stt
from app.voice.tts import get_tts

__all__ = [
    "listen_and_transcribe_local",
    "transcribe_audio_data",
    "speak_text",
    "get_available_voices",
    "VoiceListener",
    "JarvusVoice",
    "get_stt",
    "get_tts",
    "stt_available",
    "tts_available",
]
