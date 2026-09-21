"""Voice perception module for the Adaptive Multimodal HRI Framework."""

from src.agents.voice.agent import VoiceAgent
from src.agents.voice.audio_utils import (
    calculate_rms_energy,
    generate_silence,
    generate_synthetic_tone,
    is_voice_active,
    load_audio_file,
    normalize_audio,
    pcm_bytes_to_float32,
    resample_audio,
)
from src.agents.voice.recognizer import (
    BaseSpeechRecognizer,
    MockSpeechRecognizer,
    SpeechCommandsRecognizer,
    WhisperSpeechRecognizer,
)
from src.agents.voice.schemas import (
    AudioFormat,
    SpeechIntent,
    TranscribedSegment,
    UrgencyLevel,
    VoiceAgentInput,
    VoiceAgentOutput,
)

__all__ = [
    "VoiceAgent",
    "BaseSpeechRecognizer",
    "WhisperSpeechRecognizer",
    "MockSpeechRecognizer",
    "SpeechCommandsRecognizer",
    "AudioFormat",
    "UrgencyLevel",
    "SpeechIntent",
    "TranscribedSegment",
    "VoiceAgentInput",
    "VoiceAgentOutput",
    "load_audio_file",
    "pcm_bytes_to_float32",
    "resample_audio",
    "calculate_rms_energy",
    "is_voice_active",
    "normalize_audio",
    "generate_synthetic_tone",
    "generate_silence",
]
