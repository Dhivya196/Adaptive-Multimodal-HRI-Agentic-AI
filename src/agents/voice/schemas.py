"""Voice-specific schemas and data representations for speech recognition and intent extraction."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.common.schemas import BaseAgentInput, BaseAgentOutput


class AudioFormat(str, Enum):
    """Supported audio input formats."""
    NUMPY_FLOAT32 = "numpy_float32"
    RAW_PCM = "raw_pcm"
    WAV_FILE = "wav_file"
    TEXT_SIMULATION = "text_simulation"


class UrgencyLevel(str, Enum):
    """Urgency level of the detected speech command."""
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EMERGENCY = "EMERGENCY"


@dataclass
class TranscribedSegment:
    """A timestamped segment of transcribed speech."""
    start_sec: float = 0.0
    end_sec: float = 0.0
    text: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "text": self.text,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class SpeechIntent:
    """Structured intent and semantic entities extracted from speech."""
    action: str = "unknown"
    target_object: Optional[str] = None
    spatial_sector: Optional[str] = None
    proximity_cue: Optional[str] = None
    urgency: UrgencyLevel = UrgencyLevel.NORMAL
    confidence: float = 0.0
    raw_entities: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target_object": self.target_object,
            "spatial_sector": self.spatial_sector,
            "proximity_cue": self.proximity_cue,
            "urgency": self.urgency.value if isinstance(self.urgency, UrgencyLevel) else self.urgency,
            "confidence": round(self.confidence, 3),
            "raw_entities": self.raw_entities,
        }


@dataclass
class VoiceAgentInput(BaseAgentInput):
    """Input payload passed to the Voice Agent."""
    audio_data: Any = None
    sample_rate: int = 16000
    audio_format: AudioFormat = AudioFormat.NUMPY_FLOAT32
    text_override: Optional[str] = None
    language: str = "en"


@dataclass
class VoiceAgentOutput(BaseAgentOutput):
    """Structured speech recognition context emitted by the Voice Agent."""
    transcript: str = ""
    normalized_text: str = ""
    is_speech_detected: bool = False
    speech_intent: Optional[SpeechIntent] = None
    detected_keywords: List[str] = field(default_factory=list)
    segments: List[TranscribedSegment] = field(default_factory=list)
    audio_duration_sec: float = 0.0
    rms_energy: float = 0.0
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "transcript": self.transcript,
            "normalized_text": self.normalized_text,
            "is_speech_detected": self.is_speech_detected,
            "speech_intent": self.speech_intent.to_dict() if self.speech_intent else None,
            "detected_keywords": self.detected_keywords,
            "segments": [seg.to_dict() for seg in self.segments],
            "audio_duration_sec": round(self.audio_duration_sec, 3),
            "rms_energy": round(self.rms_energy, 5),
            "language": self.language,
        })
        return base_dict
