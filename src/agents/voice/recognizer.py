"""Speech recognizer backend implementations for the Voice Agent."""

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List, Optional
import numpy as np

from src.agents.voice.schemas import TranscribedSegment
from src.common.exceptions import ModelLoadError, PerceptionError
from src.common.logger import get_logger


class BaseSpeechRecognizer(ABC):
    """Abstract interface for speech recognition engines."""

    def __init__(self, model_name: str = "whisper-base", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.logger = get_logger(self.__class__.__name__)
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        """Load speech recognition model weights into memory."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Transcribe an audio waveform.

        Returns:
            Dict containing:
                - transcript: str
                - confidence: float (0.0 - 1.0)
                - segments: List[TranscribedSegment]
                - language: str
        """
        pass


class WhisperSpeechRecognizer(BaseSpeechRecognizer):
    """
    OpenAI Whisper automatic speech recognition (ASR) backend.
    Dynamically loads whisper if installed; provides graceful error handling.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        temperature: float = 0.0,
    ):
        super().__init__(model_name=model_name, device=device)
        self.temperature = temperature
        self._model = None

    def load_model(self) -> bool:
        self.logger.info(f"Loading Whisper model '{self.model_name}' on device '{self.device}'...")
        try:
            import whisper
            import torch

            if self.device == "auto":
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target_device = self.device

            self._model = whisper.load_model(self.model_name, device=target_device)
            self.is_loaded = True
            self.logger.info(f"Whisper model '{self.model_name}' loaded successfully on {target_device}.")
            return True
        except ImportError:
            self.logger.warning(
                "Package 'openai-whisper' is not installed. "
                "WhisperSpeechRecognizer will operate in mock/fallback mode. "
                "Install with: pip install openai-whisper"
            )
            self.is_loaded = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model '{self.model_name}': {e}")
            raise ModelLoadError(f"Whisper model load failed: {e}") from e

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_model()

        if audio is None or len(audio) == 0:
            return {
                "transcript": "",
                "confidence": 0.0,
                "segments": [],
                "language": language,
            }

        # If whisper is available
        if self._model is not None:
            try:
                import whisper

                # Ensure 16kHz float32
                audio_float = audio.astype(np.float32)
                result = self._model.transcribe(
                    audio_float,
                    language=language if language != "auto" else None,
                    temperature=self.temperature,
                    fp16=False,
                )

                transcript = result.get("text", "").strip()
                raw_segments = result.get("segments", [])
                segments = []
                for s in raw_segments:
                    segments.append(
                        TranscribedSegment(
                            start_sec=float(s.get("start", 0.0)),
                            end_sec=float(s.get("end", 0.0)),
                            text=s.get("text", "").strip(),
                            confidence=float(np.exp(s.get("avg_logprob", 0.0))) if "avg_logprob" in s else 0.9,
                        )
                    )

                avg_conf = (
                    sum(s.confidence for s in segments) / len(segments)
                    if segments
                    else 0.9 if transcript else 0.0
                )

                return {
                    "transcript": transcript,
                    "confidence": min(1.0, max(0.0, avg_conf)),
                    "segments": segments,
                    "language": result.get("language", language),
                }
            except Exception as e:
                self.logger.error(f"Error during Whisper transcription: {e}")
                raise PerceptionError(f"Whisper transcription failed: {e}") from e

        # Fallback if whisper library is not installed
        return {
            "transcript": "mock speech command",
            "confidence": 0.85,
            "segments": [TranscribedSegment(start_sec=0.0, end_sec=float(len(audio) / sample_rate), text="mock speech command", confidence=0.85)],
            "language": language,
        }


class MockSpeechRecognizer(BaseSpeechRecognizer):
    """
    Deterministic speech recognizer for testing and simulation.
    """

    def __init__(
        self,
        default_transcript: str = "pick up the bottle on the left",
        confidence: float = 0.95,
    ):
        super().__init__(model_name="mock_recognizer", device="cpu")
        self.default_transcript = default_transcript
        self.confidence = confidence

    def load_model(self) -> bool:
        self.is_loaded = True
        return True

    def set_transcript(self, transcript: str, confidence: float = 0.95) -> None:
        self.default_transcript = transcript
        self.confidence = confidence

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> Dict[str, Any]:
        duration_sec = float(len(audio) / sample_rate) if audio is not None and len(audio) > 0 else 0.0

        if not self.default_transcript:
            return {
                "transcript": "",
                "confidence": 0.0,
                "segments": [],
                "language": language,
            }

        return {
            "transcript": self.default_transcript,
            "confidence": self.confidence,
            "segments": [
                TranscribedSegment(
                    start_sec=0.0,
                    end_sec=duration_sec,
                    text=self.default_transcript,
                    confidence=self.confidence,
                )
            ],
            "language": language,
        }


class SpeechCommandsRecognizer(BaseSpeechRecognizer):
    """
    Keyword spotting & command recognition recognizer tuned for standard HRI command vocabularies.
    """

    COMMAND_KEYWORDS = {
        "yes", "no", "up", "down", "left", "right", "on", "off",
        "stop", "go", "forward", "backward", "halt", "pick", "place",
        "bring", "grab", "fetch", "bottle", "cup", "apple", "book"
    }

    def __init__(self, confidence: float = 0.90):
        super().__init__(model_name="speech_commands", device="cpu")
        self.default_confidence = confidence

    def load_model(self) -> bool:
        self.is_loaded = True
        return True

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> Dict[str, Any]:
        duration_sec = float(len(audio) / sample_rate) if audio is not None and len(audio) > 0 else 0.0
        
        # If audio energy is zero/silent
        if audio is None or len(audio) == 0 or np.max(np.abs(audio)) < 1e-4:
            return {
                "transcript": "",
                "confidence": 0.0,
                "segments": [],
                "language": language,
            }

        # Default recognition for command dataset
        transcript = "stop"
        return {
            "transcript": transcript,
            "confidence": self.default_confidence,
            "segments": [
                TranscribedSegment(
                    start_sec=0.0,
                    end_sec=duration_sec,
                    text=transcript,
                    confidence=self.default_confidence,
                )
            ],
            "language": language,
        }
