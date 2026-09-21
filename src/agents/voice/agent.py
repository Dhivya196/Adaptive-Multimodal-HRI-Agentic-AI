"""Voice Agent: processes speech audio and extracts structured commands and intents."""

import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.agents.base import BaseAgent
from src.agents.voice.audio_utils import (
    calculate_rms_energy,
    is_voice_active,
    load_audio_file,
    pcm_bytes_to_float32,
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
from src.common.exceptions import PerceptionError
from src.common.schemas import AgentType


class VoiceAgent(BaseAgent):
    """
    Voice perception agent.

    Responsibilities:
    - Receive speech audio (files, raw PCM bytes, NumPy float arrays, or text overrides)
    - Compute acoustic energy and perform Voice Activity Detection (VAD)
    - Transcribe speech to text using the configured ASR backend
    - Extract structured linguistic intents, actions, target objects, spatial sectors, and urgency
    - Produce standardized VoiceAgentOutput for downstream multimodal coordination
    """

    # HRI Domain Intent Keywords & Lexicons (Ordered by priority: safety/stop first)
    ACTION_KEYWORDS: Dict[str, List[str]] = {
        "stop_robot": ["stop", "halt", "freeze", "abort", "cancel", "pause", "wait", "hold", "emergency"],
        "pick_and_place": ["pick", "grab", "fetch", "bring", "get", "take", "lift", "carry", "place", "put", "deliver"],
        "navigate_to": ["go", "move", "navigate", "come", "drive", "travel", "head", "approach"],
        "inspect_object": ["look", "inspect", "examine", "find", "check", "scan", "detect", "locate"],
        "query_scene": ["what", "how many", "where", "list", "identify", "status", "tell me"],
        "follow_user": ["follow", "track", "stay with", "accompany"],
    }

    SPATIAL_KEYWORDS: Dict[str, List[str]] = {
        "LEFT": ["left", "leftside", "to the left", "leftmost", "on the left"],
        "CENTER": ["center", "middle", "front", "in front", "center-stage", "straight ahead"],
        "RIGHT": ["right", "rightside", "to the right", "rightmost", "on the right"],
    }

    PROXIMITY_KEYWORDS: Dict[str, List[str]] = {
        "NEAR": ["nearest", "closest", "near", "close", "nearby", "closer"],
        "FAR": ["farthest", "furthest", "far", "distant", "away"],
    }

    OBJECT_KEYWORDS: List[str] = [
        "bottle", "cup", "mug", "wine glass", "bowl", "banana", "apple",
        "sandwich", "orange", "chair", "couch", "table", "laptop",
        "cell phone", "phone", "book", "box", "can", "glass", "object", "item", "person"
    ]

    def __init__(
        self,
        name: str = "VoiceAgent",
        recognizer: Optional[BaseSpeechRecognizer] = None,
        config: Optional[dict] = None,
    ):
        super().__init__(
            name=name,
            agent_type=AgentType.VOICE,
            version="0.1.0",
            description="Transcribes human speech and extracts structured action commands, target objects, and spatial referents.",
            config=config or {},
        )

        self.recognizer = recognizer

        voice_cfg = self.config.get("voice", {})
        self._energy_threshold = voice_cfg.get("vad_energy_threshold", 0.005)
        self._target_sample_rate = voice_cfg.get("sample_rate", 16000)
        self._default_language = voice_cfg.get("language", "en")

    def _initialize(self) -> bool:
        """Initialize the speech recognition backend."""
        if self.recognizer is None:
            voice_cfg = self.config.get("voice", {})
            backend = voice_cfg.get("backend", "whisper").lower()

            if backend == "whisper":
                model_name = voice_cfg.get("model_name", "base")
                device = voice_cfg.get("device", "auto")
                self.recognizer = WhisperSpeechRecognizer(model_name=model_name, device=device)
            elif backend == "mock":
                self.recognizer = MockSpeechRecognizer()
            elif backend == "speech_commands":
                self.recognizer = SpeechCommandsRecognizer()
            else:
                self.recognizer = WhisperSpeechRecognizer()

        return self.recognizer.load_model()

    def normalize_text(self, text: str) -> str:
        """Clean and normalize transcript text for linguistic parsing."""
        if not text:
            return ""
        # Lowercase and remove extra punctuation
        cleaned = text.lower().strip()
        cleaned = re.sub(r"[^\w\s-]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def extract_intent(self, text: str, base_confidence: float) -> Tuple[SpeechIntent, List[str]]:
        """
        Parse normalized transcript to extract HRI action, target object, spatial sector, and urgency.
        """
        normalized = self.normalize_text(text)
        words = set(normalized.split())

        detected_keywords = []
        action = "unknown"
        target_object = None
        spatial_sector = None
        proximity_cue = None
        urgency = UrgencyLevel.NORMAL

        # 1. Action Extraction
        for act_type, synonyms in self.ACTION_KEYWORDS.items():
            for syn in synonyms:
                if re.search(r"\b" + re.escape(syn) + r"\b", normalized):
                    action = act_type
                    detected_keywords.append(syn)
                    break
            if action != "unknown":
                break

        # 2. Urgency Level
        if any(w in words for w in ["emergency", "danger", "immediately", "urgent"]):
            urgency = UrgencyLevel.EMERGENCY
        elif action == "stop_robot" or any(w in words for w in ["stop", "halt", "quick", "fast"]):
            urgency = UrgencyLevel.HIGH

        # 3. Spatial Sector Extraction
        for sector, phrases in self.SPATIAL_KEYWORDS.items():
            for p in phrases:
                if re.search(r"\b" + re.escape(p) + r"\b", normalized):
                    spatial_sector = sector
                    detected_keywords.append(p)
                    break
            if spatial_sector is not None:
                break

        # 4. Proximity Cue Extraction
        for prox, phrases in self.PROXIMITY_KEYWORDS.items():
            for p in phrases:
                if re.search(r"\b" + re.escape(p) + r"\b", normalized):
                    proximity_cue = prox
                    detected_keywords.append(p)
                    break
            if proximity_cue is not None:
                break

        # 5. Target Object Extraction
        for obj in self.OBJECT_KEYWORDS:
            if re.search(r"\b" + re.escape(obj) + r"\b", normalized):
                target_object = obj
                detected_keywords.append(obj)
                break

        # Confidence calculation
        intent_conf = base_confidence
        if action != "unknown":
            intent_conf = min(1.0, intent_conf * 1.05)
        else:
            intent_conf = intent_conf * 0.7

        speech_intent = SpeechIntent(
            action=action,
            target_object=target_object,
            spatial_sector=spatial_sector,
            proximity_cue=proximity_cue,
            urgency=urgency,
            confidence=intent_conf,
            raw_entities={
                "action": action,
                "target_object": target_object,
                "spatial_sector": spatial_sector,
                "proximity_cue": proximity_cue,
                "urgency": urgency.value,
            },
        )

        return speech_intent, detected_keywords

    def _process(self, input_data: Any) -> VoiceAgentOutput:
        """Process speech audio or text override and produce structured VoiceAgentOutput."""
        # ---------------------------------------------------------
        # 1. Parse Input
        # ---------------------------------------------------------
        text_override: Optional[str] = None
        audio_array: Optional[np.ndarray] = None
        sample_rate = self._target_sample_rate
        language = self._default_language

        if isinstance(input_data, VoiceAgentInput):
            text_override = input_data.text_override
            language = input_data.language or self._default_language
            sample_rate = input_data.sample_rate or self._target_sample_rate

            if text_override is None and input_data.audio_data is not None:
                if isinstance(input_data.audio_data, str):
                    audio_array, sample_rate = load_audio_file(input_data.audio_data, target_sr=sample_rate)
                elif isinstance(input_data.audio_data, np.ndarray):
                    audio_array = input_data.audio_data.astype(np.float32)
                elif isinstance(input_data.audio_data, bytes):
                    audio_array = pcm_bytes_to_float32(input_data.audio_data, sampwidth=2, n_channels=1)
        elif isinstance(input_data, str):
            # Could be a filepath or a text string
            if os.path.exists(input_data) and input_data.lower().endswith(".wav"):
                audio_array, sample_rate = load_audio_file(input_data, target_sr=sample_rate)
            else:
                text_override = input_data
        elif isinstance(input_data, np.ndarray):
            audio_array = input_data.astype(np.float32)
        elif isinstance(input_data, dict):
            if "text_override" in input_data:
                text_override = input_data["text_override"]
            elif "audio_file" in input_data:
                audio_array, sample_rate = load_audio_file(input_data["audio_file"], target_sr=sample_rate)
            elif "audio_data" in input_data:
                raw_aud = input_data["audio_data"]
                if isinstance(raw_aud, np.ndarray):
                    audio_array = raw_aud.astype(np.float32)
        else:
            raise PerceptionError(f"Unsupported input type for VoiceAgent: {type(input_data)}")

        # ---------------------------------------------------------
        # 2. Text Override Path (Simulation / Direct Commands)
        # ---------------------------------------------------------
        if text_override is not None:
            normalized = self.normalize_text(text_override)
            speech_intent, keywords = self.extract_intent(text_override, base_confidence=1.0)
            return VoiceAgentOutput(
                agent_name=self.name,
                agent_type=self.agent_type.value,
                success=True,
                transcript=text_override,
                normalized_text=normalized,
                is_speech_detected=bool(text_override.strip()),
                speech_intent=speech_intent,
                detected_keywords=keywords,
                segments=[TranscribedSegment(start_sec=0.0, end_sec=1.0, text=text_override, confidence=1.0)],
                audio_duration_sec=1.0,
                rms_energy=0.5,
                language=language,
                confidence=1.0,
            )

        # ---------------------------------------------------------
        # 3. Audio Validation & VAD
        # ---------------------------------------------------------
        if audio_array is None or len(audio_array) == 0:
            return VoiceAgentOutput(
                agent_name=self.name,
                agent_type=self.agent_type.value,
                success=True,
                transcript="",
                normalized_text="",
                is_speech_detected=False,
                speech_intent=None,
                detected_keywords=[],
                segments=[],
                audio_duration_sec=0.0,
                rms_energy=0.0,
                language=language,
                confidence=0.0,
            )

        # Calculate energy and duration
        rms_energy = calculate_rms_energy(audio_array)
        duration_sec = float(len(audio_array) / sample_rate)
        speech_active = is_voice_active(audio_array, energy_threshold=self._energy_threshold)

        if not speech_active:
            return VoiceAgentOutput(
                agent_name=self.name,
                agent_type=self.agent_type.value,
                success=True,
                transcript="",
                normalized_text="",
                is_speech_detected=False,
                speech_intent=None,
                detected_keywords=[],
                segments=[],
                audio_duration_sec=duration_sec,
                rms_energy=rms_energy,
                language=language,
                confidence=0.0,
            )

        # ---------------------------------------------------------
        # 4. Transcribe Audio
        # ---------------------------------------------------------
        if self.recognizer is None:
            self._initialize()

        assert self.recognizer is not None
        asr_result = self.recognizer.transcribe(audio_array, sample_rate=sample_rate, language=language)

        transcript = asr_result.get("transcript", "").strip()
        confidence = float(asr_result.get("confidence", 0.0))
        segments = asr_result.get("segments", [])
        lang = asr_result.get("language", language)

        normalized = self.normalize_text(transcript)
        speech_intent, keywords = self.extract_intent(transcript, base_confidence=confidence)

        return VoiceAgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type.value,
            success=True,
            transcript=transcript,
            normalized_text=normalized,
            is_speech_detected=bool(transcript),
            speech_intent=speech_intent,
            detected_keywords=keywords,
            segments=segments,
            audio_duration_sec=duration_sec,
            rms_energy=rms_energy,
            language=lang,
            confidence=confidence,
        )

    def _reset(self) -> None:
        """Reset Voice Agent state."""
        pass

    def _shutdown(self) -> None:
        """Release Voice Agent resources."""
        self.recognizer = None
