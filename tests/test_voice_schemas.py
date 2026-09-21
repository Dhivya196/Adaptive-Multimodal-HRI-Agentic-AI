"""Unit tests for Voice Agent data schemas and serialization."""

import json
import pytest

from src.agents.voice.schemas import (
    AudioFormat,
    SpeechIntent,
    TranscribedSegment,
    UrgencyLevel,
    VoiceAgentInput,
    VoiceAgentOutput,
)
from src.common.schemas import AgentType


def test_speech_intent_defaults_and_serialization():
    intent = SpeechIntent(
        action="pick_and_place",
        target_object="bottle",
        spatial_sector="LEFT",
        proximity_cue="NEAR",
        urgency=UrgencyLevel.NORMAL,
        confidence=0.92,
    )
    d = intent.to_dict()
    assert d["action"] == "pick_and_place"
    assert d["target_object"] == "bottle"
    assert d["spatial_sector"] == "LEFT"
    assert d["proximity_cue"] == "NEAR"
    assert d["urgency"] == "NORMAL"
    assert d["confidence"] == 0.92


def test_transcribed_segment_serialization():
    seg = TranscribedSegment(start_sec=0.5, end_sec=2.1, text="bring me the cup", confidence=0.88)
    d = seg.to_dict()
    assert d["start_sec"] == 0.5
    assert d["end_sec"] == 2.1
    assert d["text"] == "bring me the cup"
    assert d["confidence"] == 0.88


def test_voice_agent_output_serialization():
    intent = SpeechIntent(
        action="stop_robot",
        urgency=UrgencyLevel.EMERGENCY,
        confidence=0.99,
    )
    seg = TranscribedSegment(start_sec=0.0, end_sec=0.8, text="stop immediately", confidence=0.99)
    out = VoiceAgentOutput(
        agent_name="VoiceAgent",
        agent_type=AgentType.VOICE.value,
        success=True,
        transcript="stop immediately",
        normalized_text="stop immediately",
        is_speech_detected=True,
        speech_intent=intent,
        detected_keywords=["stop", "immediately"],
        segments=[seg],
        audio_duration_sec=0.8,
        rms_energy=0.045,
        language="en",
        confidence=0.99,
    )

    d = out.to_dict()
    assert d["agent_name"] == "VoiceAgent"
    assert d["agent_type"] == "voice_agent"
    assert d["is_speech_detected"] is True
    assert d["speech_intent"]["action"] == "stop_robot"
    assert d["speech_intent"]["urgency"] == "EMERGENCY"
    assert len(d["segments"]) == 1
    assert d["audio_duration_sec"] == 0.8
    assert d["detected_keywords"] == ["stop", "immediately"]

    json_str = out.to_json()
    parsed = json.loads(json_str)
    assert parsed["transcript"] == "stop immediately"
    assert parsed["speech_intent"]["action"] == "stop_robot"


def test_voice_agent_input_creation():
    inp = VoiceAgentInput(
        session_id="sess_123",
        text_override="pick up the bottle",
        audio_format=AudioFormat.TEXT_SIMULATION,
        language="en",
    )
    assert inp.session_id == "sess_123"
    assert inp.text_override == "pick up the bottle"
    assert inp.audio_format == AudioFormat.TEXT_SIMULATION
