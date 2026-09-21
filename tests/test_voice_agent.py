"""Unit tests for the VoiceAgent class and intent extraction logic."""

from pathlib import Path
import numpy as np
import pytest

from src.agents.voice.agent import VoiceAgent
from src.agents.voice.audio_utils import generate_silence, generate_synthetic_tone
from src.agents.voice.recognizer import MockSpeechRecognizer
from src.agents.voice.schemas import AudioFormat, UrgencyLevel, VoiceAgentInput, VoiceAgentOutput
from src.common.exceptions import PerceptionError
from src.common.schemas import AgentStatus, AgentType

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def mock_voice_agent():
    recognizer = MockSpeechRecognizer(default_transcript="pick up the bottle on the left", confidence=0.95)
    agent = VoiceAgent(recognizer=recognizer)
    agent.initialize()
    return agent


def test_voice_agent_initialization(mock_voice_agent):
    assert mock_voice_agent.status == AgentStatus.READY
    assert mock_voice_agent.agent_type == AgentType.VOICE
    assert mock_voice_agent.metadata.agent_name == "VoiceAgent"


def test_voice_agent_text_simulation(mock_voice_agent):
    inp = VoiceAgentInput(text_override="Pick up the bottle on the left")
    output: VoiceAgentOutput = mock_voice_agent.process(inp)

    assert output.success is True
    assert output.is_speech_detected is True
    assert output.transcript == "Pick up the bottle on the left"
    assert output.normalized_text == "pick up the bottle on the left"
    assert output.speech_intent is not None
    assert output.speech_intent.action == "pick_and_place"
    assert output.speech_intent.target_object == "bottle"
    assert output.speech_intent.spatial_sector == "LEFT"
    assert output.speech_intent.urgency == "NORMAL"
    assert "pick" in output.detected_keywords
    assert "bottle" in output.detected_keywords


def test_voice_agent_urgency_and_stop(mock_voice_agent):
    inp = VoiceAgentInput(text_override="Stop immediately, danger ahead!")
    output: VoiceAgentOutput = mock_voice_agent.process(inp)

    assert output.speech_intent is not None
    assert output.speech_intent.action == "stop_robot"
    assert output.speech_intent.urgency == UrgencyLevel.EMERGENCY.value or output.speech_intent.urgency == "EMERGENCY"


def test_voice_agent_navigation_and_center(mock_voice_agent):
    inp = VoiceAgentInput(text_override="Navigate to the couch in front")
    output: VoiceAgentOutput = mock_voice_agent.process(inp)

    assert output.speech_intent is not None
    assert output.speech_intent.action == "navigate_to"
    assert output.speech_intent.target_object == "couch"
    assert output.speech_intent.spatial_sector == "CENTER"


def test_voice_agent_audio_processing_with_mock(mock_voice_agent):
    tone = generate_synthetic_tone(freq_hz=440.0, duration_sec=1.0, amplitude=0.5)
    inp = VoiceAgentInput(audio_data=tone, sample_rate=16000)
    output: VoiceAgentOutput = mock_voice_agent.process(inp)

    assert output.success is True
    assert output.is_speech_detected is True
    assert output.transcript == "pick up the bottle on the left"
    assert output.audio_duration_sec == 1.0
    assert output.rms_energy > 0.1


def test_voice_agent_silence_vad(mock_voice_agent):
    silence = generate_silence(duration_sec=1.0)
    inp = VoiceAgentInput(audio_data=silence, sample_rate=16000)
    output: VoiceAgentOutput = mock_voice_agent.process(inp)

    assert output.success is True
    assert output.is_speech_detected is False
    assert output.transcript == ""
    assert output.speech_intent is None


def test_voice_agent_real_wav_dataset(mock_voice_agent):
    sample_wav = PROJECT_ROOT / "datasets" / "Speech_commands" / "00b01445_nohash_0.wav"
    if sample_wav.exists():
        output = mock_voice_agent.process(str(sample_wav))
        assert output.success is True
        assert output.is_speech_detected is True
        assert output.audio_duration_sec == 1.0


def test_voice_agent_invalid_input_type(mock_voice_agent):
    with pytest.raises(PerceptionError):
        mock_voice_agent.process(12345)
