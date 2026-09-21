"""Integration tests for the VoicePipeline."""

from pathlib import Path
import numpy as np
import pytest

from src.agents.voice.agent import VoiceAgent
from src.agents.voice.audio_utils import generate_synthetic_tone
from src.agents.voice.recognizer import MockSpeechRecognizer
from src.agents.voice.schemas import VoiceAgentOutput
from src.pipeline.voice_pipeline import VoicePipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_voice_pipeline_text_simulation():
    received_outputs = []

    def callback(out: VoiceAgentOutput):
        received_outputs.append(out)

    recognizer = MockSpeechRecognizer()
    agent = VoiceAgent(recognizer=recognizer)
    pipeline = VoicePipeline(agent=agent, output_callback=callback)

    output = pipeline.process_text_simulation("Bring me the apple")
    assert output.transcript == "Bring me the apple"
    assert len(received_outputs) == 1
    assert received_outputs[0].speech_intent.target_object == "apple"

    pipeline.stop()


def test_voice_pipeline_buffer_processing():
    recognizer = MockSpeechRecognizer(default_transcript="halt robot")
    agent = VoiceAgent(recognizer=recognizer)
    pipeline = VoicePipeline(agent=agent)

    tone = generate_synthetic_tone(freq_hz=440.0, duration_sec=0.5)
    output = pipeline.process_buffer(tone, sample_rate=16000)

    assert output.is_speech_detected is True
    assert output.transcript == "halt robot"
    assert output.speech_intent.action == "stop_robot"

    pipeline.stop()


def test_voice_pipeline_stream():
    recognizer = MockSpeechRecognizer(default_transcript="pick cup")
    agent = VoiceAgent(recognizer=recognizer)
    pipeline = VoicePipeline(agent=agent)

    def audio_chunk_gen():
        for _ in range(3):
            yield generate_synthetic_tone(freq_hz=440.0, duration_sec=0.2)

    results = pipeline.run_stream(audio_chunk_gen(), sample_rate=16000, max_chunks=3)
    assert len(results) == 3
    assert all(r.is_speech_detected for r in results)

    pipeline.stop()
