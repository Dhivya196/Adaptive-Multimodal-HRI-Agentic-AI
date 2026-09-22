#!/usr/bin/env python3
"""
Test Whisper transcription across short phrases and analyze audio signal quality.
"""

from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
import whisper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.voice.agent import VoiceAgent
from src.agents.voice.schemas import AudioFormat, VoiceAgentInput
from src.utils.audio_capture import MicrophoneRecorder


def test_phrase(model, voice_agent, phrase_name, duration=3.0):
    print("\n" + "=" * 60)
    print(f"TEST: '{phrase_name}' (Recording for {duration:.1f}s)")
    print(f">>> PLEASE SPEAK CLEARLY: '{phrase_name}' <<<")
    print("=" * 60)

    sample_rate = 16000
    raw_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=10,
    )
    sd.wait()
    audio = raw_data.flatten()

    # Preprocessing: remove DC offset
    audio_centered = audio - np.mean(audio)
    peak = np.max(np.abs(audio_centered))
    rms = np.sqrt(np.mean(audio_centered ** 2))
    clipped_pct = (np.sum(np.abs(audio) >= 0.95) / len(audio)) * 100.0

    print(f"  Signal metrics : RMS={rms:.4f}, Peak={peak:.4f}, Clipped={clipped_pct:.1f}%")

    # Safe normalization if peak is in valid range
    if peak > 0.01:
        audio_norm = audio_centered / max(1.0, peak)
    else:
        audio_norm = audio_centered

    # 1. Direct Whisper inference
    t0 = time.perf_counter()
    w_res = model.transcribe(
        audio_norm,
        language="en",
        fp16=False,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt="Human robot interaction: pick up the bottle, stop, move.",
    )
    w_latency = (time.perf_counter() - t0) * 1000.0
    raw_transcript = w_res.get("text", "").strip()

    # 2. VoiceAgent Process
    v_inp = VoiceAgentInput(
        audio_data=audio_norm,
        audio_format=AudioFormat.NUMPY_FLOAT32,
        sample_rate=sample_rate,
    )
    v_out = voice_agent.process(v_inp)

    print(f"  Expected       : \"{phrase_name}\"")
    print(f"  Actual Whisper : \"{raw_transcript}\"")
    print(f"  Whisper Latency: {w_latency:.1f} ms")
    print(f"  VoiceAgent Act : {v_out.speech_intent.action if v_out.speech_intent else 'none'}")
    print(f"  VoiceAgent Tgt : {v_out.speech_intent.target_object if v_out.speech_intent else 'None'}")
    print(f"  VoiceAgent Conf: {v_out.confidence:.2f}")

    return {
        "phrase": phrase_name,
        "raw_transcript": raw_transcript,
        "action": v_out.speech_intent.action if v_out.speech_intent else "none",
        "target": v_out.speech_intent.target_object if v_out.speech_intent else "None",
        "rms": rms,
        "clipped": clipped_pct,
    }


def main():
    print("Loading Whisper base model...")
    model = whisper.load_model("base", device="cpu")
    voice_agent = VoiceAgent()
    voice_agent.initialize()

    results = []
    # Test recording phrase
    res = test_phrase(model, voice_agent, "pick up the bottle", duration=3.5)
    results.append(res)


if __name__ == "__main__":
    main()
