#!/usr/bin/env python3
"""
Diagnostic script to test microphone capture and Whisper transcription in isolation.
Records 5 seconds of audio from the default audio device, computes metrics,
saves to a temporary WAV, and transcribes using the VoiceAgent Whisper model.
"""

from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.voice.agent import VoiceAgent
from src.agents.voice.schemas import AudioFormat, VoiceAgentInput
from src.utils.audio_capture import MicrophoneRecorder


def main():
    print("=" * 60)
    print("      MICROPHONE & WHISPER ISOLATED DIAGNOSTIC TEST")
    print("=" * 60)

    recorder = MicrophoneRecorder()
    dev_info = recorder.get_input_device_info()

    print("\n[AUDIO DEVICE]")
    print(f"  Index          : {dev_info.get('index')}")
    print(f"  Name           : {dev_info.get('name')}")
    print(f"  Input channels : {dev_info.get('channels')}")
    print(f"  Sample rate    : {dev_info.get('sample_rate')} Hz")

    if not recorder.is_available():
        print("\n[AUDIO] MICROPHONE FAILURE: No input device available.")
        sys.exit(1)

    duration_sec = 5.0
    sample_rate = 16000
    print(f"\n[AUDIO RECORDING]")
    print(f"  Recording for {duration_sec:.1f} seconds (Speak a test phrase, e.g. 'Pick it up')...")
    
    try:
        raw_recording = sd.rec(
            int(duration_sec * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=recorder.device,
        )
        sd.wait()
        audio = raw_recording.flatten()
        # Remove DC bias
        audio = audio - np.mean(audio)
        max_abs = np.max(np.abs(audio))
        if max_abs > 1.0:
            audio = audio / max_abs
    except Exception as e:
        print(f"\n[AUDIO] MICROPHONE FAILURE: {e}")
        sys.exit(1)

    num_samples = len(audio)
    actual_duration = num_samples / sample_rate
    rms = float(np.sqrt(np.mean(audio ** 2)))
    min_val = float(np.min(audio))
    max_val = float(np.max(audio))

    print(f"  Sample rate    : {sample_rate} Hz")
    print(f"  Samples        : {num_samples}")
    print(f"  Duration       : {actual_duration:.2f} s")
    print(f"  RMS            : {rms:.5f}")
    print(f"  Min / Max      : {min_val:.4f} / {max_val:.4f}")

    # Save to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name
        # Convert float32 [-1.0, 1.0] to int16 for standard WAV format
        int16_audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(wav_path, sample_rate, int16_audio)

    print(f"  Saved WAV      : {wav_path}")

    # Load VoiceAgent with Whisper model
    print("\n[VOICE AGENT / WHISPER INFERENCE]")
    try:
        voice_agent = VoiceAgent()
        voice_agent.initialize()
        rec_type = type(getattr(voice_agent, "recognizer", None)).__name__
        model_name = getattr(getattr(voice_agent, "recognizer", None), "model_name", "base")
        print(f"  Voice Backend  : {rec_type} ({model_name})")

        t0 = time.perf_counter()
        voice_input = VoiceAgentInput(
            audio_data=audio,
            audio_format=AudioFormat.NUMPY_FLOAT32,
            sample_rate=sample_rate,
        )
        voice_output = voice_agent.process(voice_input)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        transcript = voice_output.transcript or ""
        intent_action = voice_output.speech_intent.action if voice_output.speech_intent else "none"
        intent_target = voice_output.speech_intent.target_object if voice_output.speech_intent else "None"
        intent_conf = voice_output.speech_intent.confidence if voice_output.speech_intent else 0.0

        print("\n[AUDIO TEST]")
        print("  Recording      : OK")
        print(f"  Duration       : {actual_duration:.2f} s")
        print(f"  RMS            : {rms:.5f}")
        print(f"  Whisper model  : {model_name}")
        print(f"  Transcript     : \"{transcript}\"")
        print(f"  Latency        : {elapsed_ms:.1f} ms")

        print(f"\n[VOICE]")
        print(f"  Transcript     : \"{transcript}\"")
        print(f"  Intent         : {intent_action}")
        print(f"  Target         : {intent_target}")
        print(f"  Confidence     : {intent_conf:.2f}")

    except Exception as e:
        print(f"\n[VOICE] TRANSCRIPTION FAILURE: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary WAV
        try:
            Path(wav_path).unlink(missing_ok=True)
        except Exception:
            pass

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
