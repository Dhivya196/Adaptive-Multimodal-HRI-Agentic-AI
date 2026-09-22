#!/usr/bin/env python3
"""
Audio pipeline diagnostic tool for live HRI speech input.
Records audio from the active microphone, analyzes waveform statistics,
detects clipping, saves debug_audio.wav, and runs direct Whisper inference.
"""

from pathlib import Path
import sys
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import whisper

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.audio_capture import MicrophoneRecorder


def main():
    print("=" * 70)
    print("      HRI AUDIO PIPELINE & WHISPER SIGNAL DIAGNOSTIC")
    print("=" * 70)

    # 1. Device Inspection
    recorder = MicrophoneRecorder()
    dev_info = recorder.get_input_device_info()

    print("\n[AUDIO DEVICE]")
    print(f"  Device index   : {dev_info.get('index')}")
    print(f"  Device name    : {dev_info.get('name')}")
    print(f"  Input channels : {dev_info.get('channels')}")
    print(f"  Default SR     : {dev_info.get('sample_rate')} Hz")

    # List all PortAudio input devices for reference
    all_devices = sd.query_devices()
    print("\n[ALL INPUT DEVICES]")
    for idx, d in enumerate(all_devices):
        if d.get("max_input_channels", 0) > 0:
            print(f"  [{idx}] {d['name']} (Channels: {d['max_input_channels']}, Default SR: {d['default_samplerate']} Hz)")

    # 2. Record audio phrase
    sample_rate = 16000
    record_duration = 4.0
    print(f"\n[RECORDING] Recording {record_duration:.1f}s from microphone...")
    print(">>> SPEAK NOW: 'Pick up the bottle' <<<")

    raw_data = sd.rec(
        int(record_duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=recorder.device,
    )
    sd.wait()
    audio = raw_data.flatten()

    # 3. Analyze raw waveform
    num_samples = len(audio)
    duration = num_samples / sample_rate
    raw_rms = float(np.sqrt(np.mean(audio ** 2)))
    raw_peak = float(np.max(np.abs(audio)))
    raw_min = float(np.min(audio))
    raw_max = float(np.max(audio))
    raw_dc = float(np.mean(audio))

    # Clipping statistics (samples near saturation +/- 0.95)
    clipped_count = int(np.sum(np.abs(audio) >= 0.95))
    clipped_pct = (clipped_count / num_samples) * 100.0

    print("\n[RAW AUDIO SIGNAL ANALYSIS]")
    print(f"  Sample rate         : {sample_rate} Hz")
    print(f"  Channels            : 1 (Mono)")
    print(f"  dtype               : {audio.dtype}")
    print(f"  Samples             : {num_samples}")
    print(f"  Duration            : {duration:.2f} s")
    print(f"  DC Offset (mean)    : {raw_dc:.5f}")
    print(f"  RMS Energy          : {raw_rms:.5f}")
    print(f"  Peak Amplitude      : {raw_peak:.5f}")
    print(f"  Min / Max           : {raw_min:.5f} / {raw_max:.5f}")
    print(f"  Clipped samples     : {clipped_count} ({clipped_pct:.2f}%)")

    # 4. Save debug_audio.wav
    debug_wav_path = PROJECT_ROOT / "debug_audio.wav"
    int16_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(str(debug_wav_path), sample_rate, int16_audio)
    print(f"\n[SAVED FILE] Saved raw captured audio to: {debug_wav_path}")

    # 5. Direct Whisper Test on Raw Audio
    print("\n[DIRECT WHISPER TEST - RAW AUDIO]")
    model = whisper.load_model("base", device="cpu")
    t0 = time.perf_counter()
    result_raw = model.transcribe(audio, language="en", fp16=False, temperature=0.0)
    raw_latency = (time.perf_counter() - t0) * 1000.0
    print(f"  Model               : base")
    print(f"  Transcript (Raw)    : \"{result_raw.get('text', '').strip()}\"")
    print(f"  Latency             : {raw_latency:.1f} ms")

    # 6. Direct Whisper Test with Light Preprocessing
    # Preprocessing: remove DC offset and normalize amplitude safely if not clipping
    audio_proc = audio - np.mean(audio)
    peak_proc = np.max(np.abs(audio_proc))
    if peak_proc > 0.01:
        audio_proc = audio_proc / max(1.0, peak_proc)

    debug_proc_wav = PROJECT_ROOT / "debug_audio_processed.wav"
    wavfile.write(str(debug_proc_wav), sample_rate, (np.clip(audio_proc, -1.0, 1.0) * 32767).astype(np.int16))

    print("\n[DIRECT WHISPER TEST - PROCESSED AUDIO (DC centered + normalized)]")
    t0 = time.perf_counter()
    result_proc = model.transcribe(audio_proc, language="en", fp16=False, temperature=0.0)
    proc_latency = (time.perf_counter() - t0) * 1000.0
    print(f"  Transcript (Proc)   : \"{result_proc.get('text', '').strip()}\"")
    print(f"  Latency             : {proc_latency:.1f} ms")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
