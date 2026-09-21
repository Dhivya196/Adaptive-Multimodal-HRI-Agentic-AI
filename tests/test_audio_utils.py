"""Unit tests for audio processing and analysis utilities."""

import os
from pathlib import Path
import numpy as np
import pytest

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
from src.common.exceptions import PerceptionError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_generate_synthetic_tone():
    audio = generate_synthetic_tone(freq_hz=440.0, duration_sec=0.5, sample_rate=16000, amplitude=0.8)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == 8000
    assert np.max(audio) <= 0.81
    assert np.min(audio) >= -0.81


def test_generate_silence():
    silence = generate_silence(duration_sec=0.25, sample_rate=16000)
    assert len(silence) == 4000
    assert np.all(silence == 0.0)


def test_calculate_rms_energy():
    silence = generate_silence(1.0)
    assert calculate_rms_energy(silence) == 0.0

    tone = generate_synthetic_tone(freq_hz=440.0, duration_sec=1.0, amplitude=1.0)
    rms = calculate_rms_energy(tone)
    # Theoretical RMS of sine wave with amplitude 1.0 is ~ 1/sqrt(2) ≈ 0.707
    assert 0.70 < rms < 0.72


def test_is_voice_active():
    silence = generate_silence(1.0)
    assert is_voice_active(silence, energy_threshold=0.01) is False

    tone = generate_synthetic_tone(freq_hz=440.0, duration_sec=1.0, amplitude=0.5)
    assert is_voice_active(tone, energy_threshold=0.01) is True


def test_normalize_audio():
    tone = generate_synthetic_tone(freq_hz=440.0, duration_sec=0.5, amplitude=0.3)
    norm = normalize_audio(tone, target_peak=0.95)
    assert np.isclose(np.max(np.abs(norm)), 0.95, atol=1e-3)


def test_pcm_bytes_to_float32():
    # 16-bit signed PCM test
    int16_arr = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    raw_bytes = int16_arr.tobytes()
    float_arr = pcm_bytes_to_float32(raw_bytes, sampwidth=2, n_channels=1)
    assert float_arr.dtype == np.float32
    assert np.isclose(float_arr[0], 0.0, atol=1e-4)
    assert np.isclose(float_arr[1], 0.5, atol=1e-4)
    assert np.isclose(float_arr[2], -0.5, atol=1e-4)
    assert np.isclose(float_arr[3], 1.0, atol=1e-3)


def test_resample_audio():
    audio = generate_synthetic_tone(freq_hz=440.0, duration_sec=1.0, sample_rate=8000)
    assert len(audio) == 8000
    resampled = resample_audio(audio, orig_sr=8000, target_sr=16000)
    assert len(resampled) == 16000


def test_load_audio_file_existing():
    sample_wav = PROJECT_ROOT / "datasets" / "Speech_commands" / "00b01445_nohash_0.wav"
    if sample_wav.exists():
        audio, sr = load_audio_file(str(sample_wav), target_sr=16000)
        assert isinstance(audio, np.ndarray)
        assert sr == 16000
        assert len(audio) == 16000
        assert audio.dtype == np.float32


def test_load_audio_file_not_found():
    with pytest.raises(PerceptionError):
        load_audio_file("non_existent_file.wav")
