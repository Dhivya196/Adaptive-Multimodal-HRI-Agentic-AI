"""Audio utility functions for reading, converting, normalizing, and analyzing audio signals."""

import io
import math
import os
import struct
from typing import Tuple, Union
import wave

import numpy as np

from src.common.exceptions import PerceptionError


def load_audio_file(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load a WAV audio file, convert it to mono float32 in [-1.0, 1.0], and resample if needed.

    Args:
        file_path: Path to the WAV file.
        target_sr: Desired output sample rate (default: 16000 Hz for Whisper/speech models).

    Returns:
        Tuple of (audio_array_float32, sample_rate).
    """
    if not os.path.exists(file_path):
        raise PerceptionError(f"Audio file does not exist: {file_path}")

    try:
        with wave.open(file_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        audio = pcm_bytes_to_float32(raw_bytes, sampwidth, n_channels)

        if framerate != target_sr and len(audio) > 0:
            audio = resample_audio(audio, framerate, target_sr)
            framerate = target_sr

        return audio, framerate

    except Exception as e:
        if isinstance(e, PerceptionError):
            raise
        raise PerceptionError(f"Failed to read WAV audio file '{file_path}': {e}") from e


def pcm_bytes_to_float32(raw_bytes: bytes, sampwidth: int, n_channels: int = 1) -> np.ndarray:
    """
    Convert raw PCM bytes to a 1D mono float32 numpy array in [-1.0, 1.0].
    """
    if not raw_bytes:
        return np.empty(0, dtype=np.float32)

    if sampwidth == 1:
        # 8-bit unsigned PCM
        data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        # 16-bit signed PCM
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        data = data / 32768.0
    elif sampwidth == 3:
        # 24-bit signed PCM
        num_samples = len(raw_bytes) // 3
        int_data = np.zeros(num_samples, dtype=np.int32)
        for i in range(num_samples):
            chunk = raw_bytes[i * 3 : (i + 1) * 3]
            int_data[i] = int.from_bytes(chunk, byteorder="little", signed=True)
        data = int_data.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        # 32-bit signed PCM or float
        try:
            data = np.frombuffer(raw_bytes, dtype=np.float32)
        except Exception:
            data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise PerceptionError(f"Unsupported audio sample width: {sampwidth} bytes")

    # If multi-channel, convert to mono by averaging channels
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    return data.astype(np.float32)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample a 1D audio numpy array from orig_sr to target_sr using linear interpolation.
    """
    if orig_sr == target_sr or len(audio) == 0:
        return audio

    num_target_samples = int(round(len(audio) * float(target_sr) / float(orig_sr)))
    orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
    target_indices = np.linspace(0, len(audio) - 1, num=num_target_samples)
    resampled = np.interp(target_indices, orig_indices, audio)
    return resampled.astype(np.float32)


def calculate_rms_energy(audio: np.ndarray) -> float:
    """
    Calculate the Root-Mean-Square (RMS) energy of an audio array.
    """
    if audio is None or len(audio) == 0:
        return 0.0
    mean_sq = np.mean(np.square(audio))
    return float(np.sqrt(max(0.0, mean_sq)))


def is_voice_active(audio: np.ndarray, energy_threshold: float = 0.005) -> bool:
    """
    Energy-based Voice Activity Detection (VAD). Returns True if audio RMS energy exceeds threshold.
    """
    return calculate_rms_energy(audio) >= energy_threshold


def filter_audio_dc_and_rumble(audio: np.ndarray, fs: int = 16000, cutoff: float = 80.0) -> np.ndarray:
    """
    Apply a zero-phase high-pass filter and remove DC offset to clean up live microphone signals.
    Removes low-frequency rumble and DC bias that degrade Whisper ASR performance.
    """
    if audio is None or len(audio) < 16:
        return audio if audio is not None else np.empty(0, dtype=np.float32)

    # 1. Remove DC bias
    centered = (audio - np.mean(audio)).astype(np.float32)

    # 2. 80Hz high-pass filter using scipy.signal
    try:
        from scipy import signal
        sos = signal.butter(4, cutoff, "hp", fs=fs, output="sos")
        filtered = signal.sosfilt(sos, centered).astype(np.float32)
        return filtered
    except Exception:
        # Fallback to mean subtraction if scipy is unavailable
        return centered


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio array to target peak amplitude.
    """
    if audio is None or len(audio) == 0:
        return audio
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        return (audio / max_val * target_peak).astype(np.float32)
    return audio


def generate_synthetic_tone(
    freq_hz: float = 440.0,
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate a pure sinusoidal synthetic tone (useful for tests and pipelines).
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return waveform.astype(np.float32)


def generate_silence(duration_sec: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Generate silent audio buffer (all zeros).
    """
    return np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
