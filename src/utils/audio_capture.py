"""Live microphone capture with Voice Activity Detection (VAD) for the HRI Voice Agent."""

import math
import time
from typing import Callable, Optional, Tuple, Union
import numpy as np

from src.common.logger import get_logger


class MicrophoneRecorder:
    """
    Live audio capture handler using sounddevice with RMS-based Voice Activity Detection.
    Captures mono float32 audio at 16,000 Hz, compatible with OpenAI Whisper and VoiceAgent.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.1,
        energy_threshold: float = 0.008,
        silence_duration: float = 1.2,
        max_duration: float = 8.0,
        device: Optional[Union[int, str]] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.device = device
        self.logger = get_logger("MicrophoneRecorder")
        self._sounddevice = None
        self._check_driver()

    def _check_driver(self) -> bool:
        """Verify sounddevice library and available microphone devices."""
        try:
            import sounddevice as sd
            self._sounddevice = sd
            return True
        except ImportError:
            self.logger.warning("sounddevice is not installed. Live microphone capture disabled.")
            self._sounddevice = None
            return False
        except Exception as e:
            self.logger.warning(f"Audio driver error: {e}")
            self._sounddevice = None
            return False

    def is_available(self) -> bool:
        """Check if microphone hardware and sounddevice driver are available."""
        if self._sounddevice is None:
            return False
        try:
            devices = self._sounddevice.query_devices()
            input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
            return len(input_devices) > 0
        except Exception as e:
            self.logger.debug(f"Device query failed: {e}")
            return False

    @staticmethod
    def calculate_rms(chunk: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) acoustic energy of an audio chunk."""
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

    def record_phrase(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        timeout: float = 15.0,
    ) -> Optional[np.ndarray]:
        """
        Listen on the microphone and record a single utterance using VAD.

        Returns:
            1D NumPy float32 array normalized in [-1.0, 1.0] at 16,000 Hz, or None if timed out.
        """
        if not self.is_available():
            self.logger.warning("Microphone is not available for recording.")
            return None

        sd = self._sounddevice
        buffer = []
        is_speech_started = False
        silence_start_time = None
        start_time = time.time()
        recording_start_time = None

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                device=self.device,
            ) as stream:
                while True:
                    # Check overall timeout before speech starts
                    elapsed = time.time() - start_time
                    if not is_speech_started and elapsed > timeout:
                        return None

                    # Read a chunk from the microphone
                    data, overflowed = stream.read(self.chunk_size)
                    chunk = data.flatten()
                    rms = self.calculate_rms(chunk)

                    if rms > self.energy_threshold:
                        if not is_speech_started:
                            is_speech_started = True
                            recording_start_time = time.time()
                            if on_speech_start:
                                on_speech_start()
                        buffer.append(chunk)
                        silence_start_time = None
                    elif is_speech_started:
                        # Append trailing silence buffer to avoid clipping word ends
                        buffer.append(chunk)
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif time.time() - silence_start_time > self.silence_duration:
                            # Conclude phrase recording
                            break

                        # Check maximum phrase duration
                        if recording_start_time and (time.time() - recording_start_time > self.max_duration):
                            break

            if buffer:
                audio_array = np.concatenate(buffer, axis=0).astype(np.float32)
                # Normalize amplitude if max absolute value > 0
                max_val = np.max(np.abs(audio_array))
                if max_val > 1.0:
                    audio_array = audio_array / max_val
                return audio_array
            return None

        except Exception as e:
            self.logger.error(f"Error recording from microphone: {e}")
            return None
