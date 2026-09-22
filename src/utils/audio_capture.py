"""Live microphone capture with Voice Activity Detection (VAD) for the HRI Voice Agent."""

import math
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
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
        energy_threshold: float = 0.015,
        silence_duration: float = 0.8,
        max_duration: float = 6.0,
        min_duration: float = 0.4,
        device: Optional[Union[int, str]] = None,
        debug: bool = False,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.device = device
        self.debug = debug
        self.last_stop_reason = "none"
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

    def get_input_device_info(self) -> Dict[str, Any]:
        """Get details about the active PortAudio input device."""
        if not self.is_available():
            return {
                "index": None,
                "name": "None (Unavailable)",
                "channels": 0,
                "sample_rate": self.sample_rate,
            }
        try:
            sd = self._sounddevice
            dev_idx = (
                self.device
                if self.device is not None
                else (sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device)
            )
            if dev_idx is None or dev_idx < 0:
                dev_info = sd.query_devices(kind="input")
            else:
                dev_info = sd.query_devices(dev_idx)
            return {
                "index": dev_info.get("index", dev_idx),
                "name": dev_info.get("name", "Default Microphone"),
                "channels": dev_info.get("max_input_channels", 1),
                "sample_rate": dev_info.get("default_samplerate", self.sample_rate),
            }
        except Exception:
            return {
                "index": 0,
                "name": "Default System Microphone",
                "channels": 1,
                "sample_rate": self.sample_rate,
            }

    def get_input_device_name(self) -> str:
        """Get the human-readable name of the active audio input device."""
        info = self.get_input_device_info()
        return info.get("name", "Default Microphone")

    @staticmethod
    def calculate_rms(chunk: np.ndarray, remove_dc: bool = False) -> float:
        """
        Calculate Root Mean Square (RMS) acoustic energy of an audio chunk.
        Optionally removes DC bias before computing RMS.
        """
        if chunk.size == 0:
            return 0.0
        audio = chunk.astype(np.float32)
        if remove_dc:
            audio = audio - np.mean(audio)
        return float(np.sqrt(np.mean(audio ** 2)))

    def record_phrase(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        timeout: float = 8.0,
        show_debug: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """
        Listen on the microphone and record a single utterance using VAD.
        Terminates cleanly on silence or when max_duration is reached.

        Returns:
            1D NumPy float32 array normalized in [-1.0, 1.0] at 16,000 Hz, or None if timed out.
        """
        if not self.is_available():
            self.logger.warning("Microphone is not available for recording.")
            self.last_stop_reason = "device unavailable"
            return None

        debug_active = self.debug if show_debug is None else show_debug
        sd = self._sounddevice
        buffer = []
        pre_roll = []  # Ring buffer storing last 2 chunks (~200ms) before speech trigger
        is_speech_started = False
        silence_start_time = None
        start_time = time.time()
        recording_start_time = None
        last_debug_print_time = 0.0
        self.last_stop_reason = "timeout"

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                device=self.device,
            ) as stream:
                while True:
                    now = time.time()

                    # 1. Check pre-speech initial silence timeout
                    if not is_speech_started and (now - start_time > timeout):
                        self.last_stop_reason = "initial silence timeout"
                        return None

                    # 2. Read chunk from microphone
                    data, overflowed = stream.read(self.chunk_size)
                    chunk = data.flatten()
                    # Calculate RMS with DC offset removed for accurate speech/silence detection
                    rms = self.calculate_rms(chunk, remove_dc=True)

                    # 3. Voice Activity Detection logic
                    if rms > self.energy_threshold:
                        if not is_speech_started:
                            is_speech_started = True
                            recording_start_time = now
                            if on_speech_start:
                                on_speech_start()
                            # Prepend pre-roll buffer to preserve initial consonant
                            buffer.extend(pre_roll)
                        buffer.append(chunk)
                        silence_start_time = None
                        silence_elapsed = 0.0
                        speech_label = "YES"
                    else:
                        if is_speech_started:
                            buffer.append(chunk)
                            if silence_start_time is None:
                                silence_start_time = now
                            silence_elapsed = now - silence_start_time
                            speech_label = "NO"
                        else:
                            # Keep sliding 200ms pre-roll buffer
                            pre_roll.append(chunk)
                            if len(pre_roll) > 2:
                                pre_roll.pop(0)
                            silence_elapsed = 0.0
                            speech_label = "NO"

                    # 4. Periodic audio debug output (approx 3 times per second)
                    if is_speech_started and (now - last_debug_print_time >= 0.3):
                        last_debug_print_time = now
                        if debug_active:
                            print(f"\n[AUDIO DEBUG]")
                            print(f"RMS = {rms:.4f}")
                            print(f"speech = {speech_label}")
                            print(f"silence_time = {silence_elapsed:.1f} s")

                    # 5. Check termination conditions (ALWAYS evaluated)
                    if is_speech_started:
                        # Condition A: Silence duration reached after speech
                        if silence_start_time is not None and (now - silence_start_time >= self.silence_duration):
                            self.last_stop_reason = "silence"
                            break

                        # Condition B: Safe maximum recording duration reached
                        if recording_start_time is not None and (now - recording_start_time >= self.max_duration):
                            self.last_stop_reason = "maximum duration reached"
                            break

            if buffer:
                audio_array = np.concatenate(buffer, axis=0).astype(np.float32)
                # Remove DC bias from the concatenated phrase
                audio_array = audio_array - np.mean(audio_array)
                max_val = np.max(np.abs(audio_array))
                if max_val > 1.0:
                    audio_array = audio_array / max_val
                return audio_array
            return None

        except Exception as e:
            self.logger.error(f"Error recording from microphone: {e}")
            self.last_stop_reason = f"error: {e}"
            return None
