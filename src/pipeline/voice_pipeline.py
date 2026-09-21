"""Audio processing pipeline for batch and stream speech command recognition."""

from typing import Callable, Generator, List, Optional, Union
import numpy as np

from src.agents.voice.agent import VoiceAgent
from src.agents.voice.audio_utils import generate_synthetic_tone, load_audio_file
from src.agents.voice.schemas import AudioFormat, VoiceAgentInput, VoiceAgentOutput
from src.common.logger import get_logger
from src.utils.config import load_config


class VoicePipeline:
    """End-to-end speech processing pipeline connecting audio sources, VoiceAgent, and event callbacks."""

    def __init__(
        self,
        config: Optional[dict] = None,
        agent: Optional[VoiceAgent] = None,
        output_callback: Optional[Callable[[VoiceAgentOutput], None]] = None,
    ):
        self.config = config or load_config()
        self.logger = get_logger("VoicePipeline")
        self.agent = agent or VoiceAgent(config=self.config)
        self.output_callback = output_callback
        self._is_running = False

    def initialize(self) -> None:
        """Initialize pipeline and voice agent."""
        self.logger.info("Initializing Voice Pipeline...")
        self.agent.initialize()
        self.logger.info("Voice Pipeline ready.")

    def process_file(self, audio_path: str) -> VoiceAgentOutput:
        """Process a single audio file (.wav)."""
        if not self.agent.status.value == "ready":
            self.initialize()

        self.logger.info(f"Processing audio file: {audio_path}")
        agent_input = VoiceAgentInput(
            audio_data=audio_path,
            audio_format=AudioFormat.WAV_FILE,
        )
        output: VoiceAgentOutput = self.agent.process(agent_input)

        if self.output_callback:
            self.output_callback(output)

        return output

    def process_text_simulation(self, text_command: str) -> VoiceAgentOutput:
        """Process simulated text input as a speech command."""
        if not self.agent.status.value == "ready":
            self.initialize()

        agent_input = VoiceAgentInput(
            text_override=text_command,
            audio_format=AudioFormat.TEXT_SIMULATION,
        )
        output: VoiceAgentOutput = self.agent.process(agent_input)

        if self.output_callback:
            self.output_callback(output)

        return output

    def process_buffer(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> VoiceAgentOutput:
        """Process an in-memory audio buffer (NumPy float32 array)."""
        if not self.agent.status.value == "ready":
            self.initialize()

        agent_input = VoiceAgentInput(
            audio_data=audio_buffer,
            sample_rate=sample_rate,
            audio_format=AudioFormat.NUMPY_FLOAT32,
        )
        output: VoiceAgentOutput = self.agent.process(agent_input)

        if self.output_callback:
            self.output_callback(output)

        return output

    def run_stream(
        self,
        audio_stream_generator: Generator[np.ndarray, None, None],
        sample_rate: int = 16000,
        max_chunks: Optional[int] = None,
    ) -> List[VoiceAgentOutput]:
        """Process a continuous stream of audio chunks."""
        if not self.agent.status.value == "ready":
            self.initialize()

        self._is_running = True
        self.logger.info("Starting continuous audio stream processing...")
        results = []
        chunk_count = 0

        try:
            for chunk in audio_stream_generator:
                if not self._is_running:
                    break

                chunk_count += 1
                out = self.process_buffer(chunk, sample_rate=sample_rate)
                results.append(out)

                if max_chunks and chunk_count >= max_chunks:
                    self.logger.info(f"Reached max chunk count ({max_chunks}). Stopping stream.")
                    break
        except KeyboardInterrupt:
            self.logger.info("Audio stream interrupted by user.")
        finally:
            self.stop()

        return results

    def stop(self) -> None:
        """Stop running stream and shutdown agent."""
        self._is_running = False
        if self.agent:
            self.agent.shutdown()
        self.logger.info("Voice Pipeline stopped cleanly.")
