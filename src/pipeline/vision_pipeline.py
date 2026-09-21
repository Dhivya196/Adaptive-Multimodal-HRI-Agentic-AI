"""Live camera stream processing loop integrating capture, VisionAgent, and visualizer."""

from typing import Callable, Optional

from src.agents.vision.agent import VisionAgent
from src.agents.vision.schemas import VisionAgentInput, VisionAgentOutput
from src.agents.vision.visualizer import VisionVisualizer
from src.common.logger import get_logger
from src.utils.camera import CameraManager
from src.utils.config import load_config


class VisionPipeline:
    """End-to-end perception loop connecting video capture, VisionAgent, and visual display."""

    def __init__(
        self,
        config: Optional[dict] = None,
        agent: Optional[VisionAgent] = None,
        camera: Optional[CameraManager] = None,
        visualizer: Optional[VisionVisualizer] = None,
        output_callback: Optional[Callable[[VisionAgentOutput], None]] = None,
    ):
        self.config = config or load_config()
        self.logger = get_logger("VisionPipeline")
        self.agent = agent or VisionAgent(config=self.config)
        self.camera = camera
        self.visualizer = visualizer
        self.output_callback = output_callback
        self._is_running = False

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        self.logger.info("Initializing Vision Pipeline...")

        self.agent.initialize()

        if self.camera is None:
            cam_cfg = self.config.get("camera", {})
            self.camera = CameraManager(
                source=cam_cfg.get("device_id", 0),
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480),
                fps=cam_cfg.get("fps", 30),
            )

        if not self.camera._is_synthetic and (self.camera.cap is None or not self.camera.cap.isOpened()):
            if not self.camera.open():
                self.logger.warning("No physical webcam accessible. Falling back to synthetic test stream.")
                self.camera.enable_synthetic_mode()

        viz_cfg = self.config.get("visualizer", {})
        if self.visualizer is None and viz_cfg.get("enable_display", True):
            spatial_cfg = self.config.get("spatial", {})
            self.visualizer = VisionVisualizer(
                window_name=viz_cfg.get("window_name", "HRI Multimodal Perception - Vision Agent"),
                show_spatial_sectors=viz_cfg.get("show_spatial_sectors", True),
                show_labels=viz_cfg.get("show_labels", True),
                show_summary_banner=viz_cfg.get("show_summary_banner", True),
                sector_split=spatial_cfg.get("sector_split", [0.33, 0.67]),
            )

        self.logger.info("Vision Pipeline ready.")

    def run(self, max_frames: Optional[int] = None) -> None:
        """Execute continuous perception loop."""
        if not self.agent.status.value == "ready":
            self.initialize()

        self._is_running = True
        self.logger.info("Starting Vision Pipeline perception loop (press 'q' to stop)...")

        processed_count = 0
        try:
            while self._is_running:
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    self.logger.info("End of video stream or camera disconnected.")
                    break

                processed_count += 1

                agent_input = VisionAgentInput(
                    frame=frame,
                    frame_id=processed_count,
                    source_name=str(self.camera.source),
                )

                output: VisionAgentOutput = self.agent.process(agent_input)

                if self.output_callback:
                    self.output_callback(output)

                if self.visualizer:
                    annotated_frame = self.visualizer.render(frame, output)
                    key = self.visualizer.show(annotated_frame)
                    if key == ord("q") or key == 27:
                        self.logger.info("User requested termination.")
                        break

                if max_frames and processed_count >= max_frames:
                    self.logger.info(f"Reached maximum frame limit ({max_frames}).")
                    break

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received. Stopping pipeline...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully release camera, visualizer, and agent resources."""
        self._is_running = False
        if self.camera:
            self.camera.release()
        if self.visualizer:
            self.visualizer.close()
        if self.agent:
            self.agent.shutdown()
        self.logger.info("Vision Pipeline stopped cleanly.")
