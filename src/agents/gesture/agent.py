"""Gesture Agent: processes visual frames and hand landmarks to produce structured gesture perception."""

from typing import Any, Dict, List, Optional
import numpy as np

from src.agents.base import BaseAgent
from src.agents.gesture.recognizer import (
    BaseGestureRecognizer,
    HagridGestureRecognizer,
    LightweightGestureRecognizer,
    MockGestureRecognizer,
)
from src.agents.gesture.schemas import (
    GestureAgentInput,
    GestureAgentOutput,
    GestureDirection,
    GestureType,
    RecognizedGesture,
)
from src.common.exceptions import PerceptionError
from src.common.schemas import AgentType


class GestureAgent(BaseAgent):
    """
    Gesture perception agent for Human-Robot Interaction.

    Responsibilities:
    - Receive camera frames or hand landmark data
    - Detect and classify human gestures (STOP, POINT, WAVE, THUMBS_UP, etc.)
    - Ground pointing directions (LEFT, RIGHT, FORWARD, UP, DOWN)
    - Produce structured, timestamped gesture output for the Coordinator

    The Gesture Agent does NOT decide robot actions.
    """

    # Direction map for quick lookups
    DIRECTION_MAP: Dict[str, str] = {
        GestureType.POINT_LEFT.value: GestureDirection.LEFT.value,
        GestureType.POINT_RIGHT.value: GestureDirection.RIGHT.value,
        GestureType.POINT_FORWARD.value: GestureDirection.FORWARD.value,
        GestureType.THUMBS_UP.value: GestureDirection.UP.value,
        GestureType.THUMBS_DOWN.value: GestureDirection.DOWN.value,
        GestureType.STOP.value: GestureDirection.NONE.value,
        GestureType.WAVE.value: GestureDirection.NONE.value,
        GestureType.OK.value: GestureDirection.NONE.value,
        GestureType.ROCK.value: GestureDirection.NONE.value,
        GestureType.ONE.value: GestureDirection.FORWARD.value,
        GestureType.THREE.value: GestureDirection.NONE.value,
        GestureType.MUTE.value: GestureDirection.NONE.value,
        GestureType.UNKNOWN.value: GestureDirection.UNKNOWN.value,
        GestureType.NO_GESTURE.value: GestureDirection.NONE.value,
    }

    def __init__(
        self,
        name: str = "GestureAgent",
        recognizer: Optional[BaseGestureRecognizer] = None,
        config: Optional[dict] = None,
    ):
        super().__init__(
            name=name,
            agent_type=AgentType.GESTURE,
            version="0.1.0",
            description="Perceives human hand gestures and pointing directions for HRI tasks.",
            config=config or {},
        )

        self.recognizer = recognizer
        gesture_cfg = self.config.get("gesture", {})
        self._confidence_threshold = gesture_cfg.get("confidence_threshold", 0.50)

    def _initialize(self) -> bool:
        """Initialize the configured gesture recognizer backend."""
        if self.recognizer is None:
            gesture_cfg = self.config.get("gesture", {})
            backend = gesture_cfg.get("backend", "hagrid").lower()
            conf_thresh = gesture_cfg.get("confidence_threshold", self._confidence_threshold)

            if backend == "mock":
                self.recognizer = MockGestureRecognizer(
                    confidence_threshold=conf_thresh
                )
            elif backend == "lightweight":
                self.recognizer = LightweightGestureRecognizer(
                    confidence_threshold=conf_thresh
                )
            elif backend == "hagrid":
                model_path = gesture_cfg.get("model_path", "models/hagrid/yolov10n_hagrid.pt")
                device = gesture_cfg.get("device", "cpu")
                self.recognizer = HagridGestureRecognizer(
                    model_path=model_path,
                    confidence_threshold=conf_thresh,
                    device=device,
                )
            else:
                self.logger.warning(
                    f"Unknown gesture backend '{backend}'. Falling back to LightweightGestureRecognizer."
                )
                self.recognizer = LightweightGestureRecognizer(
                    confidence_threshold=conf_thresh
                )

        return self.recognizer.load_model()

    def _process(self, input_data: Any) -> GestureAgentOutput:
        """Process input frame, landmarks, or simulation override and generate structured gesture context."""
        frame = None
        frame_id = 0
        gesture_override = None
        direction_override = None
        landmarks = None

        # ---------------------------------------------------------
        # 1. Parse and validate input data
        # ---------------------------------------------------------
        if isinstance(input_data, GestureAgentInput):
            frame = input_data.frame
            frame_id = input_data.frame_id
            gesture_override = input_data.gesture_override
            direction_override = input_data.direction_override
            landmarks = input_data.landmarks

        elif isinstance(input_data, np.ndarray):
            frame = input_data
            frame_id = 0

        elif isinstance(input_data, dict):
            frame = input_data.get("frame")
            frame_id = input_data.get("frame_id", 0)
            gesture_override = input_data.get("gesture_override")
            direction_override = input_data.get("direction_override")
            landmarks = input_data.get("landmarks")

        elif isinstance(input_data, str):
            gesture_override = input_data

        else:
            raise PerceptionError(
                f"Unsupported input type for GestureAgent: {type(input_data)}. "
                "Expected GestureAgentInput, np.ndarray, dict, or str."
            )

        # Validate empty/corrupt frames
        if frame is not None:
            if not isinstance(frame, np.ndarray):
                raise PerceptionError("GestureAgent expects image frames as NumPy arrays.")
            if frame.size == 0:
                raise PerceptionError("Received an empty image frame in GestureAgent.")

        if self.recognizer is None:
            raise PerceptionError("Gesture recognizer is not initialized.")

        # ---------------------------------------------------------
        # 2. Handle simulation override (for fast testing / mocking)
        # ---------------------------------------------------------
        if gesture_override is not None:
            g_str = gesture_override.strip().upper()

            # Map common names
            try:
                g_type = GestureType(g_str)
            except ValueError:
                g_type = GestureType.UNKNOWN

            if direction_override is not None:
                try:
                    d_type = GestureDirection(direction_override.strip().upper())
                except ValueError:
                    d_type = GestureDirection.UNKNOWN
            else:
                d_type = GestureDirection(self.DIRECTION_MAP.get(g_type.value, GestureDirection.NONE.value))

            recognized = [
                RecognizedGesture(
                    gesture=g_type,
                    direction=d_type,
                    confidence=0.95,
                    metadata={"source": "simulation_override"},
                )
            ]
        else:
            # ---------------------------------------------------------
            # 3. Run gesture recognition backend
            # ---------------------------------------------------------
            recognized = self.recognizer.recognize(frame=frame, landmarks=landmarks)

        # ---------------------------------------------------------
        # 4. Construct structured perception output
        # ---------------------------------------------------------
        if recognized:
            # Sort by confidence descending
            recognized.sort(key=lambda g: g.confidence, reverse=True)
            top = recognized[0]

            primary_gesture = top.gesture.value if isinstance(top.gesture, GestureType) else str(top.gesture)
            primary_direction = top.direction.value if isinstance(top.direction, GestureDirection) else str(top.direction)
            primary_conf = float(top.confidence)
            is_detected = primary_gesture not in (GestureType.NO_GESTURE.value, GestureType.UNKNOWN.value)

            summary_text = (
                f"Detected gesture '{primary_gesture}' "
                f"with direction '{primary_direction}' "
                f"(conf={primary_conf:.2f})."
            )
        else:
            primary_gesture = GestureType.NO_GESTURE.value
            primary_direction = GestureDirection.NONE.value
            primary_conf = 0.0
            is_detected = False
            summary_text = "No human gesture detected in scene."

        return GestureAgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type.value,
            success=True,
            confidence=primary_conf if is_detected else 1.0,
            gesture=primary_gesture,
            direction=primary_direction,
            is_gesture_detected=is_detected,
            recognized_gestures=recognized,
            frame_id=frame_id,
            summary_text=summary_text,
        )

    def _reset(self) -> None:
        """Reset Gesture Agent internal state."""
        pass

    def _shutdown(self) -> None:
        """Release Gesture Agent resources."""
        self.recognizer = None
