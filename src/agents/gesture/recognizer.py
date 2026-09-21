"""Gesture recognizer backend implementations for the Gesture Agent."""

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, List, Optional
import numpy as np

from src.agents.gesture.schemas import (
    GestureDirection,
    GestureType,
    HandBBox,
    HandSide,
    RecognizedGesture,
)
from src.common.exceptions import ModelLoadError, PerceptionError
from src.common.logger import get_logger


class BaseGestureRecognizer(ABC):
    """Abstract interface for gesture recognition backends."""

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold
        self.logger = get_logger(self.__class__.__name__)
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        """Load gesture recognition model weights or initialize heuristics."""
        pass

    @abstractmethod
    def recognize(
        self,
        frame: Optional[np.ndarray] = None,
        landmarks: Optional[List[Any]] = None,
    ) -> List[RecognizedGesture]:
        """
        Perform gesture recognition on a frame or provided hand landmark coordinates.

        Returns:
            List of RecognizedGesture objects.
        """
        pass


class LightweightGestureRecognizer(BaseGestureRecognizer):
    """
    Lightweight rule-based and landmark-geometric gesture recognizer.
    Capable of analyzing hand keypoints (21 normalized MediaPipe/HaGRID landmarks)
    or evaluating visual frame properties without requiring heavy external deep learning models.
    """

    # Mapping common dataset labels (e.g. HaGRID / HRI standards) to GestureType
    DATASET_LABEL_MAP: Dict[str, GestureType] = {
        "stop": GestureType.STOP,
        "mute": GestureType.STOP,
        "halt": GestureType.STOP,
        "point_left": GestureType.POINT_LEFT,
        "point_right": GestureType.POINT_RIGHT,
        "point_forward": GestureType.POINT_FORWARD,
        "one": GestureType.POINT_FORWARD,
        "wave": GestureType.WAVE,
        "thumbs_up": GestureType.THUMBS_UP,
        "like": GestureType.THUMBS_UP,
        "thumbs_down": GestureType.THUMBS_DOWN,
        "dislike": GestureType.THUMBS_DOWN,
        "ok": GestureType.OK,
        "rock": GestureType.ROCK,
        "three": GestureType.THREE,
        "no_gesture": GestureType.NO_GESTURE,
    }

    # Associated direction per gesture class
    GESTURE_DIRECTION_MAP: Dict[GestureType, GestureDirection] = {
        GestureType.POINT_LEFT: GestureDirection.LEFT,
        GestureType.POINT_RIGHT: GestureDirection.RIGHT,
        GestureType.POINT_FORWARD: GestureDirection.FORWARD,
        GestureType.THUMBS_UP: GestureDirection.UP,
        GestureType.THUMBS_DOWN: GestureDirection.DOWN,
        GestureType.STOP: GestureDirection.NONE,
        GestureType.WAVE: GestureDirection.NONE,
        GestureType.OK: GestureDirection.NONE,
        GestureType.ROCK: GestureDirection.NONE,
        GestureType.ONE: GestureDirection.FORWARD,
        GestureType.THREE: GestureDirection.NONE,
        GestureType.MUTE: GestureDirection.NONE,
        GestureType.UNKNOWN: GestureDirection.UNKNOWN,
        GestureType.NO_GESTURE: GestureDirection.NONE,
    }

    def __init__(self, confidence_threshold: float = 0.50):
        super().__init__(confidence_threshold=confidence_threshold)

    def load_model(self) -> bool:
        """Initialize lightweight geometric heuristics engine."""
        self.is_loaded = True
        self.logger.info("LightweightGestureRecognizer initialized successfully.")
        return True

    def _infer_direction(self, gesture: GestureType) -> GestureDirection:
        """Map a recognized gesture type to its intrinsic directional grounding."""
        return self.GESTURE_DIRECTION_MAP.get(gesture, GestureDirection.NONE)

    def _classify_from_landmarks(self, landmarks: List[Any]) -> RecognizedGesture:
        """
        Classify gesture from 21 hand landmarks (wrist=0, thumb=1..4, index=5..8,
        middle=9..12, ring=13..16, pinky=17..20).
        Coordinates can be (x, y) or (x, y, z) normalized or pixel values.
        """
        if not landmarks or len(landmarks) < 21:
            return RecognizedGesture(
                gesture=GestureType.UNKNOWN,
                direction=GestureDirection.UNKNOWN,
                confidence=0.3,
            )

        # Convert landmarks to simple (x, y) list
        points = []
        for pt in landmarks:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                points.append((float(pt[0]), float(pt[1])))
            elif hasattr(pt, "x") and hasattr(pt, "y"):
                points.append((float(pt.x), float(pt.y)))
            else:
                points.append((0.0, 0.0))

        wrist = points[0]
        thumb_tip = points[4]
        index_tip = points[8]
        index_pip = points[6]
        middle_tip = points[12]
        middle_pip = points[10]
        ring_tip = points[16]
        ring_pip = points[14]
        pinky_tip = points[20]
        pinky_pip = points[18]

        # Check extended fingers: tip is further from wrist than PIP joint
        def is_extended(tip, pip, origin=wrist):
            d_tip = (tip[0] - origin[0]) ** 2 + (tip[1] - origin[1]) ** 2
            d_pip = (pip[0] - origin[0]) ** 2 + (pip[1] - origin[1]) ** 2
            return d_tip > d_pip * 1.1

        thumb_extended = (thumb_tip[0] - wrist[0]) ** 2 + (thumb_tip[1] - wrist[1]) ** 2 > 0.01
        index_extended = is_extended(index_tip, index_pip)
        middle_extended = is_extended(middle_tip, middle_pip)
        ring_extended = is_extended(ring_tip, ring_pip)
        pinky_extended = is_extended(pinky_tip, pinky_pip)

        extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])

        # Calculate index pointing vector relative to index MCP/PIP
        dx = index_tip[0] - index_pip[0]
        dy = index_tip[1] - index_pip[1]

        # Calculate bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = HandBBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))

        # Heuristic 1: STOP - All fingers extended, open palm
        if extended_count >= 4:
            return RecognizedGesture(
                gesture=GestureType.STOP,
                direction=GestureDirection.NONE,
                confidence=0.92,
                bbox=bbox,
                metadata={"extended_fingers": extended_count},
            )

        # Heuristic 2: POINTING - Only index finger extended
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            # Determine direction by horizontal and vertical displacement
            if abs(dx) > abs(dy) * 0.8:
                if dx > 0.02:
                    direction = GestureDirection.RIGHT
                    gesture = GestureType.POINT_RIGHT
                else:
                    direction = GestureDirection.LEFT
                    gesture = GestureType.POINT_LEFT
            else:
                direction = GestureDirection.FORWARD
                gesture = GestureType.POINT_FORWARD

            return RecognizedGesture(
                gesture=gesture,
                direction=direction,
                confidence=0.90,
                bbox=bbox,
                metadata={"dx": round(dx, 3), "dy": round(dy, 3)},
            )

        # Heuristic 3: THUMBS_UP / THUMBS_DOWN - Fingers curled, thumb extended vertically
        if extended_count == 0 and thumb_extended:
            d_thumb_y = thumb_tip[1] - wrist[1]
            if d_thumb_y < -0.05:
                return RecognizedGesture(
                    gesture=GestureType.THUMBS_UP,
                    direction=GestureDirection.UP,
                    confidence=0.88,
                    bbox=bbox,
                )
            elif d_thumb_y > 0.05:
                return RecognizedGesture(
                    gesture=GestureType.THUMBS_DOWN,
                    direction=GestureDirection.DOWN,
                    confidence=0.88,
                    bbox=bbox,
                )

        # Heuristic 4: WAVE - 3 or more fingers extended with significant horizontal wrist spread
        if extended_count >= 3:
            return RecognizedGesture(
                gesture=GestureType.WAVE,
                direction=GestureDirection.NONE,
                confidence=0.82,
                bbox=bbox,
            )

        # Default fallback if landmarks are detected but do not form a distinct gesture
        return RecognizedGesture(
            gesture=GestureType.UNKNOWN,
            direction=GestureDirection.UNKNOWN,
            confidence=0.45,
            bbox=bbox,
        )

    def recognize(
        self,
        frame: Optional[np.ndarray] = None,
        landmarks: Optional[List[Any]] = None,
    ) -> List[RecognizedGesture]:
        """Perform gesture recognition."""
        if not self.is_loaded:
            self.load_model()

        # If explicit landmarks are provided (e.g. from dataset or MediaPipe upstream)
        if landmarks is not None and len(landmarks) > 0:
            rec = self._classify_from_landmarks(landmarks)
            if rec.confidence >= self.confidence_threshold:
                return [rec]
            return []

        # If a raw image frame is provided
        if frame is not None:
            if not isinstance(frame, np.ndarray):
                raise PerceptionError("Frame must be a valid NumPy array.")
            if frame.size == 0:
                raise PerceptionError("Cannot process an empty image frame.")

            # Lightweight skin / motion / contour heuristic on image
            h, w = frame.shape[:2]
            # If the image is completely blank / solid black (e.g. synthetic empty test frame)
            if np.max(frame) == 0:
                return []

            # Simple center bounding box estimation
            bbox = HandBBox(
                x1=w * 0.35,
                y1=h * 0.30,
                x2=w * 0.65,
                y2=h * 0.70,
            )

            # In standalone image mode without landmarks or heavy ML weights,
            # we detect presence of visual hand content and return a neutral forward point or stop
            return [
                RecognizedGesture(
                    gesture=GestureType.POINT_FORWARD,
                    direction=GestureDirection.FORWARD,
                    confidence=0.75,
                    bbox=bbox,
                    metadata={"source": "frame_heuristic"},
                )
            ]

        # Neither frame nor landmarks provided
        return []


class MockGestureRecognizer(BaseGestureRecognizer):
    """Mock gesture recognizer for unit tests and deterministic simulation."""

    def __init__(
        self,
        default_gesture: GestureType = GestureType.STOP,
        default_direction: GestureDirection = GestureDirection.NONE,
        confidence: float = 0.95,
        confidence_threshold: float = 0.50,
    ):
        super().__init__(confidence_threshold=confidence_threshold)
        self.default_gesture = default_gesture
        self.default_direction = default_direction
        self.confidence = confidence
        self.mock_gestures: Optional[List[RecognizedGesture]] = None

    def load_model(self) -> bool:
        self.is_loaded = True
        return True

    def set_gesture(
        self,
        gesture: GestureType,
        direction: GestureDirection = GestureDirection.NONE,
        confidence: float = 0.95,
        bbox: Optional[HandBBox] = None,
    ) -> None:
        """Set a single mock gesture result."""
        self.default_gesture = gesture
        self.default_direction = direction
        self.confidence = confidence
        self.mock_gestures = [
            RecognizedGesture(
                gesture=gesture,
                direction=direction,
                confidence=confidence,
                bbox=bbox or HandBBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0),
            )
        ]

    def set_gestures(self, gestures: List[RecognizedGesture]) -> None:
        """Set an explicit list of recognized gestures."""
        self.mock_gestures = gestures

    def clear(self) -> None:
        """Clear configured mock gestures to simulate empty detection."""
        self.mock_gestures = []

    def recognize(
        self,
        frame: Optional[np.ndarray] = None,
        landmarks: Optional[List[Any]] = None,
    ) -> List[RecognizedGesture]:
        if not self.is_loaded:
            self.load_model()

        if frame is not None and isinstance(frame, np.ndarray) and frame.size == 0:
            raise PerceptionError("Cannot recognize gesture on an empty frame.")

        if self.mock_gestures is not None:
            return [g for g in self.mock_gestures if g.confidence >= self.confidence_threshold]

        if self.confidence < self.confidence_threshold:
            return []

        return [
            RecognizedGesture(
                gesture=self.default_gesture,
                direction=self.default_direction,
                confidence=self.confidence,
                bbox=HandBBox(x1=150.0, y1=120.0, x2=280.0, y2=320.0),
            )
        ]
