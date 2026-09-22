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

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        model_asset_path: Optional[str] = "models/mediapipe/hand_landmarker.task",
    ):
        super().__init__(confidence_threshold=confidence_threshold)
        self.model_asset_path = model_asset_path
        self._landmarker = None

    def load_model(self) -> bool:
        """Initialize MediaPipe hand landmarker and geometric heuristics engine."""
        self.is_loaded = True
        try:
            import urllib.request
            from pathlib import Path
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options

            path = Path(self.model_asset_path) if self.model_asset_path else None
            if path and not path.is_absolute():
                project_root = Path(__file__).resolve().parent.parent.parent.parent
                path = project_root / self.model_asset_path

            if path:
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Downloading official MediaPipe hand landmarker task asset to '{path}'...")
                    asset_url = (
                        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                        "hand_landmarker/float16/1/hand_landmarker.task"
                    )
                    urllib.request.urlretrieve(asset_url, str(path))
                    self.logger.info("MediaPipe hand landmarker task asset downloaded successfully.")

                options = vision.HandLandmarkerOptions(
                    base_options=base_options.BaseOptions(model_asset_path=str(path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.4,
                    min_hand_presence_confidence=0.4,
                )
                self._landmarker = vision.HandLandmarker.create_from_options(options)
                self.logger.info("MediaPipe HandLandmarker detector initialized successfully.")
        except Exception as e:
            self.logger.warning(
                f"MediaPipe HandLandmarker could not be initialized ({e}). "
                "LightweightGestureRecognizer will operate in landmark-only mode."
            )
            self._landmarker = None

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
        if landmarks is None or len(landmarks) < 21:
            return RecognizedGesture(
                gesture=GestureType.UNKNOWN,
                direction=GestureDirection.UNKNOWN,
                confidence=0.3,
            )

        # Convert landmarks to simple (x, y) list
        points = []
        for pt in landmarks:
            if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
                points.append((float(pt[0]), float(pt[1])))
            elif hasattr(pt, "x") and hasattr(pt, "y"):
                points.append((float(pt.x), float(pt.y)))
            else:
                points.append((0.0, 0.0))

        wrist = points[0]
        thumb_tip = points[4]
        index_mcp = points[5]
        index_pip = points[6]
        index_tip = points[8]
        middle_mcp = points[9]
        middle_pip = points[10]
        middle_tip = points[12]
        ring_mcp = points[13]
        ring_pip = points[14]
        ring_tip = points[16]
        pinky_mcp = points[17]
        pinky_pip = points[18]
        pinky_tip = points[20]

        # Robust finger extension check: compare tip distance to both MCP knuckle and wrist
        def is_extended(tip, pip, mcp, origin=wrist):
            d_tip_mcp = (tip[0] - mcp[0]) ** 2 + (tip[1] - mcp[1]) ** 2
            d_pip_mcp = (pip[0] - mcp[0]) ** 2 + (pip[1] - mcp[1]) ** 2
            d_tip_wrist = (tip[0] - origin[0]) ** 2 + (tip[1] - origin[1]) ** 2
            d_pip_wrist = (pip[0] - origin[0]) ** 2 + (pip[1] - origin[1]) ** 2
            return (d_tip_mcp > d_pip_mcp * 1.05) and (d_tip_wrist > d_pip_wrist * 1.02)

        thumb_extended = (thumb_tip[0] - wrist[0]) ** 2 + (thumb_tip[1] - wrist[1]) ** 2 > 0.01
        index_extended = is_extended(index_tip, index_pip, index_mcp)
        middle_extended = is_extended(middle_tip, middle_pip, middle_mcp)
        ring_extended = is_extended(ring_tip, ring_pip, ring_mcp)
        pinky_extended = is_extended(pinky_tip, pinky_pip, pinky_mcp)

        extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])

        # Calculate index pointing vector relative to index MCP
        dx = index_tip[0] - index_mcp[0]
        dy = index_tip[1] - index_mcp[1]

        # Calculate bounding box in landmark space
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = HandBBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))

        debug_info = {
            "wrist": (round(wrist[0], 3), round(wrist[1], 3)),
            "index_extended": index_extended,
            "middle_extended": middle_extended,
            "ring_extended": ring_extended,
            "pinky_extended": pinky_extended,
            "thumb_extended": thumb_extended,
            "extended_count": extended_count,
            "dx": round(dx, 3),
            "dy": round(dy, 3),
        }

        # -------------------------------------------------------------
        # PRIORITY 1: ROCK GESTURE (Index & pinky extended, middle & ring curled)
        # -------------------------------------------------------------
        if index_extended and pinky_extended and not middle_extended and not ring_extended:
            return RecognizedGesture(
                gesture=GestureType.ROCK,
                direction=GestureDirection.NONE,
                confidence=0.88,
                bbox=bbox,
                metadata={"debug": debug_info},
            )

        # -------------------------------------------------------------
        # PRIORITY 2: POINTING (Index extended, middle, ring, pinky curled)
        # -------------------------------------------------------------
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            if abs(dx) > abs(dy) * 0.5:
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
                confidence=0.92,
                bbox=bbox,
                metadata={"dx": round(dx, 3), "dy": round(dy, 3), "debug": debug_info},
            )

        # -------------------------------------------------------------
        # PRIORITY 3: OK GESTURE (Thumb tip touches index tip, others extended)
        # -------------------------------------------------------------
        d_thumb_index = (thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2
        if d_thumb_index < 0.005 and middle_extended and ring_extended and pinky_extended:
            return RecognizedGesture(
                gesture=GestureType.OK,
                direction=GestureDirection.NONE,
                confidence=0.89,
                bbox=bbox,
                metadata={"debug": debug_info},
            )

        # -------------------------------------------------------------
        # PRIORITY 4: THUMBS_UP / THUMBS_DOWN (All 4 fingers curled, thumb extended)
        # -------------------------------------------------------------
        if extended_count == 0 and thumb_extended:
            d_thumb_y = thumb_tip[1] - wrist[1]
            if d_thumb_y < -0.05:
                return RecognizedGesture(
                    gesture=GestureType.THUMBS_UP,
                    direction=GestureDirection.UP,
                    confidence=0.88,
                    bbox=bbox,
                    metadata={"debug": debug_info},
                )
            elif d_thumb_y > 0.05:
                return RecognizedGesture(
                    gesture=GestureType.THUMBS_DOWN,
                    direction=GestureDirection.DOWN,
                    confidence=0.88,
                    bbox=bbox,
                    metadata={"debug": debug_info},
                )

        # -------------------------------------------------------------
        # PRIORITY 5: STOP (Open palm with all 4 fingers extended)
        # -------------------------------------------------------------
        if extended_count >= 4:
            return RecognizedGesture(
                gesture=GestureType.STOP,
                direction=GestureDirection.NONE,
                confidence=0.92,
                bbox=bbox,
                metadata={"extended_fingers": extended_count, "debug": debug_info},
            )

        # -------------------------------------------------------------
        # PRIORITY 6: WAVE (3 or more fingers extended)
        # -------------------------------------------------------------
        if extended_count >= 3:
            return RecognizedGesture(
                gesture=GestureType.WAVE,
                direction=GestureDirection.NONE,
                confidence=0.82,
                bbox=bbox,
                metadata={"debug": debug_info},
            )

        # Fallback if landmarks detected but no specific gesture template matched
        return RecognizedGesture(
            gesture=GestureType.UNKNOWN,
            direction=GestureDirection.UNKNOWN,
            confidence=0.45,
            bbox=bbox,
            metadata={"debug": debug_info},
        )

    def recognize(
        self,
        frame: Optional[np.ndarray] = None,
        landmarks: Optional[List[Any]] = None,
    ) -> List[RecognizedGesture]:
        """
        Perform gesture recognition.
        
        If landmarks are explicitly passed, evaluates landmark geometry directly.
        If an image frame is passed, runs MediaPipe HandLandmarker to extract 21 keypoints
        per detected hand and classifies each hand's gesture.
        """
        if not self.is_loaded:
            self.load_model()

        # 1. Explicit landmarks provided
        if landmarks is not None and len(landmarks) > 0:
            rec = self._classify_from_landmarks(landmarks)
            if rec.confidence >= self.confidence_threshold:
                return [rec]
            return []

        # 2. Raw image frame provided
        if frame is not None:
            if not isinstance(frame, np.ndarray):
                raise PerceptionError("Frame must be a valid NumPy array.")
            if frame.size == 0:
                raise PerceptionError("Cannot process an empty image frame.")

            # If the image is completely blank / solid black, return no detections
            if np.max(frame) == 0:
                return []

            h, w = frame.shape[:2]

            # If MediaPipe HandLandmarker is available, process the real image frame
            if self._landmarker is not None:
                try:
                    import cv2
                    import mediapipe as mp

                    # Ensure RGB format for MediaPipe Image
                    if len(frame.shape) == 2:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    elif frame.shape[2] == 4:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    elif frame.shape[2] == 3:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_frame = frame

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    detection_result = self._landmarker.detect(mp_image)

                    recognized: List[RecognizedGesture] = []
                    if detection_result and detection_result.hand_landmarks:
                        total_detected_hands = len(detection_result.hand_landmarks)
                        for idx, hand_lms in enumerate(detection_result.hand_landmarks):
                            rec = self._classify_from_landmarks(hand_lms)
                            
                            # Convert normalized landmark bounding box to pixel coordinates
                            xs = [pt.x * w if hasattr(pt, "x") else float(pt[0]) * w for pt in hand_lms]
                            ys = [pt.y * h if hasattr(pt, "y") else float(pt[1]) * h for pt in hand_lms]
                            rec.bbox = HandBBox(
                                x1=max(0.0, min(xs)),
                                y1=max(0.0, min(ys)),
                                x2=min(float(w), max(xs)),
                                y2=min(float(h), max(ys)),
                            )
                            rec.metadata["source"] = "mediapipe_hand_landmarker"
                            rec.metadata["hand_count"] = total_detected_hands
                            rec.metadata["landmarks"] = [
                                (
                                    float(pt.x) if hasattr(pt, "x") else float(pt[0]),
                                    float(pt.y) if hasattr(pt, "y") else float(pt[1]),
                                    float(getattr(pt, "z", 0.0)) if hasattr(pt, "z") else (float(pt[2]) if len(pt) > 2 else 0.0),
                                )
                                for pt in hand_lms
                            ]

                            # Handedness extraction if available
                            if (
                                getattr(detection_result, "handedness", None)
                                and idx < len(detection_result.handedness)
                                and detection_result.handedness[idx]
                            ):
                                side_name = detection_result.handedness[idx][0].category_name.lower()
                                rec.hand_side = (
                                    HandSide.LEFT
                                    if "left" in side_name
                                    else (HandSide.RIGHT if "right" in side_name else HandSide.ANY)
                                )
                                rec.metadata["handedness_score"] = round(
                                    float(detection_result.handedness[idx][0].score), 3
                                )

                            if rec.confidence >= self.confidence_threshold:
                                recognized.append(rec)

                        # Sort recognized gestures by confidence descending
                        recognized.sort(key=lambda g: g.confidence, reverse=True)
                        return recognized

                except Exception as e:
                    self.logger.debug(f"MediaPipe detection failed on frame: {e}")

            # If MediaPipe didn't detect any hands or is not available
            return []

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


# Export HagridGestureRecognizer for unified backend access
from src.agents.gesture.hagrid_gesture_recognizer import HagridGestureRecognizer  # noqa: E402

__all__ = [
    "BaseGestureRecognizer",
    "LightweightGestureRecognizer",
    "MockGestureRecognizer",
    "HagridGestureRecognizer",
]

