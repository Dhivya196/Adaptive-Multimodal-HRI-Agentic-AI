"""Gesture-specific schemas and data representations for gesture perception."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.common.schemas import BaseAgentInput, BaseAgentOutput


class GestureType(str, Enum):
    """Supported gesture classes for the HRI perception layer."""
    STOP = "STOP"
    POINT_LEFT = "POINT_LEFT"
    POINT_RIGHT = "POINT_RIGHT"
    POINT_FORWARD = "POINT_FORWARD"
    WAVE = "WAVE"
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DOWN = "THUMBS_DOWN"
    OK = "OK"
    ROCK = "ROCK"
    ONE = "ONE"
    THREE = "THREE"
    MUTE = "MUTE"
    UNKNOWN = "UNKNOWN"
    NO_GESTURE = "NO_GESTURE"


class GestureDirection(str, Enum):
    """Directional grounding associated with pointing or hand gestures."""
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FORWARD = "FORWARD"
    UP = "UP"
    DOWN = "DOWN"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class HandSide(str, Enum):
    """Hand laterality (left or right hand)."""
    LEFT = "left"
    RIGHT = "right"
    ANY = "any"


@dataclass
class HandBBox:
    """Bounding box for a detected hand in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "center": {"x": round(self.center[0], 1), "y": round(self.center[1], 1)},
        }


@dataclass
class RecognizedGesture:
    """Structured representation of an individual recognized hand gesture."""
    gesture: GestureType = GestureType.UNKNOWN
    direction: GestureDirection = GestureDirection.NONE
    confidence: float = 0.0
    bbox: Optional[HandBBox] = None
    hand_side: HandSide = HandSide.ANY
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gesture": self.gesture.value if isinstance(self.gesture, GestureType) else str(self.gesture),
            "direction": self.direction.value if isinstance(self.direction, GestureDirection) else str(self.direction),
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "hand_side": self.hand_side.value if isinstance(self.hand_side, HandSide) else str(self.hand_side),
            "metadata": self.metadata,
        }


@dataclass
class GestureAgentInput(BaseAgentInput):
    """Input payload passed to the Gesture Agent."""
    frame: Any = None
    frame_id: int = 0
    gesture_override: Optional[str] = None
    direction_override: Optional[str] = None
    landmarks: Optional[List[Any]] = None
    source_name: str = "camera"


@dataclass
class GestureAgentOutput(BaseAgentOutput):
    """Structured gesture perception context emitted by the Gesture Agent for the Coordinator."""
    gesture: str = GestureType.NO_GESTURE.value
    direction: str = GestureDirection.NONE.value
    confidence: float = 0.0
    is_gesture_detected: bool = False
    recognized_gestures: List[RecognizedGesture] = field(default_factory=list)
    frame_id: int = 0
    summary_text: str = ""

    @property
    def summary(self) -> str:
        """Alias for summary_text to maintain full schema compliance."""
        return self.summary_text

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "gesture": self.gesture,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "is_gesture_detected": self.is_gesture_detected,
            "recognized_gestures": [g.to_dict() for g in self.recognized_gestures],
            "frame_id": self.frame_id,
            "summary_text": self.summary_text,
            "summary": self.summary_text,
        })
        return base_dict
