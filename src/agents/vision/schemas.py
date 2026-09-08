"""Vision-specific schemas and data representations for object detection and spatial grounding."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from src.common.schemas import BaseAgentInput, BaseAgentOutput


class SpatialSector(str, Enum):
    """Horizontal spatial sector of detected object relative to robot camera."""
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"
    UNKNOWN = "UNKNOWN"


class ProximityLevel(str, Enum):
    """Estimated visual proximity level based on bounding box size / area ratio."""
    NEAR = "NEAR"
    MEDIUM = "MEDIUM"
    FAR = "FAR"


@dataclass
class Point2D:
    """2D pixel coordinate point."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))


@dataclass
class BoundingBox:
    """2D Axis-aligned bounding box."""
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
    def center(self) -> Point2D:
        return Point2D(x=(self.x1 + self.x2) / 2.0, y=(self.y1 + self.y2) / 2.0)

    def to_xyxy_int(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "center": {"x": round(self.center.x, 1), "y": round(self.center.y, 1)},
        }


@dataclass
class DetectedObject:
    """Structured representation of an object detected in the visual scene."""
    object_id: int
    label: str
    confidence: float
    bbox: BoundingBox
    spatial_sector: SpatialSector = SpatialSector.UNKNOWN
    proximity: ProximityLevel = ProximityLevel.MEDIUM
    area_ratio: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.to_dict(),
            "spatial_sector": self.spatial_sector.value,
            "proximity": self.proximity.value,
            "area_ratio": round(self.area_ratio, 4),
            "attributes": self.attributes,
        }


@dataclass
class RawDetection:
    """Unprocessed detection directly from the object detector backend."""
    label: str
    confidence: float
    bbox: BoundingBox
    class_id: int = -1


@dataclass
class VisionAgentInput(BaseAgentInput):
    """Input payload passed to the Vision Agent."""
    frame: Any = None
    frame_id: int = 0
    source_name: str = "camera"


@dataclass
class VisionAgentOutput(BaseAgentOutput):
    """Structured visual context emitted by the Vision Agent for downstream reasoning."""
    frame_id: int = 0
    image_width: int = 0
    image_height: int = 0
    detected_objects: List[DetectedObject] = field(default_factory=list)
    target_candidates: List[str] = field(default_factory=list)
    spatial_breakdown: Dict[str, List[str]] = field(default_factory=dict)
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "frame_id": self.frame_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detected_objects": [obj.to_dict() for obj in self.detected_objects],
            "target_candidates": self.target_candidates,
            "spatial_breakdown": self.spatial_breakdown,
            "summary_text": self.summary_text,
        })
        return base_dict
