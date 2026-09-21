"""Vision perception agent package."""

from src.agents.vision.agent import VisionAgent
from src.agents.vision.detector import BaseObjectDetector, YOLOObjectDetector, MockObjectDetector
from src.agents.vision.schemas import (
    BoundingBox,
    DetectedObject,
    Point2D,
    ProximityLevel,
    RawDetection,
    SpatialSector,
    VisionAgentInput,
    VisionAgentOutput,
)
from src.agents.vision.visualizer import VisionVisualizer

__all__ = [
    "VisionAgent",
    "BaseObjectDetector",
    "YOLOObjectDetector",
    "MockObjectDetector",
    "BoundingBox",
    "DetectedObject",
    "Point2D",
    "ProximityLevel",
    "RawDetection",
    "SpatialSector",
    "VisionAgentInput",
    "VisionAgentOutput",
    "VisionVisualizer",
]
