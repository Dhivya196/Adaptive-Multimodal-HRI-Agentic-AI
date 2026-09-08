"""Vision Agent: processes camera frames and produces structured visual context."""

from typing import Any, Dict, List, Optional
import numpy as np

from src.agents.base import BaseAgent
from src.agents.vision.detector import BaseObjectDetector, YOLOObjectDetector
from src.agents.vision.schemas import (
    DetectedObject,
    ProximityLevel,
    RawDetection,
    SpatialSector,
    VisionAgentInput,
    VisionAgentOutput,
)
from src.common.exceptions import PerceptionError
from src.common.schemas import AgentType


class VisionAgent(BaseAgent):
    """
    Vision perception agent.

    Responsibilities:
    - Receive an image/frame
    - Detect objects using the configured detector
    - Determine horizontal spatial location
    - Estimate visual proximity
    - Produce structured visual context

    The Vision Agent does NOT decide robot actions.
    """

    def __init__(
        self,
        name: str = "VisionAgent",
        detector: Optional[BaseObjectDetector] = None,
        config: Optional[dict] = None,
    ):
        super().__init__(
            name=name,
            agent_type=AgentType.VISION,
            version="0.1.0",
            description="Perceives objects, locations, and spatial context for HRI tasks.",
            config=config or {},
        )

        self.detector = detector

        spatial_config = self.config.get("spatial", {})

        self._sector_split = spatial_config.get(
            "sector_split",
            [0.33, 0.67]
        )

        self._near_thresh = spatial_config.get(
            "near_area_ratio_threshold",
            0.15
        )

        self._med_thresh = spatial_config.get(
            "medium_area_ratio_threshold",
            0.05
        )

    def _initialize(self) -> bool:
        """Initialize the configured object detector."""

        if self.detector is None:

            vision_cfg = self.config.get("vision", {})

            model_name = vision_cfg.get(
                "model_name",
                "yolo26n.pt"
            )

            device = vision_cfg.get(
                "device",
                "auto"
            )

            confidence_threshold = vision_cfg.get(
                "confidence_threshold",
                0.45
            )

            iou_threshold = vision_cfg.get(
                "iou_threshold",
                0.45
            )

            target_classes = vision_cfg.get(
                "target_classes",
                None
            )

            self.detector = YOLOObjectDetector(
                model_name=model_name,
                device=device,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                target_classes=target_classes,
            )

        self.detector.load_model()

        return True

    def _determine_spatial_sector(
        self,
        center_x: float,
        frame_width: int
    ) -> SpatialSector:
        """
        Determine whether an object is on the left,
        center, or right side of the image.
        """

        if frame_width <= 0:
            return SpatialSector.UNKNOWN

        x_ratio = center_x / frame_width

        left_split, right_split = self._sector_split

        if x_ratio < left_split:
            return SpatialSector.LEFT

        if x_ratio <= right_split:
            return SpatialSector.CENTER

        return SpatialSector.RIGHT

    def _determine_proximity(
        self,
        area_ratio: float
    ) -> ProximityLevel:
        """
        Estimate visual proximity using bounding-box
        area relative to the complete image.
        """

        if area_ratio >= self._near_thresh:
            return ProximityLevel.NEAR

        if area_ratio >= self._med_thresh:
            return ProximityLevel.MEDIUM

        return ProximityLevel.FAR

    def _process(self, input_data: Any) -> VisionAgentOutput:
        """Process one frame and generate structured visual context."""

        # ---------------------------------------------------------
        # 1. Extract frame
        # ---------------------------------------------------------

        if isinstance(input_data, VisionAgentInput):

            frame = input_data.frame
            frame_id = input_data.frame_id

        elif isinstance(input_data, np.ndarray):

            frame = input_data
            frame_id = 0

        elif isinstance(input_data, dict) and "frame" in input_data:

            frame = input_data["frame"]
            frame_id = input_data.get("frame_id", 0)

        else:

            raise PerceptionError(
                f"Unsupported input type for VisionAgent: "
                f"{type(input_data)}"
            )

        # ---------------------------------------------------------
        # 2. Validate frame
        # ---------------------------------------------------------

        if frame is None:
            raise PerceptionError(
                "Received None frame in VisionAgent."
            )

        if not isinstance(frame, np.ndarray):
            raise PerceptionError(
                "VisionAgent expects an image as a NumPy array."
            )

        if frame.size == 0:
            raise PerceptionError(
                "Received an empty image frame."
            )

        # ---------------------------------------------------------
        # 3. Validate detector
        # ---------------------------------------------------------

        if self.detector is None:
            raise PerceptionError(
                "Vision detector is not initialized."
            )

        # ---------------------------------------------------------
        # 4. Get image dimensions
        # ---------------------------------------------------------

        height, width = frame.shape[:2]

        frame_area = max(
            1.0,
            float(width * height)
        )

        # ---------------------------------------------------------
        # 5. Run object detection
        # ---------------------------------------------------------

        raw_detections: List[RawDetection] = (
            self.detector.detect(frame)
        )

        # ---------------------------------------------------------
        # 6. Convert raw detections into structured objects
        # ---------------------------------------------------------

        detected_objects: List[DetectedObject] = []

        spatial_breakdown: Dict[str, List[str]] = {
            SpatialSector.LEFT.value: [],
            SpatialSector.CENTER.value: [],
            SpatialSector.RIGHT.value: [],
        }

        target_candidates: List[str] = []

        for object_id, raw in enumerate(
            raw_detections,
            start=1
        ):

            center = raw.bbox.center

            sector = self._determine_spatial_sector(
                center.x,
                width
            )

            area_ratio = (
                raw.bbox.area / frame_area
            )

            proximity = self._determine_proximity(
                area_ratio
            )

            detected_object = DetectedObject(
                object_id=object_id,
                label=raw.label,
                confidence=raw.confidence,
                bbox=raw.bbox,
                spatial_sector=sector,
                proximity=proximity,
                area_ratio=area_ratio,
            )

            detected_objects.append(
                detected_object
            )

            # Spatial grouping
            if sector.value in spatial_breakdown:
                spatial_breakdown[
                    sector.value
                ].append(raw.label)

            # Unique target candidates
            if raw.label not in target_candidates:
                target_candidates.append(raw.label)

        # ---------------------------------------------------------
        # 7. Generate scene summary
        # ---------------------------------------------------------

        if detected_objects:

            descriptions = []

            for obj in detected_objects:

                description = (
                    f"{obj.label} in "
                    f"{obj.spatial_sector.value} sector "
                    f"({obj.proximity.value.lower()}, "
                    f"conf={obj.confidence:.2f})"
                )

                descriptions.append(description)

            summary_text = (
                "Visual scene contains: "
                + ", ".join(descriptions)
                + "."
            )

        else:

            summary_text = (
                "Visual scene contains no "
                "recognized target objects."
            )

        # ---------------------------------------------------------
        # 8. Construct structured output
        # ---------------------------------------------------------

        return VisionAgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type.value,
            success=True,
            frame_id=frame_id,
            image_width=width,
            image_height=height,
            detected_objects=detected_objects,
            target_candidates=target_candidates,
            spatial_breakdown=spatial_breakdown,
            summary_text=summary_text,
            confidence=(
                1.0
                if detected_objects
                else 0.5
            ),
        )

    def _reset(self) -> None:
        """Reset Vision Agent state."""
        pass

    def _shutdown(self) -> None:
        """Release Vision Agent resources."""

        self.detector = None