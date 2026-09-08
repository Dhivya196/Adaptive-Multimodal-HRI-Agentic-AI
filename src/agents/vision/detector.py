"""Object detector interface and YOLO / Mock implementations for the Vision Agent."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from src.agents.vision.schemas import BoundingBox, RawDetection
from src.common.exceptions import ModelLoadError, PerceptionError
from src.common.logger import get_logger


class BaseObjectDetector(ABC):
    """Abstract interface for object detection backends."""

    def __init__(self, confidence_threshold: float = 0.45, iou_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.logger = get_logger(self.__class__.__name__)
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load detection model weights into memory."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        """Perform object detection on an input BGR image/frame."""
        pass


class YOLOObjectDetector(BaseObjectDetector):
    """Ultralytics YOLO (YOLO26n / YOLOv8) object detector."""

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[str]] = None,
    ):
        super().__init__(confidence_threshold, iou_threshold)
        self.model_name = model_name
        self.device = device
        self.target_classes = target_classes or []
        self.model = None

    def load_model(self) -> None:
        """Load YOLO model from Ultralytics."""
        self.logger.info(f"Loading YOLO model '{self.model_name}' on device '{self.device}'...")
        try:
            # pyrefly: ignore [missing-import]
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            self.is_loaded = True
            self.logger.info(f"Successfully loaded YOLO model: {self.model_name}")
        except ImportError as e:
            self.logger.warning(
                "Ultralytics library not found. Install via 'pip install ultralytics'. "
                "Detector will raise an error unless MockObjectDetector is used."
            )
            raise ModelLoadError("ultralytics package is required for YOLOObjectDetector.") from e
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model '{self.model_name}': {e}")
            raise ModelLoadError(f"Failed to load YOLO model '{self.model_name}': {e}") from e

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        """Run YOLO inference on an input image frame."""
        if not self.is_loaded or self.model is None:
            self.load_model()

        if frame is None or frame.size == 0:
            raise PerceptionError("Cannot run detection on empty or None image frame.")

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device if self.device != "auto" else None,
            )

            detections: List[RawDetection] = []
            if not results:
                return detections

            result = results[0]
            boxes = result.boxes
            if boxes is None:
                return detections

            names = result.names

            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                label = names.get(cls_id, f"class_{cls_id}")

                if self.target_classes and label not in self.target_classes:
                    continue

                bbox = BoundingBox(
                    x1=xyxy[0],
                    y1=xyxy[1],
                    x2=xyxy[2],
                    y2=xyxy[3],
                )
                detections.append(
                    RawDetection(
                        label=label,
                        confidence=conf,
                        bbox=bbox,
                        class_id=cls_id,
                    )
                )

            return detections
        except Exception as e:
            self.logger.error(f"Error during YOLO detection: {e}")
            raise PerceptionError(f"YOLO inference failed: {e}") from e


class MockObjectDetector(BaseObjectDetector):
    """Mock detector for unit tests and headless testing environments."""

    def __init__(
        self,
        mock_detections: Optional[List[RawDetection]] = None,
        confidence_threshold: float = 0.45,
    ):
        super().__init__(confidence_threshold=confidence_threshold)
        self.mock_detections = mock_detections or []

    def load_model(self) -> None:
        self.is_loaded = True
        self.logger.info("Loaded MockObjectDetector successfully.")

    def set_mock_detections(self, detections: List[RawDetection]) -> None:
        """Update mock detections returned on subsequent detect() calls."""
        self.mock_detections = detections

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        """Return configured mock detections filtered by confidence."""
        if not self.is_loaded:
            self.load_model()

        if frame is None or (isinstance(frame, np.ndarray) and frame.size == 0):
            raise PerceptionError("Cannot run detection on empty or None image frame.")

        return [d for d in self.mock_detections if d.confidence >= self.confidence_threshold]
