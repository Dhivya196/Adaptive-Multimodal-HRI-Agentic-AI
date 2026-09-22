"""Pretrained HaGRID Gesture Detector backend for the Gesture Agent.

Utilizes a pretrained HaGRID detection model (e.g. YOLOv10n / YOLO detector trained
on the official HaGRID dataset) to recognize hand gestures, extract bounding boxes,
map classes to the HRI project schema, and perform application-level spatial grounding
for pointing gestures.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.agents.gesture.recognizer import BaseGestureRecognizer
from src.agents.gesture.schemas import (
    GestureDirection,
    GestureType,
    HandBBox,
    HandSide,
    RecognizedGesture,
)
from src.common.exceptions import ModelLoadError, PerceptionError


class HagridGestureRecognizer(BaseGestureRecognizer):
    """
    Pretrained HaGRID Gesture Recognizer.

    Pipeline:
    1. Receive RGB frame as NumPy array.
    2. Run inference via loaded pretrained detector.
    3. Filter detections by confidence_threshold.
    4. Convert detection bounding boxes to HandBBox.
    5. Map official HaGRID class names to project GestureType.
    6. Perform spatial grounding on 'point' gestures based on bbox centroid.
    7. Sort recognized gestures by confidence descending.
    """

    # Official HaGRID gesture class name to project GestureType mapping
    HAGRID_CLASS_MAP: Dict[str, GestureType] = {
        "stop": GestureType.STOP,
        "like": GestureType.THUMBS_UP,
        "dislike": GestureType.THUMBS_DOWN,
        "ok": GestureType.OK,
        "point": GestureType.POINT_FORWARD,  # Spatially grounded by application
        "one": GestureType.ONE,
        "rock": GestureType.ROCK,
        "three": GestureType.THREE,
        "three2": GestureType.THREE,
        "three3": GestureType.THREE,
        "mute": GestureType.MUTE,
        "no_gesture": GestureType.NO_GESTURE,
        # Other official HaGRID classes without direct project action equivalent map to UNKNOWN
        "call": GestureType.UNKNOWN,
        "fist": GestureType.UNKNOWN,
        "four": GestureType.UNKNOWN,
        "peace": GestureType.UNKNOWN,
        "peace_inverted": GestureType.UNKNOWN,
        "palm": GestureType.UNKNOWN,
        "stop_inverted": GestureType.UNKNOWN,
        "two_up": GestureType.UNKNOWN,
        "two_up_inverted": GestureType.UNKNOWN,
        "grabbing": GestureType.UNKNOWN,
        "grip": GestureType.UNKNOWN,
        "holy": GestureType.UNKNOWN,
        "timeout": GestureType.UNKNOWN,
        "xsign": GestureType.UNKNOWN,
        "hand_heart": GestureType.UNKNOWN,
        "hand_heart2": GestureType.UNKNOWN,
        "little_finger": GestureType.UNKNOWN,
        "middle_finger": GestureType.UNKNOWN,
        "take_picture": GestureType.UNKNOWN,
        "three_gun": GestureType.UNKNOWN,
        "pinch": GestureType.UNKNOWN,
    }

    # Default intrinsic directions for non-pointing gestures
    GESTURE_DEFAULT_DIRECTIONS: Dict[GestureType, GestureDirection] = {
        GestureType.STOP: GestureDirection.NONE,
        GestureType.THUMBS_UP: GestureDirection.UP,
        GestureType.THUMBS_DOWN: GestureDirection.DOWN,
        GestureType.OK: GestureDirection.NONE,
        GestureType.ROCK: GestureDirection.NONE,
        GestureType.ONE: GestureDirection.FORWARD,
        GestureType.THREE: GestureDirection.NONE,
        GestureType.MUTE: GestureDirection.NONE,
        GestureType.NO_GESTURE: GestureDirection.NONE,
        GestureType.UNKNOWN: GestureDirection.UNKNOWN,
    }

    def __init__(
        self,
        model_path: Optional[str] = "models/hagrid/yolov10n_hagrid.pt",
        confidence_threshold: float = 0.50,
        device: str = "cpu",
        model_backend: Optional[Any] = None,
    ):
        super().__init__(confidence_threshold=confidence_threshold)
        self.model_path = model_path
        self.device = device
        self.model = model_backend

    def load_model(self) -> bool:
        """
        Load the pretrained HaGRID model weights.
        Loads once and caches the model instance.
        """
        if self.model is not None:
            self.is_loaded = True
            return True

        if not self.model_path:
            raise ModelLoadError("No model_path provided for HagridGestureRecognizer.")

        path = Path(self.model_path)
        if not path.is_absolute():
            # Resolve relative to project root
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            path = project_root / self.model_path

        if not path.exists():
            raise ModelLoadError(
                f"HaGRID pretrained model file not found at '{path}'. "
                f"Please place the pretrained model weights (e.g. 'yolov10n_hagrid.pt') "
                f"in 'models/hagrid/' or configure model_path in configs/gesture_config.yaml."
            )

        self.logger.info(f"Loading pretrained HaGRID detector from '{path}' on device '{self.device}'...")
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(path))
            self.is_loaded = True
            self.logger.info("HaGRID pretrained model loaded successfully.")
            return True
        except Exception as e:
            self.is_loaded = False
            raise ModelLoadError(f"Failed to load HaGRID model from '{path}': {e}") from e

    def _ground_pointing_direction(
        self,
        bbox: HandBBox,
        image_width: int,
    ) -> Tuple[GestureType, GestureDirection]:
        """
        Perform application-level spatial grounding for 'point' gestures.

        Grounding Rule:
        - x_center < 0.33 * width       -> POINT_LEFT (direction: LEFT)
        - 0.33 * width <= x_center <= 0.67 * width -> POINT_FORWARD (direction: FORWARD)
        - x_center > 0.67 * width       -> POINT_RIGHT (direction: RIGHT)
        """
        center_x = bbox.center[0]
        left_bound = 0.33 * image_width
        right_bound = 0.67 * image_width

        if center_x < left_bound:
            return GestureType.POINT_LEFT, GestureDirection.LEFT
        elif center_x > right_bound:
            return GestureType.POINT_RIGHT, GestureDirection.RIGHT
        else:
            return GestureType.POINT_FORWARD, GestureDirection.FORWARD

    def recognize(
        self,
        frame: Optional[np.ndarray] = None,
        landmarks: Optional[List[Any]] = None,
    ) -> List[RecognizedGesture]:
        """
        Run inference on an input frame using the pretrained HaGRID model.

        Parameters:
            frame: Input RGB/BGR image frame as a NumPy array (H x W x C).
            landmarks: Optional landmarks (ignored by the visual HaGRID detector).

        Returns:
            List of RecognizedGesture objects sorted by confidence descending.
        """
        if frame is None:
            return []

        if not isinstance(frame, np.ndarray):
            raise PerceptionError("HagridGestureRecognizer expects an image frame as a NumPy ndarray.")

        if frame.size == 0:
            raise PerceptionError("Cannot process an empty image frame.")

        if not self.is_loaded:
            self.load_model()

        img_height, img_width = frame.shape[:2]

        try:
            # Run inference via the loaded model
            if hasattr(self.model, "predict"):
                raw_results = self.model.predict(
                    frame,
                    conf=self.confidence_threshold,
                    device=self.device,
                    verbose=False,
                )
            elif callable(self.model):
                raw_results = self.model(frame)
            else:
                raise PerceptionError(f"Loaded model '{type(self.model)}' is not callable or recognizable.")

            detections: List[RecognizedGesture] = []

            for result in raw_results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue

                xyxy_attr = getattr(boxes, "xyxy", None)
                if xyxy_attr is None:
                    continue

                xyxy_list = xyxy_attr.cpu().numpy() if hasattr(xyxy_attr, "cpu") else np.array(xyxy_attr)
                if len(xyxy_list) == 0:
                    continue

                # Extract class names mapping if available on result/model
                names = getattr(result, "names", {}) or getattr(self.model, "names", {})

                conf_attr = getattr(boxes, "conf", None)
                cls_attr = getattr(boxes, "cls", None)

                conf_list = conf_attr.cpu().numpy() if hasattr(conf_attr, "cpu") else np.array(conf_attr)
                cls_list = cls_attr.cpu().numpy() if hasattr(cls_attr, "cpu") else np.array(cls_attr)

                for i in range(len(xyxy_list)):
                    conf = float(conf_list[i])
                    if conf < self.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = xyxy_list[i]
                    class_id = int(cls_list[i])

                    # Determine raw class name from model names dictionary
                    if isinstance(names, dict) and class_id in names:
                        raw_class_name = str(names[class_id]).lower().strip()
                    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                        raw_class_name = str(names[class_id]).lower().strip()
                    else:
                        raw_class_name = str(class_id)

                    # Create bounding box
                    bbox = HandBBox(
                        x1=float(max(0.0, x1)),
                        y1=float(max(0.0, y1)),
                        x2=float(min(img_width, x2)),
                        y2=float(min(img_height, y2)),
                    )

                    # Map class to project GestureType
                    mapped_gesture = self.HAGRID_CLASS_MAP.get(raw_class_name, GestureType.UNKNOWN)

                    # Metadata
                    metadata = {
                        "source": "hagrid_pretrained",
                        "model": str(self.model_path),
                        "dataset": "HaGRID",
                        "class_name": raw_class_name,
                        "class_id": class_id,
                    }

                    # Handle pointing gesture with spatial grounding
                    if raw_class_name == "point" or mapped_gesture == GestureType.POINT_FORWARD:
                        grounded_gesture, grounded_direction = self._ground_pointing_direction(
                            bbox=bbox,
                            image_width=img_width,
                        )
                        mapped_gesture = grounded_gesture
                        direction = grounded_direction
                        metadata["direction_source"] = "bbox_spatial_grounding"
                    else:
                        direction = self.GESTURE_DEFAULT_DIRECTIONS.get(
                            mapped_gesture,
                            GestureDirection.NONE,
                        )

                    rec = RecognizedGesture(
                        gesture=mapped_gesture,
                        direction=direction,
                        confidence=conf,
                        bbox=bbox,
                        hand_side=HandSide.ANY,
                        metadata=metadata,
                    )
                    detections.append(rec)

            # Sort detections by confidence descending
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections

        except PerceptionError:
            raise
        except Exception as e:
            self.logger.error(f"Error during HaGRID gesture recognition: {e}")
            raise PerceptionError(f"HaGRID inference error: {e}") from e
