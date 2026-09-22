"""Comprehensive unit tests for GestureAgent, HagridGestureRecognizer, and backends."""

from dataclasses import dataclass
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.agents.gesture.agent import GestureAgent
from src.agents.gesture.hagrid_gesture_recognizer import HagridGestureRecognizer
from src.agents.gesture.recognizer import LightweightGestureRecognizer, MockGestureRecognizer
from src.agents.gesture.schemas import (
    GestureAgentInput,
    GestureDirection,
    GestureType,
    HandBBox,
    RecognizedGesture,
)
from src.common.exceptions import ModelLoadError, PerceptionError
from src.common.schemas import AgentStatus, AgentType


class DummyBoxes:
    """Mock container for Ultralytics YOLO Boxes object."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.array(xyxy, dtype=np.float32)
        self.conf = np.array(conf, dtype=np.float32)
        self.cls = np.array(cls, dtype=np.int64)

    def __len__(self):
        return len(self.xyxy)


class DummyResult:
    """Mock container for Ultralytics YOLO Results object."""

    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class DummyYOLOModel:
    """Mock YOLO model for deterministic testing of HagridGestureRecognizer."""

    def __init__(self, names, detections=None):
        self.names = names
        self.detections = detections or []

    def predict(self, frame, conf=0.5, device="cpu", verbose=False):
        if not self.detections:
            boxes = DummyBoxes(xyxy=[], conf=[], cls=[])
        else:
            xyxy = [d["xyxy"] for d in self.detections]
            confs = [d["conf"] for d in self.detections]
            classes = [d["cls"] for d in self.detections]
            boxes = DummyBoxes(xyxy=xyxy, conf=confs, cls=classes)
        return [DummyResult(boxes=boxes, names=self.names)]


class TestGestureAgent(unittest.TestCase):
    """Tests for GestureAgent lifecycle and processing."""

    def setUp(self):
        self.recognizer = MockGestureRecognizer()
        self.agent = GestureAgent(recognizer=self.recognizer)
        self.agent.initialize()

    def tearDown(self):
        self.agent.shutdown()

    def test_lifecycle_and_simulation_override(self):
        self.assertEqual(self.agent.status, AgentStatus.READY)
        self.assertEqual(self.agent.agent_type, AgentType.GESTURE)

        output = self.agent.process(
            GestureAgentInput(gesture_override="POINT_LEFT", frame_id=7)
        )

        self.assertTrue(output.success)
        self.assertTrue(output.is_gesture_detected)
        self.assertEqual(output.gesture, GestureType.POINT_LEFT.value)
        self.assertEqual(output.direction, GestureDirection.LEFT.value)
        self.assertEqual(output.frame_id, 7)

    def test_empty_frame_has_no_detection(self):
        self.recognizer.clear()
        output = self.agent.process(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertFalse(output.is_gesture_detected)
        self.assertEqual(output.gesture, GestureType.NO_GESTURE.value)
        self.assertEqual(output.direction, GestureDirection.NONE.value)

    def test_mock_recognizer_output_is_forwarded(self):
        self.recognizer.set_gesture(GestureType.STOP, confidence=0.91)
        output = self.agent.process({"frame_id": 3})

        self.assertEqual(output.gesture, GestureType.STOP.value)
        self.assertAlmostEqual(output.confidence, 0.91)
        self.assertEqual(output.recognized_gestures[0].gesture, GestureType.STOP)

    def test_backend_selection_mock(self):
        agent = GestureAgent(config={"gesture": {"backend": "mock", "confidence_threshold": 0.6}})
        self.assertTrue(agent.initialize())
        self.assertIsInstance(agent.recognizer, MockGestureRecognizer)
        agent.shutdown()

    def test_backend_selection_lightweight(self):
        agent = GestureAgent(config={"gesture": {"backend": "lightweight"}})
        self.assertTrue(agent.initialize())
        self.assertIsInstance(agent.recognizer, LightweightGestureRecognizer)
        agent.shutdown()

    def test_backend_selection_hagrid_with_mocked_model(self):
        mock_model = DummyYOLOModel(names={0: "stop"}, detections=[{"xyxy": [100, 100, 300, 300], "conf": 0.88, "cls": 0}])
        rec = HagridGestureRecognizer(model_backend=mock_model)
        agent = GestureAgent(recognizer=rec)
        self.assertTrue(agent.initialize())
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        output = agent.process(frame)
        self.assertTrue(output.is_gesture_detected)
        self.assertEqual(output.gesture, GestureType.STOP.value)
        self.assertAlmostEqual(output.confidence, 0.88)
        agent.shutdown()


class TestHagridGestureRecognizer(unittest.TestCase):
    """Deep unit tests for HagridGestureRecognizer."""

    def setUp(self):
        self.names_map = {
            0: "stop",
            1: "like",
            2: "dislike",
            3: "ok",
            4: "point",
            5: "one",
            6: "rock",
            7: "three",
            8: "mute",
            9: "no_gesture",
            10: "call",
            11: "palm",
            12: "stop_inverted",
            13: "peace",
        }

    def test_init_parameters(self):
        rec = HagridGestureRecognizer(
            model_path="models/hagrid/custom.pt",
            confidence_threshold=0.65,
            device="cpu",
        )
        self.assertEqual(rec.model_path, "models/hagrid/custom.pt")
        self.assertEqual(rec.confidence_threshold, 0.65)
        self.assertEqual(rec.device, "cpu")
        self.assertFalse(rec.is_loaded)

    def test_missing_model_file_raises_model_load_error(self):
        rec = HagridGestureRecognizer(model_path="non_existent_path_12345.pt")
        with self.assertRaises(ModelLoadError) as ctx:
            rec.load_model()
        self.assertIn("not found", str(ctx.exception).lower())

    def test_empty_model_path_raises_model_load_error(self):
        rec = HagridGestureRecognizer(model_path="")
        with self.assertRaises(ModelLoadError):
            rec.load_model()

    def test_empty_frame_raises_perception_error(self):
        mock_model = DummyYOLOModel(names=self.names_map)
        rec = HagridGestureRecognizer(model_backend=mock_model)
        rec.load_model()
        with self.assertRaises(PerceptionError):
            rec.recognize(frame=np.array([]))

    def test_invalid_frame_type_raises_perception_error(self):
        mock_model = DummyYOLOModel(names=self.names_map)
        rec = HagridGestureRecognizer(model_backend=mock_model)
        rec.load_model()
        with self.assertRaises(PerceptionError):
            rec.recognize(frame="not_a_numpy_array")

    def test_none_frame_returns_empty_list(self):
        mock_model = DummyYOLOModel(names=self.names_map)
        rec = HagridGestureRecognizer(model_backend=mock_model)
        rec.load_model()
        res = rec.recognize(frame=None)
        self.assertEqual(res, [])

    def test_confidence_threshold_filtering(self):
        # One high confidence (0.85) and one low confidence (0.40) detection
        detections = [
            {"xyxy": [50, 50, 200, 200], "conf": 0.85, "cls": 0},  # stop
            {"xyxy": [300, 300, 450, 450], "conf": 0.40, "cls": 1},  # like (filtered)
        ]
        mock_model = DummyYOLOModel(names=self.names_map, detections=detections)
        rec = HagridGestureRecognizer(confidence_threshold=0.50, model_backend=mock_model)
        rec.load_model()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        results = rec.recognize(frame)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].gesture, GestureType.STOP)
        self.assertAlmostEqual(results[0].confidence, 0.85)

    def test_bounding_box_conversion_to_handbbox(self):
        detections = [
            {"xyxy": [120.5, 80.2, 280.8, 320.4], "conf": 0.92, "cls": 0}
        ]
        mock_model = DummyYOLOModel(names=self.names_map, detections=detections)
        rec = HagridGestureRecognizer(model_backend=mock_model)
        rec.load_model()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        results = rec.recognize(frame)

        self.assertEqual(len(results), 1)
        bbox = results[0].bbox
        self.assertIsNotNone(bbox)
        self.assertAlmostEqual(bbox.x1, 120.5, places=1)
        self.assertAlmostEqual(bbox.y1, 80.2, places=1)
        self.assertAlmostEqual(bbox.x2, 280.8, places=1)
        self.assertAlmostEqual(bbox.y2, 320.4, places=1)
        self.assertAlmostEqual(bbox.width, 160.3, places=1)

    def test_hagrid_class_mapping_core_gestures(self):
        """Verify exact mapping for stop, like, dislike, ok, one, rock, three, mute, no_gesture."""
        test_cases = [
            (0, "stop", GestureType.STOP, GestureDirection.NONE),
            (1, "like", GestureType.THUMBS_UP, GestureDirection.UP),
            (2, "dislike", GestureType.THUMBS_DOWN, GestureDirection.DOWN),
            (3, "ok", GestureType.OK, GestureDirection.NONE),
            (5, "one", GestureType.ONE, GestureDirection.FORWARD),
            (6, "rock", GestureType.ROCK, GestureDirection.NONE),
            (7, "three", GestureType.THREE, GestureDirection.NONE),
            (8, "mute", GestureType.MUTE, GestureDirection.NONE),
            (9, "no_gesture", GestureType.NO_GESTURE, GestureDirection.NONE),
        ]

        for cls_id, cls_name, expected_type, expected_dir in test_cases:
            detections = [{"xyxy": [100, 100, 200, 200], "conf": 0.90, "cls": cls_id}]
            mock_model = DummyYOLOModel(names={cls_id: cls_name}, detections=detections)
            rec = HagridGestureRecognizer(model_backend=mock_model)
            rec.load_model()

            frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
            results = rec.recognize(frame)

            self.assertEqual(len(results), 1, f"Failed for {cls_name}")
            self.assertEqual(results[0].gesture, expected_type, f"Class mapping failed for {cls_name}")
            self.assertEqual(results[0].direction, expected_dir, f"Direction mapping failed for {cls_name}")

    def test_hagrid_unmapped_classes_map_to_unknown(self):
        """Verify that palm, stop_inverted, call, peace, etc. are NOT mapped to STOP and become UNKNOWN."""
        unmapped_cases = [
            (10, "call"),
            (11, "palm"),
            (12, "stop_inverted"),
            (13, "peace"),
        ]

        for cls_id, cls_name in unmapped_cases:
            detections = [{"xyxy": [100, 100, 200, 200], "conf": 0.88, "cls": cls_id}]
            mock_model = DummyYOLOModel(names={cls_id: cls_name}, detections=detections)
            rec = HagridGestureRecognizer(model_backend=mock_model)
            rec.load_model()

            frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
            results = rec.recognize(frame)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].gesture, GestureType.UNKNOWN, f"{cls_name} should map to UNKNOWN")
            self.assertEqual(results[0].direction, GestureDirection.UNKNOWN)

    def test_spatial_pointing_grounding(self):
        """Verify spatial grounding of 'point' gesture into LEFT, FORWARD, and RIGHT."""
        frame_width = 600
        frame_height = 400
        frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 100

        # 1. Left Sector: centroid x < 0.33 * 600 = 198 (e.g. bbox from 50 to 150, center = 100)
        detections_left = [{"xyxy": [50, 100, 150, 250], "conf": 0.89, "cls": 4}]
        rec_left = HagridGestureRecognizer(model_backend=DummyYOLOModel(names={4: "point"}, detections=detections_left))
        rec_left.load_model()
        res_left = rec_left.recognize(frame)
        self.assertEqual(res_left[0].gesture, GestureType.POINT_LEFT)
        self.assertEqual(res_left[0].direction, GestureDirection.LEFT)
        self.assertEqual(res_left[0].metadata.get("direction_source"), "bbox_spatial_grounding")

        # 2. Center Sector: 198 <= centroid x <= 402 (e.g. bbox from 250 to 350, center = 300)
        detections_center = [{"xyxy": [250, 100, 350, 250], "conf": 0.91, "cls": 4}]
        rec_center = HagridGestureRecognizer(model_backend=DummyYOLOModel(names={4: "point"}, detections=detections_center))
        rec_center.load_model()
        res_center = rec_center.recognize(frame)
        self.assertEqual(res_center[0].gesture, GestureType.POINT_FORWARD)
        self.assertEqual(res_center[0].direction, GestureDirection.FORWARD)

        # 3. Right Sector: centroid x > 0.67 * 600 = 402 (e.g. bbox from 450 to 550, center = 500)
        detections_right = [{"xyxy": [450, 100, 550, 250], "conf": 0.93, "cls": 4}]
        rec_right = HagridGestureRecognizer(model_backend=DummyYOLOModel(names={4: "point"}, detections=detections_right))
        rec_right.load_model()
        res_right = rec_right.recognize(frame)
        self.assertEqual(res_right[0].gesture, GestureType.POINT_RIGHT)
        self.assertEqual(res_right[0].direction, GestureDirection.RIGHT)

    def test_multiple_detections_sorted_descending(self):
        """Verify that multiple recognized gestures are sorted by confidence descending."""
        detections = [
            {"xyxy": [50, 50, 150, 150], "conf": 0.60, "cls": 0},   # stop (0.60)
            {"xyxy": [250, 50, 350, 150], "conf": 0.95, "cls": 1},  # like (0.95)
            {"xyxy": [450, 50, 550, 150], "conf": 0.78, "cls": 3},  # ok (0.78)
        ]
        mock_model = DummyYOLOModel(names=self.names_map, detections=detections)
        rec = HagridGestureRecognizer(model_backend=mock_model)
        rec.load_model()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        results = rec.recognize(frame)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].gesture, GestureType.THUMBS_UP)
        self.assertAlmostEqual(results[0].confidence, 0.95)
        self.assertEqual(results[1].gesture, GestureType.OK)
        self.assertAlmostEqual(results[1].confidence, 0.78)
        self.assertEqual(results[2].gesture, GestureType.STOP)
        self.assertAlmostEqual(results[2].confidence, 0.60)

    def test_metadata_structure(self):
        detections = [{"xyxy": [100, 100, 200, 200], "conf": 0.87, "cls": 0}]
        mock_model = DummyYOLOModel(names={0: "stop"}, detections=detections)
        rec = HagridGestureRecognizer(model_path="models/hagrid/yolov10n_hagrid.pt", model_backend=mock_model)
        rec.load_model()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        results = rec.recognize(frame)

        meta = results[0].metadata
        self.assertEqual(meta.get("source"), "hagrid_pretrained")
        self.assertEqual(meta.get("model"), "models/hagrid/yolov10n_hagrid.pt")
        self.assertEqual(meta.get("dataset"), "HaGRID")
        self.assertEqual(meta.get("class_name"), "stop")
        self.assertEqual(meta.get("class_id"), 0)


class TestLightweightGestureRecognizer(unittest.TestCase):
    """Tests for fallback LightweightGestureRecognizer."""

    def setUp(self):
        self.recognizer = LightweightGestureRecognizer(confidence_threshold=0.50)
        self.recognizer.load_model()

    def test_landmark_stop_all_extended(self):
        # 21 points with all fingers extended away from wrist at (0.5, 0.9)
        wrist = [0.5, 0.9]
        landmarks = [wrist]
        # 4 points each for 5 fingers
        for f in range(5):
            for j in range(1, 5):
                landmarks.append([0.3 + f * 0.1, 0.9 - j * 0.15])

        results = self.recognizer.recognize(landmarks=landmarks)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].gesture, GestureType.STOP)

    def test_insufficient_landmarks(self):
        results = self.recognizer.recognize(landmarks=[[0.1, 0.2]])
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()