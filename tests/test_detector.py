"""Unit tests for detector interface & mock inference."""

import unittest
import numpy as np

from src.agents.vision.detector import MockObjectDetector
from src.agents.vision.schemas import BoundingBox, RawDetection
from src.common.exceptions import PerceptionError


class TestDetector(unittest.TestCase):

    def setUp(self):
        self.mock_detections = [
            RawDetection(label="bottle", confidence=0.85, bbox=BoundingBox(x1=10, y1=10, x2=50, y2=100)),
            RawDetection(label="cup", confidence=0.35, bbox=BoundingBox(x1=60, y1=60, x2=90, y2=90)),
            RawDetection(label="person", confidence=0.95, bbox=BoundingBox(x1=150, y1=50, x2=300, y2=400)),
        ]
        self.detector = MockObjectDetector(
            mock_detections=self.mock_detections,
            confidence_threshold=0.45,
        )

    def test_detector_initialization(self):
        self.detector.load_model()
        self.assertTrue(self.detector.is_loaded)

    def test_mock_detection_filtering(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(frame)

        self.assertEqual(len(results), 2)
        labels = [r.label for r in results]
        self.assertIn("bottle", labels)
        self.assertIn("person", labels)
        self.assertNotIn("cup", labels)

    def test_empty_frame_raises_error(self):
        with self.assertRaises(PerceptionError):
            self.detector.detect(None)

        empty_array = np.array([])
        with self.assertRaises(PerceptionError):
            self.detector.detect(empty_array)


if __name__ == "__main__":
    unittest.main()
