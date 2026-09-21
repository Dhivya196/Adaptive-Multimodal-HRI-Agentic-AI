"""Unit tests for configuration loader, camera manager, and vision pipeline."""

import unittest
import numpy as np

from src.agents.vision.agent import VisionAgent
from src.agents.vision.detector import MockObjectDetector
from src.agents.vision.schemas import BoundingBox, RawDetection, VisionAgentOutput
from src.pipeline.vision_pipeline import VisionPipeline
from src.utils.camera import CameraManager
from src.utils.config import deep_merge, load_config


class TestConfigAndPipeline(unittest.TestCase):

    def test_deep_merge(self):
        d1 = {"a": 1, "nested": {"x": 10, "y": 20}}
        d2 = {"b": 2, "nested": {"y": 99, "z": 30}}
        merged = deep_merge(d1, d2)

        self.assertEqual(merged["a"], 1)
        self.assertEqual(merged["b"], 2)
        self.assertEqual(merged["nested"]["x"], 10)
        self.assertEqual(merged["nested"]["y"], 99)
        self.assertEqual(merged["nested"]["z"], 30)

    def test_load_config_default(self):
        config = load_config()
        self.assertIn("vision", config)
        self.assertIn("camera", config)
        self.assertIn("spatial", config)
        self.assertIn("visualizer", config)

    def test_camera_manager_synthetic(self):
        cam = CameraManager(width=320, height=240, fps=60)
        cam.enable_synthetic_mode()
        ret, frame = cam.read_frame()

        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (240, 320, 3))
        cam.release()

    def test_vision_pipeline_synthetic_run(self):
        mock_detections = [
            RawDetection(
                label="bottle",
                confidence=0.9,
                bbox=BoundingBox(x1=400, y1=200, x2=480, y2=340),
            )
        ]
        detector = MockObjectDetector(mock_detections=mock_detections)
        agent = VisionAgent(detector=detector)

        cam = CameraManager(width=640, height=480, fps=100)
        cam.enable_synthetic_mode()

        received_outputs = []

        def callback(out: VisionAgentOutput):
            received_outputs.append(out)

        pipeline = VisionPipeline(
            agent=agent,
            camera=cam,
            visualizer=None,
            output_callback=callback,
        )

        pipeline.run(max_frames=3)
        self.assertEqual(len(received_outputs), 3)
        self.assertTrue(received_outputs[0].success)
        self.assertEqual(len(received_outputs[0].detected_objects), 1)
        self.assertEqual(received_outputs[0].detected_objects[0].label, "bottle")


if __name__ == "__main__":
    unittest.main()
