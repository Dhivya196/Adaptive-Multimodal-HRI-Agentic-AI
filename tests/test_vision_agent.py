"""Unit tests for spatial grounding & vision agent logic."""

import unittest
import numpy as np

from src.agents.vision.agent import VisionAgent
from src.agents.vision.detector import MockObjectDetector
from src.agents.vision.schemas import (
    BoundingBox,
    RawDetection,
    SpatialSector,
    VisionAgentInput,
    VisionAgentOutput,
)
from src.common.schemas import AgentStatus


class TestVisionAgent(unittest.TestCase):

    def setUp(self):
        self.mock_detections = [
            RawDetection(label="cup", confidence=0.88, bbox=BoundingBox(x1=50, y1=100, x2=150, y2=200)),
            RawDetection(label="person", confidence=0.95, bbox=BoundingBox(x1=250, y1=50, x2=390, y2=400)),
            RawDetection(label="bottle", confidence=0.91, bbox=BoundingBox(x1=450, y1=100, x2=600, y2=450)),
        ]
        self.detector = MockObjectDetector(
            mock_detections=self.mock_detections,
            confidence_threshold=0.4,
        )
        self.agent = VisionAgent(name="TestVisionAgent", detector=self.detector)

    def test_agent_lifecycle(self):
        self.assertEqual(self.agent.status, AgentStatus.UNINITIALIZED)
        self.agent.initialize()
        self.assertEqual(self.agent.status, AgentStatus.READY)

    def test_spatial_sector_grounding(self):
        self.agent.initialize()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        output: VisionAgentOutput = self.agent.process(VisionAgentInput(frame=frame, frame_id=42))

        self.assertTrue(output.success)
        self.assertEqual(output.frame_id, 42)
        self.assertEqual(len(output.detected_objects), 3)

        obj_by_label = {obj.label: obj for obj in output.detected_objects}

        self.assertIn("cup", obj_by_label)
        self.assertEqual(obj_by_label["cup"].spatial_sector, SpatialSector.LEFT)

        self.assertIn("person", obj_by_label)
        self.assertEqual(obj_by_label["person"].spatial_sector, SpatialSector.CENTER)

        self.assertIn("bottle", obj_by_label)
        self.assertEqual(obj_by_label["bottle"].spatial_sector, SpatialSector.RIGHT)

        self.assertEqual(output.spatial_breakdown["LEFT"], ["cup"])
        self.assertEqual(output.spatial_breakdown["CENTER"], ["person"])
        self.assertEqual(output.spatial_breakdown["RIGHT"], ["bottle"])

        self.assertIn("cup in LEFT sector", output.summary_text)
        self.assertIn("person in CENTER sector", output.summary_text)
        self.assertIn("bottle in RIGHT sector", output.summary_text)

    def test_empty_scene_handling(self):
        self.detector.set_mock_detections([])
        self.agent.initialize()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        output: VisionAgentOutput = self.agent.process(frame)
        self.assertEqual(len(output.detected_objects), 0)
        self.assertIn("no recognized target objects", output.summary_text)


if __name__ == "__main__":
    unittest.main()
