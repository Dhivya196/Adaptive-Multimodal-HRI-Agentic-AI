"""Unit tests for structured data models."""

import unittest
import json

from src.agents.vision.schemas import (
    BoundingBox,
    DetectedObject,
    Point2D,
    ProximityLevel,
    SpatialSector,
    VisionAgentOutput,
)
from src.common.schemas import AgentType


class TestSchemas(unittest.TestCase):

    def test_point2d(self):
        p = Point2D(x=10.6, y=20.2)
        self.assertEqual(p.to_tuple(), (10.6, 20.2))
        self.assertEqual(p.to_int_tuple(), (11, 20))

    def test_bounding_box_geometry(self):
        bbox = BoundingBox(x1=100.0, y1=50.0, x2=300.0, y2=250.0)
        self.assertEqual(bbox.width, 200.0)
        self.assertEqual(bbox.height, 200.0)
        self.assertEqual(bbox.area, 40000.0)
        self.assertEqual(bbox.center.x, 200.0)
        self.assertEqual(bbox.center.y, 150.0)
        self.assertEqual(bbox.to_xyxy_int(), (100, 50, 300, 250))

        d = bbox.to_dict()
        self.assertIn("center", d)
        self.assertEqual(d["width"], 200.0)

    def test_detected_object_serialization(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=100.0)
        obj = DetectedObject(
            object_id=1,
            label="bottle",
            confidence=0.92,
            bbox=bbox,
            spatial_sector=SpatialSector.RIGHT,
            proximity=ProximityLevel.NEAR,
            area_ratio=0.18,
        )
        obj_dict = obj.to_dict()
        self.assertEqual(obj_dict["label"], "bottle")
        self.assertEqual(obj_dict["spatial_sector"], "RIGHT")
        self.assertEqual(obj_dict["proximity"], "NEAR")

    def test_vision_agent_output_json(self):
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        obj = DetectedObject(
            object_id=1,
            label="cup",
            confidence=0.88,
            bbox=bbox,
            spatial_sector=SpatialSector.CENTER,
            proximity=ProximityLevel.MEDIUM,
            area_ratio=0.08,
        )
        output = VisionAgentOutput(
            agent_name="VisionAgent",
            agent_type=AgentType.VISION.value,
            frame_id=1,
            image_width=640,
            image_height=480,
            detected_objects=[obj],
            target_candidates=["cup"],
            spatial_breakdown={"CENTER": ["cup"]},
            summary_text="Visual scene contains 1 cup in the center.",
        )
        json_str = output.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["frame_id"], 1)
        self.assertEqual(len(parsed["detected_objects"]), 1)
        self.assertEqual(parsed["detected_objects"][0]["label"], "cup")
        self.assertEqual(parsed["spatial_breakdown"]["CENTER"], ["cup"])


if __name__ == "__main__":
    unittest.main()
