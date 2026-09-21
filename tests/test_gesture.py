"""Focused tests for GestureAgent and its deterministic recognizer backends."""

import unittest

import numpy as np

from src.agents.gesture.agent import GestureAgent
from src.agents.gesture.recognizer import MockGestureRecognizer
from src.agents.gesture.schemas import GestureAgentInput, GestureDirection, GestureType
from src.common.schemas import AgentStatus, AgentType


class TestGestureAgent(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()