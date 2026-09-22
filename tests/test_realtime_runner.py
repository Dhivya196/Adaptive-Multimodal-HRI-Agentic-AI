"""Unit and integration tests for the Real-Time HRI Demonstration Runner."""

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.run_realtime_hri import RealTimeHRIOrchestrator
from src.agents.controller.schemas import ControllerCommand
from src.agents.coordinator.schemas import TaskStatus
from src.agents.vision.schemas import (
    BoundingBox,
    DetectedObject,
    ProximityLevel,
    SpatialSector,
    VisionAgentOutput,
)
from src.agents.gesture.schemas import GestureAgentOutput
from src.agents.voice.schemas import AudioFormat, SpeechIntent, UrgencyLevel, VoiceAgentOutput
from src.utils.audio_capture import MicrophoneRecorder


class TestRealTimeHRIOrchestrator(unittest.TestCase):
    """Tests for RealTimeHRIOrchestrator multi-agent coordination."""

    def setUp(self):
        self.orchestrator = RealTimeHRIOrchestrator(
            mode="simulation",
            cmd_vel_topic="/cmd_vel",
            max_vision_age=3.0,
            enable_gui=False,
        )

    def tearDown(self):
        self.orchestrator.shutdown()

    def test_initialization_and_agent_status(self):
        """Verify all 7 agents are initialized cleanly."""
        self.assertIsNotNone(self.orchestrator.vision_agent)
        self.assertIsNotNone(self.orchestrator.voice_agent)
        self.assertIsNotNone(self.orchestrator.coordinator_agent)
        self.assertIsNotNone(self.orchestrator.memory_agent)
        self.assertIsNotNone(self.orchestrator.planner_agent)
        self.assertIsNotNone(self.orchestrator.safety_agent)
        self.assertIsNotNone(self.orchestrator.controller_agent)
        self.assertEqual(self.orchestrator.mode, "simulation")

    def test_single_text_command_pick_and_place_flow(self):
        """Test full end-to-end execution of 'pick up the bottle' with valid synthetic vision."""
        # Inject fresh synthetic vision output with a bottle
        bottle_obj = DetectedObject(
            object_id=1,
            label="bottle",
            confidence=0.92,
            bbox=BoundingBox(100, 100, 200, 300),
            spatial_sector=SpatialSector.CENTER,
            proximity=ProximityLevel.NEAR,
        )
        mock_vision = VisionAgentOutput(
            agent_name="VisionAgent",
            agent_type="vision_agent",
            success=True,
            confidence=0.92,
            detected_objects=[bottle_obj],
        )

        voice_output = VoiceAgentOutput(
            transcript="pick up the bottle",
            confidence=0.95,
            is_speech_detected=True,
            speech_intent=SpeechIntent(
                action="pick_and_place",
                target_object="bottle",
                spatial_sector="CENTER",
                proximity_cue="NEAR",
                urgency=UrgencyLevel.NORMAL,
                confidence=0.95,
            ),
        )

        result = self.orchestrator.execute_command_pipeline(
            voice_output=voice_output,
            manual_vision=mock_vision,
            safety_override_dist=1.2,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["fused_task"].action, "pick_and_place")
        self.assertEqual(result["resolved_target"], "bottle")
        self.assertGreater(len(result["step_logs"]), 0)

    def test_pronoun_memory_resolution_two_turns(self):
        """Verify Turn 1 'go to the bottle' followed by Turn 2 'pick it up' resolves 'it' -> 'bottle'."""
        bottle_obj = DetectedObject(
            object_id=1,
            label="bottle",
            confidence=0.90,
            bbox=BoundingBox(150, 150, 250, 350),
            spatial_sector=SpatialSector.CENTER,
            proximity=ProximityLevel.MEDIUM,
        )
        mock_vision = VisionAgentOutput(
            agent_name="VisionAgent",
            agent_type="vision_agent",
            success=True,
            confidence=0.90,
            detected_objects=[bottle_obj],
        )

        # Turn 1: Explicit target "bottle"
        voice_turn1 = VoiceAgentOutput(
            transcript="go to the bottle",
            confidence=0.92,
            is_speech_detected=True,
            speech_intent=SpeechIntent(
                action="navigate_to",
                target_object="bottle",
                urgency=UrgencyLevel.NORMAL,
                confidence=0.92,
            ),
        )
        res1 = self.orchestrator.execute_command_pipeline(voice_turn1, manual_vision=mock_vision)
        self.assertTrue(res1["success"])
        self.assertEqual(res1["resolved_target"], "bottle")

        # Turn 2: Ambiguous pronoun target "it"
        voice_turn2 = VoiceAgentOutput(
            transcript="pick it up",
            confidence=0.91,
            is_speech_detected=True,
            speech_intent=SpeechIntent(
                action="pick_and_place",
                target_object="it",
                urgency=UrgencyLevel.NORMAL,
                confidence=0.91,
            ),
        )
        res2 = self.orchestrator.execute_command_pipeline(voice_turn2, manual_vision=mock_vision)
        self.assertTrue(res2["success"])
        # Pronoun 'it' should be grounded to 'bottle' from Turn 1 history
        self.assertEqual(res2["resolved_target"], "bottle")

    def test_safety_gate_blocks_on_danger_obstacle(self):
        """Verify hard safety constraint (distance 0.20m < 0.35m) blocks execution and commands STOP."""
        bottle_obj = DetectedObject(
            object_id=1,
            label="bottle",
            confidence=0.95,
            bbox=BoundingBox(100, 100, 200, 300),
            spatial_sector=SpatialSector.CENTER,
            proximity=ProximityLevel.NEAR,
        )
        mock_vision = VisionAgentOutput(
            agent_name="VisionAgent",
            agent_type="vision_agent",
            success=True,
            confidence=0.95,
            detected_objects=[bottle_obj],
        )

        voice_output = VoiceAgentOutput(
            transcript="move forward",
            confidence=0.95,
            is_speech_detected=True,
            speech_intent=SpeechIntent(
                action="navigate_to",
                target_object="bottle",
                urgency=UrgencyLevel.NORMAL,
                confidence=0.95,
            ),
        )

        # Inject danger proximity distance: 0.20m (< danger threshold 0.35m)
        result = self.orchestrator.execute_command_pipeline(
            voice_output=voice_output,
            manual_vision=mock_vision,
            safety_override_dist=0.20,
        )

        self.assertFalse(result["success"])
        # Check that the step log records SAFETY_REJECTED
        statuses = [s["status"] for s in result["step_logs"]]
        self.assertIn("SAFETY_REJECTED", statuses)

    def test_stale_vision_rejection(self):
        """Verify stale vision snapshot (> max_vision_age) results in rejection or unconfirmed target."""
        # Create an orchestrator with tiny max vision age
        short_orchestrator = RealTimeHRIOrchestrator(
            mode="simulation",
            max_vision_age=0.01,
            enable_gui=False,
        )
        # Sleep slightly so the background snapshot is definitely older than 0.01s
        time.sleep(0.05)

        voice_output = VoiceAgentOutput(
            transcript="pick up the bottle",
            confidence=0.90,
            is_speech_detected=True,
            speech_intent=SpeechIntent(
                action="pick_and_place",
                target_object="bottle",
                urgency=UrgencyLevel.NORMAL,
                confidence=0.90,
            ),
        )

        result = short_orchestrator.execute_command_pipeline(voice_output=voice_output)
        short_orchestrator.shutdown()

        # Without fresh vision confirmation of 'bottle', task should not proceed to successful execution
        self.assertFalse(result["success"])

    def test_ungrounded_none_target_does_not_crash(self):
        """Verify pipeline handles resolved_target=None without raising AttributeError."""
        bottle_obj = DetectedObject(
            object_id=1,
            label="bottle",
            confidence=0.90,
            bbox=BoundingBox(100, 100, 200, 300),
            spatial_sector=SpatialSector.RIGHT,
            proximity=ProximityLevel.FAR,
        )
        mock_vision = VisionAgentOutput(
            agent_name="VisionAgent",
            agent_type="vision_agent",
            success=True,
            confidence=0.90,
            detected_objects=[bottle_obj],
        )

        # Voice action pick_and_place with target_object=None
        voice_output = VoiceAgentOutput(
            transcript="pick something up",
            confidence=0.95,
            is_speech_detected=True,
            speech_intent=SpeechIntent(
                action="pick_and_place",
                target_object=None,
                urgency=UrgencyLevel.NORMAL,
                confidence=0.95,
            ),
        )

        # Non-pointing gesture STOP
        gesture_output = GestureAgentOutput(
            agent_name="GestureAgent",
            agent_type="gesture_agent",
            success=True,
            gesture="STOP",
            direction="NONE",
            confidence=0.92,
            is_gesture_detected=True,
        )

        result = self.orchestrator.execute_command_pipeline(
            voice_output=voice_output,
            manual_vision=mock_vision,
            manual_gesture=gesture_output,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result.get("resolved_target"), None)
        self.assertEqual(result.get("stage"), "PLANNER")

    def test_ros2_mode_mock_fallback(self):
        """Verify ROS2 mode initializes and falls back to mock when ROS2 node is not running."""
        ros_orchestrator = RealTimeHRIOrchestrator(
            mode="ros2",
            cmd_vel_topic="/test/cmd_vel",
            enable_gui=False,
        )
        self.assertIsNotNone(ros_orchestrator.controller_agent)
        ros_orchestrator.shutdown()


class TestMicrophoneRecorder(unittest.TestCase):
    """Tests for MicrophoneRecorder helper."""

    def test_rms_calculation(self):
        silence = np.zeros(1600, dtype=np.float32)
        self.assertAlmostEqual(MicrophoneRecorder.calculate_rms(silence), 0.0)

        tone = np.ones(1600, dtype=np.float32) * 0.5
        self.assertAlmostEqual(MicrophoneRecorder.calculate_rms(tone), 0.5, places=4)

    def test_availability_check_non_crashing(self):
        rec = MicrophoneRecorder()
        is_avail = rec.is_available()
        self.assertIsInstance(is_avail, bool)


if __name__ == "__main__":
    unittest.main()
