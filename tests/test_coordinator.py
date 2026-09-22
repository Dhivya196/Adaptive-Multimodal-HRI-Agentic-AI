"""Unit and integration tests for the Coordinator Agent, LangGraph workflow,
multimodal fusion, target validation, and confidence handling.
"""

from pathlib import Path
import pytest

from src.agents.coordinator.agent import CoordinatorAgent
from src.agents.coordinator.fusion import MultimodalFusionEngine
from src.agents.coordinator.graph import create_coordinator_graph
from src.agents.coordinator.schemas import (
    CoordinatorAgentInput,
    CoordinatorAgentOutput,
    GestureAgentOutput,
    MultimodalTask,
    TaskStatus,
)
from src.agents.vision.schemas import (
    BoundingBox,
    DetectedObject,
    ProximityLevel,
    SpatialSector,
    VisionAgentOutput,
)
from src.agents.voice.schemas import SpeechIntent, UrgencyLevel, VoiceAgentOutput
from src.common.schemas import AgentStatus, AgentType


@pytest.fixture
def coordinator_agent():
    agent = CoordinatorAgent(name="TestCoordinator")
    agent.initialize()
    return agent


@pytest.fixture
def sample_vision_output():
    return VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.88,
                bbox=BoundingBox(x1=250, y1=150, x2=390, y2=400),
                spatial_sector=SpatialSector.CENTER,
                proximity=ProximityLevel.MEDIUM,
            ),
            DetectedObject(
                object_id=2,
                label="cup",
                confidence=0.85,
                bbox=BoundingBox(x1=50, y1=100, x2=150, y2=200),
                spatial_sector=SpatialSector.LEFT,
                proximity=ProximityLevel.NEAR,
            ),
        ],
    )


# --------------------------------------------------------------------------
# 1. Coordinator initialization
# --------------------------------------------------------------------------
def test_1_coordinator_initialization():
    agent = CoordinatorAgent(name="InitCoordinator")
    assert agent.status == AgentStatus.UNINITIALIZED
    assert agent.agent_type == AgentType.COORDINATOR
    assert agent.name == "InitCoordinator"

    success = agent.initialize()
    assert success is True
    assert agent.status == AgentStatus.READY
    assert agent.graph is not None


# --------------------------------------------------------------------------
# 2. Voice + Vision successful fusion
# --------------------------------------------------------------------------
def test_2_voice_and_vision_successful_fusion(coordinator_agent, sample_vision_output):
    voice = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.92,
        speech_intent=SpeechIntent(
            action="navigate_to",
            target_object="bottle",
            spatial_sector="CENTER",
            urgency=UrgencyLevel.NORMAL,
            confidence=0.92,
        ),
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=sample_vision_output)
    output: CoordinatorAgentOutput = coordinator_agent.process(inp)

    assert output.success is True
    assert output.fused_task is not None
    assert output.fused_task.action == "navigate_to"
    assert output.fused_task.target_object == "bottle"
    assert output.fused_task.target_confirmed is True
    assert output.fused_task.spatial_sector == SpatialSector.CENTER
    assert output.fused_task.proximity == ProximityLevel.MEDIUM
    assert output.task_status == TaskStatus.VALID.value


# --------------------------------------------------------------------------
# 3. Voice-only input
# --------------------------------------------------------------------------
def test_3_voice_only_input(coordinator_agent):
    voice = VoiceAgentOutput(
        transcript="Go to the table",
        is_speech_detected=True,
        confidence=0.85,
        speech_intent=SpeechIntent(
            action="navigate_to",
            target_object="table",
            spatial_sector="RIGHT",
            urgency=UrgencyLevel.NORMAL,
            confidence=0.85,
        ),
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=None)
    output: CoordinatorAgentOutput = coordinator_agent.process(inp)

    assert output.success is True
    assert output.fused_task.action == "navigate_to"
    assert output.fused_task.target_object == "table"
    assert output.fused_task.target_confirmed is False
    assert output.task_status == TaskStatus.VALID.value


# --------------------------------------------------------------------------
# 4. Vision-only input
# --------------------------------------------------------------------------
def test_4_vision_only_input(coordinator_agent, sample_vision_output):
    inp = CoordinatorAgentInput(voice_output=None, vision_output=sample_vision_output)
    output: CoordinatorAgentOutput = coordinator_agent.process(inp)

    assert output.success is True
    assert output.fused_task.action == "none"
    assert output.task_status == TaskStatus.VALID.value


# --------------------------------------------------------------------------
# 5. Matching target
# --------------------------------------------------------------------------
def test_5_matching_target(coordinator_agent, sample_vision_output):
    voice = VoiceAgentOutput(
        transcript="Pick up the cup on the left",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(
            action="pick_and_place",
            target_object="cup",
            spatial_sector="LEFT",
            urgency=UrgencyLevel.NORMAL,
            confidence=0.90,
        ),
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=sample_vision_output)
    output: CoordinatorAgentOutput = coordinator_agent.process(inp)

    assert output.fused_task.target_confirmed is True
    assert output.fused_task.target_object == "cup"
    assert output.fused_task.spatial_sector == SpatialSector.LEFT
    assert output.fused_task.proximity == ProximityLevel.NEAR


# --------------------------------------------------------------------------
# 6. Target not found
# --------------------------------------------------------------------------
def test_6_target_not_found(coordinator_agent, sample_vision_output):
    voice = VoiceAgentOutput(
        transcript="Go to the laptop",
        is_speech_detected=True,
        confidence=0.88,
        speech_intent=SpeechIntent(
            action="navigate_to",
            target_object="laptop",  # Not in sample_vision_output
            confidence=0.88,
        ),
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=sample_vision_output)
    output: CoordinatorAgentOutput = coordinator_agent.process(inp)

    assert output.fused_task.target_confirmed is False
    assert output.fused_task.task_status == TaskStatus.TARGET_NOT_FOUND
    assert output.task_status == TaskStatus.TARGET_NOT_FOUND.value


# --------------------------------------------------------------------------
# 7. Low confidence
# --------------------------------------------------------------------------
def test_7_low_confidence(coordinator_agent):
    low_conf_vision = VisionAgentOutput(
        confidence=0.30,
        detected_objects=[
            DetectedObject(
                object_id=10,
                label="bottle",
                confidence=0.25,  # Below threshold
                bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
                spatial_sector=SpatialSector.CENTER,
            )
        ],
    )
    voice = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.35,
        speech_intent=SpeechIntent(
            action="navigate_to",
            target_object="bottle",
            confidence=0.35,
        ),
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=low_conf_vision)
    output: CoordinatorAgentOutput = coordinator_agent.process(inp)

    assert output.fused_task.task_status == TaskStatus.LOW_CONFIDENCE


# --------------------------------------------------------------------------
# 8. Modality conflict
# --------------------------------------------------------------------------
def test_8_modality_conflict(coordinator_agent):
    vision_right = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.90,
                bbox=BoundingBox(x1=500, y1=100, x2=600, y2=400),
                spatial_sector=SpatialSector.RIGHT,
            )
        ],
    )
    voice_left = VoiceAgentOutput(
        transcript="Go to the bottle on the left",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(
            action="navigate_to",
            target_object="bottle",
            spatial_sector="LEFT",  # Voice LEFT vs Vision RIGHT
            confidence=0.90,
        ),
    )
    inp = CoordinatorAgentInput(voice_output=voice_left, vision_output=vision_right)
    output: CoordinatorAgentOutput = coordinator_agent.process(inp)

    assert output.fused_task.task_status == TaskStatus.MODALITY_CONFLICT
    assert "Modality conflict" in output.fused_task.reasoning


# --------------------------------------------------------------------------
# 9. navigate_to action
# --------------------------------------------------------------------------
def test_9_action_navigate_to(coordinator_agent, sample_vision_output):
    voice = VoiceAgentOutput(
        transcript="Navigate to the bottle",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="navigate_to", target_object="bottle", confidence=0.90),
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=sample_vision_output)
    output = coordinator_agent.process(inp)

    assert output.fused_task.action == "navigate_to"
    assert output.fused_task.task_status == TaskStatus.VALID


# --------------------------------------------------------------------------
# 10. pick_and_place action
# --------------------------------------------------------------------------
def test_10_action_pick_and_place(coordinator_agent, sample_vision_output):
    voice = VoiceAgentOutput(
        transcript="Pick up the cup",
        is_speech_detected=True,
        confidence=0.95,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="cup", confidence=0.95),
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=sample_vision_output)
    output = coordinator_agent.process(inp)

    assert output.fused_task.action == "pick_and_place"
    assert output.fused_task.target_object == "cup"
    assert output.fused_task.target_confirmed is True


# --------------------------------------------------------------------------
# 11. stop_robot action
# --------------------------------------------------------------------------
def test_11_action_stop_robot(coordinator_agent, sample_vision_output):
    voice = VoiceAgentOutput(
        transcript="Emergency stop!",
        is_speech_detected=True,
        confidence=0.99,
        speech_intent=SpeechIntent(
            action="stop_robot",
            urgency=UrgencyLevel.EMERGENCY,
            confidence=0.99,
        ),
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=sample_vision_output)
    output = coordinator_agent.process(inp)

    assert output.fused_task.action == "stop_robot"
    assert output.fused_task.task_status == TaskStatus.VALID
    assert output.fused_task.target_confirmed is True


# --------------------------------------------------------------------------
# 12. Empty / invalid input
# --------------------------------------------------------------------------
def test_12_empty_input(coordinator_agent):
    inp = CoordinatorAgentInput(voice_output=None, vision_output=None, gesture_output=None)
    output = coordinator_agent.process(inp)

    assert output.success is True
    assert output.fused_task.task_status == TaskStatus.INVALID_INPUT
    assert output.task_status == TaskStatus.INVALID_INPUT.value


# --------------------------------------------------------------------------
# 13. LangGraph execution
# --------------------------------------------------------------------------
def test_13_langgraph_execution_nodes():
    graph = create_coordinator_graph()
    assert graph is not None

    initial_state = {
        "voice_output": None,
        "vision_output": None,
        "gesture_output": None,
        "task": None,
        "task_status": None,
        "metadata": {},
        "errors": [],
    }
    result = graph.invoke(initial_state)
    assert "task" in result
    assert "task_status" in result
    assert result["task_status"] == TaskStatus.INVALID_INPUT.value


# --------------------------------------------------------------------------
# 14. Gesture output = None
# --------------------------------------------------------------------------
def test_14_gesture_output_none(coordinator_agent, sample_vision_output):
    inp = CoordinatorAgentInput(
        voice_output=VoiceAgentOutput(
            transcript="Go to the bottle",
            is_speech_detected=True,
            confidence=0.9,
            speech_intent=SpeechIntent(action="navigate_to", target_object="bottle", confidence=0.9),
        ),
        vision_output=sample_vision_output,
        gesture_output=None,
    )
    output = coordinator_agent.process(inp)
    assert output.fused_task.target_confirmed is True
    assert output.fused_task.target_object == "bottle"


# --------------------------------------------------------------------------
# 15. Optional future Gesture output
# --------------------------------------------------------------------------
def test_15_optional_future_gesture_output(coordinator_agent, sample_vision_output):
    gesture = GestureAgentOutput(
        gesture="POINT_FORWARD",
        direction="CENTER",
        confidence=0.85,
        is_gesture_detected=True,
    )
    inp = CoordinatorAgentInput(
        voice_output=VoiceAgentOutput(
            transcript="Go to the bottle",
            is_speech_detected=True,
            confidence=0.9,
            speech_intent=SpeechIntent(action="navigate_to", target_object="bottle", confidence=0.9),
        ),
        vision_output=sample_vision_output,
        gesture_output=gesture,
    )
    output = coordinator_agent.process(inp)
    assert output.fused_task.target_confirmed is True
    assert "gesture" in output.fused_task.modality_contributions


# --------------------------------------------------------------------------
# 16. Coordinator output schema serialization
# --------------------------------------------------------------------------
def test_16_coordinator_output_schema_serialization(coordinator_agent, sample_vision_output):
    voice = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="navigate_to", target_object="bottle", confidence=0.90),
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=sample_vision_output)
    output = coordinator_agent.process(inp)

    out_dict = output.to_dict()
    assert isinstance(out_dict, dict)
    assert "fused_task" in out_dict
    assert "task_status" in out_dict
    assert out_dict["task_status"] == TaskStatus.VALID.value
    assert out_dict["agent_name"] == "TestCoordinator"
    assert out_dict["agent_type"] == AgentType.COORDINATOR.value


# --------------------------------------------------------------------------
# 17. TEST 1: Pointing selects object (LEFT)
# --------------------------------------------------------------------------
def test_17_pointing_selects_object_left(coordinator_agent):
    voice = VoiceAgentOutput(
        transcript="Pick it up",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.88,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
                proximity=ProximityLevel.NEAR,
            ),
            DetectedObject(
                object_id=2,
                label="phone",
                confidence=0.84,
                bbox=BoundingBox(500, 100, 600, 300),
                spatial_sector=SpatialSector.RIGHT,
                proximity=ProximityLevel.MEDIUM,
            ),
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_LEFT",
        direction="LEFT",
        confidence=0.91,
        is_gesture_detected=True,
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    output = coordinator_agent.process(inp)

    assert output.success is True
    assert output.fused_task is not None
    assert output.fused_task.action == "pick_and_place"
    assert output.fused_task.target_object == "bottle"
    assert output.fused_task.target_confirmed is True
    assert output.fused_task.spatial_sector == SpatialSector.LEFT
    assert output.fused_task.task_status == TaskStatus.VALID
    assert output.task_status == TaskStatus.VALID.value
    assert output.fused_task.metadata.get("multimodal_target_association") is True


# --------------------------------------------------------------------------
# 18. TEST 2: Right pointing
# --------------------------------------------------------------------------
def test_18_pointing_selects_object_right(coordinator_agent):
    voice = VoiceAgentOutput(
        transcript="Pick it up",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.88,
                bbox=BoundingBox(500, 100, 600, 300),
                spatial_sector=SpatialSector.RIGHT,
                proximity=ProximityLevel.NEAR,
            ),
            DetectedObject(
                object_id=2,
                label="cup",
                confidence=0.85,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
                proximity=ProximityLevel.MEDIUM,
            ),
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_RIGHT",
        direction="RIGHT",
        confidence=0.91,
        is_gesture_detected=True,
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    output = coordinator_agent.process(inp)

    assert output.fused_task.target_object == "bottle"
    assert output.fused_task.spatial_sector == SpatialSector.RIGHT
    assert output.fused_task.task_status == TaskStatus.VALID


# --------------------------------------------------------------------------
# 19. TEST 3: Center pointing
# --------------------------------------------------------------------------
def test_19_pointing_selects_object_center(coordinator_agent):
    voice = VoiceAgentOutput(
        transcript="Pick it up",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.88,
                bbox=BoundingBox(250, 100, 390, 300),
                spatial_sector=SpatialSector.CENTER,
                proximity=ProximityLevel.NEAR,
            ),
            DetectedObject(
                object_id=2,
                label="phone",
                confidence=0.84,
                bbox=BoundingBox(500, 100, 600, 300),
                spatial_sector=SpatialSector.RIGHT,
                proximity=ProximityLevel.MEDIUM,
            ),
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_FORWARD",
        direction="FORWARD",
        confidence=0.91,
        is_gesture_detected=True,
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    output = coordinator_agent.process(inp)

    assert output.fused_task.target_object == "bottle"
    assert output.fused_task.spatial_sector == SpatialSector.CENTER
    assert output.fused_task.task_status == TaskStatus.VALID


# --------------------------------------------------------------------------
# 20. TEST 4: Empty scene / no candidates detected -> TARGET_NOT_FOUND
# --------------------------------------------------------------------------
def test_20_empty_scene_target_not_found(coordinator_agent):
    voice = VoiceAgentOutput(
        transcript="Pick it up",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision_empty = VisionAgentOutput(
        confidence=0.0,
        detected_objects=[],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_LEFT",
        direction="LEFT",
        confidence=0.91,
        is_gesture_detected=True,
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=vision_empty, gesture_output=gesture)
    output = coordinator_agent.process(inp)

    assert output.fused_task.target_confirmed is False
    assert output.fused_task.task_status == TaskStatus.TARGET_NOT_FOUND
    assert output.task_status == TaskStatus.TARGET_NOT_FOUND.value


# --------------------------------------------------------------------------
# 21. TEST 5: Candidate exists in different sector -> MODALITY_CONFLICT
# --------------------------------------------------------------------------
def test_21_candidate_in_different_sector_modality_conflict(coordinator_agent):
    voice = VoiceAgentOutput(
        transcript="Pick it up",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision_right = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.88,
                bbox=BoundingBox(500, 100, 600, 300),
                spatial_sector=SpatialSector.RIGHT,
            )
        ],
    )
    gesture_left = GestureAgentOutput(
        gesture="POINT_LEFT",
        direction="LEFT",
        confidence=0.91,
        is_gesture_detected=True,
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=vision_right, gesture_output=gesture_left)
    output = coordinator_agent.process(inp)

    # Bottle exists in RIGHT, but gesture pointed LEFT -> MODALITY_CONFLICT
    assert output.fused_task.task_status == TaskStatus.MODALITY_CONFLICT
    assert output.task_status == TaskStatus.MODALITY_CONFLICT.value


# --------------------------------------------------------------------------
# 22. TEST 6: Multiple objects (Ambiguous Target)
# --------------------------------------------------------------------------
def test_22_multiple_objects_ambiguous(coordinator_agent):
    voice = VoiceAgentOutput(
        transcript="Pick it up",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.88,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
            ),
            DetectedObject(
                object_id=2,
                label="cup",
                confidence=0.85,
                bbox=BoundingBox(160, 100, 240, 300),
                spatial_sector=SpatialSector.LEFT,
            ),
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_LEFT",
        direction="LEFT",
        confidence=0.91,
        is_gesture_detected=True,
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    output = coordinator_agent.process(inp)

    assert output.fused_task.task_status == TaskStatus.AMBIGUOUS_TARGET
    assert output.task_status == TaskStatus.AMBIGUOUS_TARGET.value


# --------------------------------------------------------------------------
# 23. TEST 7: Explicit target + Gesture agreement
# --------------------------------------------------------------------------
def test_23_explicit_target_with_gesture_agreement(coordinator_agent):
    voice = VoiceAgentOutput(
        transcript="Pick up the bottle",
        is_speech_detected=True,
        confidence=0.95,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="bottle", confidence=0.95),
    )
    vision = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.88,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
            ),
            DetectedObject(
                object_id=2,
                label="cup",
                confidence=0.85,
                bbox=BoundingBox(500, 100, 600, 300),
                spatial_sector=SpatialSector.RIGHT,
            ),
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_LEFT",
        direction="LEFT",
        confidence=0.91,
        is_gesture_detected=True,
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    output = coordinator_agent.process(inp)

    assert output.fused_task.target_object == "bottle"
    assert output.fused_task.target_confirmed is True
    assert output.fused_task.spatial_sector == SpatialSector.LEFT
    assert output.fused_task.task_status == TaskStatus.VALID
    assert output.fused_task.metadata.get("gesture_confirmation") is True


# --------------------------------------------------------------------------
# 24. TEST 8: Low-confidence gesture fallback
# --------------------------------------------------------------------------
def test_24_low_confidence_gesture_fallback(coordinator_agent):
    voice = VoiceAgentOutput(
        transcript="Pick up the bottle",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="bottle", confidence=0.90),
    )
    vision = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.88,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
            )
        ],
    )
    gesture_low = GestureAgentOutput(
        gesture="POINT_RIGHT",
        direction="RIGHT",
        confidence=0.20,  # Below threshold 0.40 -> treated as unreliable
        is_gesture_detected=True,
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture_low)
    output = coordinator_agent.process(inp)

    assert output.fused_task.target_object == "bottle"
    assert output.fused_task.spatial_sector == SpatialSector.LEFT
    assert output.fused_task.task_status == TaskStatus.VALID


# --------------------------------------------------------------------------
# 25. TEST 9: Explicit target vs Gesture conflict
# --------------------------------------------------------------------------
def test_25_explicit_target_vs_gesture_conflict(coordinator_agent):
    voice = VoiceAgentOutput(
        transcript="Pick up the bottle",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="bottle", confidence=0.90),
    )
    vision = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.88,
                bbox=BoundingBox(500, 100, 600, 300),
                spatial_sector=SpatialSector.RIGHT,
            )
        ],
    )
    gesture_left = GestureAgentOutput(
        gesture="POINT_LEFT",
        direction="LEFT",
        confidence=0.91,  # High confidence conflict
        is_gesture_detected=True,
    )
    inp = CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture_left)
    output = coordinator_agent.process(inp)

    assert output.fused_task.task_status == TaskStatus.MODALITY_CONFLICT
    assert output.task_status == TaskStatus.MODALITY_CONFLICT.value


# --------------------------------------------------------------------------
# 26. TEST 10: Three-modal confidence calculation
# --------------------------------------------------------------------------
def test_26_three_modal_confidence_calculation():
    engine = MultimodalFusionEngine(
        voice_weight=0.55,
        vision_weight=0.35,
        gesture_weight=0.10,
    )
    voice = VoiceAgentOutput(
        transcript="Pick it up",
        is_speech_detected=True,
        confidence=0.80,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.80),
    )
    vision = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.90,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
            )
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_LEFT",
        direction="LEFT",
        confidence=0.70,
        is_gesture_detected=True,
    )
    task = engine.fuse(voice_output=voice, vision_output=vision, gesture_output=gesture)

    expected_conf = (0.55 * 0.80) + (0.35 * 0.90) + (0.10 * 0.70)
    assert abs(task.confidence - expected_conf) < 1e-5
    assert task.task_status == TaskStatus.VALID
    assert task.target_object == "bottle"


# --------------------------------------------------------------------------
# 27. TEST 11: LangGraph compilation and StateGraph node execution
# --------------------------------------------------------------------------
def test_27_langgraph_graph_execution_verification(coordinator_agent):
    """Verify that CoordinatorAgent uses a compiled LangGraph graph and invokes nodes."""
    assert coordinator_agent.graph is not None
    # Verify graph attribute is a compiled LangGraph Pregel/CompiledStateGraph
    graph_class_name = type(coordinator_agent.graph).__name__
    assert "Compiled" in graph_class_name or "Pregel" in graph_class_name or hasattr(coordinator_agent.graph, "invoke")

    # Invoke with three-modal inputs
    voice = VoiceAgentOutput(
        transcript="Pick it up",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.88,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
            )
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_LEFT",
        direction="LEFT",
        confidence=0.91,
        is_gesture_detected=True,
    )

    inp = CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    out = coordinator_agent.process(inp)

    assert out.success is True
    assert "input_validation" not in out.hri_state or out.hri_state is not None
    assert out.fused_task.target_object == "bottle"
    assert out.fused_task.task_status == TaskStatus.VALID
