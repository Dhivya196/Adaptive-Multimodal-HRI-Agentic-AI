"""Integration tests for the End-to-End Multi-Agent HRI Pipeline with LangGraph.
Tests full data flow across Perception -> Coordinator -> Memory -> Planner -> Safety -> Controller.
"""

from typing import Any, Dict, Optional
import pytest

from src.agents.controller.agent import RobotControllerAgent
from src.agents.controller.controller import RobotController
from src.agents.controller.ros2_interface import MockRobotController
from src.agents.controller.schemas import ExecutionStatus
from src.agents.coordinator.fusion import MultimodalFusionEngine
from src.agents.coordinator.schemas import (
    CoordinatorAgentInput,
    HRIState,
    MultimodalTask,
    TaskStatus,
)
from src.agents.gesture.schemas import (
    GestureAgentOutput,
    GestureDirection,
    GestureType,
    HandBBox,
    RecognizedGesture,
)
from src.agents.memory.agent import MemoryAgent
from src.agents.memory.memory_store import InMemoryStore
from src.agents.planner.agent import TaskPlannerAgent
from src.agents.safety.agent import SafetyAgent
from src.agents.safety.human_intervention import AutoApprovalStub
from src.agents.safety.llm_reasoner import MockLLMReasoner
from src.agents.safety.schemas import RiskLevel, SafetyDecisionType
from src.agents.vision.schemas import (
    BoundingBox,
    DetectedObject,
    ProximityLevel,
    SpatialSector,
    VisionAgentOutput,
)
from src.agents.voice.schemas import SpeechIntent, UrgencyLevel, VoiceAgentOutput
from src.pipeline.coordinator_integration import (
    IntegratedHRIPipeline,
    create_integrated_hri_graph,
)


@pytest.fixture
def integrated_pipeline():
    """Build an integrated pipeline with clean in-memory test agents."""
    memory = MemoryAgent(name="TestMemory", store=InMemoryStore())
    planner = TaskPlannerAgent(name="TestPlanner")
    safety = SafetyAgent(
        name="TestSafety",
        human_interface=AutoApprovalStub(auto_approve=True),
        config={
            "safety": {
                "danger_distance": 0.35,
                "safe_distance": 0.50,
                "confidence_threshold": 0.60,
                "sensor_policy": "human_approval_on_missing",
            }
        },
    )
    controller_backend = RobotController()
    controller = RobotControllerAgent(name="TestController", controller=controller_backend)

    pipeline = IntegratedHRIPipeline(
        memory_agent=memory,
        planner_agent=planner,
        safety_agent=safety,
        controller_agent=controller,
    )
    return pipeline


# --------------------------------------------------------------------------
# TEST 1: Navigation ("Go to the bottle" + bottle detected)
# --------------------------------------------------------------------------
def test_1_navigation_successful_flow(integrated_pipeline):
    voice_out = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.92,
        speech_intent=SpeechIntent(
            action="navigate_to",
            target_object="bottle",
            confidence=0.92,
        ),
    )
    vision_out = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.89,
                bbox=BoundingBox(x1=200, y1=100, x2=300, y2=400),
                spatial_sector=SpatialSector.CENTER,
                proximity=ProximityLevel.MEDIUM,
            )
        ],
    )

    state_in = {
        "voice_output": voice_out,
        "vision_output": vision_out,
        "obstacle_distance": 1.20,
    }

    final_state = integrated_pipeline.process(state_in)

    # 1. Coordinator check
    assert final_state["task"] is not None
    assert final_state["task"].action == "navigate_to"
    assert final_state["task"].target_object == "bottle"
    assert final_state["task"].task_status == TaskStatus.VALID

    # 2. Planner check
    assert final_state["task_plan"] is not None
    assert len(final_state["task_plan"].steps) > 0

    # 3. Safety check
    assert final_state["safety_output"] is not None
    assert final_state["safety_output"].approved_for_execution is True
    assert final_state["safety_decision"].decision == SafetyDecisionType.SAFE

    # 4. Controller check
    assert final_state["controller_output"] is not None
    assert final_state["controller_output"].success is True
    assert final_state["execution_result"].status == ExecutionStatus.EXECUTED


# --------------------------------------------------------------------------
# TEST 2: Pick and Place ("Pick up the phone" + phone detected)
# --------------------------------------------------------------------------
def test_2_pick_and_place_flow(integrated_pipeline):
    voice_out = VoiceAgentOutput(
        transcript="Pick up the phone",
        is_speech_detected=True,
        confidence=0.95,
        speech_intent=SpeechIntent(
            action="pick_and_place",
            target_object="phone",
            confidence=0.95,
        ),
    )
    vision_out = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="phone",
                confidence=0.91,
                bbox=BoundingBox(x1=50, y1=120, x2=150, y2=250),
                spatial_sector=SpatialSector.LEFT,
                proximity=ProximityLevel.NEAR,
            )
        ],
    )

    state_in = {
        "voice_output": voice_out,
        "vision_output": vision_out,
        "obstacle_distance": 0.80,
    }

    final_state = integrated_pipeline.process(state_in)

    assert final_state["task"].action == "pick_and_place"
    assert final_state["task"].target_object == "phone"
    assert final_state["safety_output"].approved_for_execution is True
    assert final_state["controller_output"] is not None
    assert final_state["execution_result"].command in ("PICK", "MOVE_FORWARD", "TURN_LEFT")
    assert final_state["execution_result"].status == ExecutionStatus.EXECUTED


# --------------------------------------------------------------------------
# TEST 3: Target Requested but NOT Detected -> Controller must NOT execute
# --------------------------------------------------------------------------
def test_3_target_not_detected_no_execution(integrated_pipeline):
    voice_out = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(
            action="navigate_to",
            target_object="bottle",
            confidence=0.90,
        ),
    )
    vision_out = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="cup",  # Only a cup, no bottle!
                confidence=0.85,
                bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150),
            )
        ],
    )

    state_in = {
        "voice_output": voice_out,
        "vision_output": vision_out,
        "obstacle_distance": 1.50,
    }

    final_state = integrated_pipeline.process(state_in)

    # Coordinator should mark TARGET_NOT_FOUND
    assert final_state["task"].task_status == TaskStatus.TARGET_NOT_FOUND
    assert final_state["task"].target_confirmed is False

    # Planning should be skipped
    assert final_state["task_plan"] is None

    # Robot Controller must NOT have executed
    assert final_state.get("controller_output") is None or final_state["controller_output"].success is False
    assert final_state.get("execution_result") is None or final_state["execution_result"].status != ExecutionStatus.EXECUTED


# --------------------------------------------------------------------------
# TEST 4: Hard Safety Violation (Obstacle < 0.35m) -> Safety STOP, Controller NOT CALLED
# --------------------------------------------------------------------------
def test_4_hard_safety_violation_stop(integrated_pipeline):
    voice_out = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.92,
        speech_intent=SpeechIntent(
            action="navigate_to",
            target_object="bottle",
            confidence=0.92,
        ),
    )
    vision_out = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.90,
                bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            )
        ],
    )

    # Danger distance: 0.18m strictly < 0.35m hard boundary
    state_in = {
        "voice_output": voice_out,
        "vision_output": vision_out,
        "obstacle_distance": 0.18,
    }

    final_state = integrated_pipeline.process(state_in)

    # Safety decision must be STOP and hard violation
    assert final_state["safety_decision"].decision == SafetyDecisionType.STOP
    assert final_state["safety_decision"].is_hard_violation is True
    assert final_state["safety_output"].approved_for_execution is False

    # Controller must NOT be executed
    assert final_state.get("controller_output") is None or final_state["controller_output"].success is False
    assert final_state.get("execution_result") is None or final_state["execution_result"].status != ExecutionStatus.EXECUTED


# --------------------------------------------------------------------------
# TEST 5: Human Approval Routing (Test both Approval and Rejection)
# --------------------------------------------------------------------------
def test_5a_human_approval_approved_path():
    # Low perception confidence triggers HUMAN_APPROVAL_REQUIRED
    approve_stub = AutoApprovalStub(auto_approve=True, approver_id="supervisor_alice")
    safety = SafetyAgent(
        name="ApprovalSafety",
        human_interface=approve_stub,
        config={"safety": {"confidence_threshold": 0.60}},
    )
    controller = RobotControllerAgent(name="ApprovalController", controller=RobotController())

    pipeline = IntegratedHRIPipeline(
        safety_agent=safety,
        controller_agent=controller,
    )

    voice_out = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.45,  # Low confidence < 0.60
        speech_intent=SpeechIntent(action="navigate_to", target_object="bottle", confidence=0.45),
    )
    vision_out = VisionAgentOutput(
        confidence=0.45,
        detected_objects=[
            DetectedObject(object_id=1, label="bottle", confidence=0.45, bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50))
        ],
    )

    final_state = pipeline.process({
        "voice_output": voice_out,
        "vision_output": vision_out,
        "obstacle_distance": 1.20,
        "target_confidence": 0.45,
    })

    assert final_state["human_intervention_required"] is True
    assert final_state["human_intervention_result"] is not None
    assert final_state["human_intervention_result"].approved is True
    assert final_state["controller_output"] is not None
    assert final_state["execution_result"].status == ExecutionStatus.EXECUTED


def test_5b_human_approval_rejected_path():
    # Human rejects approval
    reject_stub = AutoApprovalStub(auto_approve=False, approver_id="supervisor_bob")
    safety = SafetyAgent(
        name="RejectSafety",
        human_interface=reject_stub,
        config={"safety": {"confidence_threshold": 0.60}},
    )
    controller = RobotControllerAgent(name="RejectController", controller=RobotController())

    pipeline = IntegratedHRIPipeline(
        safety_agent=safety,
        controller_agent=controller,
    )

    voice_out = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.45,
        speech_intent=SpeechIntent(action="navigate_to", target_object="bottle", confidence=0.45),
    )
    vision_out = VisionAgentOutput(
        confidence=0.45,
        detected_objects=[
            DetectedObject(object_id=1, label="bottle", confidence=0.45, bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50))
        ],
    )

    final_state = pipeline.process({
        "voice_output": voice_out,
        "vision_output": vision_out,
        "obstacle_distance": 1.20,
        "target_confidence": 0.45,
    })

    assert final_state["human_intervention_result"] is not None
    assert final_state["human_intervention_result"].approved is False
    # Controller must NOT execute
    assert final_state.get("controller_output") is None or final_state["controller_output"].success is False


# --------------------------------------------------------------------------
# TEST 6: Memory Context (Referential Pronoun Resolution "pick it up")
# --------------------------------------------------------------------------
def test_6_memory_context_pronoun_resolution(integrated_pipeline):
    # Step 1: Pre-populate memory with previous interaction "Go to the bottle"
    integrated_pipeline.memory_agent.store_entry(
        content={
            "action": "navigate_to",
            "target": "bottle",
            "location": "LEFT",
            "status": "COMPLETED",
        },
    )

    # Step 2: User says "Pick it up"
    voice_it = VoiceAgentOutput(
        transcript="Pick it up",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(
            action="pick_and_place",
            target_object="it",  # Pronoun "it"!
            confidence=0.90,
        ),
    )

    final_state = integrated_pipeline.process({
        "voice_output": voice_it,
        "vision_output": None,
        "obstacle_distance": 0.90,
    })

    # Memory node should resolve "it" to "bottle"
    assert final_state["task"].target_object == "bottle"
    assert final_state["task"].target_confirmed is True
    assert final_state["task"].task_status == TaskStatus.VALID
    assert final_state["task_plan"] is not None
    assert final_state["task_plan"].target == "bottle"


# --------------------------------------------------------------------------
# TEST 7: Multimodal Fusion (Voice + Vision + Gesture)
# --------------------------------------------------------------------------
def test_7_multimodal_fusion_gesture_voice_vision(integrated_pipeline):
    voice_out = VoiceAgentOutput(
        transcript="Bring me that object",
        is_speech_detected=True,
        confidence=0.88,
        speech_intent=SpeechIntent(
            action="pick_and_place",
            target_object="bottle",
            confidence=0.88,
        ),
    )
    vision_out = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.85,
                bbox=BoundingBox(x1=350, y1=100, x2=450, y2=300),
                spatial_sector=SpatialSector.RIGHT,
                proximity=ProximityLevel.NEAR,
            )
        ],
    )
    gesture_out = GestureAgentOutput(
        gesture=GestureType.POINT_RIGHT.value,
        direction=GestureDirection.RIGHT.value,
        confidence=0.92,
        is_gesture_detected=True,
        recognized_gestures=[
            RecognizedGesture(
                gesture=GestureType.POINT_RIGHT,
                direction=GestureDirection.RIGHT,
                confidence=0.92,
                bbox=HandBBox(x1=100, y1=100, x2=200, y2=200),
            )
        ],
    )

    state_in = {
        "voice_output": voice_out,
        "vision_output": vision_out,
        "gesture_output": gesture_out,
        "obstacle_distance": 1.0,
    }

    final_state = integrated_pipeline.process(state_in)

    assert final_state["task"].action == "pick_and_place"
    assert final_state["task"].target_object == "bottle"
    assert final_state["task"].spatial_sector == SpatialSector.RIGHT
    assert final_state["controller_output"].success is True


# --------------------------------------------------------------------------
# TEST 8: Controller Failure Handling (Safe termination, no crashes)
# --------------------------------------------------------------------------
def test_8_controller_failure_safe_handling():
    # Controller with disconnected backend
    ctrl_backend = RobotController()
    ctrl_backend.backend.connected = False  # Disconnect backend to trigger ERROR status

    controller = RobotControllerAgent(
        name="FaultyController",
        controller=ctrl_backend,
    )
    safety = SafetyAgent(
        name="Safety",
        human_interface=AutoApprovalStub(auto_approve=True),
    )

    pipeline = IntegratedHRIPipeline(
        safety_agent=safety,
        controller_agent=controller,
    )

    voice_out = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="navigate_to", target_object="bottle", confidence=0.90),
    )
    vision_out = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(object_id=1, label="bottle", confidence=0.90, bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50))
        ],
    )

    final_state = pipeline.process({
        "voice_output": voice_out,
        "vision_output": vision_out,
        "obstacle_distance": 1.20,
    })

    # Execution result must indicate failure
    assert final_state["controller_output"].success is False
    assert final_state["execution_result"].status == ExecutionStatus.ERROR
    assert len(final_state["errors"]) > 0


# --------------------------------------------------------------------------
# TEST 9: Invalid / Unknown Safety Decision -> Failsafe STOP
# --------------------------------------------------------------------------
def test_9_invalid_or_unknown_safety_decision_failsafe():
    from src.pipeline.coordinator_integration import create_integrated_hri_graph
    from src.agents.safety.schemas import SafetyDecision, SafetyAgentOutput

    # Compile graph with a mock safety agent that returns an unapproved/invalid safety decision
    mock_bad_safety = SafetyAgent(
        name="BadSafety",
        human_interface=AutoApprovalStub(auto_approve=False),
    )
    mock_bad_safety.process = lambda inp: SafetyAgentOutput(
        agent_name="BadSafety",
        agent_type="safety",
        success=True,
        safety_decision=SafetyDecision(
            decision="UNKNOWN_INVALID_DECISION",
            risk_level=RiskLevel.CRITICAL,
            reason="Corrupted decision payload",
        ),
        approved_for_execution=False,
    )

    graph = create_integrated_hri_graph(safety_agent=mock_bad_safety)

    voice_out = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="navigate_to", target_object="bottle", confidence=0.90),
    )

    initial_state = {
        "voice_output": voice_out,
        "vision_output": None,
    }

    final_state = graph.invoke(initial_state)

    # Controller must NOT have executed
    assert final_state.get("controller_output") is None or final_state["controller_output"].success is False
    assert final_state.get("execution_result") is None or final_state["execution_result"].status != ExecutionStatus.EXECUTED
