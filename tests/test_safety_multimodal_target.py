"""Deterministic test suite for Safety-Aware Multimodal Target Selection.

Tests:
TEST 1: "Pick it up" + POINT_RIGHT + bottle RIGHT (0.88) -> target=bottle -> HIGH_CONFIDENCE -> Safety SAFE
TEST 2: "Pick it up" + POINT_RIGHT + plant RIGHT (0.49) -> target=plant -> LOW_CONFIDENCE_REFERENTIAL_GROUNDING -> HUMAN_CONFIRMATION_REQUIRED
TEST 3: "Pick it up" + POINT_RIGHT + bottle LEFT (0.85), plant RIGHT (0.49) -> candidate=plant -> LOW_CONFIDENCE -> HUMAN_CONFIRMATION_REQUIRED (MUST NOT pick bottle)
TEST 4: "Pick up the bottle" + bottle LEFT (0.85) + POINT_LEFT -> target=bottle, explicit target, spatial agreement
TEST 5: "Pick up the bottle" + bottle LEFT (0.85), plant RIGHT (0.90) + POINT_RIGHT -> target=bottle, explicit target preserved, spatial conflict flagged
TEST 6: "Pick it up" + POINT_RIGHT + bottle RIGHT (0.61), cup RIGHT (0.59) -> AMBIGUOUS -> HUMAN_CONFIRMATION_REQUIRED
TEST 7: "Pick it up" + no gesture + multiple objects -> unresolved/ambiguous -> HUMAN_CONFIRMATION_REQUIRED
TEST 8: High-confidence deictic target -> verify Safety does NOT unnecessarily require human confirmation
"""

import pytest

from src.agents.coordinator.agent import CoordinatorAgent
from src.agents.coordinator.fusion import MultimodalFusionEngine
from src.agents.coordinator.schemas import (
    CoordinatorAgentInput,
    CoordinatorAgentOutput,
    GroundingStatus,
    TaskStatus,
)
from src.agents.gesture.schemas import GestureAgentOutput
from src.agents.safety.agent import SafetyAgent
from src.agents.safety.human_intervention import AutoApprovalStub
from src.agents.safety.schemas import (
    RiskLevel,
    SafetyAgentInput,
    SafetyAgentOutput,
    SafetyDecisionType,
)
from src.agents.vision.schemas import (
    BoundingBox,
    DetectedObject,
    ProximityLevel,
    SpatialSector,
    VisionAgentOutput,
)
from src.agents.voice.schemas import SpeechIntent, UrgencyLevel, VoiceAgentOutput


@pytest.fixture
def coordinator_agent():
    agent = CoordinatorAgent(
        name="TestSafetyAwareCoordinator",
        config={
            "coordinator": {
                "min_confidence_threshold": 0.40,
                "high_confidence_threshold": 0.60,
                "ambiguity_threshold": 0.10,
                "voice_weight": 0.55,
                "vision_weight": 0.35,
                "gesture_weight": 0.10,
            }
        },
    )
    agent.initialize()
    return agent


@pytest.fixture
def safety_agent_no_auto_approve():
    # Safety agent with human_intervention_enabled=False to check raw gating decision before human override
    agent = SafetyAgent(
        name="TestGatingSafety",
        config={"safety": {"confidence_threshold": 0.60, "human_intervention_enabled": False}},
    )
    agent.initialize()
    return agent


# --------------------------------------------------------------------------
# TEST 1: High-Confidence Deictic Target Selection
# --------------------------------------------------------------------------
def test_1_high_confidence_deictic_target(coordinator_agent, safety_agent_no_auto_approve):
    """
    Voice: "Pick it up."
    Gesture: POINT_RIGHT (0.92)
    Vision: bottle RIGHT (0.88), plant CENTER (0.72)
    -> target = bottle, HIGH_CONFIDENCE -> Safety can proceed (SAFE)
    """
    voice = VoiceAgentOutput(
        transcript="Pick it up.",
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
                label="plant",
                confidence=0.72,
                bbox=BoundingBox(250, 100, 350, 300),
                spatial_sector=SpatialSector.CENTER,
                proximity=ProximityLevel.MEDIUM,
            ),
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_RIGHT",
        direction="RIGHT",
        confidence=0.92,
        is_gesture_detected=True,
    )

    coord_out: CoordinatorAgentOutput = coordinator_agent.process(
        CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    )

    task = coord_out.fused_task
    assert task is not None
    assert task.target_object == "bottle"
    assert task.spatial_sector == SpatialSector.RIGHT
    assert task.target_confidence == 0.88
    assert task.grounding_status == GroundingStatus.HIGH_CONFIDENCE
    assert task.spatial_agreement is True
    assert task.task_status == TaskStatus.VALID

    # Safety Gating Evaluation
    safety_in = SafetyAgentInput(task=task, obstacle_distance=1.20)
    safety_out: SafetyAgentOutput = safety_agent_no_auto_approve.process(safety_in)

    assert safety_out.safety_decision.decision == SafetyDecisionType.SAFE
    assert safety_out.approved_for_execution is True
    assert safety_out.safety_decision.human_intervention_required is False


# --------------------------------------------------------------------------
# TEST 2: Low-Confidence Deictic Target Selection
# --------------------------------------------------------------------------
def test_2_low_confidence_deictic_target(coordinator_agent, safety_agent_no_auto_approve):
    """
    Voice: "Pick it up."
    Gesture: POINT_RIGHT (0.92)
    Vision: potted plant RIGHT (0.49)
    -> target = potted plant candidate
    -> LOW_CONFIDENCE_REFERENTIAL_GROUNDING -> Safety: HUMAN_CONFIRMATION_REQUIRED
    """
    voice = VoiceAgentOutput(
        transcript="Pick it up.",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision = VisionAgentOutput(
        confidence=0.49,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="potted plant",
                confidence=0.49,
                bbox=BoundingBox(500, 100, 600, 300),
                spatial_sector=SpatialSector.RIGHT,
                proximity=ProximityLevel.NEAR,
            )
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_RIGHT",
        direction="RIGHT",
        confidence=0.92,
        is_gesture_detected=True,
    )

    coord_out: CoordinatorAgentOutput = coordinator_agent.process(
        CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    )

    task = coord_out.fused_task
    assert task is not None
    assert task.target_object == "potted plant"
    assert task.target_confidence == 0.49
    assert task.spatial_sector == SpatialSector.RIGHT
    assert task.spatial_agreement is True
    assert task.grounding_status in (
        GroundingStatus.LOW_CONFIDENCE,
        GroundingStatus.LOW_CONFIDENCE_REFERENTIAL_GROUNDING,
        GroundingStatus.REQUIRES_CONFIRMATION,
    )
    assert task.referential_grounding_status == "LOW_CONFIDENCE_REFERENTIAL_GROUNDING"

    # Safety Gating Evaluation
    safety_in = SafetyAgentInput(task=task, obstacle_distance=1.20)
    safety_out: SafetyAgentOutput = safety_agent_no_auto_approve.process(safety_in)

    # Must NOT automatically proceed
    assert safety_out.safety_decision.decision in (
        SafetyDecisionType.HUMAN_APPROVAL_REQUIRED,
        SafetyDecisionType.HUMAN_CONFIRMATION_REQUIRED,
    )
    assert safety_out.approved_for_execution is False
    assert safety_out.safety_decision.human_intervention_required is True
    assert "below the autonomous grounding threshold" in safety_out.safety_decision.reason


# --------------------------------------------------------------------------
# TEST 3: Competing Objects - Inspect All Candidates (Do NOT choose highest YOLO)
# --------------------------------------------------------------------------
def test_3_competing_objects_proper_referential_selection(coordinator_agent, safety_agent_no_auto_approve):
    """
    Voice: "Pick it up."
    Gesture: POINT_RIGHT (0.92)
    Vision: bottle LEFT (0.85), potted plant RIGHT (0.49)
    -> candidate MUST be plant because gesture points RIGHT (bottle has poor spatial agreement)
    -> but plant is low confidence (0.49 < 0.60)
    -> LOW_CONFIDENCE_REFERENTIAL_GROUNDING -> Safety: HUMAN_CONFIRMATION_REQUIRED
    -> MUST NOT automatically pick plant or mistakenly choose bottle
    """
    voice = VoiceAgentOutput(
        transcript="Pick it up.",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision = VisionAgentOutput(
        confidence=0.85,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.85,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
                proximity=ProximityLevel.NEAR,
            ),
            DetectedObject(
                object_id=2,
                label="potted plant",
                confidence=0.49,
                bbox=BoundingBox(500, 100, 600, 300),
                spatial_sector=SpatialSector.RIGHT,
                proximity=ProximityLevel.NEAR,
            ),
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_RIGHT",
        direction="RIGHT",
        confidence=0.92,
        is_gesture_detected=True,
    )

    coord_out: CoordinatorAgentOutput = coordinator_agent.process(
        CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    )

    task = coord_out.fused_task
    assert task is not None
    # Candidate must be plant in pointed sector RIGHT, NOT bottle on LEFT
    assert task.target_object == "potted plant"
    assert task.spatial_sector == SpatialSector.RIGHT
    assert task.target_confidence == 0.49
    assert task.referential_grounding_status == "LOW_CONFIDENCE_REFERENTIAL_GROUNDING"

    # Safety Evaluation
    safety_in = SafetyAgentInput(task=task, obstacle_distance=1.20)
    safety_out: SafetyAgentOutput = safety_agent_no_auto_approve.process(safety_in)

    # Must NOT automatically pick plant
    assert safety_out.safety_decision.decision in (
        SafetyDecisionType.HUMAN_APPROVAL_REQUIRED,
        SafetyDecisionType.HUMAN_CONFIRMATION_REQUIRED,
    )
    assert safety_out.approved_for_execution is False
    assert safety_out.safety_decision.human_intervention_required is True


# --------------------------------------------------------------------------
# TEST 4: Explicit Target with Spatial Agreement
# --------------------------------------------------------------------------
def test_4_explicit_target_with_spatial_agreement(coordinator_agent, safety_agent_no_auto_approve):
    """
    Voice: "Pick up the bottle."
    Vision: bottle LEFT (0.85)
    Gesture: POINT_LEFT (0.90)
    -> target = bottle, explicit target, spatial agreement = True -> Safety SAFE
    """
    voice = VoiceAgentOutput(
        transcript="Pick up the bottle.",
        is_speech_detected=True,
        confidence=0.92,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="bottle", confidence=0.92),
    )
    vision = VisionAgentOutput(
        confidence=0.85,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.85,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
                proximity=ProximityLevel.NEAR,
            )
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_LEFT",
        direction="LEFT",
        confidence=0.90,
        is_gesture_detected=True,
    )

    coord_out = coordinator_agent.process(
        CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    )

    task = coord_out.fused_task
    assert task.target_object == "bottle"
    assert task.target_confirmed is True
    assert task.target_confidence == 0.85
    assert task.spatial_sector == SpatialSector.LEFT
    assert task.spatial_agreement is True
    assert task.task_status == TaskStatus.VALID
    assert task.grounding_status == GroundingStatus.HIGH_CONFIDENCE

    safety_out = safety_agent_no_auto_approve.process(SafetyAgentInput(task=task, obstacle_distance=1.0))
    assert safety_out.safety_decision.decision == SafetyDecisionType.SAFE
    assert safety_out.approved_for_execution is True


# --------------------------------------------------------------------------
# TEST 5: Explicit Target with Spatial Conflict (Do NOT replace target)
# --------------------------------------------------------------------------
def test_5_explicit_target_with_spatial_conflict(coordinator_agent, safety_agent_no_auto_approve):
    """
    Voice: "Pick up the bottle."
    Vision: bottle LEFT (0.85), plant RIGHT (0.90)
    Gesture: POINT_RIGHT (0.90)
    -> Target MUST remain bottle (do not replace with plant)
    -> Flag spatial conflict -> Safety: HUMAN_CONFIRMATION_REQUIRED
    """
    voice = VoiceAgentOutput(
        transcript="Pick up the bottle.",
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
                confidence=0.85,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
                proximity=ProximityLevel.NEAR,
            ),
            DetectedObject(
                object_id=2,
                label="plant",
                confidence=0.90,
                bbox=BoundingBox(500, 100, 600, 300),
                spatial_sector=SpatialSector.RIGHT,
                proximity=ProximityLevel.NEAR,
            ),
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_RIGHT",
        direction="RIGHT",
        confidence=0.90,
        is_gesture_detected=True,
    )

    coord_out = coordinator_agent.process(
        CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    )

    task = coord_out.fused_task
    # Target MUST remain bottle!
    assert task.target_object == "bottle"
    assert task.target_object != "plant"
    assert task.spatial_sector == SpatialSector.LEFT
    assert task.spatial_agreement is False
    assert task.task_status == TaskStatus.MODALITY_CONFLICT
    assert task.grounding_status == GroundingStatus.EXPLICIT_TARGET_SPATIAL_CONFLICT

    safety_out = safety_agent_no_auto_approve.process(SafetyAgentInput(task=task, obstacle_distance=1.0))
    assert safety_out.safety_decision.decision in (
        SafetyDecisionType.HUMAN_APPROVAL_REQUIRED,
        SafetyDecisionType.HUMAN_CONFIRMATION_REQUIRED,
    )
    assert safety_out.approved_for_execution is False
    assert safety_out.safety_decision.human_intervention_required is True


# --------------------------------------------------------------------------
# TEST 6: Ambiguous Target in Pointed Sector
# --------------------------------------------------------------------------
def test_6_ambiguous_candidates_in_pointed_sector(coordinator_agent, safety_agent_no_auto_approve):
    """
    Voice: "Pick it up."
    Gesture: POINT_RIGHT (0.90)
    Vision: bottle RIGHT (0.61), cup RIGHT (0.59)
    -> Multiple objects in pointed sector with similar confidence
    -> AMBIGUOUS -> Safety: HUMAN_CONFIRMATION_REQUIRED
    """
    voice = VoiceAgentOutput(
        transcript="Pick it up.",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision = VisionAgentOutput(
        confidence=0.61,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.61,
                bbox=BoundingBox(500, 100, 550, 300),
                spatial_sector=SpatialSector.RIGHT,
                proximity=ProximityLevel.NEAR,
            ),
            DetectedObject(
                object_id=2,
                label="cup",
                confidence=0.59,
                bbox=BoundingBox(560, 100, 610, 300),
                spatial_sector=SpatialSector.RIGHT,
                proximity=ProximityLevel.NEAR,
            ),
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_RIGHT",
        direction="RIGHT",
        confidence=0.90,
        is_gesture_detected=True,
    )

    coord_out = coordinator_agent.process(
        CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=gesture)
    )

    task = coord_out.fused_task
    assert task.task_status == TaskStatus.AMBIGUOUS_TARGET
    assert task.grounding_status == GroundingStatus.AMBIGUOUS

    safety_out = safety_agent_no_auto_approve.process(SafetyAgentInput(task=task, obstacle_distance=1.0))
    assert safety_out.safety_decision.decision in (
        SafetyDecisionType.HUMAN_APPROVAL_REQUIRED,
        SafetyDecisionType.HUMAN_CONFIRMATION_REQUIRED,
    )
    assert safety_out.approved_for_execution is False


# --------------------------------------------------------------------------
# TEST 7: Deictic Command Without Pointing Gesture
# --------------------------------------------------------------------------
def test_7_deictic_without_gesture_multiple_objects(coordinator_agent, safety_agent_no_auto_approve):
    """
    Voice: "Pick it up."
    Gesture: None / no pointing
    Vision: bottle LEFT (0.85), cup RIGHT (0.80)
    -> Unresolved deictic target -> AMBIGUOUS -> Safety: HUMAN_CONFIRMATION_REQUIRED
    """
    voice = VoiceAgentOutput(
        transcript="Pick it up.",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision = VisionAgentOutput(
        confidence=0.85,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.85,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
            ),
            DetectedObject(
                object_id=2,
                label="cup",
                confidence=0.80,
                bbox=BoundingBox(500, 100, 600, 300),
                spatial_sector=SpatialSector.RIGHT,
            ),
        ],
    )

    coord_out = coordinator_agent.process(
        CoordinatorAgentInput(voice_output=voice, vision_output=vision, gesture_output=None)
    )

    task = coord_out.fused_task
    assert task.task_status == TaskStatus.AMBIGUOUS_TARGET
    assert task.grounding_status == GroundingStatus.AMBIGUOUS

    safety_out = safety_agent_no_auto_approve.process(SafetyAgentInput(task=task, obstacle_distance=1.0))
    assert safety_out.safety_decision.decision in (
        SafetyDecisionType.HUMAN_APPROVAL_REQUIRED,
        SafetyDecisionType.HUMAN_CONFIRMATION_REQUIRED,
    )
    assert safety_out.approved_for_execution is False


# --------------------------------------------------------------------------
# TEST 8: High-Confidence Deictic Target Does NOT Require Confirmation
# --------------------------------------------------------------------------
def test_8_high_confidence_deictic_does_not_unnecessarily_gate():
    """
    Verify that a high-confidence deictic target resolution with valid environment
    and distance clears Safety autonomously without requiring human intervention.
    """
    engine = MultimodalFusionEngine(
        min_confidence_threshold=0.40,
        high_confidence_threshold=0.60,
    )
    voice = VoiceAgentOutput(
        transcript="Pick it up.",
        is_speech_detected=True,
        confidence=0.95,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.95),
    )
    vision = VisionAgentOutput(
        confidence=0.92,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="cup",
                confidence=0.92,
                bbox=BoundingBox(50, 100, 150, 300),
                spatial_sector=SpatialSector.LEFT,
                proximity=ProximityLevel.NEAR,
            )
        ],
    )
    gesture = GestureAgentOutput(
        gesture="POINT_LEFT",
        direction="LEFT",
        confidence=0.95,
        is_gesture_detected=True,
    )

    task = engine.fuse(voice_output=voice, vision_output=vision, gesture_output=gesture)
    assert task.target_object == "cup"
    assert task.target_confidence == 0.92
    assert task.grounding_status == GroundingStatus.HIGH_CONFIDENCE
    assert task.task_status == TaskStatus.VALID

    safety = SafetyAgent(
        name="StrictSafety",
        human_interface=None,
        config={"safety": {"confidence_threshold": 0.60}},
    )
    safety.initialize()

    safety_out: SafetyAgentOutput = safety.process(
        SafetyAgentInput(task=task, obstacle_distance=1.50)
    )

    assert safety_out.approved_for_execution is True
    assert safety_out.safety_decision.decision == SafetyDecisionType.SAFE
    assert safety_out.safety_decision.human_intervention_required is False
    assert "All safety checks passed" in safety_out.safety_decision.reason or "RULE_6_NORMAL_SAFE" in str(safety_out.safety_decision.triggered_rules)
