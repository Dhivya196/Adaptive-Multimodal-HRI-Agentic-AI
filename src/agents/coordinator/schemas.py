"""Schemas and data models for the Coordinator Agent and LangGraph HRI State."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from src.agents.vision.schemas import DetectedObject, ProximityLevel, SpatialSector, VisionAgentOutput
from src.agents.voice.schemas import SpeechIntent, UrgencyLevel, VoiceAgentOutput
from src.common.schemas import BaseAgentInput, BaseAgentOutput


class TaskStatus(str, Enum):
    """Semantic validation status of the interpreted multimodal task."""
    VALID = "VALID"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MODALITY_CONFLICT = "MODALITY_CONFLICT"
    INVALID_INPUT = "INVALID_INPUT"
    AMBIGUOUS_TARGET = "AMBIGUOUS_TARGET"


@dataclass
class GestureAgentOutput:
    """Future perception output contract for the Gesture Agent."""
    gesture_type: str = "none"
    pointing_direction: Optional[str] = None
    confidence: float = 0.0
    detected_landmarks: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gesture_type": self.gesture_type,
            "pointing_direction": self.pointing_direction,
            "confidence": round(self.confidence, 3),
            "metadata": self.metadata,
        }


@dataclass
class MultimodalTask:
    """Structured high-level task interpreted from multimodal perception inputs."""
    action: str = "none"
    target_object: Optional[str] = None
    target_confirmed: bool = False
    spatial_sector: Optional[SpatialSector] = None
    proximity: Optional[ProximityLevel] = None
    urgency: UrgencyLevel = UrgencyLevel.NORMAL
    confidence: float = 0.0
    task_status: TaskStatus = TaskStatus.INVALID_INPUT
    matched_visual_object: Optional[DetectedObject] = None
    reasoning: str = ""
    modality_contributions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target_object": self.target_object,
            "target_confirmed": self.target_confirmed,
            "spatial_sector": (
                self.spatial_sector.value if isinstance(self.spatial_sector, SpatialSector) else self.spatial_sector
            ),
            "proximity": (
                self.proximity.value if isinstance(self.proximity, ProximityLevel) else self.proximity
            ),
            "urgency": self.urgency.value if isinstance(self.urgency, UrgencyLevel) else self.urgency,
            "confidence": round(self.confidence, 3),
            "task_status": self.task_status.value if isinstance(self.task_status, TaskStatus) else self.task_status,
            "matched_visual_object": self.matched_visual_object.to_dict() if self.matched_visual_object else None,
            "reasoning": self.reasoning,
            "modality_contributions": self.modality_contributions,
        }


class HRIState(TypedDict, total=False):
    """
    Strongly-typed shared state for the multi-agent LangGraph workflow.
    Carries multimodal inputs, coordinator fusion, memory context, planning,
    safety evaluation, human intervention, and controller execution state.
    """
    # Perception inputs
    voice_output: Optional[Any]
    vision_output: Optional[Any]
    gesture_output: Optional[Any]

    # Environment & sensor measurements for safety & hardware
    obstacle_distance: Optional[float]
    human_distance: Optional[float]
    emergency_stop: Optional[bool]
    sensor_validity: Optional[Dict[str, bool]]
    target_confidence: Optional[float]
    environment_context: Optional[Dict[str, Any]]
    robot_state: Optional[Dict[str, Any]]

    # Coordinator Orchestration
    task: Optional[MultimodalTask]
    task_status: Optional[str]

    # Memory & Context
    memory_context: Optional[Dict[str, Any]]
    memory_output: Optional[Any]

    # Task Planning
    task_plan: Optional[Any]
    planner_output: Optional[Any]

    # Safety Evaluation
    safety_decision: Optional[Any]
    safety_output: Optional[Any]
    safety_decision_type: Optional[str]

    # Human Intervention
    human_intervention_required: Optional[bool]
    human_intervention_result: Optional[Any]

    # Robot Controller Execution
    controller_input: Optional[Any]
    controller_output: Optional[Any]
    execution_result: Optional[Any]

    # Pipeline metadata & errors
    metadata: Dict[str, Any]
    errors: List[str]


@dataclass
class CoordinatorAgentInput(BaseAgentInput):
    """Input payload passed to the Coordinator Agent."""
    voice_output: Optional[VoiceAgentOutput] = None
    vision_output: Optional[VisionAgentOutput] = None
    gesture_output: Optional[GestureAgentOutput] = None


@dataclass
class CoordinatorAgentOutput(BaseAgentOutput):
    """Final orchestrated decision emitted by the Coordinator Agent."""
    fused_task: Optional[MultimodalTask] = None
    task_status: str = TaskStatus.INVALID_INPUT.value
    hri_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "fused_task": self.fused_task.to_dict() if self.fused_task else None,
            "task_status": self.task_status,
            "hri_state": self.hri_state,
        })
        return base_dict
