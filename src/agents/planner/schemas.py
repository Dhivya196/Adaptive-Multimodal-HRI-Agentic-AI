"""Schemas and data models for the Task Planner Agent."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.common.schemas import BaseAgentInput, BaseAgentOutput


class PlanAction(str, Enum):
    """Supported low-level execution actions."""
    MOVE = "MOVE"
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"
    INSPECT = "INSPECT"
    PICK = "PICK"
    PLACE = "PLACE"
    NAVIGATE = "NAVIGATE"


class PlanStepStatus(str, Enum):
    """Lifecycle status of a single plan step."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"


@dataclass
class PlanStep:
    """A single sequential step within a Task Plan."""
    step_id: int
    action: str
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_state: Optional[str] = None
    timeout: float = 10.0
    status: PlanStepStatus = PlanStepStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "target": self.target,
            "parameters": self.parameters,
            "expected_state": self.expected_state,
            "timeout": self.timeout,
            "status": self.status.value if isinstance(self.status, PlanStepStatus) else self.status,
            "metadata": self.metadata,
        }


@dataclass
class TaskPlan:
    """A structured sequential plan of execution generated from a high-level multimodal task."""
    plan_id: str
    task: str
    target: Optional[str] = None
    steps: List[PlanStep] = field(default_factory=list)
    total_estimated_duration: float = 0.0
    created_at: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task": self.task,
            "target": self.target,
            "steps": [step.to_dict() for step in self.steps],
            "total_estimated_duration": round(self.total_estimated_duration, 2),
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class TaskPlannerInput(BaseAgentInput):
    """Input payload passed to the Task Planner Agent."""
    task_input: Any = None  # Could be MultimodalTask or Dict
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlannerOutput(BaseAgentOutput):
    """Output plan emitted by the Task Planner Agent."""
    plan: Optional[TaskPlan] = None
    plan_status: str = "VALIDATION_FAILED"
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "plan": self.plan.to_dict() if self.plan else None,
            "plan_status": self.plan_status,
            "validation_errors": self.validation_errors,
        })
        return base_dict
