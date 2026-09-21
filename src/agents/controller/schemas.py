"""Schemas and data models for the Robot Controller Agent."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime

from src.common.schemas import BaseAgentInput, BaseAgentOutput


class ControllerCommand(str, Enum):
    """Supported high-level robot controller commands."""
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"
    PICK = "PICK"
    PLACE = "PLACE"
    INSPECT = "INSPECT"


class ExecutionStatus(str, Enum):
    """Lifecycle status of a command execution."""
    READY = "READY"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass
class TwistCommand:
    """Standard linear and angular velocity commands (similar to geometry_msgs/Twist)."""
    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "linear_x": self.linear_x,
            "linear_y": self.linear_y,
            "linear_z": self.linear_z,
            "angular_x": self.angular_x,
            "angular_y": self.angular_y,
            "angular_z": self.angular_z,
        }


@dataclass
class ExecutionResult:
    """Detailed result of a command execution attempt."""
    status: ExecutionStatus
    command: str
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    duration: float = 0.0
    twist: Optional[TwistCommand] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, ExecutionStatus) else self.status,
            "command": self.command,
            "timestamp": self.timestamp,
            "duration": round(self.duration, 3),
            "twist": self.twist.to_dict() if self.twist else None,
            "details": self.details,
            "error": self.error,
        }


@dataclass
class ControllerAgentInput(BaseAgentInput):
    """Input payload passed to the Robot Controller."""
    command: str = "STOP"
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_safety_approved: bool = False
    safety_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerAgentOutput(BaseAgentOutput):
    """Output execution result emitted by the Robot Controller."""
    execution_result: Optional[ExecutionResult] = None
    execution_status: str = ExecutionStatus.READY.value

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "execution_status": self.execution_status,
        })
        return base_dict
