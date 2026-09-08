"""Common data contracts and message schemas for the HRI multi-agent architecture."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import json


class AgentType(str, Enum):
    """Enumeration of all agents in the HRI architecture."""
    VISION = "vision_agent"
    GESTURE = "gesture_agent"
    VOICE = "voice_agent"
    COORDINATOR = "coordinator_agent"
    MEMORY = "memory_agent"
    TASK_PLANNER = "task_planner_agent"
    SAFETY = "safety_agent"
    ROBOT_CONTROLLER = "robot_controller"


class AgentStatus(str, Enum):
    """Lifecycle status of an agent."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class TimestampedMessage:
    """Base class for all inter-agent timestamped messages."""
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    iso_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string representation."""
        return json.dumps(self.to_dict(), default=str, indent=indent)


@dataclass
class AgentMetadata:
    """Metadata describing an agent's version, capabilities, and status."""
    agent_name: str
    agent_type: AgentType
    version: str = "0.1.0"
    status: AgentStatus = AgentStatus.UNINITIALIZED
    last_active: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type.value if isinstance(self.agent_type, AgentType) else self.agent_type,
            "version": self.version,
            "status": self.status.value if isinstance(self.status, AgentStatus) else self.status,
            "last_active": self.last_active,
            "description": self.description,
        }


@dataclass
class BaseAgentInput(TimestampedMessage):
    """Generic input container passed to an agent."""
    session_id: str = "default_session"
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseAgentOutput(TimestampedMessage):
    """Generic output container returned by an agent."""
    agent_name: str = ""
    agent_type: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    confidence: float = 1.0
