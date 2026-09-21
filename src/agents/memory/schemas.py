"""Memory-specific schemas and data representations for context tracking and retrieval."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from src.common.schemas import BaseAgentInput, BaseAgentOutput, TimestampedMessage


class MemoryEntryType(str, Enum):
    """Categorical classification of information stored in Memory."""
    TASK = "task"
    INTERACTION = "interaction"
    OBSERVATION = "observation"
    USER_PREFERENCE = "user_preference"
    CONTEXT = "context"


class TaskStatus(str, Enum):
    """Lifecycle and execution status of tracked tasks."""
    REQUESTED = "requested"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VALID = "VALID"
    INVALID = "INVALID"


class MemoryOperation(str, Enum):
    """Supported operations on the Memory Agent."""
    STORE = "store"
    RETRIEVE = "retrieve"
    GET_RECENT = "get_recent"
    UPDATE_TASK = "update_task"
    GET_CURRENT_TASK = "get_current_task"
    RESOLVE_CONTEXT = "resolve_context"
    CLEAR = "clear"


@dataclass
class MemoryEntry(TimestampedMessage):
    """Strongly typed unit of stored memory."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: MemoryEntryType = MemoryEntryType.TASK
    content: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    source: str = "coordinator"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "id": self.id,
            "entry_type": self.entry_type.value if isinstance(self.entry_type, MemoryEntryType) else str(self.entry_type),
            "content": self.content,
            "task_id": self.task_id,
            "source": self.source,
            "metadata": self.metadata,
        })
        return base_dict


@dataclass
class MemoryQuery:
    """Query parameters for retrieving relevant memory entries."""
    text: Optional[str] = None
    entry_type: Optional[MemoryEntryType] = None
    target: Optional[str] = None
    action: Optional[str] = None
    task_id: Optional[str] = None
    status: Optional[str] = None
    limit: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entry_type": self.entry_type.value if isinstance(self.entry_type, MemoryEntryType) else (str(self.entry_type) if self.entry_type else None),
            "target": self.target,
            "action": self.action,
            "task_id": self.task_id,
            "status": self.status,
            "limit": self.limit,
        }


@dataclass
class CoordinatorTaskInput:
    """Structured representation of Coordinator output passed to Memory."""
    task_status: str = "VALID"
    action: str = "unknown"
    target: Optional[str] = None
    location: Optional[str] = None
    confidence: float = 1.0
    raw_command: Optional[str] = None
    source_modalities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_status": self.task_status,
            "action": self.action,
            "target": self.target,
            "location": self.location,
            "confidence": round(self.confidence, 3),
            "raw_command": self.raw_command,
            "source_modalities": self.source_modalities,
            "metadata": self.metadata,
        }


@dataclass
class MemoryAgentInput(BaseAgentInput):
    """Input payload passed to the Memory Agent."""
    operation: MemoryOperation = MemoryOperation.RESOLVE_CONTEXT
    coordinator_output: Optional[Dict[str, Any]] = None
    task_input: Optional[CoordinatorTaskInput] = None
    query: Optional[MemoryQuery] = None
    entry_to_store: Optional[MemoryEntry] = None
    task_id: Optional[str] = None
    new_status: Optional[str] = None
    limit: int = 5


@dataclass
class MemoryAgentOutput(BaseAgentOutput):
    """Structured contextual state emitted by the Memory Agent for Task Planning."""
    operation: str = ""
    resolved_context: Dict[str, Any] = field(default_factory=dict)
    relevant_entries: List[MemoryEntry] = field(default_factory=list)
    current_task: Optional[Dict[str, Any]] = None
    last_target: Optional[str] = None
    last_action: Optional[str] = None
    last_location: Optional[str] = None
    total_entries_count: int = 0
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "operation": self.operation,
            "resolved_context": self.resolved_context,
            "relevant_entries": [e.to_dict() for e in self.relevant_entries],
            "current_task": self.current_task,
            "last_target": self.last_target,
            "last_action": self.last_action,
            "last_location": self.last_location,
            "total_entries_count": self.total_entries_count,
            "summary_text": self.summary_text,
        })
        return base_dict
