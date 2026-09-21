"""Common modules, schemas, logging, and exceptions."""

from src.common.exceptions import (
    HRIException,
    PerceptionError,
    CameraStreamError,
    ModelLoadError,
    AgentExecutionError,
    ConfigurationError,
)
from src.common.logger import setup_logger, get_logger
from src.common.schemas import (
    AgentMetadata,
    AgentStatus,
    AgentType,
    BaseAgentInput,
    BaseAgentOutput,
    TimestampedMessage,
)

__all__ = [
    "HRIException",
    "PerceptionError",
    "CameraStreamError",
    "ModelLoadError",
    "AgentExecutionError",
    "ConfigurationError",
    "setup_logger",
    "get_logger",
    "AgentMetadata",
    "AgentStatus",
    "AgentType",
    "BaseAgentInput",
    "BaseAgentOutput",
    "TimestampedMessage",
]
