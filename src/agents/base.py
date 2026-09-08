"""Abstract Base Agent definition for the Adaptive Multimodal HRI Framework."""

from abc import ABC, abstractmethod
from datetime import datetime
import time
from typing import Any, Dict, Optional

from src.common.exceptions import AgentExecutionError
from src.common.logger import get_logger
from src.common.schemas import (
    AgentMetadata,
    AgentStatus,
    AgentType,
)



class BaseAgent(ABC):
    """
    Abstract base class for all agents in the HRI multi-agent architecture.
    Provides standardized lifecycle management, status tracking, logging,
    and performance monitoring.
    """

    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        version: str = "0.1.0",
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.agent_type = agent_type
        self.version = version
        self.description = description
        self.config = config or {}
        self.status = AgentStatus.UNINITIALIZED
        self.logger = get_logger(self.name)
        self._metadata = AgentMetadata(
            agent_name=self.name,
            agent_type=self.agent_type,
            version=self.version,
            status=self.status,
            description=self.description,
        )

    @property
    def metadata(self) -> AgentMetadata:
        """Get updated agent metadata."""
        self._metadata.status = self.status
        self._metadata.last_active = datetime.utcnow().timestamp()
        return self._metadata

    def initialize(self) -> bool:
        """
        Public initialization hook. Sets lifecycle status and delegates to _initialize().
        """
        self.logger.info(f"Initializing agent '{self.name}' ({self.agent_type.value})...")
        self.status = AgentStatus.INITIALIZING
        try:
            success = self._initialize()
            if success:
                self.status = AgentStatus.READY
                self.logger.info(f"Agent '{self.name}' initialized successfully and is READY.")
                return True
            else:
                self.status = AgentStatus.ERROR
                self.logger.error(f"Agent '{self.name}' initialization failed.")
                return False
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Exception during initialization of agent '{self.name}': {e}")
            raise

    def process(self, input_data: Any) -> Any:
        """
        Execute agent reasoning/perception on input_data with timing and status management.
        """
        if self.status != AgentStatus.READY:
            if self.status == AgentStatus.UNINITIALIZED:
                self.logger.warning(
                    f"Agent '{self.name}' was not initialized before process(). Auto-initializing..."
                )
                if not self.initialize():
                    raise AgentExecutionError(f"Cannot process: agent '{self.name}' failed to initialize.")
            elif self.status == AgentStatus.SHUTDOWN:
                raise AgentExecutionError(f"Cannot process: agent '{self.name}' is shut down.")
            elif self.status == AgentStatus.ERROR:
                raise AgentExecutionError(f"Cannot process: agent '{self.name}' is in an ERROR state.")

        self.status = AgentStatus.PROCESSING
        start_time = time.perf_counter()

        try:
            output = self._process(input_data)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if hasattr(output, "execution_time_ms"):
                output.execution_time_ms = elapsed_ms
            if hasattr(output, "agent_name") and not output.agent_name:
                output.agent_name = self.name
            if hasattr(output, "agent_type") and not output.agent_type:
                output.agent_type = (
                    self.agent_type.value
                    if isinstance(self.agent_type, AgentType)
                    else str(self.agent_type)
                )

            self.status = AgentStatus.READY
            return output
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Error processing in agent '{self.name}': {e}")
            raise

    def reset(self) -> None:
        """Reset internal agent working memory or state."""
        self.logger.info(f"Resetting agent '{self.name}' state.")
        self._reset()
        if self.status not in (AgentStatus.UNINITIALIZED, AgentStatus.SHUTDOWN):
            self.status = AgentStatus.READY

    def shutdown(self) -> None:
        """Cleanly shutdown and release resources."""
        self.logger.info(f"Shutting down agent '{self.name}'...")
        try:
            self._shutdown()
        finally:
            self.status = AgentStatus.SHUTDOWN
            self.logger.info(f"Agent '{self.name}' has been shut down.")

    # ------------------------------------------------------------------
    # Abstract methods to be implemented by concrete agents
    # ------------------------------------------------------------------

    @abstractmethod
    def _initialize(self) -> bool:
        """Backend initialization logic (loading models, connections, etc.)."""
        pass

    @abstractmethod
    def _process(self, input_data: Any) -> Any:
        """Agent-specific execution / perception logic."""
        pass

    @abstractmethod
    def _reset(self) -> None:
        """Reset agent-specific state."""
        pass

    @abstractmethod
    def _shutdown(self) -> None:
        """Release agent-specific resources."""
        pass