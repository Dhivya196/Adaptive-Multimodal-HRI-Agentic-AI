"""Robot Controller Agent implementation."""

from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.agents.controller.controller import RobotController
from src.agents.controller.schemas import ControllerAgentInput, ControllerAgentOutput
from src.common.exceptions import AgentExecutionError
from src.common.schemas import AgentType


class RobotControllerAgent(BaseAgent):
    """
    Robot Controller Agent in the Execution Layer.
    Converts safety-approved high-level commands into low-level execution on the hardware.
    """

    def __init__(
        self,
        name: str = "RobotControllerAgent",
        config: Optional[Dict[str, Any]] = None,
        controller: Optional[RobotController] = None,
    ):
        super().__init__(
            name=name,
            agent_type=AgentType.ROBOT_CONTROLLER,
            version="0.1.0",
            description="Executes safety-approved hardware commands via ROS2 or Mock interface.",
            config=config or {},
        )
        self.controller = controller

    def _initialize(self) -> bool:
        """Initialize the hardware controller backend."""
        if self.controller is None:
            self.controller = RobotController(config=self.config)
        self.logger.info("RobotControllerAgent initialized successfully.")
        return True

    def _process(self, input_data: Any) -> ControllerAgentOutput:
        """Process incoming command and execute it."""
        if not self.controller:
            raise AgentExecutionError("RobotController backend not initialized.")

        cmd = "STOP"
        params = {}
        is_approved = False

        if isinstance(input_data, ControllerAgentInput):
            cmd = input_data.command
            params = input_data.parameters
            is_approved = input_data.is_safety_approved
        elif isinstance(input_data, dict):
            cmd = input_data.get("command", "STOP")
            params = input_data.get("parameters", {})
            is_approved = input_data.get("is_safety_approved", False)
        else:
            raise AgentExecutionError(f"Unsupported input type for ControllerAgent: {type(input_data)}")

        # Execute
        result = self.controller.execute_command(
            command=cmd,
            parameters=params,
            is_safety_approved=is_approved
        )

        return ControllerAgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type.value,
            success=(result.status.value == "EXECUTED"),
            execution_result=result,
            execution_status=result.status.value,
            error_message=result.error
        )

    def _reset(self) -> None:
        """Reset internal state."""
        pass

    def _shutdown(self) -> None:
        """Release hardware/ROS2 resources."""
        if self.controller:
            self.controller.shutdown()
