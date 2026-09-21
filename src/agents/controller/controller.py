"""Core robot controller translating commands into hardware actions."""

import time
from typing import Any, Dict, Optional, Tuple

from src.agents.controller.ros2_interface import BaseRobotInterface, MockRobotController, ROS2Interface
from src.agents.controller.schemas import (
    ControllerCommand,
    ExecutionResult,
    ExecutionStatus,
    TwistCommand,
)
from src.common.logger import get_logger


class RobotController:
    """
    Translates high-level safety-approved commands into executable
    hardware operations via the selected backend interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        ctrl_cfg = self.config.get("controller", {})
        
        self.mode = ctrl_cfg.get("mode", "mock")
        self.linear_speed = ctrl_cfg.get("linear_speed", 0.2)
        self.angular_speed = ctrl_cfg.get("angular_speed", 0.5)
        self.topic = ctrl_cfg.get("cmd_vel_topic", "/cmd_vel")
        self.default_duration = ctrl_cfg.get("default_duration", 1.0)
        
        self.logger = get_logger("RobotController")
        
        # Initialize Backend
        if self.mode == "ros2":
            self.backend: BaseRobotInterface = ROS2Interface(topic=self.topic)
            if not self.backend.is_connected():
                self.logger.warning("ROS2 mode requested but backend failed to connect. Falling back to mock.")
                self.backend = MockRobotController()
        else:
            self.backend = MockRobotController()

    def execute_command(self, command: str, parameters: Dict[str, Any], is_safety_approved: bool) -> ExecutionResult:
        """
        Execute a single command if it is safety approved.
        """
        start_time = time.time()
        
        # 1. SAFETY GATE
        if not is_safety_approved:
            self.logger.warning(f"Safety violation: Command '{command}' was rejected because is_safety_approved=False.")
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                command=command,
                error="Safety Agent approval is required for execution."
            )
            
        if not self.backend.is_connected():
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                command=command,
                error="Robot backend interface is not connected."
            )

        # 2. Command Translation & Execution
        twist: Optional[TwistCommand] = None
        cmd_upper = command.upper()
        
        try:
            if cmd_upper == ControllerCommand.MOVE_FORWARD.value:
                twist = TwistCommand(linear_x=self.linear_speed)
                self.backend.publish_cmd_vel(twist)
                
            elif cmd_upper == ControllerCommand.MOVE_BACKWARD.value:
                twist = TwistCommand(linear_x=-self.linear_speed)
                self.backend.publish_cmd_vel(twist)
                
            elif cmd_upper == ControllerCommand.TURN_LEFT.value:
                twist = TwistCommand(angular_z=self.angular_speed)
                self.backend.publish_cmd_vel(twist)
                
            elif cmd_upper == ControllerCommand.TURN_RIGHT.value:
                twist = TwistCommand(angular_z=-self.angular_speed)
                self.backend.publish_cmd_vel(twist)
                
            elif cmd_upper == ControllerCommand.STOP.value:
                twist = TwistCommand(linear_x=0.0, angular_z=0.0)
                self.backend.publish_cmd_vel(twist)
                
            elif cmd_upper in [ControllerCommand.PICK.value, ControllerCommand.PLACE.value, ControllerCommand.INSPECT.value]:
                # Discrete manipulation or sensor action
                self.backend.execute_action(cmd_upper, parameters)
                
            else:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    command=command,
                    error=f"Unsupported command '{command}'"
                )
                
            duration = time.time() - start_time
            return ExecutionResult(
                status=ExecutionStatus.EXECUTED,
                command=command,
                duration=duration,
                twist=twist,
                details={"parameters": parameters}
            )
            
        except Exception as e:
            self.logger.error(f"Execution failed for '{command}': {e}")
            self.backend.stop()  # Emergency stop on error
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                command=command,
                duration=time.time() - start_time,
                error=str(e)
            )

    def shutdown(self) -> None:
        """Shutdown controller and backend."""
        if self.backend:
            self.backend.shutdown()
