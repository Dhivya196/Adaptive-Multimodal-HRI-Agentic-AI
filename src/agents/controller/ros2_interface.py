"""Hardware / Simulation interfaces for the Robot Controller."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from datetime import datetime

from src.agents.controller.schemas import TwistCommand
from src.common.logger import get_logger

logger = get_logger("RobotInterface")


class BaseRobotInterface(ABC):
    """Abstract interface for robot hardware or simulation backend."""

    @abstractmethod
    def publish_cmd_vel(self, twist: TwistCommand) -> bool:
        """Publish velocity command to the robot."""
        pass

    @abstractmethod
    def execute_action(self, action: str, params: Dict[str, Any]) -> bool:
        """Execute a discrete manipulation or sensor action."""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Emergency or standard stop command."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanly shutdown the interface."""
        pass


class MockRobotController(BaseRobotInterface):
    """
    In-memory simulation interface.
    Records commands and simulated state without needing ROS2 or Gazebo.
    """

    def __init__(self):
        self.connected = True
        self.command_history: List[Dict[str, Any]] = []
        self.last_twist = TwistCommand()
        self.simulated_pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        logger.info("Initialized MockRobotController simulation backend.")

    def publish_cmd_vel(self, twist: TwistCommand) -> bool:
        self.last_twist = twist
        self.command_history.append({
            "type": "cmd_vel",
            "twist": twist.to_dict(),
            "timestamp": datetime.utcnow().timestamp()
        })
        logger.debug(f"[Mock] Published Twist: {twist.to_dict()}")
        return True

    def execute_action(self, action: str, params: Dict[str, Any]) -> bool:
        self.command_history.append({
            "type": "discrete_action",
            "action": action,
            "params": params,
            "timestamp": datetime.utcnow().timestamp()
        })
        logger.debug(f"[Mock] Executed Action: {action} with params {params}")
        return True

    def stop(self) -> bool:
        self.publish_cmd_vel(TwistCommand(linear_x=0.0, angular_z=0.0))
        logger.info("[Mock] Robot STOPPED.")
        return True

    def is_connected(self) -> bool:
        return self.connected

    def shutdown(self) -> None:
        self.stop()
        self.connected = False
        logger.info("MockRobotController shut down.")


class ROS2Interface(BaseRobotInterface):
    """
    ROS2 implementation using rclpy and geometry_msgs/Twist.
    Safely falls back if rclpy is not installed.
    """

    def __init__(self, topic: str = "/cmd_vel"):
        self.topic = topic
        self.node = None
        self.publisher = None
        self.connected = False
        self._rclpy = None
        self._twist_msg = None

        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Twist
            self._rclpy = rclpy
            self._twist_msg = Twist

            # Only init rclpy if it hasn't been initialized
            if not rclpy.ok():
                rclpy.init(args=None)

            self.node = Node('hri_robot_controller')
            self.publisher = self.node.create_publisher(Twist, self.topic, 10)
            self.connected = True
            logger.info(f"Initialized ROS2Interface on topic {self.topic}.")
            
        except ImportError:
            logger.warning("ROS2 (rclpy) is not available in the environment. ROS2Interface will be disabled.")
            self.connected = False
        except Exception as e:
            logger.error(f"Failed to initialize ROS2Interface: {e}")
            self.connected = False

    def publish_cmd_vel(self, twist: TwistCommand) -> bool:
        if not self.connected or not self.publisher:
            logger.error("Cannot publish cmd_vel: ROS2 backend is not connected.")
            return False

        msg = self._twist_msg()
        msg.linear.x = twist.linear_x
        msg.linear.y = twist.linear_y
        msg.linear.z = twist.linear_z
        msg.angular.x = twist.angular_x
        msg.angular.y = twist.angular_y
        msg.angular.z = twist.angular_z

        try:
            self.publisher.publish(msg)
            return True
        except Exception as e:
            logger.error(f"Error publishing ROS2 twist message: {e}")
            return False

    def execute_action(self, action: str, params: Dict[str, Any]) -> bool:
        # In a full ROS2 system, this would call action servers or services for manipulation
        logger.info(f"ROS2 execute_action called for {action}. (Not implemented in basic twist controller)")
        return self.connected

    def stop(self) -> bool:
        return self.publish_cmd_vel(TwistCommand(linear_x=0.0, angular_z=0.0))

    def is_connected(self) -> bool:
        return self.connected

    def shutdown(self) -> None:
        if self.connected:
            self.stop()
            if self.node:
                self.node.destroy_node()
            if self._rclpy and self._rclpy.ok():
                try:
                    self._rclpy.shutdown()
                except Exception:
                    pass
        self.connected = False
        logger.info("ROS2Interface shut down.")
