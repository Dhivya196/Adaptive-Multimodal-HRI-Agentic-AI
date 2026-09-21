#!/usr/bin/env python3
"""Standalone Demonstration runner for the Robot Controller Agent."""

import sys
from pathlib import Path
import json

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.controller.agent import RobotControllerAgent
from src.agents.controller.schemas import ControllerAgentInput
from src.utils.config import load_config


def print_divider(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title.upper()}")
        print("=" * 70)


def print_execution(scenario: str, command: str, is_approved: bool, params: dict, output):
    print_divider(scenario)
    appr_str = "✅ APPROVED" if is_approved else "❌ UNAPPROVED"
    print(f"📥 INPUT COMMAND: {command} ({appr_str}) | Params: {params}")
    print("-" * 70)
    
    res = output.execution_result
    if output.success and res:
        print(f"🟢 EXECUTION SUCCESS: {res.status.value if hasattr(res.status, 'value') else res.status}")
        if res.twist:
            t = res.twist
            print(f"   Twist -> linear({t.linear_x}, {t.linear_y}, {t.linear_z}) angular({t.angular_x}, {t.angular_y}, {t.angular_z})")
        print(f"   Duration: {res.duration:.3f}s")
    else:
        print(f"🔴 EXECUTION FAILED/REJECTED: {output.execution_status}")
        if res and res.error:
            print(f"   Error: {res.error}")
        elif output.error_message:
            print(f"   Error: {output.error_message}")


def main():
    print_divider("Robot Controller Agent Demo")
    config = load_config(PROJECT_ROOT / "configs" / "controller_config.yaml")
    # Force mock mode for the demo so it runs without ROS2
    if "controller" not in config:
        config["controller"] = {}
    config["controller"]["mode"] = "mock"
    
    controller = RobotControllerAgent(config=config)
    controller.initialize()

    # 1. Approved MOVE_FORWARD
    out_1 = controller.process(ControllerAgentInput(
        command="MOVE_FORWARD", 
        is_safety_approved=True
    ))
    print_execution("Scenario 1: Approved MOVE_FORWARD", "MOVE_FORWARD", True, {}, out_1)

    # 2. Approved TURN_LEFT
    out_2 = controller.process(ControllerAgentInput(
        command="TURN_LEFT", 
        is_safety_approved=True
    ))
    print_execution("Scenario 2: Approved TURN_LEFT", "TURN_LEFT", True, {}, out_2)

    # 3. Approved STOP
    out_3 = controller.process(ControllerAgentInput(
        command="STOP", 
        is_safety_approved=True
    ))
    print_execution("Scenario 3: Approved STOP", "STOP", True, {}, out_3)
    
    # 4. Approved PICK
    out_4 = controller.process(ControllerAgentInput(
        command="PICK", 
        parameters={"target": "bottle"},
        is_safety_approved=True
    ))
    print_execution("Scenario 4: Approved PICK (Manipulation)", "PICK", True, {"target": "bottle"}, out_4)

    # 5. Safety Rejection - Unapproved command
    out_5 = controller.process(ControllerAgentInput(
        command="MOVE_FORWARD", 
        is_safety_approved=False
    ))
    print_execution("Scenario 5: Unapproved Command (Safety Gate Rejection)", "MOVE_FORWARD", False, {}, out_5)
    
    # 6. Invalid command
    out_6 = controller.process(ControllerAgentInput(
        command="JUMP", 
        is_safety_approved=True
    ))
    print_execution("Scenario 6: Invalid/Unsupported Command", "JUMP", True, {}, out_6)
    
    # Print Mock Controller History
    print_divider("Mock Backend Command History")
    history = controller.controller.backend.command_history
    for idx, log in enumerate(history):
        type_ = log["type"]
        if type_ == "cmd_vel":
            print(f" [{idx+1}] CMD_VEL: {log['twist']}")
        else:
            print(f" [{idx+1}] DISCRETE: {log['action']} | params: {log['params']}")

if __name__ == "__main__":
    main()
