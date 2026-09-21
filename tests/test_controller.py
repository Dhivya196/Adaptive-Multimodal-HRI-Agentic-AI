"""Unit tests for the Robot Controller Agent."""

import pytest
from src.agents.controller.agent import RobotControllerAgent
from src.agents.controller.schemas import ControllerAgentInput, ControllerCommand, ExecutionStatus

def test_controller_initialization():
    # Force mock mode
    agent = RobotControllerAgent(config={"controller": {"mode": "mock"}})
    assert agent.initialize() is True
    assert agent.controller is not None
    assert agent.controller.backend.is_connected() is True

def test_controller_approved_command():
    agent = RobotControllerAgent(config={"controller": {"mode": "mock"}})
    agent.initialize()
    
    input_data = ControllerAgentInput(
        command="MOVE_FORWARD",
        is_safety_approved=True
    )
    
    output = agent.process(input_data)
    
    assert output.success is True
    assert output.execution_status == ExecutionStatus.EXECUTED.value
    assert output.execution_result is not None
    assert output.execution_result.twist.linear_x > 0.0

def test_controller_unapproved_command():
    agent = RobotControllerAgent(config={"controller": {"mode": "mock"}})
    agent.initialize()
    
    input_data = ControllerAgentInput(
        command="MOVE_FORWARD",
        is_safety_approved=False
    )
    
    output = agent.process(input_data)
    
    assert output.success is False
    assert output.execution_status == ExecutionStatus.REJECTED.value
    assert "Safety Agent approval is required" in output.error_message

def test_controller_unsupported_command():
    agent = RobotControllerAgent(config={"controller": {"mode": "mock"}})
    agent.initialize()
    
    input_data = ControllerAgentInput(
        command="JUMP",
        is_safety_approved=True
    )
    
    output = agent.process(input_data)
    
    assert output.success is False
    assert output.execution_status == ExecutionStatus.ERROR.value
    assert "Unsupported command" in output.error_message

def test_controller_discrete_action():
    agent = RobotControllerAgent(config={"controller": {"mode": "mock"}})
    agent.initialize()
    
    input_data = ControllerAgentInput(
        command="PICK",
        parameters={"target": "cup"},
        is_safety_approved=True
    )
    
    output = agent.process(input_data)
    
    assert output.success is True
    assert output.execution_status == ExecutionStatus.EXECUTED.value
    history = agent.controller.backend.command_history
    assert len(history) == 1
    assert history[0]["type"] == "discrete_action"
    assert history[0]["action"] == "PICK"
    assert history[0]["params"] == {"target": "cup"}
