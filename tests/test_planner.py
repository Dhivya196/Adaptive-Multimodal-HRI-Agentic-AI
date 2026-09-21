"""Unit tests for the Task Planner Agent."""

import pytest
from src.agents.planner.agent import TaskPlannerAgent
from src.agents.planner.schemas import TaskPlannerInput, PlanAction

def test_planner_initialization():
    agent = TaskPlannerAgent()
    assert agent.initialize() is True
    assert agent.engine is not None

def test_planner_stop_robot():
    agent = TaskPlannerAgent()
    agent.initialize()
    
    input_data = TaskPlannerInput(task_input={
        "task_status": "VALID",
        "action": "stop_robot",
    })
    
    output = agent.process(input_data)
    
    assert output.success is True
    assert output.plan is not None
    assert output.plan.task == "stop_robot"
    assert len(output.plan.steps) == 1
    assert output.plan.steps[0].action == PlanAction.STOP.value

def test_planner_navigate_to():
    agent = TaskPlannerAgent()
    agent.initialize()
    
    input_data = TaskPlannerInput(task_input={
        "task_status": "VALID",
        "action": "navigate_to",
        "target_object": "cup",
        "spatial_sector": "LEFT",
        "proximity": "MEDIUM"
    })
    
    output = agent.process(input_data)
    
    assert output.success is True
    assert output.plan is not None
    assert len(output.plan.steps) == 3
    # TURN_LEFT -> MOVE -> STOP
    assert output.plan.steps[0].action == PlanAction.TURN_LEFT.value
    assert output.plan.steps[1].action == PlanAction.MOVE.value
    assert output.plan.steps[2].action == PlanAction.STOP.value

def test_planner_rejection_missing_target():
    agent = TaskPlannerAgent()
    agent.initialize()
    
    input_data = TaskPlannerInput(task_input={
        "task_status": "VALID",
        "action": "navigate_to",
        # Missing target_object
    })
    
    output = agent.process(input_data)
    
    assert output.success is False
    assert output.plan_status == "VALIDATION_FAILED"
    assert "requires a target_object" in output.validation_errors[0]

def test_planner_rejection_invalid_task():
    agent = TaskPlannerAgent()
    agent.initialize()
    
    input_data = TaskPlannerInput(task_input={
        "task_status": "INVALID",
        "action": "navigate_to",
        "target_object": "cup"
    })
    
    output = agent.process(input_data)
    
    assert output.success is False
    assert output.plan_status == "VALIDATION_FAILED"
    assert "Task status is 'INVALID'" in output.validation_errors[0]
