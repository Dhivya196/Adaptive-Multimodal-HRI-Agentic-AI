#!/usr/bin/env python3
"""Standalone Demonstration runner for the Task Planner Agent."""

import sys
from pathlib import Path
import json

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.planner.agent import TaskPlannerAgent
from src.agents.planner.schemas import TaskPlannerInput
from src.utils.config import load_config


def print_divider(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title.upper()}")
        print("=" * 70)


def print_plan(scenario: str, task_dict: dict, output):
    print_divider(scenario)
    print("📥 INPUT TASK:")
    print(json.dumps(task_dict, indent=2))
    print("-" * 70)
    
    if output.success and output.plan:
        plan = output.plan
        print(f"📋 GENERATED PLAN: {plan.plan_id}")
        print(f"   Task: {plan.task} | Target: {plan.target}")
        print(f"   Estimated Duration: {plan.total_estimated_duration}s")
        print("\n   Steps:")
        for step in plan.steps:
            target_str = f" target='{step.target}'" if step.target else ""
            param_str = f" params={step.parameters}" if step.parameters else ""
            print(f"   [{step.step_id}] {step.action}{target_str}{param_str} -> (expect: {step.expected_state})")
    else:
        print(f"❌ PLAN GENERATION FAILED: {output.plan_status}")
        for err in output.validation_errors:
            print(f"   - {err}")
        if output.error_message:
            print(f"   Error: {output.error_message}")


def main():
    print_divider("Task Planner Agent Demo")
    config = load_config(PROJECT_ROOT / "configs" / "planner_config.yaml")
    planner = TaskPlannerAgent(config=config)
    planner.initialize()

    # Scenario 1: Navigate to target (LEFT, MEDIUM)
    task_1 = {
        "task_status": "VALID",
        "action": "navigate_to",
        "target_object": "bottle",
        "spatial_sector": "LEFT",
        "proximity": "MEDIUM",
        "confidence": 0.90
    }
    out_1 = planner.process(TaskPlannerInput(task_input=task_1))
    print_plan("Scenario 1: Navigate to Bottle on the Left", task_1, out_1)

    # Scenario 2: Pick and place (CENTER)
    task_2 = {
        "task_status": "VALID",
        "action": "pick_and_place",
        "target_object": "cup",
        "spatial_sector": "CENTER",
        "proximity": "NEAR",
        "confidence": 0.95
    }
    out_2 = planner.process(TaskPlannerInput(task_input=task_2))
    print_plan("Scenario 2: Pick and Place Cup in Center", task_2, out_2)

    # Scenario 3: Emergency Stop
    task_3 = {
        "task_status": "VALID",
        "action": "stop_robot",
        "confidence": 0.99
    }
    out_3 = planner.process(TaskPlannerInput(task_input=task_3))
    print_plan("Scenario 3: Emergency Stop", task_3, out_3)

    # Scenario 4: Inspect Object (FAR, RIGHT)
    task_4 = {
        "task_status": "VALID",
        "action": "inspect_object",
        "target_object": "book",
        "spatial_sector": "RIGHT",
        "proximity": "FAR",
        "confidence": 0.85
    }
    out_4 = planner.process(TaskPlannerInput(task_input=task_4))
    print_plan("Scenario 4: Inspect Book on the Right (Far)", task_4, out_4)

    # Scenario 5: Rejection - Target Not Found
    task_5 = {
        "task_status": "TARGET_NOT_FOUND",
        "action": "navigate_to",
        "target_object": "apple"
    }
    out_5 = planner.process(TaskPlannerInput(task_input=task_5))
    print_plan("Scenario 5: Rejection - Target Not Found", task_5, out_5)

    # Scenario 6: Rejection - Missing Target for Pick and Place
    task_6 = {
        "task_status": "VALID",
        "action": "pick_and_place"
    }
    out_6 = planner.process(TaskPlannerInput(task_input=task_6))
    print_plan("Scenario 6: Rejection - Missing Target Object", task_6, out_6)

if __name__ == "__main__":
    main()
