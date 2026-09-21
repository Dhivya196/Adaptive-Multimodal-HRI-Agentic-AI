"""Task Planner Engine for translating high-level tasks into logical plans."""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.agents.planner.schemas import PlanAction, PlanStep, TaskPlan
from src.agents.vision.schemas import SpatialSector, ProximityLevel


class TaskPlannerEngine:
    """
    Engine that translates a validated multimodal task into a sequence of
    executable steps (TaskPlan).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        planner_cfg = self.config.get("planner", {})
        
        self.supported_tasks = planner_cfg.get(
            "supported_tasks",
            ["stop_robot", "navigate_to", "pick_and_place", "inspect_object"]
        )
        self.max_plan_steps = planner_cfg.get("max_plan_steps", 10)
        self.default_timeout = planner_cfg.get("default_timeout", 15.0)
        self.step_timeouts = planner_cfg.get("step_timeouts", {})

    def _get_timeout(self, action: str) -> float:
        return self.step_timeouts.get(action, self.default_timeout)

    def validate_task(self, task_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate if the task is actionable and well-formed."""
        errors = []
        
        if not task_data:
            errors.append("Task input is empty.")
            return False, errors
            
        task_status = task_data.get("task_status")
        # Accept valid tasks. Handle Enum value or string.
        status_str = task_status.value if hasattr(task_status, "value") else str(task_status)
        if status_str != "VALID":
            errors.append(f"Task status is '{status_str}', expected 'VALID'.")
            
        action = task_data.get("action", "none")
        if action not in self.supported_tasks:
            errors.append(f"Action '{action}' is not supported by the planner.")
            
        target = task_data.get("target_object")
        
        # Action-specific validations
        if action in ["navigate_to", "pick_and_place", "inspect_object"] and not target:
            errors.append(f"Action '{action}' requires a target_object.")
            
        return len(errors) == 0, errors

    def generate_plan(self, task_data: Dict[str, Any]) -> TaskPlan:
        """Generate a sequential plan for the given task."""
        action = task_data.get("action")
        target = task_data.get("target_object")
        sector_val = task_data.get("spatial_sector")
        prox_val = task_data.get("proximity")
        
        sector = sector_val.value if hasattr(sector_val, "value") else sector_val
        proximity = prox_val.value if hasattr(prox_val, "value") else prox_val

        plan_id = f"PLAN-{uuid.uuid4().hex[:8].upper()}"
        steps: List[PlanStep] = []
        step_idx = 1
        
        def add_step(act: str, param: dict, tgt: Optional[str] = target, state: Optional[str] = None):
            nonlocal step_idx
            steps.append(PlanStep(
                step_id=step_idx,
                action=act,
                target=tgt,
                parameters=param,
                expected_state=state,
                timeout=self._get_timeout(act)
            ))
            step_idx += 1

        if action == "stop_robot":
            add_step(PlanAction.STOP.value, {"immediate": True}, tgt=None, state="robot_stopped")
            
        elif action == "navigate_to":
            if sector == "LEFT":
                add_step(PlanAction.TURN_LEFT.value, {"angle_deg": 30.0}, state="aligned_with_target")
            elif sector == "RIGHT":
                add_step(PlanAction.TURN_RIGHT.value, {"angle_deg": 30.0}, state="aligned_with_target")
                
            add_step(PlanAction.MOVE.value, {"direction": "forward", "proximity_goal": "NEAR"}, state="target_approached")
            add_step(PlanAction.STOP.value, {}, state="robot_stopped_at_target")
            
        elif action == "inspect_object":
            if sector == "LEFT":
                add_step(PlanAction.TURN_LEFT.value, {"angle_deg": 30.0}, state="aligned_with_target")
            elif sector == "RIGHT":
                add_step(PlanAction.TURN_RIGHT.value, {"angle_deg": 30.0}, state="aligned_with_target")
                
            if proximity == "FAR":
                add_step(PlanAction.MOVE.value, {"direction": "forward", "proximity_goal": "MEDIUM"}, state="target_in_range")
                
            add_step(PlanAction.STOP.value, {}, state="ready_for_inspection")
            add_step(PlanAction.INSPECT.value, {"duration": 3.0, "modalities": ["vision"]}, state="inspection_complete")
            
        elif action == "pick_and_place":
            if sector == "LEFT":
                add_step(PlanAction.TURN_LEFT.value, {"angle_deg": 30.0}, state="aligned_with_target")
            elif sector == "RIGHT":
                add_step(PlanAction.TURN_RIGHT.value, {"angle_deg": 30.0}, state="aligned_with_target")
                
            add_step(PlanAction.NAVIGATE.value, {"destination": "target"}, state="at_target")
            add_step(PlanAction.STOP.value, {}, state="robot_stopped_at_target")
            add_step(PlanAction.PICK.value, {"grip_force": 0.5}, state="object_grasped")
            add_step(PlanAction.MOVE.value, {"direction": "forward", "distance": 1.0}, state="at_destination")
            add_step(PlanAction.STOP.value, {}, state="robot_stopped_at_destination")
            add_step(PlanAction.PLACE.value, {"destination": "table"}, state="object_placed")
            add_step(PlanAction.STOP.value, {}, state="robot_idle")

        total_duration = sum(step.timeout for step in steps)
        
        return TaskPlan(
            plan_id=plan_id,
            task=action,
            target=target,
            steps=steps,
            total_estimated_duration=total_duration,
            metadata={"original_task": task_data}
        )
