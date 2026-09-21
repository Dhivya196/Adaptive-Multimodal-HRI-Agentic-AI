"""Task Planner Agent implementation."""

from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.agents.planner.planner import TaskPlannerEngine
from src.agents.planner.schemas import TaskPlannerInput, TaskPlannerOutput
from src.common.exceptions import AgentExecutionError
from src.common.schemas import AgentType


class TaskPlannerAgent(BaseAgent):
    """
    Task Planner Agent in the Decision-Making Layer.
    Translates orchestrated multimodal tasks into sequential execution plans.
    """

    def __init__(
        self,
        name: str = "TaskPlannerAgent",
        config: Optional[Dict[str, Any]] = None,
        engine: Optional[TaskPlannerEngine] = None,
    ):
        super().__init__(
            name=name,
            agent_type=AgentType.TASK_PLANNER,
            version="0.1.0",
            description="Converts validated high-level tasks into executable sequence of logical steps.",
            config=config or {},
        )
        self.engine = engine

    def _initialize(self) -> bool:
        """Initialize the planning engine."""
        if self.engine is None:
            self.engine = TaskPlannerEngine(config=self.config)
        self.logger.info("TaskPlannerAgent initialized successfully.")
        return True

    def _process(self, input_data: Any) -> TaskPlannerOutput:
        """Process incoming task and generate a plan."""
        if not self.engine:
            raise AgentExecutionError("TaskPlannerEngine not initialized.")

        # Extract task dictionary
        task_dict = {}
        if isinstance(input_data, TaskPlannerInput):
            task_dict = self._extract_task_dict(input_data.task_input)
        else:
            task_dict = self._extract_task_dict(input_data)

        # Validate
        is_valid, errors = self.engine.validate_task(task_dict)
        if not is_valid:
            self.logger.warning(f"Task validation failed: {errors}")
            return TaskPlannerOutput(
                agent_name=self.name,
                agent_type=self.agent_type.value,
                success=False,
                error_message="Task validation failed.",
                plan=None,
                plan_status="VALIDATION_FAILED",
                validation_errors=errors
            )

        # Generate Plan
        try:
            plan = self.engine.generate_plan(task_dict)
            self.logger.info(f"Generated plan {plan.plan_id} with {len(plan.steps)} steps.")
            return TaskPlannerOutput(
                agent_name=self.name,
                agent_type=self.agent_type.value,
                success=True,
                plan=plan,
                plan_status="GENERATED",
            )
        except Exception as e:
            self.logger.error(f"Failed to generate plan: {e}")
            return TaskPlannerOutput(
                agent_name=self.name,
                agent_type=self.agent_type.value,
                success=False,
                error_message=str(e),
                plan=None,
                plan_status="ERROR"
            )

    def _extract_task_dict(self, data: Any) -> Dict[str, Any]:
        """Helper to extract dictionary representation of a task."""
        if not data:
            return {}
        
        # If it's a dict, check if it has a fused_task (Coordinator output format)
        if isinstance(data, dict):
            if "fused_task" in data and data["fused_task"]:
                return data["fused_task"]
            return data
            
        # If it's a CoordinatorAgentOutput object
        if hasattr(data, "fused_task") and data.fused_task:
            if hasattr(data.fused_task, "to_dict"):
                return data.fused_task.to_dict()
            return dict(data.fused_task)
            
        # If it's a MultimodalTask object directly
        if hasattr(data, "to_dict"):
            return data.to_dict()
            
        return {}

    def _reset(self) -> None:
        """Reset internal state."""
        pass

    def _shutdown(self) -> None:
        """Release resources."""
        self.engine = None
