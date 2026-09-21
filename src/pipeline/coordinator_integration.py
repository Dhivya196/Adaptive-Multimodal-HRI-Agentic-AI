"""End-to-End Multi-Agent Integration with LangGraph for the Adaptive Multimodal HRI Framework.

Orchestration Pipeline:
START
  ↓
input_validation
  ↓
multimodal_fusion
  ↓
task_interpretation
  ↓
memory_context
  ↓
task_planning
  ↓
safety_check
  ↓
safety_decision_router
  ├── APPROVED
  │      ↓
  │   robot_execution
  │      ↓
  │     END
  │
  ├── HUMAN_APPROVAL_REQUIRED
  │      ↓
  │   human_intervention
  │      ├── APPROVED → robot_execution → END
  │      └── REJECTED → END
  │
  └── STOP / UNSAFE / UNKNOWN
         ↓
        END
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from langgraph.graph import END, START, StateGraph

from src.agents.controller.agent import RobotControllerAgent
from src.agents.controller.schemas import (
    ControllerAgentInput,
    ControllerAgentOutput,
    ControllerCommand,
    ExecutionResult,
    ExecutionStatus,
)
from src.agents.coordinator.fusion import MultimodalFusionEngine
from src.agents.coordinator.schemas import (
    CoordinatorAgentInput,
    CoordinatorAgentOutput,
    HRIState,
    MultimodalTask,
    TaskStatus,
)
from src.agents.memory.agent import MemoryAgent
from src.agents.memory.schemas import (
    CoordinatorTaskInput,
    MemoryAgentInput,
    MemoryAgentOutput,
    MemoryEntryType,
    MemoryOperation,
)
from src.agents.planner.agent import TaskPlannerAgent
from src.agents.planner.schemas import (
    PlanAction,
    TaskPlan,
    TaskPlannerInput,
    TaskPlannerOutput,
)
from src.agents.safety.agent import SafetyAgent
from src.agents.safety.human_intervention import (
    AutoApprovalStub,
    BaseHumanInterventionInterface,
)
from src.agents.safety.schemas import (
    HumanInterventionResult,
    RiskLevel,
    SafetyAgentInput,
    SafetyAgentOutput,
    SafetyDecision,
    SafetyDecisionType,
)
from src.common.logger import setup_logger
from src.common.schemas import AgentStatus, AgentType


def create_integrated_hri_graph(
    fusion_engine: Optional[MultimodalFusionEngine] = None,
    memory_agent: Optional[MemoryAgent] = None,
    planner_agent: Optional[TaskPlannerAgent] = None,
    safety_agent: Optional[SafetyAgent] = None,
    controller_agent: Optional[RobotControllerAgent] = None,
    human_interface: Optional[BaseHumanInterventionInterface] = None,
) -> Any:
    """
    Construct and compile the full Multi-Agent LangGraph StateGraph workflow.
    Reuses provided agent instances without re-instantiating on each invocation.
    """
    logger = setup_logger("IntegratedHRIGraph")

    _fusion = fusion_engine or MultimodalFusionEngine()

    _memory = memory_agent
    if _memory is None:
        _memory = MemoryAgent(name="PipelineMemory")
        _memory.initialize()
    elif _memory.status != AgentStatus.READY:
        _memory.initialize()

    _planner = planner_agent
    if _planner is None:
        _planner = TaskPlannerAgent(name="PipelinePlanner")
        _planner.initialize()
    elif _planner.status != AgentStatus.READY:
        _planner.initialize()

    _safety = safety_agent
    if _safety is None:
        _safety = SafetyAgent(name="PipelineSafety")
        _safety.initialize()
    elif _safety.status != AgentStatus.READY:
        _safety.initialize()

    _controller = controller_agent
    if _controller is None:
        _controller = RobotControllerAgent(name="PipelineController")
        _controller.initialize()
    elif _controller.status != AgentStatus.READY:
        _controller.initialize()

    _human_interface = human_interface or getattr(_safety, "human_interface", None) or AutoApprovalStub(auto_approve=True)

    builder = StateGraph(HRIState)

    # ------------------------------------------------------------------
    # 1. Input Validation Node
    # ------------------------------------------------------------------
    def input_validation_node(state: HRIState) -> Dict[str, Any]:
        errors = state.get("errors") or []
        metadata = state.get("metadata") or {}

        voice = state.get("voice_output")
        vision = state.get("vision_output")
        gesture = state.get("gesture_output")

        metadata["has_voice"] = voice is not None and getattr(voice, "is_speech_detected", False)
        metadata["has_vision"] = vision is not None and len(getattr(vision, "detected_objects", [])) > 0
        metadata["has_gesture"] = gesture is not None and getattr(gesture, "confidence", 0.0) > 0.0

        return {
            "errors": errors,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # 2. Multimodal Fusion Node
    # ------------------------------------------------------------------
    def multimodal_fusion_node(state: HRIState) -> Dict[str, Any]:
        voice = state.get("voice_output")
        vision = state.get("vision_output")
        gesture = state.get("gesture_output")

        task: MultimodalTask = _fusion.fuse(
            voice_output=voice,
            vision_output=vision,
            gesture_output=gesture,
        )
        return {"task": task}

    # ------------------------------------------------------------------
    # 3. Task Interpretation Node
    # ------------------------------------------------------------------
    def task_interpretation_node(state: HRIState) -> Dict[str, Any]:
        task: Optional[MultimodalTask] = state.get("task")
        if task is None:
            task = MultimodalTask(
                action="none",
                task_status=TaskStatus.INVALID_INPUT,
                reasoning="No task produced during fusion.",
            )

        task_status_str = (
            task.task_status.value
            if hasattr(task.task_status, "value")
            else str(task.task_status)
        )

        return {
            "task": task,
            "task_status": task_status_str,
        }

    # ------------------------------------------------------------------
    # 4. Memory Context Node (Resolves pronouns & queries context)
    # ------------------------------------------------------------------
    def memory_context_node(state: HRIState) -> Dict[str, Any]:
        task: Optional[MultimodalTask] = state.get("task")
        if task is None:
            return {"memory_context": {}, "memory_output": None}

        task_payload = {
            "action": task.action,
            "target": task.target_object,
            "location": (
                task.spatial_sector.value
                if task.spatial_sector and hasattr(task.spatial_sector, "value")
                else (str(task.spatial_sector) if task.spatial_sector else None)
            ),
            "task_status": (
                task.task_status.value
                if hasattr(task.task_status, "value")
                else str(task.task_status)
            ),
            "confidence": task.confidence,
        }

        mem_input = MemoryAgentInput(
            operation=MemoryOperation.RESOLVE_CONTEXT,
            coordinator_output=task_payload,
        )

        mem_output: MemoryAgentOutput = _memory.process(mem_input)
        resolved_context = mem_output.resolved_context or {}

        # If memory resolved a referential target (e.g. "it" -> "bottle")
        target_str = str(task.target_object or "").strip().lower()
        if target_str in MemoryAgent.REFERENTIAL_PRONOUNS and resolved_context.get("target"):
            task.target_object = resolved_context["target"]
            if resolved_context.get("location") and not task.spatial_sector:
                task.spatial_sector = resolved_context["location"]
            task.target_confirmed = True
            task.task_status = TaskStatus.VALID
            task.reasoning += f" (Target resolved from memory context: {task.target_object})"

        return {
            "task": task,
            "memory_context": resolved_context,
            "memory_output": mem_output,
        }

    # ------------------------------------------------------------------
    # 5. Task Planning Node
    # ------------------------------------------------------------------
    def task_planning_node(state: HRIState) -> Dict[str, Any]:
        task: Optional[MultimodalTask] = state.get("task")

        # Do not plan for invalid, ungrounded or conflicting tasks
        if task is None or task.task_status != TaskStatus.VALID:
            status_val = task.task_status.value if (task and hasattr(task.task_status, "value")) else "INVALID_TASK"
            planner_output = TaskPlannerOutput(
                agent_name=_planner.name,
                agent_type=_planner.agent_type.value,
                success=False,
                error_message=f"Planning skipped for task status: {status_val}",
                plan=None,
                plan_status="SKIPPED_INVALID_TASK",
                validation_errors=[f"Task status is {status_val}"],
            )
            return {
                "task_plan": None,
                "planner_output": planner_output,
            }

        planner_input = TaskPlannerInput(
            task_input=task.to_dict(),
            context=state.get("memory_context") or {},
        )

        planner_output: TaskPlannerOutput = _planner.process(planner_input)
        return {
            "task_plan": planner_output.plan,
            "planner_output": planner_output,
        }

    # ------------------------------------------------------------------
    # 6. Safety Check Node
    # ------------------------------------------------------------------
    def safety_check_node(state: HRIState) -> Dict[str, Any]:
        task = state.get("task")
        planner_out = state.get("planner_output")

        conf_val = state.get("target_confidence")
        if conf_val is None and task is not None:
            conf_val = task.confidence

        safety_input = SafetyAgentInput(
            task=task,
            planner_output=planner_out.to_dict() if planner_out else None,
            obstacle_distance=state.get("obstacle_distance"),
            human_distance=state.get("human_distance"),
            emergency_stop=bool(state.get("emergency_stop", False)),
            sensor_validity=state.get("sensor_validity", {"camera": True, "proximity": True}),
            target_confidence=conf_val,
            environment_context=state.get("environment_context") or {},
            robot_state=state.get("robot_state") or {},
        )

        safety_output: SafetyAgentOutput = _safety.process(safety_input)
        safety_dec = safety_output.safety_decision
        decision_str = safety_dec.decision.value if (safety_dec and hasattr(safety_dec.decision, "value")) else "UNSAFE"

        requires_human = False
        human_res = None
        if safety_dec:
            requires_human = (safety_dec.decision == SafetyDecisionType.HUMAN_APPROVAL_REQUIRED) or safety_dec.human_intervention_required
            human_res = safety_dec.human_intervention_result

        return {
            "safety_output": safety_output,
            "safety_decision": safety_dec,
            "safety_decision_type": decision_str,
            "human_intervention_required": requires_human,
            "human_intervention_result": human_res,
        }

    # ------------------------------------------------------------------
    # 7. Safety Decision Router (Conditional Edge)
    # ------------------------------------------------------------------
    def safety_decision_router(state: HRIState) -> str:
        safety_output: Optional[SafetyAgentOutput] = state.get("safety_output")
        safety_dec: Optional[SafetyDecision] = state.get("safety_decision")

        if safety_output is None or safety_dec is None:
            logger.warning("No safety output in state; routing to fail-safe STOP.")
            return "STOP"

        # Fail-safe check: Hard violations NEVER pass
        if safety_dec.is_hard_violation:
            logger.warning(f"Hard safety violation triggered ({safety_dec.reason}); routing to STOP.")
            return "STOP"

        if safety_dec.decision == SafetyDecisionType.HUMAN_APPROVAL_REQUIRED:
            return "HUMAN_APPROVAL_REQUIRED"

        if safety_output.approved_for_execution and safety_dec.decision == SafetyDecisionType.SAFE:
            return "APPROVED"

        if safety_dec.decision in (SafetyDecisionType.STOP, SafetyDecisionType.UNSAFE):
            return "STOP"

        # Any unrecognized state is routed to STOP for fail-safe behavior
        logger.warning(f"Unrecognized safety decision: {safety_dec.decision}; routing to STOP.")
        return "STOP"

    # ------------------------------------------------------------------
    # 8. Human Intervention Node
    # ------------------------------------------------------------------
    def human_intervention_node(state: HRIState) -> Dict[str, Any]:
        task = state.get("task")
        safety_dec = state.get("safety_decision")
        safety_out = state.get("safety_output")

        # If already resolved by SafetyAgent
        existing_res = state.get("human_intervention_result")
        if existing_res is not None:
            return {
                "human_intervention_result": existing_res,
                "safety_output": safety_out,
                "safety_decision": safety_dec,
            }

        task_desc = f"{task.action} ({task.target_object})" if task else "Unspecified Task"
        reason_str = safety_dec.reason if safety_dec else "Human confirmation required."
        risk_lvl = safety_dec.risk_level if safety_dec else RiskLevel.MEDIUM
        measurements = safety_dec.relevant_measurements if safety_dec else {}

        human_res: HumanInterventionResult = _human_interface.request_approval(
            task_summary=task_desc,
            reason=reason_str,
            risk_level=risk_lvl,
            measurements=measurements,
            is_hard_violation=bool(safety_dec and safety_dec.is_hard_violation),
        )

        if safety_out is not None:
            if human_res.approved:
                safety_out.approved_for_execution = True
                if safety_dec:
                    safety_dec.decision = SafetyDecisionType.SAFE
                    safety_dec.human_intervention_result = human_res
            else:
                safety_out.approved_for_execution = False
                if safety_dec:
                    safety_dec.decision = SafetyDecisionType.UNSAFE
                    safety_dec.human_intervention_result = human_res

        return {
            "human_intervention_result": human_res,
            "safety_output": safety_out,
            "safety_decision": safety_dec,
        }

    # ------------------------------------------------------------------
    # 8B. Human Intervention Router (Conditional Edge)
    # ------------------------------------------------------------------
    def human_intervention_router(state: HRIState) -> str:
        human_res: Optional[HumanInterventionResult] = state.get("human_intervention_result")
        if human_res and human_res.approved:
            return "APPROVED"
        return "REJECTED"

    # ------------------------------------------------------------------
    # 9. Robot Execution Node
    # ------------------------------------------------------------------
    def robot_execution_node(state: HRIState) -> Dict[str, Any]:
        safety_out: Optional[SafetyAgentOutput] = state.get("safety_output")
        errors = state.get("errors") or []

        # Strict safety assertion
        if not safety_out or not safety_out.approved_for_execution:
            err = "Controller execution blocked: Safety Agent did not approve task."
            logger.error(err)
            errors.append(err)
            rejected_res = ExecutionResult(
                status=ExecutionStatus.REJECTED,
                command="STOP",
                error=err,
            )
            return {
                "errors": errors,
                "execution_result": rejected_res,
                "controller_output": ControllerAgentOutput(
                    agent_name=_controller.name,
                    agent_type=_controller.agent_type.value,
                    success=False,
                    execution_result=rejected_res,
                    execution_status=ExecutionStatus.REJECTED.value,
                    error_message=err,
                ),
            }

        plan: Optional[TaskPlan] = state.get("task_plan")
        task: Optional[MultimodalTask] = state.get("task")

        # Map plan or task to controller command
        cmd_str = "MOVE_FORWARD"
        params: Dict[str, Any] = {}

        if plan and plan.steps:
            first_step = plan.steps[0]
            step_action = str(first_step.action).upper()
            params = dict(first_step.parameters)

            if "STOP" in step_action:
                cmd_str = "STOP"
            elif "PICK" in step_action:
                cmd_str = "PICK"
            elif "PLACE" in step_action:
                cmd_str = "PLACE"
            elif "TURN_LEFT" in step_action:
                cmd_str = "TURN_LEFT"
            elif "TURN_RIGHT" in step_action:
                cmd_str = "TURN_RIGHT"
            elif "NAVIGATE" in step_action or "MOVE" in step_action:
                cmd_str = "MOVE_FORWARD"
            elif "INSPECT" in step_action:
                cmd_str = "INSPECT"
            else:
                cmd_str = step_action
        elif task:
            action_name = str(task.action).lower()
            if "stop" in action_name:
                cmd_str = "STOP"
            elif "pick" in action_name:
                cmd_str = "PICK"
            elif "place" in action_name:
                cmd_str = "PLACE"
            elif "inspect" in action_name:
                cmd_str = "INSPECT"
            else:
                cmd_str = "MOVE_FORWARD"
            params = {"target": task.target_object}

        ctrl_input = ControllerAgentInput(
            command=cmd_str,
            parameters=params,
            is_safety_approved=True,
            safety_metadata={
                "safety_state": safety_out.safety_state,
                "decision": state.get("safety_decision_type", "SAFE"),
            },
        )

        ctrl_output: ControllerAgentOutput = _controller.process(ctrl_input)
        exec_res = ctrl_output.execution_result

        # Store completed execution feedback in Memory
        if ctrl_output.success and task:
            _memory.store_entry(
                content={
                    "action": task.action,
                    "target": task.target_object,
                    "status": "COMPLETED",
                    "execution_status": ctrl_output.execution_status,
                },
                entry_type=MemoryEntryType.TASK,
                source="controller_feedback",
            )
        elif not ctrl_output.success:
            err_msg = f"Robot Controller execution failed: {ctrl_output.error_message}"
            logger.error(err_msg)
            errors.append(err_msg)

        return {
            "controller_input": ctrl_input,
            "controller_output": ctrl_output,
            "execution_result": exec_res,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Graph Wiring
    # ------------------------------------------------------------------
    builder.add_node("input_validation", input_validation_node)
    builder.add_node("multimodal_fusion", multimodal_fusion_node)
    builder.add_node("task_interpretation", task_interpretation_node)
    builder.add_node("memory_context", memory_context_node)
    builder.add_node("task_planning", task_planning_node)
    builder.add_node("safety_check", safety_check_node)
    builder.add_node("human_intervention", human_intervention_node)
    builder.add_node("robot_execution", robot_execution_node)

    # Sequential edges
    builder.add_edge(START, "input_validation")
    builder.add_edge("input_validation", "multimodal_fusion")
    builder.add_edge("multimodal_fusion", "task_interpretation")
    builder.add_edge("task_interpretation", "memory_context")
    builder.add_edge("memory_context", "task_planning")
    builder.add_edge("task_planning", "safety_check")

    # Conditional safety router
    builder.add_conditional_edges(
        "safety_check",
        safety_decision_router,
        {
            "APPROVED": "robot_execution",
            "HUMAN_APPROVAL_REQUIRED": "human_intervention",
            "STOP": END,
        },
    )

    # Conditional human intervention router
    builder.add_conditional_edges(
        "human_intervention",
        human_intervention_router,
        {
            "APPROVED": "robot_execution",
            "REJECTED": END,
        },
    )

    builder.add_edge("robot_execution", END)

    return builder.compile()


class IntegratedHRIPipeline:
    """
    High-level orchestrator interface running the compiled multi-agent LangGraph workflow.
    """

    def __init__(
        self,
        fusion_engine: Optional[MultimodalFusionEngine] = None,
        memory_agent: Optional[MemoryAgent] = None,
        planner_agent: Optional[TaskPlannerAgent] = None,
        safety_agent: Optional[SafetyAgent] = None,
        controller_agent: Optional[RobotControllerAgent] = None,
        human_interface: Optional[BaseHumanInterventionInterface] = None,
    ):
        self.fusion_engine = fusion_engine or MultimodalFusionEngine()
        self.memory_agent = memory_agent
        self.planner_agent = planner_agent
        self.safety_agent = safety_agent
        self.controller_agent = controller_agent
        self.human_interface = human_interface

        # Initialize agents once
        if self.memory_agent and self.memory_agent.status != AgentStatus.READY:
            self.memory_agent.initialize()
        if self.planner_agent and self.planner_agent.status != AgentStatus.READY:
            self.planner_agent.initialize()
        if self.safety_agent and self.safety_agent.status != AgentStatus.READY:
            self.safety_agent.initialize()
        if self.controller_agent and self.controller_agent.status != AgentStatus.READY:
            self.controller_agent.initialize()

        self.graph = create_integrated_hri_graph(
            fusion_engine=self.fusion_engine,
            memory_agent=self.memory_agent,
            planner_agent=self.planner_agent,
            safety_agent=self.safety_agent,
            controller_agent=self.controller_agent,
            human_interface=self.human_interface,
        )

    def process(self, input_data: Union[HRIState, CoordinatorAgentInput, Dict[str, Any]]) -> HRIState:
        """
        Execute the complete multi-agent HRI workflow.
        """
        if isinstance(input_data, CoordinatorAgentInput):
            initial_state: HRIState = {
                "voice_output": input_data.voice_output,
                "vision_output": input_data.vision_output,
                "gesture_output": input_data.gesture_output,
                "obstacle_distance": None,
                "human_distance": None,
                "emergency_stop": False,
                "sensor_validity": {"camera": True, "proximity": True},
                "target_confidence": None,
                "task": None,
                "task_status": None,
                "metadata": {"session_id": input_data.session_id},
                "errors": [],
            }
        elif isinstance(input_data, dict):
            initial_state = {
                "voice_output": input_data.get("voice_output"),
                "vision_output": input_data.get("vision_output"),
                "gesture_output": input_data.get("gesture_output"),
                "obstacle_distance": input_data.get("obstacle_distance"),
                "human_distance": input_data.get("human_distance"),
                "emergency_stop": input_data.get("emergency_stop", False),
                "sensor_validity": input_data.get("sensor_validity", {"camera": True, "proximity": True}),
                "target_confidence": input_data.get("target_confidence"),
                "environment_context": input_data.get("environment_context", {}),
                "robot_state": input_data.get("robot_state", {}),
                "task": input_data.get("task"),
                "task_status": input_data.get("task_status"),
                "metadata": input_data.get("metadata", {}),
                "errors": input_data.get("errors", []),
            }
        else:
            initial_state = input_data

        final_state: HRIState = self.graph.invoke(initial_state)
        return final_state
