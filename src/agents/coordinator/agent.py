"""Coordinator Agent: central orchestration component in the Decision-Making Layer."""

from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.agents.coordinator.fusion import MultimodalFusionEngine
from src.agents.coordinator.graph import create_coordinator_graph
from src.agents.coordinator.schemas import (
    CoordinatorAgentInput,
    CoordinatorAgentOutput,
    HRIState,
    TaskStatus,
)
from src.common.exceptions import AgentExecutionError
from src.common.schemas import AgentType


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent in the Decision-Making Layer.
    Orchestrates multimodal perception outputs (Voice, Vision, and optional Gesture)
    via LangGraph StateGraph, performs multimodal fusion, validates target objects,
    and produces structured high-level tasks for future downstream agents.
    """

    def __init__(
        self,
        name: str = "CoordinatorAgent",
        config: Optional[Dict[str, Any]] = None,
        fusion_engine: Optional[MultimodalFusionEngine] = None,
    ):
        super().__init__(
            name=name,
            agent_type=AgentType.COORDINATOR,
            version="0.1.0",
            description="Coordinates multimodal perception outputs and determines structured high-level tasks.",
            config=config or {},
        )

        self.fusion_engine = fusion_engine
        self.graph = None

    def _initialize(self) -> bool:
        """Initialize fusion engine and compile LangGraph StateGraph workflow."""
        coord_cfg = self.config.get("coordinator", {})

        if self.fusion_engine is None:
            min_conf = coord_cfg.get("min_confidence_threshold", 0.40)
            high_conf = coord_cfg.get("high_confidence_threshold", 0.60)
            ambig_thresh = coord_cfg.get("ambiguity_threshold", 0.10)
            v_weight = coord_cfg.get("voice_weight", 0.55)
            vis_weight = coord_cfg.get("vision_weight", 0.35)
            g_weight = coord_cfg.get("gesture_weight", 0.10)
            req_vis = coord_cfg.get("require_visual_target_confirmation", True)
            self.fusion_engine = MultimodalFusionEngine(
                min_confidence_threshold=min_conf,
                high_confidence_threshold=high_conf,
                ambiguity_threshold=ambig_thresh,
                voice_weight=v_weight,
                vision_weight=vis_weight,
                gesture_weight=g_weight,
                require_visual_target_confirmation=req_vis,
            )

        self.graph = create_coordinator_graph(fusion_engine=self.fusion_engine)
        self.logger.info("CoordinatorAgent initialized and LangGraph StateGraph compiled successfully.")
        return True

    def _process(self, input_data: Any) -> CoordinatorAgentOutput:
        """
        Execute LangGraph orchestration workflow on multimodal perception inputs.
        """
        if self.graph is None:
            raise AgentExecutionError("CoordinatorAgent graph is not compiled. Please initialize() the agent first.")

        if not isinstance(input_data, CoordinatorAgentInput):
            if isinstance(input_data, dict):
                input_data = CoordinatorAgentInput(
                    session_id=input_data.get("session_id", "default_session"),
                    voice_output=input_data.get("voice_output"),
                    vision_output=input_data.get("vision_output"),
                    gesture_output=input_data.get("gesture_output"),
                )
            else:
                raise AgentExecutionError(
                    f"CoordinatorAgent expects CoordinatorAgentInput or dict, got {type(input_data).__name__}"
                )

        initial_state: HRIState = {
            "voice_output": input_data.voice_output,
            "vision_output": input_data.vision_output,
            "gesture_output": input_data.gesture_output,
            "task": None,
            "task_status": None,
            "metadata": {"session_id": input_data.session_id},
            "errors": [],
        }

        final_state: HRIState = self.graph.invoke(initial_state)

        fused_task = final_state.get("task")
        task_status_str = final_state.get("task_status") or (
            fused_task.task_status.value
            if fused_task and hasattr(fused_task.task_status, "value")
            else TaskStatus.INVALID_INPUT.value
        )
        confidence_val = fused_task.confidence if fused_task else 0.0

        return CoordinatorAgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type.value,
            success=True,
            confidence=confidence_val,
            fused_task=fused_task,
            task_status=task_status_str,
            hri_state=final_state,
        )

    def _reset(self) -> None:
        """Reset agent internal state."""
        pass

    def _shutdown(self) -> None:
        """Release Coordinator resources."""
        self.graph = None
