"""LangGraph StateGraph workflow definition for the Coordinator Agent."""

from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from src.agents.coordinator.fusion import MultimodalFusionEngine
from src.agents.coordinator.schemas import HRIState, MultimodalTask, TaskStatus


def create_coordinator_graph(
    fusion_engine: Optional[MultimodalFusionEngine] = None,
) -> Any:
    """
    Construct and compile the LangGraph StateGraph workflow for Coordinator task orchestration.
    
    Graph Architecture:
        START
          ↓
        input_validation
          ↓
        multimodal_fusion
          ↓
        task_interpretation
          ↓
        END
    """
    _fusion = fusion_engine or MultimodalFusionEngine()

    builder = StateGraph(HRIState)

    # ------------------------------------------------------------------
    # Node Definitions
    # ------------------------------------------------------------------

    def input_validation_node(state: HRIState) -> Dict[str, Any]:
        """Validate presence of perception inputs and initialize metadata."""
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

    def multimodal_fusion_node(state: HRIState) -> Dict[str, Any]:
        """Fuse voice, vision, and optional gesture inputs into a candidate MultimodalTask."""
        voice = state.get("voice_output")
        vision = state.get("vision_output")
        gesture = state.get("gesture_output")

        task: MultimodalTask = _fusion.fuse(
            voice_output=voice,
            vision_output=vision,
            gesture_output=gesture,
        )
        return {"task": task}

    def task_interpretation_node(state: HRIState) -> Dict[str, Any]:
        """Finalize task status and structure the task for downstream consumers."""
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
    # Edge Wiring
    # ------------------------------------------------------------------

    builder.add_node("input_validation", input_validation_node)
    builder.add_node("multimodal_fusion", multimodal_fusion_node)
    builder.add_node("task_interpretation", task_interpretation_node)

    builder.add_edge(START, "input_validation")
    builder.add_edge("input_validation", "multimodal_fusion")
    builder.add_edge("multimodal_fusion", "task_interpretation")
    builder.add_edge("task_interpretation", END)

    return builder.compile()
