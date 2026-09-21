#!/usr/bin/env python3
"""
End-to-End Integrated LangGraph Pipeline Demonstration.

Demonstrates all 7 core HRI scenarios without requiring physical hardware:
1. Voice + Vision → Navigation
2. Voice + Vision → Pick and Place
3. Target Not Found (No False Execution)
4. Hard Safety Violation (Distance < 0.35m -> Emergency Stop)
5. Human Approval Required (Contextual Risk -> Human Approved & Rejected)
6. Memory Context (Pronoun Resolution: "Go to bottle" -> "Pick it up")
7. Robot Controller Failure Handling (Safe Termination)
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.controller.agent import RobotControllerAgent
from src.agents.controller.controller import RobotController
from src.agents.controller.ros2_interface import MockRobotController
from src.agents.controller.schemas import ControllerAgentInput, ExecutionStatus
from src.agents.gesture.schemas import (
    GestureAgentOutput,
    GestureDirection,
    GestureType,
    HandBBox,
    RecognizedGesture,
)
from src.agents.memory.agent import MemoryAgent
from src.agents.memory.memory_store import InMemoryStore
from src.agents.planner.agent import TaskPlannerAgent
from src.agents.safety.agent import SafetyAgent
from src.agents.safety.human_intervention import AutoApprovalStub
from src.agents.safety.llm_reasoner import MockLLMReasoner
from src.agents.safety.schemas import (
    HumanInterventionResult,
    RiskLevel,
    SafetyDecisionType,
)
from src.agents.vision.schemas import (
    BoundingBox,
    DetectedObject,
    ProximityLevel,
    SpatialSector,
    VisionAgentOutput,
)
from src.agents.voice.schemas import SpeechIntent, UrgencyLevel, VoiceAgentOutput
from src.pipeline.coordinator_integration import IntegratedHRIPipeline


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def print_step(icon: str, name: str, details: list[str]):
    print(f"\n{icon} [{name.upper()}]")
    for d in details:
        print(f"    • {d}")


def display_pipeline_execution(title: str, state_out: dict):
    print_header(title)

    # 1. Perception
    perception_details = []
    if state_out.get("voice_output"):
        v = state_out["voice_output"]
        perception_details.append(f'Voice Transcript : "{v.transcript}" (Conf: {v.confidence:.2f})')
        if v.speech_intent:
            perception_details.append(f"Voice Intent     : {v.speech_intent.action} -> {v.speech_intent.target_object}")
    if state_out.get("vision_output"):
        vis = state_out["vision_output"]
        objs = [f"{o.label} ({o.spatial_sector.value}, {o.proximity.value})" for o in vis.detected_objects]
        perception_details.append(f"Vision Objects   : {', '.join(objs) if objs else 'None detected'}")
    if state_out.get("gesture_output"):
        g = state_out["gesture_output"]
        g_name = g.gesture.value if hasattr(g.gesture, "value") else str(g.gesture)
        g_dir = g.direction.value if hasattr(g.direction, "value") else str(g.direction)
        perception_details.append(f"Gesture          : {g_name} -> direction: {g_dir}")

    print_step("🎙️👁️", "1. Perception Inputs", perception_details)

    # 2. Coordinator
    coord_details = []
    task = state_out.get("task")
    if task:
        coord_details.append(f"Interpreted Task : {task.action} -> {task.target_object}")
        coord_details.append(f"Task Status      : {task.task_status.value}")
        coord_details.append(f"Target Confirmed : {task.target_confirmed}")
        coord_details.append(f"Reasoning        : {task.reasoning}")
    else:
        coord_details.append("No task produced")
    print_step("🧠", "2. Coordinator Fusion & Interpretation", coord_details)

    # 3. Memory
    mem_details = []
    mem_ctx = state_out.get("memory_context")
    if mem_ctx:
        mem_details.append(f"Context Entity   : {mem_ctx.get('current_focus', 'None')}")
        mem_details.append(f"Recent Entities  : {mem_ctx.get('recent_entities', [])}")
        mem_details.append(f"Resolved Pronoun : {mem_ctx.get('resolved_target', 'N/A')}")
    else:
        mem_details.append("No prior memory context required")
    print_step("💾", "3. Memory Agent Context", mem_details)

    # 4. Planner
    plan_details = []
    plan = state_out.get("task_plan")
    if plan:
        plan_details.append(f"Plan ID          : {plan.plan_id}")
        plan_details.append(f"Estimated Cost   : {plan.total_estimated_duration:.2f}s")
        plan_details.append(f"Generated Steps  : {len(plan.steps)} step(s)")
        for idx, s in enumerate(plan.steps, 1):
            act_name = s.action.value if hasattr(s.action, "value") else str(s.action)
            plan_details.append(f"    [{idx}] {act_name} -> target: {s.target} (params: {s.parameters})")
    else:
        plan_details.append("No plan generated (pipeline stopped or target missing)")
    print_step("📋", "4. Task Planner", plan_details)

    # 5. Safety
    safety_details = []
    s_out = state_out.get("safety_output")
    s_dec = state_out.get("safety_decision")
    if s_dec:
        safety_details.append(f"Safety Decision  : {s_dec.decision.value}")
        safety_details.append(f"Risk Level       : {s_dec.risk_level.value}")
        safety_details.append(f"Approved For Exec: {s_out.approved_for_execution if s_out else False}")
        safety_details.append(f"Triggered Rules  : {s_dec.triggered_rules or 'None'}")
        safety_details.append(f"Reason           : {s_dec.reason}")
    else:
        safety_details.append("Safety check was not reached (bypassed safely)")
    print_step("🛡️", "5. Safety Agent", safety_details)

    # 6. Human Intervention (if triggered)
    if state_out.get("human_intervention_required"):
        hi = state_out.get("human_intervention_result")
        hi_details = [
            "Human Approval Triggered: Yes",
            f"Intervention Reason    : {hi.reason if hi else 'None'}",
            f"Approved By Operator   : {hi.approved if hi else False}",
        ]
        print_step("👤", "6. Human Intervention", hi_details)

    # 7. Controller
    ctrl_details = []
    ctrl_out = state_out.get("controller_output")
    exec_res = state_out.get("execution_result")
    if ctrl_out:
        ctrl_details.append(f"Execution Success: {ctrl_out.success}")
        ctrl_details.append(f"Execution Status : {exec_res.status.value if exec_res else 'N/A'}")
        ctrl_details.append(f"Executed Command : {exec_res.command if exec_res else 'N/A'}")
        ctrl_details.append(f"Twist / Movement : {exec_res.twist if exec_res else 'N/A'}")
    else:
        ctrl_details.append("Robot Controller was NOT invoked (Safe state preserved)")
    print_step("🤖", "7. Robot Controller", ctrl_details)

    # 8. Final Result
    errors = state_out.get("errors", [])
    result_status = "SUCCESS" if ctrl_out and ctrl_out.success else ("STOPPED SAFELY" if not errors else "FAILED")
    print(f"\n🏁 [FINAL RESULT]: {result_status}")
    if errors:
        print(f"    Errors Encountered: {errors}")


def create_demo_pipeline(human_approver=None) -> IntegratedHRIPipeline:
    memory = MemoryAgent(name="DemoMemory", store=InMemoryStore())
    planner = TaskPlannerAgent(name="DemoPlanner")
    safety = SafetyAgent(
        name="DemoSafety",
        human_interface=human_approver or AutoApprovalStub(auto_approve=True),
        config={
            "safety": {
                "danger_distance": 0.35,
                "safe_distance": 0.50,
                "confidence_threshold": 0.60,
                "sensor_policy": "human_approval_on_missing",
            }
        },
    )
    controller_backend = RobotController()
    controller_agent = RobotControllerAgent(name="DemoController", controller=controller_backend)

    return IntegratedHRIPipeline(
        memory_agent=memory,
        planner_agent=planner,
        safety_agent=safety,
        controller_agent=controller_agent,
    )


def main():
    print_header("Adaptive Multimodal HRI - End-to-End LangGraph Pipeline Demo")
    print("Setting up all 8 system agents into unified LangGraph pipeline...")

    pipeline = create_demo_pipeline()
    print("Pipeline compiled successfully. Running 7 demonstration scenarios...\n")

    # =========================================================================
    # SCENARIO 1: Voice + Vision -> Navigation
    # =========================================================================
    voice_1 = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.92,
        speech_intent=SpeechIntent(action="navigate_to", target_object="bottle", confidence=0.92),
    )
    vision_1 = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.91,
                bbox=BoundingBox(x1=200, y1=100, x2=300, y2=400),
                spatial_sector=SpatialSector.CENTER,
                proximity=ProximityLevel.MEDIUM,
            )
        ],
    )
    s1 = pipeline.process({"voice_output": voice_1, "vision_output": vision_1, "obstacle_distance": 1.50})
    display_pipeline_execution("Scenario 1: Voice + Vision -> Navigation (Clear Path)", s1)

    # =========================================================================
    # SCENARIO 2: Voice + Vision -> Pick and Place
    # =========================================================================
    voice_2 = VoiceAgentOutput(
        transcript="Pick up the phone",
        is_speech_detected=True,
        confidence=0.95,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="phone", confidence=0.95),
    )
    vision_2 = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=2,
                label="phone",
                confidence=0.94,
                bbox=BoundingBox(x1=80, y1=120, x2=180, y2=250),
                spatial_sector=SpatialSector.LEFT,
                proximity=ProximityLevel.NEAR,
            )
        ],
    )
    s2 = pipeline.process({"voice_output": voice_2, "vision_output": vision_2, "obstacle_distance": 0.80})
    display_pipeline_execution("Scenario 2: Voice + Vision -> Pick and Place", s2)

    # =========================================================================
    # SCENARIO 3: Target Not Detected (Voice says bottle, vision sees cup only)
    # =========================================================================
    voice_3 = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="navigate_to", target_object="bottle", confidence=0.90),
    )
    vision_3 = VisionAgentOutput(
        confidence=0.85,
        detected_objects=[
            DetectedObject(
                object_id=9,
                label="cup",
                confidence=0.85,
                bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150),
                spatial_sector=SpatialSector.RIGHT,
                proximity=ProximityLevel.FAR,
            )
        ],
    )
    s3 = pipeline.process({"voice_output": voice_3, "vision_output": vision_3, "obstacle_distance": 1.50})
    display_pipeline_execution("Scenario 3: Target Not Found (Safe Halt - No False Movement)", s3)

    # =========================================================================
    # SCENARIO 4: Hard Safety Violation (Obstacle Distance = 0.20m < 0.35m Stop Threshold)
    # =========================================================================
    voice_4 = VoiceAgentOutput(
        transcript="Go to the bottle",
        is_speech_detected=True,
        confidence=0.92,
        speech_intent=SpeechIntent(action="navigate_to", target_object="bottle", confidence=0.92),
    )
    vision_4 = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.91,
                bbox=BoundingBox(x1=200, y1=100, x2=300, y2=400),
                spatial_sector=SpatialSector.CENTER,
                proximity=ProximityLevel.NEAR,
            )
        ],
    )
    s4 = pipeline.process({"voice_output": voice_4, "vision_output": vision_4, "obstacle_distance": 0.20})
    display_pipeline_execution("Scenario 4: Hard Safety Violation (Obstacle 0.20m < 0.35m -> Emergency Stop)", s4)

    # =========================================================================
    # SCENARIO 5: Human Approval (Uncertain Perception -> Operator Approves & Rejects)
    # =========================================================================
    pipeline_human_appr = create_demo_pipeline(
        human_approver=AutoApprovalStub(auto_approve=True, approver_id="supervisor_alice")
    )
    voice_5 = VoiceAgentOutput(
        transcript="Navigate to chair",
        is_speech_detected=True,
        confidence=0.45,
        speech_intent=SpeechIntent(action="navigate_to", target_object="chair", confidence=0.45),
    )
    vision_5 = VisionAgentOutput(
        confidence=0.45,
        detected_objects=[
            DetectedObject(
                object_id=3,
                label="chair",
                confidence=0.45,
                bbox=BoundingBox(x1=200, y1=100, x2=300, y2=400),
                spatial_sector=SpatialSector.CENTER,
                proximity=ProximityLevel.MEDIUM,
            )
        ],
    )
    s5_approved = pipeline_human_appr.process(
        {"voice_output": voice_5, "vision_output": vision_5, "obstacle_distance": 1.20}
    )
    display_pipeline_execution("Scenario 5A: Uncertain Perception -> Human Approval Required & APPROVED", s5_approved)

    pipeline_human_rej = create_demo_pipeline(
        human_approver=AutoApprovalStub(auto_approve=False, approver_id="safety_auditor_bob")
    )
    s5_rejected = pipeline_human_rej.process(
        {"voice_output": voice_5, "vision_output": vision_5, "obstacle_distance": 1.20}
    )
    display_pipeline_execution("Scenario 5B: Uncertain Perception -> Human Approval Required & REJECTED", s5_rejected)

    # =========================================================================
    # SCENARIO 6: Memory Context & Pronoun Resolution ("Go to bottle" -> "Pick it up")
    # =========================================================================
    # Step A: User references bottle
    pipeline.process({"voice_output": voice_1, "vision_output": vision_1, "obstacle_distance": 1.50})
    # Step B: User says "Pick it up"
    voice_6 = VoiceAgentOutput(
        transcript="Pick it up",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(action="pick_and_place", target_object="it", confidence=0.90),
    )
    vision_6 = VisionAgentOutput(
        confidence=0.90,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.91,
                bbox=BoundingBox(x1=200, y1=100, x2=300, y2=400),
                spatial_sector=SpatialSector.CENTER,
                proximity=ProximityLevel.NEAR,
            )
        ],
    )
    s6 = pipeline.process({"voice_output": voice_6, "vision_output": vision_6, "obstacle_distance": 0.80})
    display_pipeline_execution("Scenario 6: Memory Context & Pronoun Resolution ('it' -> 'bottle')", s6)

    # =========================================================================
    # SCENARIO 7: Controller Failure Handling (Simulated Actuator Error)
    # =========================================================================
    ctrl_backend = RobotController()
    ctrl_backend.backend.connected = False  # Disconnect hardware backend to simulate failure

    faulty_ctrl = RobotControllerAgent(name="FaultyCtrl", controller=ctrl_backend)

    pipeline_faulty = IntegratedHRIPipeline(
        memory_agent=MemoryAgent(name="M", store=InMemoryStore()),
        planner_agent=TaskPlannerAgent(name="P"),
        safety_agent=SafetyAgent(name="S", human_interface=AutoApprovalStub(auto_approve=True)),
        controller_agent=faulty_ctrl,
    )
    s7 = pipeline_faulty.process({"voice_output": voice_1, "vision_output": vision_1, "obstacle_distance": 1.50})
    display_pipeline_execution("Scenario 7: Robot Controller Failure Handled Safely", s7)

    print_header("All 7 Scenarios Demonstrated Successfully!")


if __name__ == "__main__":
    main()
