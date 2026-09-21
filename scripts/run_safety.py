#!/usr/bin/env python3
"""Standalone Demonstration runner for the Safety Agent in the Adaptive Multimodal HRI Framework."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.coordinator.schemas import MultimodalTask, TaskStatus
from src.agents.safety.agent import SafetyAgent
from src.agents.safety.human_intervention import AutoApprovalStub
from src.agents.safety.llm_reasoner import MockLLMReasoner
from src.agents.safety.schemas import (
    RiskLevel,
    SafetyAgentInput,
    SafetyAgentOutput,
    SafetyDecisionType,
    SafetyState,
)
from src.agents.vision.schemas import ProximityLevel, SpatialSector
from src.agents.voice.schemas import UrgencyLevel
from src.utils.config import load_config


def print_divider(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title.upper()}")
        print("=" * 70)


def print_decision(scenario_title: str, safety_in: SafetyAgentInput, output: SafetyAgentOutput):
    print_divider(scenario_title)
    task_desc = f"{safety_in.task.action} (target: {safety_in.task.target_object})" if safety_in.task else "[No Task]"
    print(f"📋  REQUESTED TASK     : {task_desc}")
    print(f"📏  OBSTACLE DISTANCE  : {safety_in.obstacle_distance if safety_in.obstacle_distance is not None else 'N/A'} m")
    print(f"🎯  TARGET CONFIDENCE  : {safety_in.target_confidence if safety_in.target_confidence is not None else (safety_in.task.confidence if safety_in.task else 'N/A')}")
    print(f"🛑  EMERGENCY STOP     : {safety_in.emergency_stop}")
    print("-" * 70)
    dec = output.safety_decision

    if dec:
        print(f"🛡️  SAFETY DECISION    : {dec.decision.value if hasattr(dec.decision, 'value') else dec.decision}")
        print(f"⚠️  RISK LEVEL         : {dec.risk_level.value if hasattr(dec.risk_level, 'value') else dec.risk_level}")
        print(f"🚦  EXECUTION APPROVED : {output.approved_for_execution}")
        print(f"⚡  HARD VIOLATION     : {dec.is_hard_violation}")
        print(f"🤖  LLM USED           : {dec.llm_used}")
        print(f"👤  HUMAN INTERVENTION : {dec.human_intervention_required} (Approved: {dec.human_intervention_result.approved if dec.human_intervention_result else 'N/A'})")
        print(f"📝  EXPLANATION REASON : {dec.reason}")
        if dec.triggered_rules:
            print(f"📌  TRIGGERED RULES    : {', '.join(dec.triggered_rules)}")
    else:
        print("    [None]")


def main():
    print_divider("Adaptive Multimodal HRI Framework - Safety Agent Demo")
    print("Initializing Safety Agent with deterministic rules, hysteresis monitor, and mock reasoner...")

    config = load_config(PROJECT_ROOT / "configs" / "safety_config.yaml")
    human_stub = AutoApprovalStub(auto_approve=True)
    safety_agent = SafetyAgent(name="SafetyDemo", config=config, human_interface=human_stub)
    safety_agent.initialize()

    # ------------------------------------------------------------------
    # SCENARIO 1: Normal Safe Condition (Navigate to bottle, clear distance)
    # ------------------------------------------------------------------
    task_1 = MultimodalTask(
        action="navigate_to",
        target_object="bottle",
        target_confirmed=True,
        spatial_sector=SpatialSector.CENTER,
        proximity=ProximityLevel.MEDIUM,
        confidence=0.92,
        task_status=TaskStatus.VALID,
    )
    in_1 = SafetyAgentInput(task=task_1, obstacle_distance=1.20, target_confidence=0.92)
    out_1 = safety_agent.process(in_1)
    print_decision("Scenario 1: Normal Safe Condition (Clear obstacle, high confidence)", in_1, out_1)

    # ------------------------------------------------------------------
    # SCENARIO 2: Critical Obstacle Distance (0.20m < danger_threshold 0.35m)
    # ------------------------------------------------------------------
    task_2 = MultimodalTask(
        action="navigate_to",
        target_object="bottle",
        target_confirmed=True,
        confidence=0.90,
        task_status=TaskStatus.VALID,
    )
    in_2 = SafetyAgentInput(task=task_2, obstacle_distance=0.20, target_confidence=0.90)
    out_2 = safety_agent.process(in_2)
    print_decision("Scenario 2: Critical Obstacle Violation (Hard Stop at 0.20m)", in_2, out_2)

    # ------------------------------------------------------------------
    # SCENARIO 3: Low Target Confidence (0.42 < 0.60 threshold)
    # ------------------------------------------------------------------
    task_3 = MultimodalTask(
        action="pick_and_place",
        target_object="cup",
        target_confirmed=True,
        confidence=0.42,
        task_status=TaskStatus.LOW_CONFIDENCE,
    )
    in_3 = SafetyAgentInput(task=task_3, obstacle_distance=1.00, target_confidence=0.42)
    out_3 = safety_agent.process(in_3)
    print_decision("Scenario 3: Low Target Confidence (Escalated to Human Review)", in_3, out_3)

    # ------------------------------------------------------------------
    # SCENARIO 4: Conflicting Multimodal Task (Modality Conflict)
    # ------------------------------------------------------------------
    task_4 = MultimodalTask(
        action="navigate_to",
        target_object="bottle",
        target_confirmed=True,
        spatial_sector=SpatialSector.RIGHT,
        confidence=0.75,
        task_status=TaskStatus.MODALITY_CONFLICT,
        reasoning="Voice requested LEFT, but vision detected target at RIGHT.",
    )
    in_4 = SafetyAgentInput(task=task_4, obstacle_distance=1.10)
    out_4 = safety_agent.process(in_4)
    print_decision("Scenario 4: Conflicting Multimodal Input (Human Approval Required)", in_4, out_4)

    # ------------------------------------------------------------------
    # SCENARIO 5: Emergency Stop Active
    # ------------------------------------------------------------------
    task_5 = MultimodalTask(
        action="navigate_to",
        target_object="chair",
        target_confirmed=True,
        confidence=0.95,
        task_status=TaskStatus.VALID,
    )
    in_5 = SafetyAgentInput(task=task_5, obstacle_distance=1.50, emergency_stop=True)
    out_5 = safety_agent.process(in_5)
    print_decision("Scenario 5: Hardware Emergency Stop Active (Hard Stop)", in_5, out_5)

    # ------------------------------------------------------------------
    # SCENARIO 6: Threshold + Hysteresis Proximity in Action
    # ------------------------------------------------------------------
    print_divider("Scenario 6: Threshold + Hysteresis Proximity Deadband (0.42m)")
    print("Step 6A: Previous state was SAFE, obstacle at 0.42m (in [0.35, 0.50]m deadband)...")
    in_6a = SafetyAgentInput(task=task_1, obstacle_distance=0.42, previous_safety_state=SafetyState.SAFE)
    out_6a = safety_agent.process(in_6a)
    print(f"  -> Step 6A Result: Decision = {out_6a.safety_decision.decision.value}, State = {out_6a.safety_state} (Retained SAFE)")

    print("\nStep 6B: Previous state was UNSAFE/STOPPED, obstacle at 0.42m (in [0.35, 0.50]m deadband)...")
    in_6b = SafetyAgentInput(task=task_1, obstacle_distance=0.42, previous_safety_state=SafetyState.UNSAFE)
    out_6b = safety_agent.process(in_6b)
    print(f"  -> Step 6B Result: Decision = {out_6b.safety_decision.decision.value}, State = {out_6b.safety_state} (Retained UNSAFE/STOP until clearance above 0.50m)")

    # ------------------------------------------------------------------
    # SCENARIO 7: Contextual LLM Reasoning Layer
    # ------------------------------------------------------------------
    mock_llm = MockLLMReasoner(
        risk_level=RiskLevel.MEDIUM,
        contextually_safe=False,
        uncertain=True,
        reason="Contextual ambiguity: user command lacks specific placement destination in cluttered environment.",
        confidence=0.65,
        requires_human=True,
    )
    safety_agent_llm = SafetyAgent(
        name="SafetyWithLLM",
        config=config,
        llm_reasoner=mock_llm,
        human_interface=human_stub,
    )
    safety_agent_llm.initialize()

    task_7 = MultimodalTask(
        action="pick_and_place",
        target_object="glass",
        target_confirmed=True,
        confidence=0.88,
        task_status=TaskStatus.VALID,
    )
    in_7 = SafetyAgentInput(task=task_7, obstacle_distance=0.90)
    out_7 = safety_agent_llm.process(in_7)
    print_decision("Scenario 7: LLM Contextual Risk Evaluation with Human Intervention", in_7, out_7)

    print_divider("Safety Agent Demo completed successfully!")


if __name__ == "__main__":
    main()
