"""Comprehensive unit and integration tests for the Safety Agent, deterministic rules,
threshold + hysteresis proximity monitor, LLM contextual reasoning, and human-in-the-loop review.
"""

from pathlib import Path
import pytest

from src.agents.coordinator.schemas import MultimodalTask, TaskStatus
from src.agents.safety.agent import SafetyAgent
from src.agents.safety.human_intervention import AutoApprovalStub
from src.agents.safety.llm_reasoner import MockLLMReasoner, OllamaLLMReasoner
from src.agents.safety.monitor import HysteresisMonitor
from src.agents.safety.rules import DeterministicRuleEngine
from src.agents.safety.schemas import (
    RiskLevel,
    SafetyAgentInput,
    SafetyAgentOutput,
    SafetyDecisionType,
    SafetyState,
)
from src.agents.vision.schemas import ProximityLevel, SpatialSector
from src.common.schemas import AgentStatus, AgentType


@pytest.fixture
def default_task():
    return MultimodalTask(
        action="navigate_to",
        target_object="bottle",
        target_confirmed=True,
        spatial_sector=SpatialSector.CENTER,
        proximity=ProximityLevel.MEDIUM,
        confidence=0.90,
        task_status=TaskStatus.VALID,
    )


@pytest.fixture
def safety_agent():
    agent = SafetyAgent(
        name="TestSafetyAgent",
        config={
            "safety": {
                "danger_distance": 0.35,
                "safe_distance": 0.50,
                "confidence_threshold": 0.60,
                "sensor_policy": "human_approval_on_missing",
            }
        },
        human_interface=AutoApprovalStub(auto_approve=True),
    )
    agent.initialize()
    return agent


# --------------------------------------------------------------------------
# TEST 1: Normal safe condition -> Expected: SAFE
# --------------------------------------------------------------------------
def test_1_normal_safe_condition(safety_agent, default_task):
    inp = SafetyAgentInput(task=default_task, obstacle_distance=1.20, target_confidence=0.90)
    out: SafetyAgentOutput = safety_agent.process(inp)

    assert out.success is True
    assert out.safety_decision.decision == SafetyDecisionType.SAFE
    assert out.approved_for_execution is True
    assert out.safety_decision.risk_level == RiskLevel.LOW
    assert out.safety_state == SafetyState.SAFE.value


# --------------------------------------------------------------------------
# TEST 2: Obstacle below danger threshold -> Expected: STOP / UNSAFE
# --------------------------------------------------------------------------
def test_2_obstacle_below_danger_threshold(safety_agent, default_task):
    inp = SafetyAgentInput(task=default_task, obstacle_distance=0.20, target_confidence=0.90)
    out: SafetyAgentOutput = safety_agent.process(inp)

    assert out.safety_decision.decision == SafetyDecisionType.STOP
    assert out.approved_for_execution is False
    assert out.safety_decision.is_hard_violation is True
    assert out.safety_decision.risk_level == RiskLevel.CRITICAL


# --------------------------------------------------------------------------
# TEST 3: Obstacle above safe threshold -> Expected: SAFE
# --------------------------------------------------------------------------
def test_3_obstacle_above_safe_threshold(safety_agent, default_task):
    inp = SafetyAgentInput(task=default_task, obstacle_distance=0.85, target_confidence=0.90)
    out: SafetyAgentOutput = safety_agent.process(inp)

    assert out.safety_decision.decision == SafetyDecisionType.SAFE
    assert out.approved_for_execution is True


# --------------------------------------------------------------------------
# TEST 4: Hysteresis state retention
# --------------------------------------------------------------------------
def test_4_hysteresis_state_retention(safety_agent, default_task):
    # Case A: Previous state was SAFE, distance 0.42m in [0.35, 0.50] deadband -> Remain SAFE
    inp_safe_prev = SafetyAgentInput(
        task=default_task, obstacle_distance=0.42, previous_safety_state=SafetyState.SAFE
    )
    out_safe = safety_agent.process(inp_safe_prev)
    assert out_safe.safety_decision.decision == SafetyDecisionType.SAFE
    assert out_safe.approved_for_execution is True

    # Case B: Previous state was UNSAFE/STOPPED, distance 0.42m -> Remain UNSAFE / STOP
    inp_unsafe_prev = SafetyAgentInput(
        task=default_task, obstacle_distance=0.42, previous_safety_state=SafetyState.UNSAFE
    )
    out_unsafe = safety_agent.process(inp_unsafe_prev)
    assert out_unsafe.safety_decision.decision in (SafetyDecisionType.UNSAFE, SafetyDecisionType.STOP)
    assert out_unsafe.approved_for_execution is False
    assert out_unsafe.safety_state == SafetyState.UNSAFE.value


# --------------------------------------------------------------------------
# TEST 5: Emergency stop -> Expected: STOP
# --------------------------------------------------------------------------
def test_5_emergency_stop_hard_constraint(safety_agent, default_task):
    inp = SafetyAgentInput(task=default_task, obstacle_distance=1.50, emergency_stop=True)
    out: SafetyAgentOutput = safety_agent.process(inp)

    assert out.safety_decision.decision == SafetyDecisionType.STOP
    assert out.safety_decision.is_hard_violation is True
    assert out.approved_for_execution is False
    assert "RULE_1_EMERGENCY_STOP" in str(out.safety_decision.triggered_rules) or "Emergency Stop" in str(out.safety_decision.triggered_rules)


# --------------------------------------------------------------------------
# TEST 6: Low target confidence -> Expected: HUMAN_APPROVAL_REQUIRED
# --------------------------------------------------------------------------
def test_6_low_target_confidence(default_task):
    # Auto-reject stub to test escalation
    stub_reject = AutoApprovalStub(auto_approve=False)
    agent = SafetyAgent(human_interface=stub_reject)
    agent.initialize()

    inp = SafetyAgentInput(task=default_task, obstacle_distance=1.00, target_confidence=0.45)
    out = agent.process(inp)

    assert out.safety_decision.human_intervention_required is True
    assert out.approved_for_execution is False
    assert out.safety_decision.decision == SafetyDecisionType.UNSAFE


# --------------------------------------------------------------------------
# TEST 7: Missing / invalid sensor data -> Expected: HUMAN_APPROVAL_REQUIRED
# --------------------------------------------------------------------------
def test_7_missing_sensor_data(safety_agent, default_task):
    inp = SafetyAgentInput(
        task=default_task,
        obstacle_distance=1.0,
        sensor_validity={"camera": False, "proximity": True},
    )
    out = safety_agent.process(inp)

    # Escalated to human review due to sensor degradation
    assert out.safety_decision.human_intervention_required is True


# --------------------------------------------------------------------------
# TEST 8: Invalid planner task / unconfirmed target -> Expected: UNSAFE
# --------------------------------------------------------------------------
def test_8_invalid_planner_task(safety_agent):
    invalid_task = MultimodalTask(
        action="pick_and_place",
        target_object="cup",
        target_confirmed=False,
        task_status=TaskStatus.TARGET_NOT_FOUND,
    )
    inp = SafetyAgentInput(task=invalid_task, obstacle_distance=1.0)
    out = safety_agent.process(inp)

    assert out.safety_decision.decision == SafetyDecisionType.UNSAFE
    assert out.approved_for_execution is False


# --------------------------------------------------------------------------
# TEST 9: LLM unavailable -> Deterministic rules continue safely
# --------------------------------------------------------------------------
def test_9_llm_unavailable_fallback(default_task):
    # SafetyAgent configured with no LLM
    agent = SafetyAgent(llm_reasoner=None, human_interface=AutoApprovalStub(auto_approve=True))
    agent.initialize()

    inp = SafetyAgentInput(task=default_task, obstacle_distance=1.20)
    out = agent.process(inp)

    assert out.success is True
    assert out.safety_decision.llm_used is False
    assert out.safety_decision.decision == SafetyDecisionType.SAFE


# --------------------------------------------------------------------------
# TEST 10: LLM recommends SAFE but deterministic rule says STOP -> Expected: STOP
# (Crucial test: LLM must NEVER override hard deterministic safety violation!)
# --------------------------------------------------------------------------
def test_10_llm_safe_recommendation_overridden_by_hard_deterministic_stop(default_task):
    # Malicious or misaligned LLM says SAFE and low risk despite imminent collision
    optimistic_llm = MockLLMReasoner(
        risk_level=RiskLevel.LOW,
        contextually_safe=True,
        uncertain=False,
        reason="LLM believes everything is fine.",
    )
    agent = SafetyAgent(llm_reasoner=optimistic_llm, human_interface=AutoApprovalStub(auto_approve=True))
    agent.initialize()

    # Obstacle distance 0.15m is strictly dangerous (< 0.35m)
    inp = SafetyAgentInput(task=default_task, obstacle_distance=0.15)
    out = agent.process(inp)

    assert out.safety_decision.decision == SafetyDecisionType.STOP
    assert out.approved_for_execution is False
    assert out.safety_decision.is_hard_violation is True


# --------------------------------------------------------------------------
# TEST 11: LLM identifies contextual ambiguity -> Expected: HUMAN_APPROVAL_REQUIRED
# --------------------------------------------------------------------------
def test_11_llm_identifies_contextual_ambiguity(default_task):
    ambiguous_llm = MockLLMReasoner(
        risk_level=RiskLevel.MEDIUM,
        contextually_safe=False,
        uncertain=True,
        reason="Contextual ambiguity: multiple cups present, target unspecified.",
        requires_human=True,
    )
    agent = SafetyAgent(llm_reasoner=ambiguous_llm, human_interface=AutoApprovalStub(auto_approve=False))
    agent.initialize()

    inp = SafetyAgentInput(task=default_task, obstacle_distance=1.0)
    out = agent.process(inp)

    assert out.safety_decision.human_intervention_required is True
    assert out.safety_decision.llm_used is True
    assert out.approved_for_execution is False


# --------------------------------------------------------------------------
# TEST 12: Human approves uncertain task -> Expected: SAFE / approved execution
# --------------------------------------------------------------------------
def test_12_human_approves_uncertain_task(default_task):
    human_approve_stub = AutoApprovalStub(auto_approve=True, approver_id="supervisor_alice")
    agent = SafetyAgent(human_interface=human_approve_stub)
    agent.initialize()

    inp = SafetyAgentInput(task=default_task, obstacle_distance=1.0, target_confidence=0.45)
    out = agent.process(inp)

    assert out.safety_decision.human_intervention_required is True
    assert out.safety_decision.human_intervention_result.approved is True
    assert out.approved_for_execution is True
    assert out.safety_decision.decision == SafetyDecisionType.SAFE


# --------------------------------------------------------------------------
# TEST 13: Human rejects uncertain task -> Expected: UNSAFE / STOP
# --------------------------------------------------------------------------
def test_13_human_rejects_uncertain_task(default_task):
    human_reject_stub = AutoApprovalStub(auto_approve=False, approver_id="supervisor_bob")
    agent = SafetyAgent(human_interface=human_reject_stub)
    agent.initialize()

    inp = SafetyAgentInput(task=default_task, obstacle_distance=1.0, target_confidence=0.45)
    out = agent.process(inp)

    assert out.safety_decision.human_intervention_required is True
    assert out.safety_decision.human_intervention_result.approved is False
    assert out.approved_for_execution is False
    assert out.safety_decision.decision == SafetyDecisionType.UNSAFE


# --------------------------------------------------------------------------
# TEST 14: Conflicting multimodal information -> Expected: HUMAN_APPROVAL_REQUIRED
# --------------------------------------------------------------------------
def test_14_conflicting_multimodal_information():
    conflict_task = MultimodalTask(
        action="navigate_to",
        target_object="bottle",
        target_confirmed=True,
        confidence=0.85,
        task_status=TaskStatus.MODALITY_CONFLICT,
        reasoning="Voice requested LEFT, but vision detected target at RIGHT.",
    )
    agent = SafetyAgent(human_interface=AutoApprovalStub(auto_approve=False))
    agent.initialize()

    inp = SafetyAgentInput(task=conflict_task, obstacle_distance=1.0)
    out = agent.process(inp)

    assert out.safety_decision.human_intervention_required is True
    assert out.approved_for_execution is False


# --------------------------------------------------------------------------
# TEST 15: LLM timeout / error -> Safe fallback behavior
# --------------------------------------------------------------------------
def test_15_llm_error_safe_fallback(default_task):
    faulty_llm = MockLLMReasoner(should_raise_error=True)
    agent = SafetyAgent(
        llm_reasoner=faulty_llm,
        human_interface=AutoApprovalStub(auto_approve=False),
        config={"safety": {"default_on_llm_failure": "HUMAN_APPROVAL_REQUIRED"}},
    )
    agent.initialize()

    inp = SafetyAgentInput(task=default_task, obstacle_distance=1.0)
    out = agent.process(inp)

    # Safe fallback triggered without crashing
    assert out.success is True
    assert out.safety_decision.human_intervention_required is True


# --------------------------------------------------------------------------
# TEST 16: Human approval cannot override hard emergency stop or obstacle violation
# --------------------------------------------------------------------------
def test_16_human_approval_cannot_override_hard_safety(default_task):
    always_approve = AutoApprovalStub(auto_approve=True)
    agent = SafetyAgent(human_interface=always_approve)
    agent.initialize()

    # Hard obstacle violation (< 0.35m)
    inp = SafetyAgentInput(task=default_task, obstacle_distance=0.10)
    out = agent.process(inp)

    assert out.safety_decision.decision == SafetyDecisionType.STOP
    assert out.approved_for_execution is False
    assert out.safety_decision.is_hard_violation is True


# --------------------------------------------------------------------------
# TEST 17: Safety Agent lifecycle and schema serialization
# --------------------------------------------------------------------------
def test_17_lifecycle_and_serialization(default_task):
    agent = SafetyAgent(name="SerializationSafety")
    assert agent.status == AgentStatus.UNINITIALIZED
    assert agent.agent_type == AgentType.SAFETY

    agent.initialize()
    assert agent.status == AgentStatus.READY

    inp = SafetyAgentInput(task=default_task, obstacle_distance=1.50)
    out = agent.process(inp)

    out_dict = out.to_dict()
    assert isinstance(out_dict, dict)
    assert "safety_decision" in out_dict
    assert out_dict["approved_for_execution"] is True
    assert out_dict["agent_name"] == "SerializationSafety"
    assert out_dict["agent_type"] == AgentType.SAFETY.value


# --------------------------------------------------------------------------
# TEST 18: Ollama Reasoner offline fallback (daemon unreachable)
# --------------------------------------------------------------------------
def test_18_ollama_reasoner_offline_fallback():
    # Points to a port guaranteed to have no server running
    reasoner = OllamaLLMReasoner(model_name="llama3", host="http://127.0.0.1:59999", timeout_sec=0.5)
    assessment = reasoner.assess_risk({"task": "navigate_to bottle"})

    assert assessment.risk_level == RiskLevel.LOW
    assert assessment.contextually_safe is True
    assert assessment.uncertain is False
    assert "unreachable" in assessment.reason.lower()


# --------------------------------------------------------------------------
# TEST 19: Ollama Reasoner response parsing with mocked API
# --------------------------------------------------------------------------
def test_19_ollama_reasoner_mocked_response():
    from unittest.mock import patch, MagicMock
    import json
    import io

    mock_ollama_reply = {
        "message": {
            "role": "assistant",
            "content": json.dumps({
                "risk_level": "HIGH",
                "contextually_safe": False,
                "uncertain": True,
                "reason": "Fragile glassware in close proximity.",
                "confidence": 0.88,
                "requires_human": True,
            }),
        }
    }

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(mock_ollama_reply).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        reasoner = OllamaLLMReasoner(model_name="llama3", host="http://localhost:11434")
        assessment = reasoner.assess_risk({"task": "pick_and_place glass"})

        assert assessment.risk_level == RiskLevel.HIGH
        assert assessment.contextually_safe is False
        assert assessment.uncertain is True
        assert assessment.requires_human is True
        assert assessment.confidence == 0.88
        assert "glassware" in assessment.reason


# --------------------------------------------------------------------------
# TEST 20: SafetyAgent initialized with Ollama provider in config
# --------------------------------------------------------------------------
def test_20_safety_agent_ollama_initialization(default_task):
    agent = SafetyAgent(
        name="OllamaSafetyAgent",
        config={
            "safety": {
                "llm_enabled": True,
            },
            "llm": {
                "provider": "ollama",
                "model": "llama3",
                "host": "http://localhost:11434",
            },
        },
    )
    agent.initialize()

    assert isinstance(agent.llm_reasoner, OllamaLLMReasoner)
    assert agent.llm_reasoner.model_name == "llama3"
    assert agent.llm_reasoner.host == "http://localhost:11434"
