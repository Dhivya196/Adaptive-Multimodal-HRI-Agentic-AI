"""Schemas and data models for the Safety Agent, risk evaluation, and decision hierarchy."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.agents.coordinator.schemas import MultimodalTask, TaskStatus
from src.common.schemas import BaseAgentInput, BaseAgentOutput


class SafetyDecisionType(str, Enum):
    """Primary safety decision outcome."""
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"
    STOP = "STOP"


class RiskLevel(str, Enum):
    """Categorical risk assessment level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SafetyState(str, Enum):
    """Explicit lifecycle/tracking safety states."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    UNSAFE = "UNSAFE"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"
    STOPPED = "STOPPED"


@dataclass
class RuleEvaluationResult:
    """Outcome of evaluating an individual deterministic safety rule."""
    rule_id: str
    rule_name: str
    passed: bool
    triggered_decision: Optional[SafetyDecisionType] = None
    reason: str = ""
    is_hard_violation: bool = False
    severity: RiskLevel = RiskLevel.LOW
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "passed": self.passed,
            "triggered_decision": self.triggered_decision.value if self.triggered_decision else None,
            "reason": self.reason,
            "is_hard_violation": self.is_hard_violation,
            "severity": self.severity.value if isinstance(self.severity, RiskLevel) else self.severity,
            "details": self.details,
        }


@dataclass
class LLMContextualRiskAssessment:
    """Structured response from LLM contextual risk reasoner."""
    risk_level: RiskLevel = RiskLevel.LOW
    contextually_safe: bool = True
    uncertain: bool = False
    reason: str = "No contextual risk detected."
    confidence: float = 1.0
    requires_human: bool = False
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level.value if isinstance(self.risk_level, RiskLevel) else self.risk_level,
            "contextually_safe": self.contextually_safe,
            "uncertain": self.uncertain,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "requires_human": self.requires_human,
        }


@dataclass
class HumanInterventionResult:
    """Record of human review / approval."""
    requested: bool = False
    approved: bool = False
    approver_id: Optional[str] = None
    reason: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "approved": self.approved,
            "approver_id": self.approver_id,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class SafetyDecision:
    """Comprehensive safety evaluation result returned by the Safety Agent."""
    decision: SafetyDecisionType = SafetyDecisionType.UNSAFE
    risk_level: RiskLevel = RiskLevel.HIGH
    reason: str = "Uninitialized safety decision."
    triggered_rules: List[str] = field(default_factory=list)
    confidence: float = 0.0
    uncertainty: float = 1.0
    relevant_measurements: Dict[str, Any] = field(default_factory=dict)
    is_hard_violation: bool = False
    llm_used: bool = False
    llm_assessment: Optional[LLMContextualRiskAssessment] = None
    human_intervention_required: bool = False
    human_intervention_result: Optional[HumanInterventionResult] = None
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value if isinstance(self.decision, SafetyDecisionType) else self.decision,
            "risk_level": self.risk_level.value if isinstance(self.risk_level, RiskLevel) else self.risk_level,
            "reason": self.reason,
            "triggered_rules": self.triggered_rules,
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "relevant_measurements": self.relevant_measurements,
            "is_hard_violation": self.is_hard_violation,
            "llm_used": self.llm_used,
            "llm_assessment": self.llm_assessment.to_dict() if self.llm_assessment else None,
            "human_intervention_required": self.human_intervention_required,
            "human_intervention_result": (
                self.human_intervention_result.to_dict() if self.human_intervention_result else None
            ),
            "timestamp": self.timestamp,
        }


@dataclass
class SafetyAgentInput(BaseAgentInput):
    """Input data container consumed by the Safety Agent."""
    task: Optional[MultimodalTask] = None
    planner_output: Optional[Dict[str, Any]] = None
    obstacle_distance: Optional[float] = None
    human_distance: Optional[float] = None
    emergency_stop: bool = False
    sensor_validity: Dict[str, bool] = field(default_factory=lambda: {"camera": True, "proximity": True})
    target_confidence: Optional[float] = None
    environment_context: Dict[str, Any] = field(default_factory=dict)
    robot_state: Dict[str, Any] = field(default_factory=dict)
    previous_safety_state: Optional[SafetyState] = None


@dataclass
class SafetyAgentOutput(BaseAgentOutput):
    """Output container produced by the Safety Agent for downstream Robot Controller."""
    safety_decision: Optional[SafetyDecision] = None
    approved_for_execution: bool = False
    safety_state: str = SafetyState.UNSAFE.value

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "safety_decision": self.safety_decision.to_dict() if self.safety_decision else None,
            "approved_for_execution": self.approved_for_execution,
            "safety_state": self.safety_state,
        })
        return base_dict
