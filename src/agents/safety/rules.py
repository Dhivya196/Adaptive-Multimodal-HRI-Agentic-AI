"""Deterministic safety rules engine for immediate, explainable safety violation checking."""

from typing import Any, Dict, List, Optional, Tuple

from src.agents.coordinator.schemas import MultimodalTask, TaskStatus
from src.agents.safety.schemas import (
    RiskLevel,
    RuleEvaluationResult,
    SafetyAgentInput,
    SafetyDecisionType,
)


class DeterministicRuleEngine:
    """
    Evaluates deterministic safety constraints in a strict priority hierarchy.
    Hard safety violations (such as emergency stop and collision boundary violations)
    cannot be overridden by LLM contextual reasoning or human approval.
    """

    def __init__(
        self,
        danger_distance: float = 0.35,
        confidence_threshold: float = 0.60,
        sensor_policy: str = "human_approval_on_missing",
    ):
        self.danger_distance = danger_distance
        self.confidence_threshold = confidence_threshold
        self.sensor_policy = sensor_policy

    def evaluate_rules(self, safety_input: SafetyAgentInput) -> List[RuleEvaluationResult]:
        """
        Evaluate all deterministic safety rules in order of priority.
        Returns a list of all rule evaluation results.
        """
        results: List[RuleEvaluationResult] = []

        # ------------------------------------------------------------------
        # RULE 1: Emergency Stop Check (Highest Priority Hard Constraint)
        # ------------------------------------------------------------------
        if safety_input.emergency_stop:
            results.append(
                RuleEvaluationResult(
                    rule_id="RULE_1_EMERGENCY_STOP",
                    rule_name="Emergency Stop",
                    passed=False,
                    triggered_decision=SafetyDecisionType.STOP,
                    is_hard_violation=True,
                    severity=RiskLevel.CRITICAL,
                    reason="Hardware or software emergency stop is active.",
                    details={"emergency_stop": True},
                )
            )

        # Also handle emergency stop task
        if safety_input.task and safety_input.task.action == "stop_robot":
            results.append(
                RuleEvaluationResult(
                    rule_id="RULE_1_STOP_TASK",
                    rule_name="Stop Command Request",
                    passed=True,
                    triggered_decision=SafetyDecisionType.STOP,
                    is_hard_violation=False,
                    severity=RiskLevel.LOW,
                    reason="Stop action requested by user command.",
                    details={"action": "stop_robot"},
                )
            )

        # ------------------------------------------------------------------
        # RULE 2: Critical Obstacle Distance (Hard Collision Constraint)
        # ------------------------------------------------------------------
        obs_dist = safety_input.obstacle_distance
        if obs_dist is not None:
            if obs_dist < self.danger_distance:
                results.append(
                    RuleEvaluationResult(
                        rule_id="RULE_2_OBSTACLE_DANGER",
                        rule_name="Obstacle Danger Distance Violation",
                        passed=False,
                        triggered_decision=SafetyDecisionType.STOP,
                        is_hard_violation=True,
                        severity=RiskLevel.CRITICAL,
                        reason=f"Obstacle distance ({obs_dist:.2f} m) is below danger threshold ({self.danger_distance:.2f} m).",
                        details={"obstacle_distance": obs_dist, "danger_distance": self.danger_distance},
                    )
                )

        # ------------------------------------------------------------------
        # RULE 3: Sensor Validity & Data Integrity Check
        # ------------------------------------------------------------------
        sensor_validity = safety_input.sensor_validity or {}
        invalid_sensors = [name for name, is_valid in sensor_validity.items() if not is_valid]

        if invalid_sensors:
            decision = (
                SafetyDecisionType.STOP
                if self.sensor_policy == "strict_stop"
                else SafetyDecisionType.HUMAN_APPROVAL_REQUIRED
            )
            is_hard = self.sensor_policy == "strict_stop"
            results.append(
                RuleEvaluationResult(
                    rule_id="RULE_3_SENSOR_INVALID",
                    rule_name="Sensor Data Invalid or Missing",
                    passed=False,
                    triggered_decision=decision,
                    is_hard_violation=is_hard,
                    severity=RiskLevel.HIGH,
                    reason=f"Critical sensors reported invalid or unavailable status: {', '.join(invalid_sensors)}.",
                    details={"invalid_sensors": invalid_sensors},
                )
            )

        # ------------------------------------------------------------------
        # RULE 4: Perception & Target Confidence Threshold
        # ------------------------------------------------------------------
        target_conf = safety_input.target_confidence
        task = safety_input.task

        # Check explicit target confidence or task confidence
        effective_conf = target_conf if target_conf is not None else (task.confidence if task else None)

        if effective_conf is not None and task and task.action != "stop_robot":
            if effective_conf < self.confidence_threshold:
                results.append(
                    RuleEvaluationResult(
                        rule_id="RULE_4_LOW_CONFIDENCE",
                        rule_name="Low Perception Confidence",
                        passed=False,
                        triggered_decision=SafetyDecisionType.HUMAN_APPROVAL_REQUIRED,
                        is_hard_violation=False,
                        severity=RiskLevel.MEDIUM,
                        reason=(
                            f"Perception confidence ({effective_conf:.2f}) is below the required "
                            f"autonomous safety threshold ({self.confidence_threshold:.2f})."
                        ),
                        details={"confidence": effective_conf, "threshold": self.confidence_threshold},
                    )
                )

        # ------------------------------------------------------------------
        # RULE 5: Task Validity & Multimodal Conflict Checking
        # ------------------------------------------------------------------
        if task is not None:
            if task.task_status == TaskStatus.TARGET_NOT_FOUND and task.action != "stop_robot":
                results.append(
                    RuleEvaluationResult(
                        rule_id="RULE_5_TARGET_NOT_FOUND",
                        rule_name="Target Object Not Found",
                        passed=False,
                        triggered_decision=SafetyDecisionType.UNSAFE,
                        is_hard_violation=False,
                        severity=RiskLevel.HIGH,
                        reason=f"Target object '{task.target_object}' was not confirmed by vision perception.",
                        details={"target_object": task.target_object},
                    )
                )
            elif task.task_status == TaskStatus.MODALITY_CONFLICT:
                results.append(
                    RuleEvaluationResult(
                        rule_id="RULE_5_MODALITY_CONFLICT",
                        rule_name="Conflicting Multimodal Input",
                        passed=False,
                        triggered_decision=SafetyDecisionType.HUMAN_APPROVAL_REQUIRED,
                        is_hard_violation=False,
                        severity=RiskLevel.HIGH,
                        reason="Conflicting multimodal information was detected (e.g. conflicting directions).",
                        details={"reasoning": task.reasoning},
                    )
                )
            elif task.task_status == TaskStatus.INVALID_INPUT:
                results.append(
                    RuleEvaluationResult(
                        rule_id="RULE_5_INVALID_TASK",
                        rule_name="Invalid Multimodal Task",
                        passed=False,
                        triggered_decision=SafetyDecisionType.UNSAFE,
                        is_hard_violation=False,
                        severity=RiskLevel.HIGH,
                        reason="Task contains invalid or missing action parameters.",
                        details={"task_status": task.task_status.value},
                    )
                )
            elif task.task_status == TaskStatus.LOW_CONFIDENCE:
                results.append(
                    RuleEvaluationResult(
                        rule_id="RULE_5_TASK_LOW_CONFIDENCE",
                        rule_name="Task Low Confidence Flag",
                        passed=False,
                        triggered_decision=SafetyDecisionType.HUMAN_APPROVAL_REQUIRED,
                        is_hard_violation=False,
                        severity=RiskLevel.MEDIUM,
                        reason="Task was flagged with low perception confidence by upstream coordinator.",
                        details={"confidence": task.confidence},
                    )
                )

        # ------------------------------------------------------------------
        # RULE 6: Normal Safe Condition (Default if no violations triggered)
        # ------------------------------------------------------------------
        if not results:
            results.append(
                RuleEvaluationResult(
                    rule_id="RULE_6_NORMAL_SAFE",
                    rule_name="Normal Safe Operation",
                    passed=True,
                    triggered_decision=SafetyDecisionType.SAFE,
                    is_hard_violation=False,
                    severity=RiskLevel.LOW,
                    reason="All deterministic safety criteria are satisfied.",
                )
            )

        return results
