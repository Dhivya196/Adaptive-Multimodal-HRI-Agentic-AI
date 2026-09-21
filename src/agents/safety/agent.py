"""Safety Agent: independent safety layer for deterministic and contextual HRI risk evaluation."""

from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import BaseAgent
from src.agents.coordinator.schemas import MultimodalTask, TaskStatus
from src.agents.safety.human_intervention import (
    AutoApprovalStub,
    BaseHumanInterventionInterface,
    CLIHumanIntervention,
)
from src.agents.safety.llm_reasoner import (
    BaseLLMReasoner,
    MockLLMReasoner,
    OllamaLLMReasoner,
    ProviderLLMReasoner,
)
from src.agents.safety.monitor import HysteresisMonitor
from src.agents.safety.rules import DeterministicRuleEngine
from src.agents.safety.schemas import (
    HumanInterventionResult,
    LLMContextualRiskAssessment,
    RiskLevel,
    RuleEvaluationResult,
    SafetyAgentInput,
    SafetyAgentOutput,
    SafetyDecision,
    SafetyDecisionType,
    SafetyState,
)
from src.common.exceptions import AgentExecutionError
from src.common.schemas import AgentType


class SafetyAgent(BaseAgent):
    """
    Safety Agent in the Decision-Making Layer.
    Combines deterministic safety rules, threshold + hysteresis proximity monitoring,
    LLM contextual risk reasoning, uncertainty quantification, and human-in-the-loop review.
    
    The LLM is NOT the sole or final safety authority.
    Deterministic hard safety rules (e.g. Emergency Stop and distance < danger_distance)
    strictly take precedence and cannot be overridden by the LLM or human approval.
    """

    def __init__(
        self,
        name: str = "SafetyAgent",
        config: Optional[Dict[str, Any]] = None,
        rule_engine: Optional[DeterministicRuleEngine] = None,
        hysteresis_monitor: Optional[HysteresisMonitor] = None,
        llm_reasoner: Optional[BaseLLMReasoner] = None,
        human_interface: Optional[BaseHumanInterventionInterface] = None,
    ):
        super().__init__(
            name=name,
            agent_type=AgentType.SAFETY,
            version="0.1.0",
            description="Evaluates deterministic safety constraints, proximity hysteresis, and contextual risks.",
            config=config or {},
        )

        self.rule_engine = rule_engine
        self.hysteresis_monitor = hysteresis_monitor
        self.llm_reasoner = llm_reasoner
        self.human_interface = human_interface
        self._previous_safety_state: SafetyState = SafetyState.SAFE

    def _initialize(self) -> bool:
        """Initialize deterministic rules, hysteresis monitor, LLM reasoner, and human review interface."""
        safety_cfg = self.config.get("safety", {})
        llm_cfg = self.config.get("llm", {})

        danger_dist = safety_cfg.get("danger_distance", 0.35)
        safe_dist = safety_cfg.get("safe_distance", 0.50)
        conf_thresh = safety_cfg.get("confidence_threshold", 0.60)
        sensor_policy = safety_cfg.get("sensor_policy", "human_approval_on_missing")

        # 1. Deterministic Rule Engine
        if self.rule_engine is None:
            self.rule_engine = DeterministicRuleEngine(
                danger_distance=danger_dist,
                confidence_threshold=conf_thresh,
                sensor_policy=sensor_policy,
            )

        # 2. Hysteresis Proximity Monitor
        if self.hysteresis_monitor is None:
            self.hysteresis_monitor = HysteresisMonitor(
                danger_distance=danger_dist,
                safe_distance=safe_dist,
                default_in_band_state=SafetyState.SAFE,
            )

        # 3. LLM Contextual Risk Reasoner
        if self.llm_reasoner is None:
            llm_enabled = safety_cfg.get("llm_enabled", False)
            if llm_enabled:
                provider = str(llm_cfg.get("provider", "ollama")).lower()
                if provider == "ollama":
                    model_name = llm_cfg.get("model", "llama3")
                    host = llm_cfg.get("host", "http://localhost:11434")
                    temp = llm_cfg.get("temperature", 0.0)
                    timeout = llm_cfg.get("timeout", 5.0)
                    self.llm_reasoner = OllamaLLMReasoner(
                        model_name=model_name,
                        host=host,
                        temperature=temp,
                        timeout_sec=timeout,
                    )
                else:
                    model_name = llm_cfg.get("model", "gpt-4o-mini")
                    temp = llm_cfg.get("temperature", 0.0)
                    timeout = llm_cfg.get("timeout", 3.0)
                    api_key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
                    self.llm_reasoner = ProviderLLMReasoner(
                        model_name=model_name,
                        temperature=temp,
                        timeout_sec=timeout,
                        api_key_env=api_key_env,
                    )
            else:
                self.llm_reasoner = None

        # 4. Human Intervention Interface
        if self.human_interface is None:
            human_enabled = safety_cfg.get("human_intervention_enabled", True)
            if human_enabled:
                mode = safety_cfg.get("human_intervention_mode", "auto_stub")
                if mode == "cli":
                    self.human_interface = CLIHumanIntervention()
                else:
                    self.human_interface = AutoApprovalStub(auto_approve=True)
            else:
                self.human_interface = None

        self.logger.info("SafetyAgent initialized successfully with multi-layer risk hierarchy.")
        return True

    def _process(self, input_data: Any) -> SafetyAgentOutput:
        """
        Execute safety evaluation following the strict safety hierarchy.
        """
        if isinstance(input_data, dict):
            safety_input = SafetyAgentInput(
                session_id=input_data.get("session_id", "default_session"),
                task=input_data.get("task"),
                planner_output=input_data.get("planner_output"),
                obstacle_distance=input_data.get("obstacle_distance"),
                human_distance=input_data.get("human_distance"),
                emergency_stop=input_data.get("emergency_stop", False),
                sensor_validity=input_data.get("sensor_validity", {"camera": True, "proximity": True}),
                target_confidence=input_data.get("target_confidence"),
                environment_context=input_data.get("environment_context", {}),
                robot_state=input_data.get("robot_state", {}),
                previous_safety_state=input_data.get("previous_safety_state"),
            )
        elif isinstance(input_data, SafetyAgentInput):
            safety_input = input_data
        else:
            raise AgentExecutionError(
                f"SafetyAgent expects SafetyAgentInput or dict, got {type(input_data).__name__}"
            )

        prev_state = safety_input.previous_safety_state or self._previous_safety_state
        measurements: Dict[str, Any] = {
            "obstacle_distance": safety_input.obstacle_distance,
            "human_distance": safety_input.human_distance,
            "emergency_stop": safety_input.emergency_stop,
            "sensor_validity": safety_input.sensor_validity,
            "target_confidence": safety_input.target_confidence,
        }

        triggered_rule_names: List[str] = []
        is_hard_violation = False
        decision_type = SafetyDecisionType.SAFE
        risk_level = RiskLevel.LOW
        reasons: List[str] = []

        # ==================================================================
        # STEP 1: Deterministic Safety Rules
        # ==================================================================
        rule_results: List[RuleEvaluationResult] = self.rule_engine.evaluate_rules(safety_input)

        for res in rule_results:
            if not res.passed or res.triggered_decision in (SafetyDecisionType.STOP, SafetyDecisionType.UNSAFE, SafetyDecisionType.HUMAN_APPROVAL_REQUIRED):
                triggered_rule_names.append(res.rule_name)
                reasons.append(res.reason)
                if res.is_hard_violation:
                    is_hard_violation = True
                    decision_type = res.triggered_decision or SafetyDecisionType.STOP
                    risk_level = res.severity
                    break
                else:
                    if res.triggered_decision == SafetyDecisionType.UNSAFE:
                        decision_type = SafetyDecisionType.UNSAFE
                        risk_level = max(risk_level, res.severity, key=lambda r: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(r.value))
                    elif res.triggered_decision == SafetyDecisionType.HUMAN_APPROVAL_REQUIRED:
                        if decision_type != SafetyDecisionType.UNSAFE:
                            decision_type = SafetyDecisionType.HUMAN_APPROVAL_REQUIRED
                            risk_level = max(risk_level, res.severity, key=lambda r: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(r.value))

        # ==================================================================
        # STEP 2: Threshold + Hysteresis Evaluation (if no hard E-Stop)
        # ==================================================================
        if not is_hard_violation and safety_input.obstacle_distance is not None:
            hyst_decision, new_hyst_state, hyst_reason = self.hysteresis_monitor.evaluate_distance(
                safety_input.obstacle_distance, previous_state=prev_state
            )
            if hyst_decision in (SafetyDecisionType.STOP, SafetyDecisionType.UNSAFE):
                is_hard_violation = (safety_input.obstacle_distance < self.rule_engine.danger_distance)
                decision_type = hyst_decision
                risk_level = RiskLevel.CRITICAL if is_hard_violation else RiskLevel.HIGH
                triggered_rule_names.append("Hysteresis Proximity Monitor")
                reasons.append(hyst_reason)

        # ==================================================================
        # STEP 3: LLM Contextual Risk Reasoning (Only if no hard violation)
        # ==================================================================
        llm_used = False
        llm_assessment: Optional[LLMContextualRiskAssessment] = None

        if not is_hard_violation and self.llm_reasoner is not None:
            llm_used = True
            context_payload = {
                "task": safety_input.task.to_dict() if safety_input.task else None,
                "obstacle_distance": safety_input.obstacle_distance,
                "human_distance": safety_input.human_distance,
                "measurements": measurements,
                "environment_context": safety_input.environment_context,
                "rule_evaluation": [r.to_dict() for r in rule_results],
            }
            try:
                llm_assessment = self.llm_reasoner.assess_risk(context_payload)
            except Exception as e:
                self.logger.warning(f"LLM reasoner exception: {e}; applying fallback policy.")
                default_fallback = self.config.get("safety", {}).get(
                    "default_on_llm_failure", "HUMAN_APPROVAL_REQUIRED"
                )
                llm_assessment = LLMContextualRiskAssessment(
                    risk_level=RiskLevel.MEDIUM,
                    contextually_safe=False,
                    uncertain=True,
                    reason=f"LLM reasoning failed ({e}); default fallback policy triggered.",
                    confidence=0.5,
                    requires_human=(default_fallback == "HUMAN_APPROVAL_REQUIRED"),
                )

            # Integrate LLM contextual reasoning
            if llm_assessment:
                if llm_assessment.requires_human or llm_assessment.uncertain:
                    if decision_type == SafetyDecisionType.SAFE:
                        decision_type = SafetyDecisionType.HUMAN_APPROVAL_REQUIRED
                        risk_level = max(risk_level, llm_assessment.risk_level, key=lambda r: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(r.value))
                        reasons.append(f"LLM Contextual Reasoning: {llm_assessment.reason}")
                elif not llm_assessment.contextually_safe:
                    if decision_type == SafetyDecisionType.SAFE:
                        decision_type = SafetyDecisionType.UNSAFE
                        risk_level = max(risk_level, llm_assessment.risk_level, key=lambda r: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(r.value))
                        reasons.append(f"LLM flagged contextual hazard: {llm_assessment.reason}")

        # ==================================================================
        # STEP 4: Uncertainty Evaluation
        # ==================================================================
        conf_val = 1.0
        if safety_input.target_confidence is not None:
            conf_val = safety_input.target_confidence
        elif safety_input.task is not None:
            conf_val = safety_input.task.confidence

        uncertainty_val = max(0.0, min(1.0, 1.0 - conf_val))

        # Check for ambiguity in multimodal task
        if safety_input.task and safety_input.task.task_status in (
            TaskStatus.MODALITY_CONFLICT,
            TaskStatus.AMBIGUOUS_TARGET,
            TaskStatus.LOW_CONFIDENCE,
        ):
            if decision_type == SafetyDecisionType.SAFE:
                decision_type = SafetyDecisionType.HUMAN_APPROVAL_REQUIRED
                reasons.append(f"Task uncertainty detected: {safety_input.task.task_status.value}.")

        # ==================================================================
        # STEP 5: Human Intervention (Only for uncertainty / non-hard cases)
        # ==================================================================
        human_intervention_required = (decision_type == SafetyDecisionType.HUMAN_APPROVAL_REQUIRED)
        human_result: Optional[HumanInterventionResult] = None
        approved_for_execution = (decision_type == SafetyDecisionType.SAFE)

        if is_hard_violation:
            # Hard violations NEVER prompt for human override to proceed
            human_intervention_required = False
            approved_for_execution = False
        elif human_intervention_required and self.human_interface is not None:
            task_desc = (
                f"{safety_input.task.action} ({safety_input.task.target_object})"
                if safety_input.task
                else "Unspecified Task"
            )
            human_result = self.human_interface.request_approval(
                task_summary=task_desc,
                reason="; ".join(reasons) or "Uncertain multimodal context.",
                risk_level=risk_level,
                measurements=measurements,
                is_hard_violation=is_hard_violation,
            )
            if human_result.approved:
                decision_type = SafetyDecisionType.SAFE
                approved_for_execution = True
                reasons.append(f"Approved by human reviewer ({human_result.approver_id}).")
            else:
                decision_type = SafetyDecisionType.UNSAFE
                approved_for_execution = False
                reasons.append("Rejected by human reviewer.")

        # Update persistent safety state
        if decision_type == SafetyDecisionType.SAFE:
            self._previous_safety_state = SafetyState.SAFE
        elif decision_type == SafetyDecisionType.STOP:
            self._previous_safety_state = SafetyState.STOPPED
        elif decision_type == SafetyDecisionType.UNSAFE:
            self._previous_safety_state = SafetyState.UNSAFE
        else:
            self._previous_safety_state = SafetyState.HUMAN_APPROVAL_REQUIRED

        final_reason = "; ".join(reasons) if reasons else "All safety checks passed."

        safety_decision = SafetyDecision(
            decision=decision_type,
            risk_level=risk_level,
            reason=final_reason,
            triggered_rules=triggered_rule_names,
            confidence=conf_val,
            uncertainty=uncertainty_val,
            relevant_measurements=measurements,
            is_hard_violation=is_hard_violation,
            llm_used=llm_used,
            llm_assessment=llm_assessment,
            human_intervention_required=human_intervention_required,
            human_intervention_result=human_result,
        )

        self.logger.info(
            f"Safety evaluation complete. Decision: {decision_type.value}, Risk: {risk_level.value}, "
            f"Execution Approved: {approved_for_execution}, Reason: {final_reason}"
        )

        return SafetyAgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type.value,
            success=True,
            confidence=conf_val,
            safety_decision=safety_decision,
            approved_for_execution=approved_for_execution,
            safety_state=self._previous_safety_state.value,
        )

    def _reset(self) -> None:
        """Reset internal monitor state."""
        self._previous_safety_state = SafetyState.SAFE
        if self.hysteresis_monitor:
            self.hysteresis_monitor.reset(SafetyState.SAFE)

    def _shutdown(self) -> None:
        """Cleanly release resources."""
        pass
