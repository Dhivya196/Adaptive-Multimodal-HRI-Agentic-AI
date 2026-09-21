"""Human-in-the-loop intervention and approval interface."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import sys
from typing import Any, Dict, Optional

from src.agents.safety.schemas import HumanInterventionResult, RiskLevel


class BaseHumanInterventionInterface(ABC):
    """
    Abstract interface for human-in-the-loop review and approval.
    Enables future substitution with GUI modal, ROS2 action server,
    or speech dialogue without modifying safety agent core logic.
    """

    @abstractmethod
    def request_approval(
        self,
        task_summary: str,
        reason: str,
        risk_level: RiskLevel,
        measurements: Dict[str, Any],
        is_hard_violation: bool = False,
    ) -> HumanInterventionResult:
        """Prompt human reviewer and return intervention result."""
        pass


class AutoApprovalStub(BaseHumanInterventionInterface):
    """
    Automated intervention stub for headless integration and unit testing.
    Can be configured to automatically approve or reject requests.
    """

    def __init__(self, auto_approve: bool = True, approver_id: str = "test_operator"):
        self.auto_approve = auto_approve
        self.approver_id = approver_id
        self.last_request: Optional[Dict[str, Any]] = None

    def request_approval(
        self,
        task_summary: str,
        reason: str,
        risk_level: RiskLevel,
        measurements: Dict[str, Any],
        is_hard_violation: bool = False,
    ) -> HumanInterventionResult:
        self.last_request = {
            "task_summary": task_summary,
            "reason": reason,
            "risk_level": risk_level,
            "measurements": measurements,
            "is_hard_violation": is_hard_violation,
        }

        # Human approval cannot bypass a hard safety violation
        if is_hard_violation:
            return HumanInterventionResult(
                requested=True,
                approved=False,
                approver_id=self.approver_id,
                reason="Hard safety violations cannot be overridden by human approval.",
            )

        return HumanInterventionResult(
            requested=True,
            approved=self.auto_approve,
            approver_id=self.approver_id,
            reason="Automated test stub response.",
        )


class CLIHumanIntervention(BaseHumanInterventionInterface):
    """
    Interactive command-line interface for human review.
    Displays task summary, risk assessment, and sensor readings before requesting [y/n] input.
    """

    def request_approval(
        self,
        task_summary: str,
        reason: str,
        risk_level: RiskLevel,
        measurements: Dict[str, Any],
        is_hard_violation: bool = False,
    ) -> HumanInterventionResult:
        if is_hard_violation:
            print("\n" + "!" * 60)
            print("  CRITICAL HARD SAFETY VIOLATION: HUMAN OVERRIDE BLOCKED")
            print("!" * 60)
            print(f"Reason: {reason}\n")
            return HumanInterventionResult(
                requested=True,
                approved=False,
                approver_id="system",
                reason="Hard safety violations cannot be overridden by human approval.",
            )

        print("\n" + "=" * 60)
        print("  ⚠️  SAFETY REVIEW REQUIRED (HUMAN-IN-THE-LOOP)")
        print("=" * 60)
        print(f"Task Summary : {task_summary}")
        print(f"Risk Level   : {risk_level.value if hasattr(risk_level, 'value') else risk_level}")
        print(f"Reason       : {reason}")
        if measurements:
            print("Measurements :")
            for k, v in measurements.items():
                print(f"  • {k}: {v}")
        print("-" * 60)

        try:
            choice = input("Approve robot execution? [y/N]: ").strip().lower()
            approved = choice in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            approved = False

        status_str = "APPROVED" if approved else "REJECTED / HALTED"
        print(f"Decision registered: {status_str}\n" + "=" * 60 + "\n")

        return HumanInterventionResult(
            requested=True,
            approved=approved,
            approver_id="cli_operator",
            reason=f"Operator selected {status_str}.",
        )
