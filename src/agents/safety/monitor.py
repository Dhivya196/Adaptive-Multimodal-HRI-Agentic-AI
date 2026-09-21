"""Threshold and hysteresis monitoring for continuous proximity and safety metrics."""

from typing import Any, Dict, Optional, Tuple

from src.agents.safety.schemas import (
    RiskLevel,
    RuleEvaluationResult,
    SafetyDecisionType,
    SafetyState,
)


class HysteresisMonitor:
    """
    Threshold and hysteresis monitor for proximity and continuous safety measurements.
    Implements a dual-threshold deadband to prevent rapid switching between SAFE and UNSAFE
    states when measurements fluctuate near boundary values.
    
    Behavior:
      - distance < danger_distance (e.g. 0.35m)       -> UNSAFE / STOP
      - distance > safe_distance   (e.g. 0.50m)       -> SAFE
      - danger_distance <= distance <= safe_distance   -> Retain previous state
    """

    def __init__(
        self,
        danger_distance: float = 0.35,
        safe_distance: float = 0.50,
        default_in_band_state: SafetyState = SafetyState.SAFE,
    ):
        if danger_distance >= safe_distance:
            raise ValueError(
                f"danger_distance ({danger_distance}) must be strictly less than safe_distance ({safe_distance})"
            )
        self.danger_distance = danger_distance
        self.safe_distance = safe_distance
        self.default_in_band_state = default_in_band_state
        self._current_state: SafetyState = default_in_band_state

    @property
    def current_state(self) -> SafetyState:
        return self._current_state

    def reset(self, initial_state: Optional[SafetyState] = None) -> None:
        """Reset internal state tracker."""
        self._current_state = initial_state or self.default_in_band_state

    def evaluate_distance(
        self,
        distance: Optional[float],
        previous_state: Optional[SafetyState] = None,
    ) -> Tuple[SafetyDecisionType, SafetyState, str]:
        """
        Evaluate distance against hysteresis thresholds.
        
        Returns:
            Tuple of (SafetyDecisionType, new_SafetyState, explanation_reason)
        """
        if distance is None:
            # If no distance measurement provided, pass through without updating state
            return SafetyDecisionType.SAFE, self._current_state, "No obstacle distance measurement provided."

        active_prev_state = previous_state or self._current_state

        # Zone 1: Below Danger Threshold (Critical Hazard)
        if distance < self.danger_distance:
            self._current_state = SafetyState.UNSAFE
            return (
                SafetyDecisionType.STOP,
                SafetyState.UNSAFE,
                f"Obstacle distance ({distance:.2f} m) is strictly below danger threshold ({self.danger_distance:.2f} m).",
            )

        # Zone 2: Above Safe Threshold (Clear Safety Zone)
        elif distance > self.safe_distance:
            self._current_state = SafetyState.SAFE
            return (
                SafetyDecisionType.SAFE,
                SafetyState.SAFE,
                f"Obstacle distance ({distance:.2f} m) is comfortably above safe threshold ({self.safe_distance:.2f} m).",
            )

        # Zone 3: Hysteresis Deadband [danger_distance, safe_distance]
        else:
            if active_prev_state in (SafetyState.UNSAFE, SafetyState.STOPPED):
                self._current_state = active_prev_state
                decision = SafetyDecisionType.STOP if active_prev_state == SafetyState.STOPPED else SafetyDecisionType.UNSAFE
                return (
                    decision,
                    active_prev_state,
                    (
                        f"Obstacle distance ({distance:.2f} m) is in hysteresis deadband "
                        f"[{self.danger_distance:.2f}, {self.safe_distance:.2f}] m. "
                        f"Retaining previous {active_prev_state.value} state until clearance above {self.safe_distance:.2f} m."
                    ),
                )
            else:
                self._current_state = SafetyState.SAFE
                return (
                    SafetyDecisionType.SAFE,
                    SafetyState.SAFE,
                    (
                        f"Obstacle distance ({distance:.2f} m) is in hysteresis deadband "
                        f"[{self.danger_distance:.2f}, {self.safe_distance:.2f}] m. "
                        f"Retaining previous SAFE state until distance drops below {self.danger_distance:.2f} m."
                    ),
                )
