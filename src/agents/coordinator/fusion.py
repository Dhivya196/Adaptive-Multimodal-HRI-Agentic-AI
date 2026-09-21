"""Dedicated Multimodal Fusion component for combining Voice, Vision, and Gesture perception."""

from typing import Any, Dict, List, Optional

from src.agents.coordinator.schemas import (
    GestureAgentOutput,
    MultimodalTask,
    TaskStatus,
)
from src.agents.vision.schemas import DetectedObject, ProximityLevel, SpatialSector, VisionAgentOutput
from src.agents.voice.schemas import SpeechIntent, UrgencyLevel, VoiceAgentOutput


class MultimodalFusionEngine:
    """
    Multimodal Fusion Engine for the Coordinator Agent.
    Fuses structured perception outputs while preserving confidence, detecting cross-modal conflicts,
    grounding spatial context, and validating target existence.
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.40,
        voice_weight: float = 0.55,
        vision_weight: float = 0.35,
        gesture_weight: float = 0.10,
        require_visual_target_confirmation: bool = True,
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.voice_weight = voice_weight
        self.vision_weight = vision_weight
        self.gesture_weight = gesture_weight
        self.require_visual_target_confirmation = require_visual_target_confirmation

    def fuse(
        self,
        voice_output: Optional[VoiceAgentOutput] = None,
        vision_output: Optional[VisionAgentOutput] = None,
        gesture_output: Optional[GestureAgentOutput] = None,
    ) -> MultimodalTask:
        """
        Execute multimodal fusion across perception inputs.
        """
        has_voice = (
            voice_output is not None
            and voice_output.is_speech_detected
            and voice_output.speech_intent is not None
        )
        has_vision = (
            vision_output is not None
            and len(vision_output.detected_objects) > 0
        )
        has_gesture = (
            gesture_output is not None
            and gesture_output.confidence > 0.0
        )

        # 1. Empty or invalid input
        if not has_voice and not has_vision and not has_gesture:
            if voice_output is not None and voice_output.transcript:
                return MultimodalTask(
                    action="none",
                    confidence=max(0.1, voice_output.confidence),
                    task_status=TaskStatus.INVALID_INPUT,
                    reasoning="Speech was transcribed but no actionable intent was parsed.",
                    modality_contributions={"voice": voice_output.confidence},
                )
            return MultimodalTask(
                action="none",
                confidence=0.0,
                task_status=TaskStatus.INVALID_INPUT,
                reasoning="No valid perception data received from Voice, Vision, or Gesture agents.",
            )

        # 2. Emergency / Stop Command from Voice (Highest priority, target independent)
        if has_voice:
            intent = voice_output.speech_intent
            if intent.action == "stop_robot" or intent.urgency in (UrgencyLevel.EMERGENCY, "EMERGENCY"):
                return MultimodalTask(
                    action="stop_robot",
                    target_object=intent.target_object,
                    target_confirmed=True,
                    urgency=UrgencyLevel.EMERGENCY,
                    confidence=voice_output.confidence,
                    task_status=TaskStatus.VALID,
                    reasoning="Immediate stop command received with critical urgency.",
                    modality_contributions={"voice": voice_output.confidence},
                )

        # 3. Multimodal Fusion (Voice + Vision)
        if has_voice and has_vision:
            return self._fuse_voice_and_vision(voice_output, vision_output, gesture_output)

        # 4. Voice-only Input
        if has_voice and not has_vision:
            return self._fuse_voice_only(voice_output, gesture_output)

        # 5. Vision-only Input
        if has_vision and not has_voice:
            return self._fuse_vision_only(vision_output, gesture_output)

        # Fallback
        return MultimodalTask(
            action="none",
            task_status=TaskStatus.INVALID_INPUT,
            reasoning="Unable to fuse inputs with current perception state.",
        )

    def _fuse_voice_and_vision(
        self,
        voice_output: VoiceAgentOutput,
        vision_output: VisionAgentOutput,
        gesture_output: Optional[GestureAgentOutput] = None,
    ) -> MultimodalTask:
        """Cross-correlate Voice intent and entities with visual detections."""
        intent: SpeechIntent = voice_output.speech_intent
        action = intent.action
        target_name = intent.target_object
        voice_conf = max(0.0, min(1.0, voice_output.confidence if voice_output.confidence > 0 else intent.confidence))

        # Check for target object matching in visual detections
        matched_obj: Optional[DetectedObject] = None
        candidates: List[DetectedObject] = []

        if target_name:
            target_clean = target_name.lower().strip()
            for obj in vision_output.detected_objects:
                label_clean = obj.label.lower().strip()
                if target_clean in label_clean or label_clean in target_clean:
                    candidates.append(obj)

            if candidates:
                # Sector disambiguation if sector is specified in Voice or Gesture
                desired_sector = intent.spatial_sector
                if gesture_output and gesture_output.pointing_direction:
                    desired_sector = gesture_output.pointing_direction

                if desired_sector:
                    sector_matches = [
                        c for c in candidates
                        if c.spatial_sector.value.upper() == desired_sector.upper()
                    ]
                    if sector_matches:
                        matched_obj = max(sector_matches, key=lambda o: o.confidence)
                    else:
                        matched_obj = max(candidates, key=lambda o: o.confidence)
                else:
                    matched_obj = max(candidates, key=lambda o: o.confidence)

        # Target requested by voice was NOT found in vision
        if target_name and matched_obj is None:
            return MultimodalTask(
                action=action,
                target_object=target_name,
                target_confirmed=False,
                urgency=intent.urgency,
                confidence=voice_conf * 0.5,
                task_status=TaskStatus.TARGET_NOT_FOUND,
                reasoning=(
                    f"Voice command requested target '{target_name}', but no matching object was "
                    f"detected in the current visual scene ({len(vision_output.detected_objects)} other objects present)."
                ),
                modality_contributions={
                    "voice": voice_conf,
                    "vision": vision_output.confidence,
                },
            )

        # Target specified and confirmed in vision
        if target_name and matched_obj is not None:
            vision_conf = matched_obj.confidence
            fused_conf = (self.voice_weight * voice_conf) + (self.vision_weight * vision_conf)
            modality_contrib = {"voice": voice_conf, "vision": vision_conf}

            # Gesture contribution if available
            if gesture_output and gesture_output.confidence > 0:
                gesture_conf = gesture_output.confidence
                fused_conf = (0.5 * voice_conf) + (0.35 * vision_conf) + (0.15 * gesture_conf)
                modality_contrib["gesture"] = gesture_conf

            # Modality conflict: Voice specified sector vs Visual detection sector
            if intent.spatial_sector and matched_obj.spatial_sector:
                if intent.spatial_sector.upper() != matched_obj.spatial_sector.value.upper():
                    return MultimodalTask(
                        action=action,
                        target_object=target_name,
                        target_confirmed=True,
                        spatial_sector=matched_obj.spatial_sector,
                        proximity=matched_obj.proximity,
                        urgency=intent.urgency,
                        confidence=fused_conf * 0.7,
                        task_status=TaskStatus.MODALITY_CONFLICT,
                        matched_visual_object=matched_obj,
                        reasoning=(
                            f"Modality conflict: Voice specified sector '{intent.spatial_sector}', "
                            f"but target '{target_name}' was detected in sector '{matched_obj.spatial_sector.value}'."
                        ),
                        modality_contributions=modality_contrib,
                    )

            # Low confidence check
            if fused_conf < self.min_confidence_threshold or vision_conf < self.min_confidence_threshold:
                return MultimodalTask(
                    action=action,
                    target_object=target_name,
                    target_confirmed=True,
                    spatial_sector=matched_obj.spatial_sector,
                    proximity=matched_obj.proximity,
                    urgency=intent.urgency,
                    confidence=fused_conf,
                    task_status=TaskStatus.LOW_CONFIDENCE,
                    matched_visual_object=matched_obj,
                    reasoning=f"Detection confidence for '{target_name}' ({vision_conf:.2f}) is below threshold.",
                    modality_contributions=modality_contrib,
                )

            return MultimodalTask(
                action=action,
                target_object=target_name,
                target_confirmed=True,
                spatial_sector=matched_obj.spatial_sector,
                proximity=matched_obj.proximity,
                urgency=intent.urgency,
                confidence=fused_conf,
                task_status=TaskStatus.VALID,
                matched_visual_object=matched_obj,
                reasoning=(
                    f"Successfully fused voice intent '{action}' with target '{target_name}' "
                    f"grounded at sector {matched_obj.spatial_sector.value} and proximity {matched_obj.proximity.value}."
                ),
                modality_contributions=modality_contrib,
            )

        # Voice command with no explicit target (e.g. query, inspect general scene)
        return MultimodalTask(
            action=action,
            target_object=None,
            target_confirmed=False,
            spatial_sector=None,
            proximity=None,
            urgency=intent.urgency,
            confidence=voice_conf,
            task_status=TaskStatus.VALID,
            reasoning=f"Voice action '{action}' validated without specific target object requirement.",
            modality_contributions={"voice": voice_conf, "vision": vision_output.confidence},
        )

    def _fuse_voice_only(
        self,
        voice_output: VoiceAgentOutput,
        gesture_output: Optional[GestureAgentOutput] = None,
    ) -> MultimodalTask:
        """Handle scenario where only voice input is available."""
        intent: SpeechIntent = voice_output.speech_intent
        action = intent.action
        voice_conf = voice_output.confidence if voice_output.confidence > 0 else intent.confidence

        spatial = None
        if intent.spatial_sector:
            try:
                spatial = SpatialSector(intent.spatial_sector.upper())
            except ValueError:
                spatial = SpatialSector.UNKNOWN

        return MultimodalTask(
            action=action,
            target_object=intent.target_object,
            target_confirmed=False,  # Unconfirmed visually
            spatial_sector=spatial,
            urgency=intent.urgency,
            confidence=voice_conf * 0.85,
            task_status=TaskStatus.VALID,
            reasoning=f"Interpreted from voice-only input: action '{action}', target '{intent.target_object}'.",
            modality_contributions={"voice": voice_conf},
        )

    def _fuse_vision_only(
        self,
        vision_output: VisionAgentOutput,
        gesture_output: Optional[GestureAgentOutput] = None,
    ) -> MultimodalTask:
        """Handle scenario where only visual perception is available without spoken instruction."""
        n_objects = len(vision_output.detected_objects)
        return MultimodalTask(
            action="none",
            target_object=None,
            target_confirmed=False,
            confidence=vision_output.confidence,
            task_status=TaskStatus.VALID,
            reasoning=f"Passive scene perception: {n_objects} objects detected across sectors.",
            modality_contributions={"vision": vision_output.confidence},
        )
