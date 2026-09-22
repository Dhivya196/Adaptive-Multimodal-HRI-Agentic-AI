"""Dedicated Multimodal Fusion component for combining Voice, Vision, and Gesture perception."""

import logging
from typing import Any, Dict, List, Optional

from src.agents.coordinator.schemas import (
    GestureAgentOutput,
    MultimodalTask,
    TaskStatus,
)
from src.agents.vision.schemas import DetectedObject, ProximityLevel, SpatialSector, VisionAgentOutput
from src.agents.voice.schemas import SpeechIntent, UrgencyLevel, VoiceAgentOutput

logger = logging.getLogger("MultimodalFusionEngine")

REFERENTIAL_TARGETS = {"it", "this", "that", "the object", "the item", "one", "the same"}

GESTURE_DIRECTION_TO_SECTOR = {
    "POINT_LEFT": SpatialSector.LEFT,
    "LEFT": SpatialSector.LEFT,
    "POINT_FORWARD": SpatialSector.CENTER,
    "FORWARD": SpatialSector.CENTER,
    "CENTER": SpatialSector.CENTER,
    "POINT_RIGHT": SpatialSector.RIGHT,
    "RIGHT": SpatialSector.RIGHT,
}


class MultimodalFusionEngine:
    """
    Multimodal Fusion Engine for the Coordinator Agent.
    Fuses structured perception outputs while preserving confidence, detecting cross-modal conflicts,
    grounding spatial context, performing deictic target association, and validating target existence.
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

    def associate_pointing_target(
        self,
        vision_output: Optional[VisionAgentOutput],
        gesture_output: Optional[GestureAgentOutput],
    ) -> Dict[str, Any]:
        """
        Spatially ground a pointing gesture against detected visual candidate objects.
        
        Mapping:
            POINT_LEFT / LEFT       -> SpatialSector.LEFT
            POINT_FORWARD / FORWARD -> SpatialSector.CENTER
            POINT_RIGHT / RIGHT     -> SpatialSector.RIGHT
        """
        if gesture_output is None or getattr(gesture_output, "confidence", 0.0) < self.min_confidence_threshold:
            return {
                "is_pointing": False,
                "status": None,
                "reason": "No reliable gesture detected or gesture confidence is below threshold.",
                "matched_object": None,
                "matching_candidates": [],
                "target_sector": None,
                "direction_str": "NONE",
                "gesture_name": "NONE",
                "sector_str": "NONE",
            }

        g_type = str(getattr(gesture_output, "gesture", None) or getattr(gesture_output, "gesture_type", "") or "").upper()
        g_dir = str(getattr(gesture_output, "direction", None) or getattr(gesture_output, "pointing_direction", "") or "").upper()

        # Check if gesture or direction indicates pointing
        is_pointing = (
            g_type in ("POINT_LEFT", "POINT_RIGHT", "POINT_FORWARD", "POINTING", "POINT")
            or g_dir in ("LEFT", "FORWARD", "RIGHT", "CENTER")
            or g_type.startswith("POINT")
        )

        if not is_pointing:
            return {
                "is_pointing": False,
                "status": None,
                "reason": f"Gesture '{g_type}' is not a directional pointing gesture.",
                "matched_object": None,
                "matching_candidates": [],
                "target_sector": None,
                "direction_str": g_dir or g_type or "NONE",
                "gesture_name": g_type or "NONE",
                "sector_str": "NONE",
            }

        # Resolve mapped spatial sector
        mapped_sector: Optional[SpatialSector] = None
        direction_str = "NONE"

        if g_dir in GESTURE_DIRECTION_TO_SECTOR:
            mapped_sector = GESTURE_DIRECTION_TO_SECTOR[g_dir]
            direction_str = g_dir
        elif g_type in GESTURE_DIRECTION_TO_SECTOR:
            mapped_sector = GESTURE_DIRECTION_TO_SECTOR[g_type]
            direction_str = g_type
        else:
            return {
                "is_pointing": False,
                "status": None,
                "reason": f"Unknown directional pointing grounding for gesture '{g_type}' / direction '{g_dir}'.",
                "matched_object": None,
                "matching_candidates": [],
                "target_sector": None,
                "direction_str": g_dir or g_type,
                "gesture_name": g_type,
                "sector_str": "NONE",
            }

        sector_str = mapped_sector.value if isinstance(mapped_sector, SpatialSector) else str(mapped_sector)

        # Obtain candidates in that sector from VisionAgent
        all_objects = vision_output.detected_objects if vision_output else []
        matching_candidates = [
            obj for obj in all_objects
            if obj.spatial_sector == mapped_sector or (
                isinstance(obj.spatial_sector, SpatialSector) and obj.spatial_sector.value.upper() == sector_str.upper()
            )
        ]

        if len(matching_candidates) == 0:
            if len(all_objects) > 0:
                # Objects exist in visual scene, but in DIFFERENT sector(s) than pointed (MODALITY_CONFLICT)
                status = TaskStatus.MODALITY_CONFLICT
                cand_locs = ", ".join([
                    f"{o.label} ({o.spatial_sector.value if isinstance(o.spatial_sector, SpatialSector) else o.spatial_sector})"
                    for o in all_objects
                ])
                reason = (
                    f"Modality conflict: Gesture pointed {direction_str} -> {sector_str}, "
                    f"but detected objects [{cand_locs}] occupy different sector(s)."
                )
            else:
                # No objects detected anywhere in the visual scene (TARGET_NOT_FOUND)
                status = TaskStatus.TARGET_NOT_FOUND
                reason = f"Gesture pointed {direction_str} -> {sector_str}, but no visual candidates were detected in the scene."
            matched_object = None
        elif len(matching_candidates) == 1:
            status = TaskStatus.VALID
            matched_object = matching_candidates[0]
            reason = f"Gesture pointed {direction_str} -> {sector_str}, successfully matched candidate '{matched_object.label}'."
        else:
            status = TaskStatus.AMBIGUOUS_TARGET
            matched_object = None
            cand_names = ", ".join([c.label for c in matching_candidates])
            reason = f"Gesture pointed {direction_str} -> {sector_str}, but multiple objects ({cand_names}) occupy sector {sector_str}."

        return {
            "is_pointing": True,
            "status": status,
            "reason": reason,
            "matched_object": matched_object,
            "matching_candidates": matching_candidates,
            "target_sector": mapped_sector,
            "direction_str": direction_str,
            "gesture_name": g_type or "POINTING",
            "sector_str": sector_str,
        }

    def _print_debug_association(
        self,
        voice_action: str,
        voice_target: str,
        vision_objects: List[DetectedObject],
        gesture_name: str,
        gesture_conf: float,
        sector_str: str,
        candidates: List[DetectedObject],
        selected_target: Optional[str],
        final_action: str,
        final_target: Optional[str],
        final_status: str,
    ):
        """Print concise Coordinator debug output for live demonstration."""
        print("\n" + "=" * 50)
        print("[COORDINATOR]")
        print(f"Voice action: {voice_action.upper()}")
        print(f"Voice target: {voice_target or 'it'}")
        print("\nVision candidates:")
        if vision_objects:
            for obj in vision_objects:
                sec = obj.spatial_sector.value if isinstance(obj.spatial_sector, SpatialSector) else obj.spatial_sector
                print(f"  {obj.label:<10} -> {sec} ({obj.confidence:.2f})")
        else:
            print("  (None)")
        print(f"\nGesture:\n  {gesture_name} ({gesture_conf:.2f})")
        print("\nTarget association:")
        print(f"  {gesture_name} -> {sector_str}")
        cands_str = ", ".join([c.label for c in candidates]) if candidates else "none"
        print(f"  {sector_str} candidates -> {cands_str}")
        print(f"  Selected target -> {selected_target or 'None'}")
        print("\nFusion:")
        print(f"  action = {final_action.upper()}")
        print(f"  target = {final_target or 'None'}")
        print(f"  status = {final_status}")
        print("=" * 50 + "\n", flush=True)

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
            and getattr(gesture_output, "confidence", 0.0) >= self.min_confidence_threshold
            and (
                getattr(gesture_output, "is_gesture_detected", False)
                or getattr(gesture_output, "confidence", 0.0) > 0.0
            )
        )

        # 1. Empty or invalid input
        if not has_voice and not has_vision and not (gesture_output and gesture_output.confidence > 0):
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

        # 3. Three-Modal Fusion (Voice + Vision + Gesture)
        if has_voice and (vision_output is not None) and has_gesture:
            return self._fuse_three_modal(voice_output, vision_output, gesture_output)

        # 4. Multimodal Fusion (Voice + Vision, no reliable gesture)
        if has_voice and has_vision:
            return self._fuse_voice_and_vision(voice_output, vision_output, gesture_output)

        # 5. Voice-only Input
        if has_voice and not has_vision:
            return self._fuse_voice_only(voice_output, gesture_output)

        # 6. Vision-only Input
        if has_vision and not has_voice:
            return self._fuse_vision_only(vision_output, gesture_output)

        # Fallback
        return MultimodalTask(
            action="none",
            task_status=TaskStatus.INVALID_INPUT,
            reasoning="Unable to fuse inputs with current perception state.",
        )

    def _fuse_three_modal(
        self,
        voice_output: VoiceAgentOutput,
        vision_output: VisionAgentOutput,
        gesture_output: GestureAgentOutput,
    ) -> MultimodalTask:
        """Execute three-modal deictic association across Voice, Vision, and Gesture."""
        intent: SpeechIntent = voice_output.speech_intent
        action = intent.action
        raw_target = intent.target_object or ""
        target_clean = raw_target.lower().strip()
        is_referential = target_clean in REFERENTIAL_TARGETS or not target_clean

        voice_conf = max(0.0, min(1.0, voice_output.confidence if voice_output.confidence > 0 else intent.confidence))
        vision_conf = max(0.0, min(1.0, vision_output.confidence))
        gesture_conf = max(0.0, min(1.0, gesture_output.confidence))

        # Check pointing association
        assoc = self.associate_pointing_target(vision_output, gesture_output)

        # If gesture is not a pointing gesture, fallback to voice + vision with 3-modal confidence
        if not assoc["is_pointing"]:
            return self._fuse_voice_and_vision(voice_output, vision_output, gesture_output)

        # ==============================================================
        # Case A: Referential / Pronoun Target ("Pick it up", "this", etc.)
        # ==============================================================
        if is_referential:
            if assoc["status"] == TaskStatus.VALID:
                matched_obj: DetectedObject = assoc["matched_object"]
                fused_conf = (
                    (self.voice_weight * voice_conf)
                    + (self.vision_weight * matched_obj.confidence)
                    + (self.gesture_weight * gesture_conf)
                )
                modality_contrib = {
                    "voice": voice_conf,
                    "vision": matched_obj.confidence,
                    "gesture": gesture_conf,
                }
                meta = {
                    "target_source": "voice + vision + gesture",
                    "gesture_used": True,
                    "gesture_direction": assoc["direction_str"],
                    "pointing_direction": assoc["direction_str"],
                    "gesture_confirmation": True,
                    "visual_target": matched_obj.label,
                    "visual_target_confirmation": True,
                    "multimodal_target_association": True,
                }

                if fused_conf < self.min_confidence_threshold or matched_obj.confidence < self.min_confidence_threshold:
                    status = TaskStatus.LOW_CONFIDENCE
                else:
                    status = TaskStatus.VALID

                self._print_debug_association(
                    voice_action=action,
                    voice_target=raw_target or "it",
                    vision_objects=vision_output.detected_objects,
                    gesture_name=assoc["gesture_name"],
                    gesture_conf=gesture_conf,
                    sector_str=assoc["sector_str"],
                    candidates=assoc["matching_candidates"],
                    selected_target=matched_obj.label,
                    final_action=action,
                    final_target=matched_obj.label,
                    final_status=status.value,
                )

                return MultimodalTask(
                    action=action,
                    target_object=matched_obj.label,
                    target_confirmed=True,
                    spatial_sector=matched_obj.spatial_sector,
                    proximity=matched_obj.proximity,
                    urgency=intent.urgency,
                    confidence=fused_conf,
                    task_status=status,
                    matched_visual_object=matched_obj,
                    reasoning=(
                        f"Multimodal target association: referential target '{raw_target or 'it'}' resolved to "
                        f"'{matched_obj.label}' in sector {matched_obj.spatial_sector.value} via gesture {assoc['gesture_name']} ({assoc['direction_str']})."
                    ),
                    modality_contributions=modality_contrib,
                    metadata=meta,
                )

            elif assoc["status"] == TaskStatus.AMBIGUOUS_TARGET:
                fused_conf = (
                    (self.voice_weight * voice_conf)
                    + (self.vision_weight * vision_conf)
                    + (self.gesture_weight * gesture_conf)
                )
                self._print_debug_association(
                    voice_action=action,
                    voice_target=raw_target or "it",
                    vision_objects=vision_output.detected_objects,
                    gesture_name=assoc["gesture_name"],
                    gesture_conf=gesture_conf,
                    sector_str=assoc["sector_str"],
                    candidates=assoc["matching_candidates"],
                    selected_target=None,
                    final_action=action,
                    final_target=raw_target or "it",
                    final_status=TaskStatus.AMBIGUOUS_TARGET.value,
                )

                return MultimodalTask(
                    action=action,
                    target_object=raw_target or "ambiguous_target",
                    target_confirmed=False,
                    spatial_sector=assoc["target_sector"],
                    urgency=intent.urgency,
                    confidence=fused_conf * 0.6,
                    task_status=TaskStatus.AMBIGUOUS_TARGET,
                    reasoning=assoc["reason"],
                    modality_contributions={"voice": voice_conf, "vision": vision_conf, "gesture": gesture_conf},
                    metadata={
                        "gesture_used": True,
                        "gesture_direction": assoc["direction_str"],
                        "ambiguous_candidates": [c.label for c in assoc["matching_candidates"]],
                    },
                )

            elif assoc["status"] == TaskStatus.MODALITY_CONFLICT:
                fused_conf = (
                    (self.voice_weight * voice_conf)
                    + (self.vision_weight * vision_conf)
                    + (self.gesture_weight * gesture_conf)
                )
                self._print_debug_association(
                    voice_action=action,
                    voice_target=raw_target or "it",
                    vision_objects=vision_output.detected_objects,
                    gesture_name=assoc["gesture_name"],
                    gesture_conf=gesture_conf,
                    sector_str=assoc["sector_str"],
                    candidates=[],
                    selected_target=None,
                    final_action=action,
                    final_target=raw_target or "it",
                    final_status=TaskStatus.MODALITY_CONFLICT.value,
                )

                return MultimodalTask(
                    action=action,
                    target_object=raw_target or "it",
                    target_confirmed=False,
                    spatial_sector=assoc["target_sector"],
                    urgency=intent.urgency,
                    confidence=fused_conf * 0.7,
                    task_status=TaskStatus.MODALITY_CONFLICT,
                    reasoning=assoc["reason"],
                    modality_contributions={"voice": voice_conf, "vision": vision_conf, "gesture": gesture_conf},
                    metadata={
                        "gesture_used": True,
                        "gesture_direction": assoc["direction_str"],
                        "conflict": "gesture_points_empty_sector_while_objects_exist_elsewhere",
                    },
                )

            else:  # TARGET_NOT_FOUND (no objects in visual scene at all)
                fused_conf = (
                    (self.voice_weight * voice_conf)
                    + (self.vision_weight * vision_conf)
                    + (self.gesture_weight * gesture_conf)
                )
                self._print_debug_association(
                    voice_action=action,
                    voice_target=raw_target or "it",
                    vision_objects=vision_output.detected_objects,
                    gesture_name=assoc["gesture_name"],
                    gesture_conf=gesture_conf,
                    sector_str=assoc["sector_str"],
                    candidates=[],
                    selected_target=None,
                    final_action=action,
                    final_target=raw_target or "it",
                    final_status=TaskStatus.TARGET_NOT_FOUND.value,
                )

                return MultimodalTask(
                    action=action,
                    target_object=raw_target or "unknown_target",
                    target_confirmed=False,
                    spatial_sector=assoc["target_sector"],
                    urgency=intent.urgency,
                    confidence=fused_conf * 0.4,
                    task_status=TaskStatus.TARGET_NOT_FOUND,
                    reasoning=assoc["reason"],
                    modality_contributions={"voice": voice_conf, "vision": vision_conf, "gesture": gesture_conf},
                    metadata={
                        "gesture_used": True,
                        "gesture_direction": assoc["direction_str"],
                    },
                )

        # ==============================================================
        # Case B: Explicit Voice Target (e.g. "Pick up the bottle")
        # ==============================================================
        matching_voice_objs = [
            o for o in vision_output.detected_objects
            if target_clean in o.label.lower() or o.label.lower() in target_clean
        ]

        if not matching_voice_objs:
            # Explicit target named by voice is not in the visual scene
            return MultimodalTask(
                action=action,
                target_object=raw_target,
                target_confirmed=False,
                urgency=intent.urgency,
                confidence=voice_conf * 0.5,
                task_status=TaskStatus.TARGET_NOT_FOUND,
                reasoning=(
                    f"Voice command requested target '{raw_target}', but no matching object was "
                    f"detected in vision ({len(vision_output.detected_objects)} other objects present)."
                ),
                modality_contributions={"voice": voice_conf, "vision": vision_conf, "gesture": gesture_conf},
            )

        pointed_sector = assoc["target_sector"]
        sector_str = assoc["sector_str"]
        matching_in_sector = [
            o for o in matching_voice_objs
            if o.spatial_sector == pointed_sector or (
                isinstance(o.spatial_sector, SpatialSector) and o.spatial_sector.value.upper() == sector_str.upper()
            )
        ]

        if matching_in_sector:
            # Agreement between voice target and gesture pointing direction
            matched_obj = max(matching_in_sector, key=lambda o: o.confidence)
            fused_conf = (
                (self.voice_weight * voice_conf)
                + (self.vision_weight * matched_obj.confidence)
                + (self.gesture_weight * gesture_conf)
            )
            modality_contrib = {
                "voice": voice_conf,
                "vision": matched_obj.confidence,
                "gesture": gesture_conf,
            }
            meta = {
                "target_source": "voice + vision + gesture",
                "gesture_used": True,
                "gesture_direction": assoc["direction_str"],
                "pointing_direction": assoc["direction_str"],
                "gesture_confirmation": True,
                "visual_target": matched_obj.label,
                "visual_target_confirmation": True,
                "multimodal_target_association": True,
            }

            if fused_conf < self.min_confidence_threshold or matched_obj.confidence < self.min_confidence_threshold:
                status = TaskStatus.LOW_CONFIDENCE
            else:
                status = TaskStatus.VALID

            self._print_debug_association(
                voice_action=action,
                voice_target=raw_target,
                vision_objects=vision_output.detected_objects,
                gesture_name=assoc["gesture_name"],
                gesture_conf=gesture_conf,
                sector_str=sector_str,
                candidates=matching_in_sector,
                selected_target=matched_obj.label,
                final_action=action,
                final_target=matched_obj.label,
                final_status=status.value,
            )

            return MultimodalTask(
                action=action,
                target_object=matched_obj.label,
                target_confirmed=True,
                spatial_sector=matched_obj.spatial_sector,
                proximity=matched_obj.proximity,
                urgency=intent.urgency,
                confidence=fused_conf,
                task_status=status,
                matched_visual_object=matched_obj,
                reasoning=(
                    f"Explicit voice target '{raw_target}' confirmed in visual sector {matched_obj.spatial_sector.value} "
                    f"with matching gesture {assoc['gesture_name']} ({assoc['direction_str']})."
                ),
                modality_contributions=modality_contrib,
                metadata=meta,
            )
        else:
            # Modality conflict: target object detected in a different sector than pointed by gesture
            matched_obj = max(matching_voice_objs, key=lambda o: o.confidence)
            fused_conf = (
                (self.voice_weight * voice_conf)
                + (self.vision_weight * matched_obj.confidence)
                + (self.gesture_weight * gesture_conf)
            )
            self._print_debug_association(
                voice_action=action,
                voice_target=raw_target,
                vision_objects=vision_output.detected_objects,
                gesture_name=assoc["gesture_name"],
                gesture_conf=gesture_conf,
                sector_str=sector_str,
                candidates=matching_voice_objs,
                selected_target=None,
                final_action=action,
                final_target=raw_target,
                final_status=TaskStatus.MODALITY_CONFLICT.value,
            )

            return MultimodalTask(
                action=action,
                target_object=raw_target,
                target_confirmed=True,
                spatial_sector=matched_obj.spatial_sector,
                proximity=matched_obj.proximity,
                urgency=intent.urgency,
                confidence=fused_conf * 0.7,
                task_status=TaskStatus.MODALITY_CONFLICT,
                matched_visual_object=matched_obj,
                reasoning=(
                    f"Modality conflict: Voice target '{raw_target}' detected in sector '{matched_obj.spatial_sector.value}', "
                    f"but gesture pointed '{assoc['direction_str']}'."
                ),
                modality_contributions={"voice": voice_conf, "vision": matched_obj.confidence, "gesture": gesture_conf},
                metadata={
                    "gesture_used": True,
                    "gesture_direction": assoc["direction_str"],
                    "conflict": "voice_vision_sector_vs_gesture_direction",
                },
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

        # Check for referential pronouns without gesture
        is_referential = False
        if target_name:
            target_clean = target_name.lower().strip()
            is_referential = target_clean in REFERENTIAL_TARGETS

        # Check for target object matching in visual detections
        matched_obj: Optional[DetectedObject] = None
        candidates: List[DetectedObject] = []

        if target_name and not is_referential:
            target_clean = target_name.lower().strip()
            for obj in vision_output.detected_objects:
                label_clean = obj.label.lower().strip()
                if target_clean in label_clean or label_clean in target_clean:
                    candidates.append(obj)

            if candidates:
                # Sector disambiguation if sector is specified in Voice
                desired_sector = intent.spatial_sector
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

        # Referential pronoun with no gesture -> leave unconfirmed for MemoryAgent cross-turn resolution
        if is_referential:
            fused_conf = (self.voice_weight * voice_conf) + (self.vision_weight * vision_output.confidence)
            return MultimodalTask(
                action=action,
                target_object=target_name,
                target_confirmed=False,
                urgency=intent.urgency,
                confidence=fused_conf,
                task_status=TaskStatus.VALID,
                reasoning=f"Referential target '{target_name}' passed to MemoryAgent for cross-turn context resolution.",
                modality_contributions={"voice": voice_conf, "vision": vision_output.confidence},
            )

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

            # Optional low-confidence / non-pointing gesture contribution
            if gesture_output and gesture_output.confidence > 0:
                gesture_conf = gesture_output.confidence
                fused_conf = (self.voice_weight * voice_conf) + (self.vision_weight * vision_conf) + (self.gesture_weight * gesture_conf)
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
