"""Dedicated Multimodal Fusion component for combining Voice, Vision, and Gesture perception."""

import logging
from typing import Any, Dict, List, Optional

from src.agents.coordinator.schemas import (
    GestureAgentOutput,
    GroundingStatus,
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
        high_confidence_threshold: float = 0.60,
        ambiguity_threshold: float = 0.10,
        voice_weight: float = 0.55,
        vision_weight: float = 0.35,
        gesture_weight: float = 0.10,
        require_visual_target_confirmation: bool = True,
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.voice_weight = voice_weight
        self.vision_weight = vision_weight
        self.gesture_weight = gesture_weight
        self.require_visual_target_confirmation = require_visual_target_confirmation

    def _get_proximity_score(self, proximity: Optional[ProximityLevel]) -> float:
        """Convert proximity category into normalized numerical grounding score."""
        if proximity == ProximityLevel.NEAR:
            return 1.0
        elif proximity == ProximityLevel.MEDIUM:
            return 0.85
        elif proximity == ProximityLevel.FAR:
            return 0.70
        return 0.80

    def calculate_referential_grounding_score(
        self,
        object_confidence: float,
        gesture_confidence: float,
        spatial_agreement: bool,
        proximity: Optional[ProximityLevel] = None,
    ) -> float:
        """
        Calculate a composite referential grounding score based on:
        - object detection confidence
        - spatial agreement with gesture
        - gesture confidence
        - proximity factor
        """
        if not spatial_agreement:
            return 0.0

        prox_score = self._get_proximity_score(proximity)
        # Weighted composite score: 50% object detection, 35% gesture pointing, 15% proximity
        score = (0.50 * object_confidence) + (0.35 * gesture_confidence) + (0.15 * prox_score)
        return max(0.0, min(1.0, score))

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
                "grounding_status": GroundingStatus.UNRESOLVED,
                "reason": "No reliable gesture detected or gesture confidence is below threshold.",
                "matched_object": None,
                "matching_candidates": [],
                "all_scored_candidates": [],
                "target_sector": None,
                "direction_str": "NONE",
                "gesture_name": "NONE",
                "sector_str": "NONE",
                "referential_score": 0.0,
            }

        g_type = str(getattr(gesture_output, "gesture", None) or getattr(gesture_output, "gesture_type", "") or "").upper()
        g_dir = str(getattr(gesture_output, "direction", None) or getattr(gesture_output, "pointing_direction", "") or "").upper()
        g_conf = max(0.0, min(1.0, getattr(gesture_output, "confidence", 0.0)))

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
                "grounding_status": GroundingStatus.UNRESOLVED,
                "reason": f"Gesture '{g_type}' is not a directional pointing gesture.",
                "matched_object": None,
                "matching_candidates": [],
                "all_scored_candidates": [],
                "target_sector": None,
                "direction_str": g_dir or g_type or "NONE",
                "gesture_name": g_type or "NONE",
                "sector_str": "NONE",
                "referential_score": 0.0,
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
                "grounding_status": GroundingStatus.UNRESOLVED,
                "reason": f"Unknown directional pointing grounding for gesture '{g_type}' / direction '{g_dir}'.",
                "matched_object": None,
                "matching_candidates": [],
                "all_scored_candidates": [],
                "target_sector": None,
                "direction_str": g_dir or g_type,
                "gesture_name": g_type,
                "sector_str": "NONE",
                "referential_score": 0.0,
            }

        sector_str = mapped_sector.value if isinstance(mapped_sector, SpatialSector) else str(mapped_sector)

        # Inspect ALL candidates in the visual scene and score each candidate
        all_objects = vision_output.detected_objects if vision_output else []
        all_scored_candidates = []
        matching_candidates: List[DetectedObject] = []

        for obj in all_objects:
            obj_sec = obj.spatial_sector.value if isinstance(obj.spatial_sector, SpatialSector) else str(obj.spatial_sector)
            sec_match = (
                obj.spatial_sector == mapped_sector
                or (isinstance(obj.spatial_sector, SpatialSector) and obj.spatial_sector.value.upper() == sector_str.upper())
                or (str(obj_sec).upper() == sector_str.upper())
            )
            ref_score = self.calculate_referential_grounding_score(
                object_confidence=obj.confidence,
                gesture_confidence=g_conf,
                spatial_agreement=sec_match,
                proximity=obj.proximity,
            )
            all_scored_candidates.append({
                "object": obj,
                "spatial_agreement": sec_match,
                "object_confidence": obj.confidence,
                "referential_score": ref_score,
            })
            if sec_match:
                matching_candidates.append(obj)

        matched_object: Optional[DetectedObject] = None
        matched_ref_score: float = 0.0

        if len(matching_candidates) == 0:
            if len(all_objects) > 0:
                # Objects exist in visual scene, but in DIFFERENT sector(s) than pointed (MODALITY_CONFLICT)
                status = TaskStatus.MODALITY_CONFLICT
                grounding_status = GroundingStatus.MODALITY_CONFLICT
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
                grounding_status = GroundingStatus.UNRESOLVED
                reason = f"Gesture pointed {direction_str} -> {sector_str}, but no visual candidates were detected in the scene."
            matched_object = None

        elif len(matching_candidates) == 1:
            matched_object = matching_candidates[0]
            matched_ref_score = self.calculate_referential_grounding_score(
                object_confidence=matched_object.confidence,
                gesture_confidence=g_conf,
                spatial_agreement=True,
                proximity=matched_object.proximity,
            )
            if matched_object.confidence >= self.high_confidence_threshold:
                grounding_status = GroundingStatus.HIGH_CONFIDENCE
                status = TaskStatus.VALID
                reason = (
                    f"Gesture pointed {direction_str} -> {sector_str}, successfully matched candidate "
                    f"'{matched_object.label}' with high confidence ({matched_object.confidence:.2f})."
                )
            else:
                grounding_status = GroundingStatus.LOW_CONFIDENCE
                status = TaskStatus.VALID
                reason = (
                    f"Gesture pointed {direction_str} -> {sector_str}, associated candidate "
                    f"'{matched_object.label}' but object confidence ({matched_object.confidence:.2f}) is below threshold ({self.high_confidence_threshold:.2f})."
                )

        else:
            # Multiple objects present in the pointed sector
            matching_candidates.sort(key=lambda o: o.confidence, reverse=True)
            top_obj = matching_candidates[0]
            second_obj = matching_candidates[1]

            # Check if candidates have ambiguous / close confidence
            if abs(top_obj.confidence - second_obj.confidence) <= self.ambiguity_threshold:
                status = TaskStatus.AMBIGUOUS_TARGET
                grounding_status = GroundingStatus.AMBIGUOUS
                matched_object = None
                cand_names = ", ".join([f"{c.label} ({c.confidence:.2f})" for c in matching_candidates])
                reason = f"Multiple objects are present in the pointed sector ({sector_str}) with similar confidence: {cand_names}."
            else:
                # Distinct dominant candidate
                matched_object = top_obj
                matched_ref_score = self.calculate_referential_grounding_score(
                    object_confidence=matched_object.confidence,
                    gesture_confidence=g_conf,
                    spatial_agreement=True,
                    proximity=matched_object.proximity,
                )
                if matched_object.confidence >= self.high_confidence_threshold:
                    grounding_status = GroundingStatus.HIGH_CONFIDENCE
                else:
                    grounding_status = GroundingStatus.LOW_CONFIDENCE
                status = TaskStatus.VALID
                reason = f"Gesture pointed {direction_str} -> {sector_str}, selected dominant candidate '{matched_object.label}'."

        return {
            "is_pointing": True,
            "status": status,
            "grounding_status": grounding_status,
            "reason": reason,
            "matched_object": matched_object,
            "matching_candidates": matching_candidates,
            "all_scored_candidates": all_scored_candidates,
            "target_sector": mapped_sector,
            "direction_str": direction_str,
            "gesture_name": g_type or "POINTING",
            "gesture_confidence": g_conf,
            "sector_str": sector_str,
            "referential_score": matched_ref_score,
        }

    def _print_debug_association(
        self,
        voice_ref: str,
        gesture_name: str,
        gesture_conf: float,
        candidate_name: Optional[str],
        object_conf: Optional[float],
        spatial_agreement: bool,
        grounding_status: str,
    ):
        """Print concise Coordinator debug output for live demonstration."""
        print("\n" + "=" * 50)
        print("[COORDINATOR]")
        print(f"  Voice reference       : \"{voice_ref}\"")
        if gesture_name and gesture_name != "NONE":
            print(f"  Gesture               : {gesture_name} ({gesture_conf:.2f})")
        else:
            print("  Gesture               : NONE")
        print(f"  Candidate             : {candidate_name or 'None'}")
        if object_conf is not None:
            print(f"  Object confidence     : {object_conf:.2f}")
        else:
            print("  Object confidence     : N/A")
        print(f"  Spatial agreement     : {str(spatial_agreement).upper()}")
        print(f"  Grounding status      : {grounding_status}")
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
                    grounding_status=GroundingStatus.UNRESOLVED,
                    reasoning="Speech was transcribed but no actionable intent was parsed.",
                    modality_contributions={"voice": voice_output.confidence},
                )
            return MultimodalTask(
                action="none",
                confidence=0.0,
                task_status=TaskStatus.INVALID_INPUT,
                grounding_status=GroundingStatus.UNRESOLVED,
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
                    grounding_status=GroundingStatus.HIGH_CONFIDENCE,
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
            grounding_status=GroundingStatus.UNRESOLVED,
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
                target_conf = matched_obj.confidence
                fused_conf = (
                    (self.voice_weight * voice_conf)
                    + (self.vision_weight * target_conf)
                    + (self.gesture_weight * gesture_conf)
                )
                modality_contrib = {
                    "voice": voice_conf,
                    "vision": target_conf,
                    "gesture": gesture_conf,
                }

                if target_conf >= self.high_confidence_threshold:
                    grounding_st = GroundingStatus.HIGH_CONFIDENCE
                    ref_grounding_st = "HIGH_CONFIDENCE"
                else:
                    grounding_st = GroundingStatus.LOW_CONFIDENCE
                    ref_grounding_st = GroundingStatus.LOW_CONFIDENCE_REFERENTIAL_GROUNDING.value

                if fused_conf < self.min_confidence_threshold or target_conf < self.min_confidence_threshold:
                    task_st = TaskStatus.LOW_CONFIDENCE
                else:
                    task_st = TaskStatus.VALID

                meta = {
                    "target_source": "voice + vision + gesture",
                    "voice_reference": raw_target or "it",
                    "gesture_used": True,
                    "gesture_direction": assoc["direction_str"],
                    "gesture_confidence": gesture_conf,
                    "pointing_direction": assoc["direction_str"],
                    "gesture_confirmation": True,
                    "visual_target": matched_obj.label,
                    "visual_target_confidence": target_conf,
                    "perception_confidence": target_conf,
                    "spatial_agreement": True,
                    "grounding_status": grounding_st.value,
                    "referential_grounding_status": ref_grounding_st,
                    "multimodal_target_association": True,
                }

                self._print_debug_association(
                    voice_ref=raw_target or "it",
                    gesture_name=f"{assoc['gesture_name']} ({assoc['direction_str']})",
                    gesture_conf=gesture_conf,
                    candidate_name=matched_obj.label,
                    object_conf=target_conf,
                    spatial_agreement=True,
                    grounding_status=grounding_st.value,
                )

                return MultimodalTask(
                    action=action,
                    target_object=matched_obj.label,
                    target_confirmed=True,
                    target_confidence=target_conf,
                    spatial_sector=matched_obj.spatial_sector,
                    proximity=matched_obj.proximity,
                    urgency=intent.urgency,
                    confidence=fused_conf,
                    grounding_status=grounding_st,
                    referential_grounding_status=ref_grounding_st,
                    referential_grounding_score=assoc.get("referential_score", 0.0),
                    spatial_agreement=True,
                    task_status=task_st,
                    matched_visual_object=matched_obj,
                    reasoning=(
                        f"Multimodal target association: referential target '{raw_target or 'it'}' resolved to "
                        f"'{matched_obj.label}' (conf: {target_conf:.2f}) in sector {matched_obj.spatial_sector.value} "
                        f"via gesture {assoc['gesture_name']} ({assoc['direction_str']}). Grounding: {grounding_st.value}."
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
                    voice_ref=raw_target or "it",
                    gesture_name=f"{assoc['gesture_name']} ({assoc['direction_str']})",
                    gesture_conf=gesture_conf,
                    candidate_name=None,
                    object_conf=None,
                    spatial_agreement=True,
                    grounding_status=GroundingStatus.AMBIGUOUS.value,
                )

                return MultimodalTask(
                    action=action,
                    target_object=raw_target or "ambiguous_target",
                    target_confirmed=False,
                    target_confidence=vision_conf,
                    spatial_sector=assoc["target_sector"],
                    urgency=intent.urgency,
                    confidence=fused_conf * 0.6,
                    grounding_status=GroundingStatus.AMBIGUOUS,
                    referential_grounding_status=GroundingStatus.AMBIGUOUS.value,
                    spatial_agreement=True,
                    task_status=TaskStatus.AMBIGUOUS_TARGET,
                    reasoning=assoc["reason"],
                    modality_contributions={"voice": voice_conf, "vision": vision_conf, "gesture": gesture_conf},
                    metadata={
                        "voice_reference": raw_target or "it",
                        "gesture_used": True,
                        "gesture_direction": assoc["direction_str"],
                        "gesture_confidence": gesture_conf,
                        "spatial_agreement": True,
                        "grounding_status": GroundingStatus.AMBIGUOUS.value,
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
                    voice_ref=raw_target or "it",
                    gesture_name=f"{assoc['gesture_name']} ({assoc['direction_str']})",
                    gesture_conf=gesture_conf,
                    candidate_name=None,
                    object_conf=None,
                    spatial_agreement=False,
                    grounding_status=GroundingStatus.MODALITY_CONFLICT.value,
                )

                return MultimodalTask(
                    action=action,
                    target_object=raw_target or "it",
                    target_confirmed=False,
                    spatial_sector=assoc["target_sector"],
                    urgency=intent.urgency,
                    confidence=fused_conf * 0.7,
                    grounding_status=GroundingStatus.MODALITY_CONFLICT,
                    referential_grounding_status=GroundingStatus.MODALITY_CONFLICT.value,
                    spatial_agreement=False,
                    task_status=TaskStatus.MODALITY_CONFLICT,
                    reasoning=assoc["reason"],
                    modality_contributions={"voice": voice_conf, "vision": vision_conf, "gesture": gesture_conf},
                    metadata={
                        "voice_reference": raw_target or "it",
                        "gesture_used": True,
                        "gesture_direction": assoc["direction_str"],
                        "gesture_confidence": gesture_conf,
                        "spatial_agreement": False,
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
                    voice_ref=raw_target or "it",
                    gesture_name=f"{assoc['gesture_name']} ({assoc['direction_str']})",
                    gesture_conf=gesture_conf,
                    candidate_name=None,
                    object_conf=None,
                    spatial_agreement=False,
                    grounding_status=GroundingStatus.UNRESOLVED.value,
                )

                return MultimodalTask(
                    action=action,
                    target_object=raw_target or "unknown_target",
                    target_confirmed=False,
                    spatial_sector=assoc["target_sector"],
                    urgency=intent.urgency,
                    confidence=fused_conf * 0.4,
                    grounding_status=GroundingStatus.UNRESOLVED,
                    referential_grounding_status=GroundingStatus.UNRESOLVED.value,
                    spatial_agreement=False,
                    task_status=TaskStatus.TARGET_NOT_FOUND,
                    reasoning=assoc["reason"],
                    modality_contributions={"voice": voice_conf, "vision": vision_conf, "gesture": gesture_conf},
                    metadata={
                        "voice_reference": raw_target or "it",
                        "gesture_used": True,
                        "gesture_direction": assoc["direction_str"],
                        "gesture_confidence": gesture_conf,
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
                grounding_status=GroundingStatus.UNRESOLVED,
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
            # Spatial agreement between explicit voice target and gesture pointing direction
            matched_obj = max(matching_in_sector, key=lambda o: o.confidence)
            target_conf = matched_obj.confidence
            fused_conf = (
                (self.voice_weight * voice_conf)
                + (self.vision_weight * target_conf)
                + (self.gesture_weight * gesture_conf)
            )
            modality_contrib = {
                "voice": voice_conf,
                "vision": target_conf,
                "gesture": gesture_conf,
            }

            if target_conf >= self.high_confidence_threshold:
                grounding_st = GroundingStatus.HIGH_CONFIDENCE
            else:
                grounding_st = GroundingStatus.LOW_CONFIDENCE

            if fused_conf < self.min_confidence_threshold or target_conf < self.min_confidence_threshold:
                status = TaskStatus.LOW_CONFIDENCE
            else:
                status = TaskStatus.VALID

            meta = {
                "target_source": "voice + vision + gesture",
                "gesture_used": True,
                "gesture_direction": assoc["direction_str"],
                "gesture_confidence": gesture_conf,
                "pointing_direction": assoc["direction_str"],
                "gesture_confirmation": True,
                "visual_target": matched_obj.label,
                "visual_target_confidence": target_conf,
                "spatial_agreement": True,
                "grounding_status": grounding_st.value,
                "visual_target_confirmation": True,
                "multimodal_target_association": True,
            }

            self._print_debug_association(
                voice_ref=raw_target,
                gesture_name=f"{assoc['gesture_name']} ({assoc['direction_str']})",
                gesture_conf=gesture_conf,
                candidate_name=matched_obj.label,
                object_conf=target_conf,
                spatial_agreement=True,
                grounding_status=grounding_st.value,
            )

            return MultimodalTask(
                action=action,
                target_object=matched_obj.label,
                target_confirmed=True,
                target_confidence=target_conf,
                spatial_sector=matched_obj.spatial_sector,
                proximity=matched_obj.proximity,
                urgency=intent.urgency,
                confidence=fused_conf,
                grounding_status=grounding_st,
                spatial_agreement=True,
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
            # Spatial conflict: explicit target object detected in a different sector than pointed by gesture
            # Retain the explicit target name; DO NOT replace with unrelated objects in the pointed sector
            matched_obj = max(matching_voice_objs, key=lambda o: o.confidence)
            target_conf = matched_obj.confidence
            fused_conf = (
                (self.voice_weight * voice_conf)
                + (self.vision_weight * target_conf)
                + (self.gesture_weight * gesture_conf)
            )

            self._print_debug_association(
                voice_ref=raw_target,
                gesture_name=f"{assoc['gesture_name']} ({assoc['direction_str']})",
                gesture_conf=gesture_conf,
                candidate_name=matched_obj.label,
                object_conf=target_conf,
                spatial_agreement=False,
                grounding_status=GroundingStatus.EXPLICIT_TARGET_SPATIAL_CONFLICT.value,
            )

            return MultimodalTask(
                action=action,
                target_object=raw_target,
                target_confirmed=True,
                target_confidence=target_conf,
                spatial_sector=matched_obj.spatial_sector,
                proximity=matched_obj.proximity,
                urgency=intent.urgency,
                confidence=fused_conf * 0.7,
                grounding_status=GroundingStatus.EXPLICIT_TARGET_SPATIAL_CONFLICT,
                spatial_agreement=False,
                task_status=TaskStatus.MODALITY_CONFLICT,
                matched_visual_object=matched_obj,
                reasoning=(
                    f"Explicit target spatial conflict: Voice target '{raw_target}' detected in sector '{matched_obj.spatial_sector.value}', "
                    f"but gesture pointed '{assoc['direction_str']}'."
                ),
                modality_contributions={"voice": voice_conf, "vision": target_conf, "gesture": gesture_conf},
                metadata={
                    "gesture_used": True,
                    "gesture_direction": assoc["direction_str"],
                    "gesture_confidence": gesture_conf,
                    "spatial_agreement": False,
                    "conflict": "explicit_target_spatial_conflict",
                    "target_sector": (
                        matched_obj.spatial_sector.value
                        if isinstance(matched_obj.spatial_sector, SpatialSector)
                        else str(matched_obj.spatial_sector)
                    ),
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

        # Referential pronoun with no gesture
        if is_referential:
            fused_conf = (self.voice_weight * voice_conf) + (self.vision_weight * vision_output.confidence)
            detected_objs = vision_output.detected_objects or []

            if len(detected_objs) > 1:
                # Ambiguous deictic target without gesture pointing to resolve it
                return MultimodalTask(
                    action=action,
                    target_object=target_name or "it",
                    target_confirmed=False,
                    target_confidence=vision_output.confidence,
                    urgency=intent.urgency,
                    confidence=fused_conf * 0.6,
                    grounding_status=GroundingStatus.AMBIGUOUS,
                    referential_grounding_status=GroundingStatus.AMBIGUOUS.value,
                    task_status=TaskStatus.AMBIGUOUS_TARGET,
                    reasoning=f"Referential target '{target_name or 'it'}' is ambiguous with {len(detected_objs)} visual candidates and no pointing gesture.",
                    modality_contributions={"voice": voice_conf, "vision": vision_output.confidence},
                    metadata={"ambiguous_candidates": [o.label for o in detected_objs]},
                )
            elif len(detected_objs) == 1:
                single_obj = detected_objs[0]
                return MultimodalTask(
                    action=action,
                    target_object=single_obj.label,
                    target_confirmed=True,
                    target_confidence=single_obj.confidence,
                    spatial_sector=single_obj.spatial_sector,
                    proximity=single_obj.proximity,
                    urgency=intent.urgency,
                    confidence=fused_conf,
                    grounding_status=(
                        GroundingStatus.HIGH_CONFIDENCE
                        if single_obj.confidence >= self.high_confidence_threshold
                        else GroundingStatus.LOW_CONFIDENCE
                    ),
                    task_status=TaskStatus.VALID,
                    matched_visual_object=single_obj,
                    reasoning=f"Referential target '{target_name or 'it'}' resolved to sole visual object '{single_obj.label}'.",
                    modality_contributions={"voice": voice_conf, "vision": single_obj.confidence},
                )
            else:
                return MultimodalTask(
                    action=action,
                    target_object=target_name or "it",
                    target_confirmed=False,
                    urgency=intent.urgency,
                    confidence=fused_conf,
                    grounding_status=GroundingStatus.UNRESOLVED,
                    task_status=TaskStatus.TARGET_NOT_FOUND,
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
                grounding_status=GroundingStatus.UNRESOLVED,
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
                        target_confidence=vision_conf,
                        spatial_sector=matched_obj.spatial_sector,
                        proximity=matched_obj.proximity,
                        urgency=intent.urgency,
                        confidence=fused_conf * 0.7,
                        grounding_status=GroundingStatus.MODALITY_CONFLICT,
                        task_status=TaskStatus.MODALITY_CONFLICT,
                        matched_visual_object=matched_obj,
                        reasoning=(
                            f"Modality conflict: Voice specified sector '{intent.spatial_sector}', "
                            f"but target '{target_name}' was detected in sector '{matched_obj.spatial_sector.value}'."
                        ),
                        modality_contributions=modality_contrib,
                    )

            # High / Low confidence check
            grounding_st = (
                GroundingStatus.HIGH_CONFIDENCE
                if vision_conf >= self.high_confidence_threshold
                else GroundingStatus.LOW_CONFIDENCE
            )

            if fused_conf < self.min_confidence_threshold or vision_conf < self.min_confidence_threshold:
                return MultimodalTask(
                    action=action,
                    target_object=target_name,
                    target_confirmed=True,
                    target_confidence=vision_conf,
                    spatial_sector=matched_obj.spatial_sector,
                    proximity=matched_obj.proximity,
                    urgency=intent.urgency,
                    confidence=fused_conf,
                    grounding_status=grounding_st,
                    task_status=TaskStatus.LOW_CONFIDENCE,
                    matched_visual_object=matched_obj,
                    reasoning=f"Detection confidence for '{target_name}' ({vision_conf:.2f}) is below threshold.",
                    modality_contributions=modality_contrib,
                )

            return MultimodalTask(
                action=action,
                target_object=target_name,
                target_confirmed=True,
                target_confidence=vision_conf,
                spatial_sector=matched_obj.spatial_sector,
                proximity=matched_obj.proximity,
                urgency=intent.urgency,
                confidence=fused_conf,
                grounding_status=grounding_st,
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
            grounding_status=GroundingStatus.HIGH_CONFIDENCE,
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
            grounding_status=GroundingStatus.UNRESOLVED,
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
            grounding_status=GroundingStatus.HIGH_CONFIDENCE,
            task_status=TaskStatus.VALID,
            reasoning=f"Passive scene perception: {n_objects} objects detected across sectors.",
            modality_contributions={"vision": vision_output.confidence},
        )
