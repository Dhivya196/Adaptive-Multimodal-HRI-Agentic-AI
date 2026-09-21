#!/usr/bin/env python3
"""Standalone Demonstration runner for the Coordinator Agent in the Adaptive Multimodal HRI Framework."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.coordinator.agent import CoordinatorAgent
from src.agents.coordinator.schemas import (
    CoordinatorAgentInput,
    CoordinatorAgentOutput,
    TaskStatus,
)
from src.agents.vision.schemas import BoundingBox, DetectedObject, ProximityLevel, SpatialSector, VisionAgentOutput
from src.agents.voice.schemas import SpeechIntent, UrgencyLevel, VoiceAgentOutput
from src.utils.config import load_config


def print_divider(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title.upper()}")
        print("=" * 70)


def print_result(scenario_title: str, voice_text: str, vision_summary: str, output: CoordinatorAgentOutput):
    print_divider(scenario_title)
    print(f"🎙️  VOICE INPUT     : \"{voice_text}\"")
    print(f"👁️  VISION DETECTIONS: {vision_summary}")
    print("-" * 70)
    task = output.fused_task

    print(f"🧠  COORDINATOR OUTPUT (MULTIMODAL TASK):")
    if task:
        print(f"    • Action          : {task.action}")
        print(f"    • Target Object   : {task.target_object} (Confirmed: {task.target_confirmed})")
        print(f"    • Spatial Sector  : {task.spatial_sector.value if hasattr(task.spatial_sector, 'value') else task.spatial_sector}")
        print(f"    • Proximity Level : {task.proximity.value if hasattr(task.proximity, 'value') else task.proximity}")
        print(f"    • Task Status     : {task.task_status.value if hasattr(task.task_status, 'value') else task.task_status}")
        print(f"    • Confidence Score: {task.confidence:.2f}")
        print(f"    • Fusion Reasoning: {task.reasoning}")
    else:
        print("    [None]")


def main():
    print_divider("Adaptive Multimodal HRI Framework - Coordinator Agent Demo")
    print("Initializing Coordinator Agent and compiling LangGraph StateGraph...")

    config = load_config(PROJECT_ROOT / "configs" / "coordinator_config.yaml")
    coordinator = CoordinatorAgent(name="CoordinatorDemo", config=config)
    coordinator.initialize()

    # ------------------------------------------------------------------
    # SCENARIO 1: Voice + Vision Target Match (Go to the bottle)
    # ------------------------------------------------------------------
    voice_1 = VoiceAgentOutput(
        transcript="Go to the bottle",
        normalized_text="go to the bottle",
        is_speech_detected=True,
        confidence=0.91,
        speech_intent=SpeechIntent(
            action="navigate_to",
            target_object="bottle",
            spatial_sector="CENTER",
            urgency=UrgencyLevel.NORMAL,
            confidence=0.91,
        ),
    )
    vision_1 = VisionAgentOutput(
        confidence=0.87,
        detected_objects=[
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.87,
                bbox=BoundingBox(x1=250, y1=150, x2=390, y2=400),
                spatial_sector=SpatialSector.CENTER,
                proximity=ProximityLevel.MEDIUM,
            )
        ],
    )
    out_1 = coordinator.process(CoordinatorAgentInput(voice_output=voice_1, vision_output=vision_1))
    print_result("Scenario 1: Target Match (Navigate to Bottle)", voice_1.transcript, "bottle (CENTER, MEDIUM, conf=0.87)", out_1)

    # ------------------------------------------------------------------
    # SCENARIO 2: Voice + Vision Pick and Place (Pick up the phone)
    # ------------------------------------------------------------------
    voice_2 = VoiceAgentOutput(
        transcript="Pick up the phone",
        normalized_text="pick up the phone",
        is_speech_detected=True,
        confidence=0.95,
        speech_intent=SpeechIntent(
            action="pick_and_place",
            target_object="phone",
            spatial_sector="LEFT",
            urgency=UrgencyLevel.NORMAL,
            confidence=0.95,
        ),
    )
    vision_2 = VisionAgentOutput(
        confidence=0.92,
        detected_objects=[
            DetectedObject(
                object_id=2,
                label="phone",
                confidence=0.92,
                bbox=BoundingBox(x1=50, y1=100, x2=180, y2=260),
                spatial_sector=SpatialSector.LEFT,
                proximity=ProximityLevel.NEAR,
            )
        ],
    )
    out_2 = coordinator.process(CoordinatorAgentInput(voice_output=voice_2, vision_output=vision_2))
    print_result("Scenario 2: Pick and Place (Pick up Phone)", voice_2.transcript, "phone (LEFT, NEAR, conf=0.92)", out_2)

    # ------------------------------------------------------------------
    # SCENARIO 3: Target Not Found (Requested bottle, but only chair detected)
    # ------------------------------------------------------------------
    voice_3 = VoiceAgentOutput(
        transcript="Go to the bottle",
        normalized_text="go to the bottle",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(
            action="navigate_to",
            target_object="bottle",
            confidence=0.90,
        ),
    )
    vision_3 = VisionAgentOutput(
        confidence=0.85,
        detected_objects=[
            DetectedObject(
                object_id=3,
                label="chair",
                confidence=0.85,
                bbox=BoundingBox(x1=400, y1=100, x2=600, y2=450),
                spatial_sector=SpatialSector.RIGHT,
                proximity=ProximityLevel.FAR,
            )
        ],
    )
    out_3 = coordinator.process(CoordinatorAgentInput(voice_output=voice_3, vision_output=vision_3))
    print_result("Scenario 3: Target Not Found (Requested bottle, vision saw chair)", voice_3.transcript, "chair (RIGHT, FAR, conf=0.85)", out_3)

    # ------------------------------------------------------------------
    # SCENARIO 4: Stop the Robot Command
    # ------------------------------------------------------------------
    voice_4 = VoiceAgentOutput(
        transcript="Stop the robot",
        normalized_text="stop the robot",
        is_speech_detected=True,
        confidence=0.98,
        speech_intent=SpeechIntent(
            action="stop_robot",
            urgency=UrgencyLevel.EMERGENCY,
            confidence=0.98,
        ),
    )
    out_4 = coordinator.process(CoordinatorAgentInput(voice_output=voice_4, vision_output=vision_1))
    print_result("Scenario 4: Stop Robot Command (Emergency Priority)", voice_4.transcript, "bottle (CENTER)", out_4)

    # ------------------------------------------------------------------
    # SCENARIO 5: Modality Conflict (Voice sector LEFT vs Vision sector RIGHT)
    # ------------------------------------------------------------------
    voice_5 = VoiceAgentOutput(
        transcript="Go to the bottle on the left",
        normalized_text="go to the bottle on the left",
        is_speech_detected=True,
        confidence=0.90,
        speech_intent=SpeechIntent(
            action="navigate_to",
            target_object="bottle",
            spatial_sector="LEFT",  # Spoken LEFT
            confidence=0.90,
        ),
    )
    vision_5 = VisionAgentOutput(
        confidence=0.88,
        detected_objects=[
            DetectedObject(
                object_id=5,
                label="bottle",
                confidence=0.88,
                bbox=BoundingBox(x1=480, y1=120, x2=620, y2=420),
                spatial_sector=SpatialSector.RIGHT,  # Detected RIGHT
                proximity=ProximityLevel.MEDIUM,
            )
        ],
    )
    out_5 = coordinator.process(CoordinatorAgentInput(voice_output=voice_5, vision_output=vision_5))
    print_result("Scenario 5: Modality Conflict (Voice LEFT vs Vision RIGHT)", voice_5.transcript, "bottle (RIGHT, MEDIUM, conf=0.88)", out_5)

    print_divider("Demo completed successfully!")


if __name__ == "__main__":
    main()
