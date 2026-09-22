#!/usr/bin/env python3
"""
Real-Time End-to-End HRI Demonstration Runner for Adaptive Multimodal Human-Robot Interaction.

Orchestrates the complete multi-agent pipeline:
  Camera (VisionAgent) + Microphone (VoiceAgent)
    ↓
  Multimodal Synchronization & Latest Snapshot
    ↓
  CoordinatorAgent (LangGraph Fusion)
    ↓
  MemoryAgent (Context & Pronoun Grounding)
    ↓
  TaskPlannerAgent (Primitive Step Decomposition)
    ↓
  SafetyAgent (Deterministic Rules + Hysteresis + Risk Reasoning)
    ↓
  RobotControllerAgent (ROS2 / Mock Controller)
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.controller.agent import RobotControllerAgent
from src.agents.controller.controller import RobotController
from src.agents.controller.ros2_interface import MockRobotController, ROS2Interface
from src.agents.controller.schemas import (
    ControllerAgentInput,
    ControllerAgentOutput,
    ControllerCommand,
    ExecutionStatus,
)
from src.agents.coordinator.agent import CoordinatorAgent
from src.agents.coordinator.schemas import (
    CoordinatorAgentInput,
    CoordinatorAgentOutput,
    MultimodalTask,
    TaskStatus,
)
from src.agents.memory.agent import MemoryAgent
from src.agents.memory.schemas import (
    CoordinatorTaskInput,
    MemoryAgentInput,
    MemoryAgentOutput,
    MemoryEntryType,
    MemoryOperation,
)
from src.agents.planner.agent import TaskPlannerAgent
from src.agents.planner.schemas import PlanAction, TaskPlan, TaskPlannerInput, TaskPlannerOutput
from src.agents.safety.agent import SafetyAgent
from src.agents.safety.schemas import (
    RiskLevel,
    SafetyAgentInput,
    SafetyAgentOutput,
    SafetyDecisionType,
)
from src.agents.gesture.agent import GestureAgent
from src.agents.gesture.schemas import (
    GestureAgentInput,
    GestureAgentOutput,
    GestureDirection,
    GestureType,
)
from src.agents.vision.agent import VisionAgent
from src.agents.vision.schemas import (
    BoundingBox,
    DetectedObject,
    ProximityLevel,
    SpatialSector,
    VisionAgentInput,
    VisionAgentOutput,
)
from src.agents.voice.agent import VoiceAgent
from src.agents.voice.schemas import AudioFormat, SpeechIntent, UrgencyLevel, VoiceAgentInput, VoiceAgentOutput
from src.common.logger import setup_logger
from src.utils.audio_capture import MicrophoneRecorder
from src.utils.camera import CameraManager
from src.utils.config import load_config


class VisionWorker(threading.Thread):
    """Background worker continuously capturing camera frames and running VisionAgent."""

    def __init__(
        self,
        agent: VisionAgent,
        camera: CameraManager,
        target_fps: float = 8.0,
        enable_gui: bool = False,
    ):
        super().__init__(daemon=True, name="VisionWorker")
        self.agent = agent
        self.camera = camera
        self.target_fps = target_fps
        self.enable_gui = enable_gui
        self.running = False

        self._lock = threading.Lock()
        self._latest_output: Optional[VisionAgentOutput] = None
        self._latest_timestamp: float = 0.0
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id = 0

    def start_worker(self):
        self.running = True
        self.start()

    def stop_worker(self):
        self.running = False
        if self.enable_gui:
            cv2.destroyAllWindows()

    def get_latest_snapshot(self) -> Tuple[Optional[VisionAgentOutput], float]:
        """Thread-safely retrieve the latest VisionAgentOutput and its capture timestamp."""
        with self._lock:
            return self._latest_output, self._latest_timestamp

    def run(self):
        interval = 1.0 / max(1.0, self.target_fps)
        while self.running:
            start_t = time.time()
            ret, frame = self.camera.read_frame()
            if not ret or frame is None:
                if not getattr(self.camera, "_is_synthetic", False):
                    self.camera.enable_synthetic_mode()
                    ret, frame = self.camera.read_frame()

            if ret and frame is not None:
                self._frame_id += 1
                try:
                    inp = VisionAgentInput(frame=frame, frame_id=self._frame_id, source_name="camera")
                    output = self.agent.process(inp)
                    # If synthetic camera is active and no detections produced by YOLO on drawing, provide synthetic demonstration objects
                    if getattr(self.camera, "_is_synthetic", False) and len(output.detected_objects) == 0:
                        synthetic_bottle = DetectedObject(
                            object_id=1,
                            label="bottle",
                            confidence=0.88,
                            bbox=BoundingBox(50, 150, 180, 350),
                            spatial_sector=SpatialSector.LEFT,
                            proximity=ProximityLevel.NEAR,
                            area_ratio=0.18,
                        )
                        synthetic_phone = DetectedObject(
                            object_id=2,
                            label="phone",
                            confidence=0.84,
                            bbox=BoundingBox(450, 180, 560, 320),
                            spatial_sector=SpatialSector.RIGHT,
                            proximity=ProximityLevel.MEDIUM,
                            area_ratio=0.12,
                        )
                        output.detected_objects.extend([synthetic_bottle, synthetic_phone])
                        output.success = True
                        output.confidence = 0.88

                    now = time.time()
                    with self._lock:
                        self._latest_output = output
                        self._latest_timestamp = now
                        self._latest_frame = frame

                    if self.enable_gui:
                        self._render_gui(frame, output)
                except Exception as e:
                    import traceback
                    print(f"[VISION WORKER ERROR]: {e}\n{traceback.format_exc()}", flush=True)

            elapsed = time.time() - start_t
            sleep_t = max(0.005, interval - elapsed)
            time.sleep(sleep_t)

    def _render_gui(self, frame: np.ndarray, output: VisionAgentOutput):
        """Render visualization overlay on camera frame."""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        # Sector grid lines
        cv2.line(vis_frame, (int(w * 0.33), 0), (int(w * 0.33), h), (80, 80, 80), 1)
        cv2.line(vis_frame, (int(w * 0.67), 0), (int(w * 0.67), h), (80, 80, 80), 1)
        cv2.putText(vis_frame, "LEFT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        cv2.putText(vis_frame, "CENTER", (int(w * 0.45), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        cv2.putText(vis_frame, "RIGHT", (int(w * 0.88), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        for obj in output.detected_objects:
            if obj.bbox:
                x1, y1, x2, y2 = int(obj.bbox.x1), int(obj.bbox.y1), int(obj.bbox.x2), int(obj.bbox.y2)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
                label = f"{obj.label} ({obj.confidence:.2f}) [{obj.spatial_sector.value}, {obj.proximity.value}]"
                cv2.putText(vis_frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 2)

        status = f"VisionAgent: {len(output.detected_objects)} object(s) detected"
        cv2.putText(vis_frame, status, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("HRI Real-Time Vision Feed", vis_frame)
        cv2.waitKey(1)


class RealTimeHRIOrchestrator:
    """
    Main Real-Time Orchestrator managing multimodal streams, synchronization,
    and agent pipeline execution.
    """

    # Mapping TaskPlanner primitive actions to RobotController commands
    ACTION_TO_CONTROLLER_CMD = {
        "NAVIGATE_TO": ControllerCommand.MOVE_FORWARD.value,
        "NAVIGATE": ControllerCommand.MOVE_FORWARD.value,
        "APPROACH_OBJECT": ControllerCommand.MOVE_FORWARD.value,
        "MOVE": ControllerCommand.MOVE_FORWARD.value,
        "MOVE_FORWARD": ControllerCommand.MOVE_FORWARD.value,
        "MOVE_BACKWARD": ControllerCommand.MOVE_BACKWARD.value,
        "TURN_LEFT": ControllerCommand.TURN_LEFT.value,
        "TURN_RIGHT": ControllerCommand.TURN_RIGHT.value,
        "STOP": ControllerCommand.STOP.value,
        "ALIGN_BASE": ControllerCommand.TURN_LEFT.value,
        "ALIGN_GRIPPER": ControllerCommand.PICK.value,
        "GRASP": ControllerCommand.PICK.value,
        "PICK": ControllerCommand.PICK.value,
        "LIFT": ControllerCommand.PLACE.value,
        "PLACE": ControllerCommand.PLACE.value,
        "RETRACT": ControllerCommand.STOP.value,
        "INSPECT_OBJECT": ControllerCommand.INSPECT.value,
        "INSPECT": ControllerCommand.INSPECT.value,
    }

    # Actions considered physical manipulation (discrete / simulated without robotic arm)
    MANIPULATION_ACTIONS = {"PICK", "PLACE", "ALIGN_GRIPPER", "GRASP", "LIFT"}

    def __init__(
        self,
        mode: str = "simulation",
        cmd_vel_topic: str = "/cmd_vel",
        max_vision_age: float = 3.0,
        safety_demo_distance: Optional[float] = None,
        enable_gui: bool = False,
        config_path: Optional[str] = None,
    ):
        self.mode = mode
        self.cmd_vel_topic = cmd_vel_topic
        self.max_vision_age = max_vision_age
        self.safety_demo_distance = safety_demo_distance
        self.enable_gui = enable_gui

        self.logger = setup_logger("RealTimeHRI")

        # Load configs
        self.config = load_config(config_path) if config_path else {}
        self._init_agents()

    def _init_agents(self):
        """Initialize all 7 HRI agents cleanly."""
        self.logger.info("=" * 60)
        self.logger.info("Initializing HRI Multi-Agent Subsystems...")
        self.logger.info("=" * 60)

        # 1. Vision Agent
        self.vision_agent = VisionAgent(config=self.config)
        self.vision_agent.initialize()

        # 2. Voice Agent
        self.voice_agent = VoiceAgent(config=self.config)
        self.voice_agent.initialize()

        # 3. Gesture Agent
        gesture_cfg = self.config.copy()
        if "gesture" not in gesture_cfg:
            gesture_cfg["gesture"] = {}
        
        backend_choice = gesture_cfg["gesture"].get("backend", "lightweight")
        if backend_choice == "hagrid":
            model_path = Path(gesture_cfg["gesture"].get("model_path", "models/hagrid/yolov10n_hagrid.pt"))
            if not model_path.exists():
                self.logger.info("HaGRID checkpoint not present. Defaulting live gesture demo to MediaPipe/Lightweight backend.")
                gesture_cfg["gesture"]["backend"] = "lightweight"
        else:
            gesture_cfg["gesture"]["backend"] = backend_choice

        self.gesture_agent = GestureAgent(config=gesture_cfg)
        self.gesture_agent.initialize()

        # 4. Coordinator Agent
        self.coordinator_agent = CoordinatorAgent(config=self.config)
        self.coordinator_agent.initialize()

        # 4. Memory Agent
        self.memory_agent = MemoryAgent(config=self.config)
        self.memory_agent.initialize()

        # 5. Task Planner Agent
        self.planner_agent = TaskPlannerAgent(config=self.config)
        self.planner_agent.initialize()

        # 6. Safety Agent
        self.safety_agent = SafetyAgent(config=self.config)
        self.safety_agent.initialize()

        # 7. Robot Controller Agent
        ctrl_config = self.config.copy()
        if "controller" not in ctrl_config:
            ctrl_config["controller"] = {}

        if self.mode == "simulation":
            ctrl_config["controller"]["mode"] = "mock"
            backend = MockRobotController()
        else:  # full or ros2
            ctrl_config["controller"]["mode"] = "ros2"
            ctrl_config["controller"]["cmd_vel_topic"] = self.cmd_vel_topic
            backend = ROS2Interface(topic=self.cmd_vel_topic)
            if not backend.is_connected():
                self.logger.warning(
                    f"ROS2 backend could not connect on '{self.cmd_vel_topic}'. Falling back to Mock controller."
                )
                backend = MockRobotController()

        controller = RobotController(config=ctrl_config)
        controller.backend = backend
        self.controller_agent = RobotControllerAgent(config=ctrl_config, controller=controller)
        self.controller_agent.initialize()

        # Audio capture helper
        self.mic_recorder = MicrophoneRecorder()

        # Camera manager & background vision worker
        cam_cfg = self.config.get("camera", {})
        self.camera_manager = CameraManager(
            source=cam_cfg.get("device_id", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
        )

        opened = self.camera_manager.open()
        if opened:
            test_ret, test_frame = self.camera_manager.read_frame()
            if not test_ret or test_frame is None:
                opened = False

        if not opened:
            self.logger.warning("Physical webcam unavailable or busy. Enabling synthetic test camera stream.")
            self.camera_manager.enable_synthetic_mode()

        self.vision_worker = VisionWorker(
            agent=self.vision_agent,
            camera=self.camera_manager,
            target_fps=8.0,
            enable_gui=self.enable_gui,
        )
        self.vision_worker.start_worker()

        # Wait briefly for initial snapshot capture and detector warmup
        warmup_start = time.time()
        while time.time() - warmup_start < 6.0:
            snap, t_snap = self.vision_worker.get_latest_snapshot()
            if snap is not None and t_snap > 0.0:
                break
            time.sleep(0.1)

        self.logger.info("All HRI agents initialized and ready.")

    def shutdown(self):
        """Cleanly shutdown all background workers and agents."""
        self.logger.info("Shutting down RealTime HRI Orchestrator...")
        self.vision_worker.stop_worker()
        self.camera_manager.release()
        self.vision_agent.shutdown()
        self.gesture_agent.shutdown()
        self.voice_agent.shutdown()
        self.coordinator_agent.shutdown()
        self.memory_agent.shutdown()
        self.planner_agent.shutdown()
        self.safety_agent.shutdown()
        self.controller_agent.shutdown()
        self.logger.info("Shutdown complete.")

    def _sync_vision_snapshot(self, wait_timeout: float = 1.0) -> Tuple[Optional[VisionAgentOutput], float, bool]:
        """
        Retrieve the latest Vision snapshot and verify freshness against max_vision_age.
        Returns: (snapshot, age_seconds, is_fresh)
        """
        start_wait = time.time()
        while time.time() - start_wait < wait_timeout:
            snapshot, capture_time = self.vision_worker.get_latest_snapshot()
            if snapshot is not None and capture_time > 0.0:
                break
            time.sleep(0.05)

        snapshot, capture_time = self.vision_worker.get_latest_snapshot()
        if snapshot is None or capture_time == 0.0:
            return None, 0.0, False

        age = time.time() - capture_time
        is_fresh = age <= self.max_vision_age
        return snapshot, age, is_fresh

    def execute_command_pipeline(
        self,
        voice_output: VoiceAgentOutput,
        manual_vision: Optional[VisionAgentOutput] = None,
        manual_gesture: Optional[GestureAgentOutput] = None,
        safety_override_dist: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full chronological multi-agent pipeline for a single command.
        """
        timings = {}
        total_start = time.perf_counter()

        print("\n" + "=" * 70)
        print("         REAL-TIME MULTIMODAL HRI DEMONSTRATION PIPELINE")
        print("=" * 70)

        # -------------------------------------------------------------
        # 1. VOICE AGENT OUTPUT
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [VOICE AGENT]")
        print("-" * 70)
        transcript = voice_output.transcript or "(None)"
        intent_action = voice_output.speech_intent.action if voice_output.speech_intent else "unknown"
        intent_target = voice_output.speech_intent.target_object if voice_output.speech_intent else "None"
        intent_urgency = voice_output.speech_intent.urgency.value if voice_output.speech_intent else "NORMAL"
        voice_conf = voice_output.confidence

        print(f"  • Transcript : \"{transcript}\"")
        print(f"  • Intent     : {intent_action}")
        print(f"  • Target     : {intent_target}")
        print(f"  • Urgency    : {intent_urgency}")
        print(f"  • Confidence : {voice_conf:.2f}")

        # -------------------------------------------------------------
        # 2. GESTURE AGENT OUTPUT
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [GESTURE AGENT]")
        print("-" * 70)
        gesture_output: Optional[GestureAgentOutput] = None
        if manual_gesture is not None:
            gesture_output = manual_gesture
            source_str = "CLI SIMULATION"
        else:
            rec_type = type(getattr(self.gesture_agent, "recognizer", None)).__name__
            source_str = "HaGRID" if "Hagrid" in rec_type else ("MediaPipe/Lightweight" if "Lightweight" in rec_type else "GestureAgent")
            with self.vision_worker._lock:
                cur_frame = self.vision_worker._latest_frame
                cur_frame_id = self.vision_worker._frame_id
            if cur_frame is not None:
                try:
                    g_inp = GestureAgentInput(frame=cur_frame, frame_id=cur_frame_id, source_name="camera")
                    gesture_output = self.gesture_agent.process(g_inp)
                except Exception as e:
                    self.logger.debug(f"Live gesture inference skipped: {e}")
                    gesture_output = None

        if gesture_output and gesture_output.confidence > 0.0:
            g_name = gesture_output.gesture
            g_dir = gesture_output.direction or "NONE"
            print(f"  • Source     : {source_str}")
            print(f"  • Gesture    : {g_name}")
            print(f"  • Direction  : {g_dir}")
            print(f"  • Confidence : {gesture_output.confidence:.2f}")
        else:
            print(f"  • Source     : {source_str}")
            print("  • Gesture    : NONE / NOT DETECTED")
            print("  • Confidence : 0.00")

        # -------------------------------------------------------------
        # 3. VISION SNAPSHOT SYNCHRONIZATION
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [VISION AGENT SNAPSHOT]")
        print("-" * 70)

        if manual_vision is not None:
            vision_output = manual_vision
            vision_age = 0.05
            is_fresh = True
        else:
            vision_output, vision_age, is_fresh = self._sync_vision_snapshot()

        if vision_output and is_fresh:
            detected_names = [f"{o.label} ({o.spatial_sector.value}, {o.proximity.value})" for o in vision_output.detected_objects]
            detected_str = ", ".join(detected_names) if detected_names else "None"
            print(f"  • Snapshot Status : FRESH (Age: {vision_age:.2f}s, Threshold: {self.max_vision_age}s)")
            print(f"  • Detected Objects: {detected_str}")
            print(f"  • Scene Confidence: {vision_output.confidence:.2f}")
        else:
            age_str = f"{vision_age:.2f}s" if (vision_output and vision_age > 0) else "N/A"
            print(f"  • Snapshot Status : STALE / UNAVAILABLE (Age: {age_str} > {self.max_vision_age}s)")
            print("  • Warning         : No fresh visual target confirmation available.")
            vision_output = VisionAgentOutput(
                agent_name="VisionAgent",
                agent_type="vision_agent",
                success=False,
                confidence=0.0,
                detected_objects=[],
            )

        # Fail safely on stale/unavailable vision when target-specific action is required
        if not is_fresh and intent_action != "stop_robot":
            print("\n" + "=" * 70)
            print("  [TASK RESULT]: BLOCKED DUE TO STALE VISION")
            print("  Reason: Visual target confirmation is required but visual snapshot is stale/unavailable.")
            print("=" * 70 + "\n")
            return {
                "success": False,
                "stage": "VISION_SYNC",
                "reason": "STALE_VISION",
                "timings": timings,
            }

        # -------------------------------------------------------------
        # 4. COORDINATOR MULTIMODAL FUSION
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [COORDINATOR AGENT - MULTIMODAL FUSION]")
        print("-" * 70)

        t0 = time.perf_counter()
        coord_input = CoordinatorAgentInput(
            session_id="realtime_session",
            voice_output=voice_output,
            vision_output=vision_output,
            gesture_output=gesture_output,
        )
        coord_output: CoordinatorAgentOutput = self.coordinator_agent.process(coord_input)
        timings["coordinator_ms"] = (time.perf_counter() - t0) * 1000.0

        fused_task = coord_output.fused_task
        task_status = coord_output.task_status
        coord_conf = coord_output.confidence

        is_pronoun_target = (
            intent_target is not None
            and intent_target.strip().lower() in MemoryAgent.REFERENTIAL_PRONOUNS
        )

        if fused_task:
            sec_val = (
                fused_task.spatial_sector.value
                if isinstance(fused_task.spatial_sector, SpatialSector)
                else (fused_task.spatial_sector or "UNKNOWN")
            )
            prox_val = (
                fused_task.proximity.value
                if isinstance(fused_task.proximity, ProximityLevel)
                else (fused_task.proximity or "UNKNOWN")
            )
            print(f"  • Interpreted Task : {fused_task.action} -> {fused_task.target_object}")
            print(f"  • Fusion Status    : {task_status.value if hasattr(task_status, 'value') else task_status}")
            print(f"  • Target Confirmed : {fused_task.target_confirmed}")
            print(f"  • Spatial Sector   : {sec_val}")
            print(f"  • Proximity        : {prox_val}")
            print(f"  • Unified Conf     : {coord_conf:.2f}")
            print(f"  • Fusion Reasoning : {fused_task.reasoning}")
        else:
            sec_val = "UNKNOWN"
            prox_val = "UNKNOWN"
            print(f"  • Fusion Status    : {task_status}")
            print(f"  • Reason           : {coord_output.error_message or 'Failed to produce task.'}")

        # If coordinator blocked and target is NOT an unresolved pronoun, reject early
        if (not coord_output.success or not fused_task) and not is_pronoun_target:
            print("\n" + "=" * 70)
            print("  [TASK RESULT]: BLOCKED AT COORDINATION")
            print(f"  Reason: Task is {task_status}. Cannot safely proceed without valid grounded intent.")
            print("=" * 70 + "\n")
            return {
                "success": False,
                "stage": "COORDINATOR",
                "reason": str(task_status),
                "timings": timings,
            }

        if task_status != TaskStatus.VALID and not is_pronoun_target:
            print("\n" + "=" * 70)
            print("  [TASK RESULT]: BLOCKED AT COORDINATION")
            print(f"  Reason: Task is {task_status}. Cannot safely proceed without valid grounded intent.")
            print("=" * 70 + "\n")
            return {
                "success": False,
                "stage": "COORDINATOR",
                "reason": str(task_status),
                "timings": timings,
            }

        # -------------------------------------------------------------
        # 4. CONTEXT MEMORY & PRONOUN GROUNDING
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [MEMORY AGENT - CONTEXT GROUNDING]")
        print("-" * 70)

        t0 = time.perf_counter()
        assigned_task_id = f"task_{uuid.uuid4().hex[:8]}"
        coord_task_dict = {
            "task_id": assigned_task_id,
            "action": fused_task.action if fused_task else intent_action,
            "target": fused_task.target_object if fused_task else intent_target,
            "location": sec_val,
            "task_status": "VALID",
            "confidence": fused_task.confidence if fused_task else voice_conf,
            "raw_command": transcript,
        }

        # Ground task against memory history
        mem_input = MemoryAgentInput(
            operation=MemoryOperation.RESOLVE_CONTEXT,
            coordinator_output=coord_task_dict,
        )
        mem_output: MemoryAgentOutput = self.memory_agent.process(mem_input)
        timings["memory_ms"] = (time.perf_counter() - t0) * 1000.0

        resolved_dict = mem_output.resolved_context
        resolved_target = resolved_dict.get("target") or (fused_task.target_object if fused_task else intent_target)
        was_resolved = resolved_dict.get("resolved_from_previous", False)
        original_target = resolved_dict.get("original_target", (fused_task.target_object if fused_task else intent_target))
        stored_task_id = resolved_dict.get("task_id", assigned_task_id)

        # If pronoun was resolved to a real object, re-ground with latest vision snapshot if possible
        if was_resolved and resolved_target and vision_output and vision_output.detected_objects:
            for obj in vision_output.detected_objects:
                if resolved_target.lower() in obj.label.lower() or obj.label.lower() in resolved_target.lower():
                    sec_val = obj.spatial_sector.value if isinstance(obj.spatial_sector, SpatialSector) else obj.spatial_sector
                    prox_val = obj.proximity.value if isinstance(obj.proximity, ProximityLevel) else obj.proximity
                    break

        print(f"  • Stored Task ID   : {stored_task_id}")
        print(f"  • Original Target  : {original_target}")
        print(f"  • Resolved Target  : {resolved_target}")
        if was_resolved:
            print(f"  • Pronoun Grounding: True ('{original_target}' -> '{resolved_target}')")
        else:
            print("  • Pronoun Grounding: Not needed (Explicit target provided)")

        # Verify we only block if an explicit pronoun was requested but remained unresolved
        if is_pronoun_target and (not resolved_target or resolved_target.strip().lower() in MemoryAgent.REFERENTIAL_PRONOUNS):
            print("\n" + "=" * 70)
            print("  [TASK RESULT]: BLOCKED AT MEMORY GROUNDING")
            print(f"  Reason: Target pronoun '{original_target}' could not be resolved from historical context.")
            print("=" * 70 + "\n")
            return {
                "success": False,
                "stage": "MEMORY",
                "reason": "UNRESOLVED_PRONOUN",
                "timings": timings,
            }

        # For generic directional motion without specific object, set target to 'forward' for planner decomposition
        if not resolved_target and (fused_task.action if fused_task else intent_action) in ("navigate_to", "move_forward", "move"):
            resolved_target = "forward"

        # Construct fully grounded task for planning and safety verification
        matched_obj = None
        if vision_output and vision_output.detected_objects:
            for obj in vision_output.detected_objects:
                if resolved_target.lower() in obj.label.lower() or obj.label.lower() in resolved_target.lower():
                    matched_obj = obj
                    break

        sec_enum = (
            matched_obj.spatial_sector
            if matched_obj
            else (fused_task.spatial_sector if fused_task else SpatialSector.CENTER)
        )
        prox_enum = (
            matched_obj.proximity
            if matched_obj
            else (fused_task.proximity if fused_task else ProximityLevel.MEDIUM)
        )
        target_conf = matched_obj.confidence if matched_obj else voice_conf
        unified_conf = (0.55 * voice_conf) + (0.35 * target_conf) if matched_obj else voice_conf

        eval_task = MultimodalTask(
            action=fused_task.action if fused_task else intent_action,
            target_object=resolved_target,
            target_confirmed=(matched_obj is not None),
            spatial_sector=sec_enum,
            proximity=prox_enum,
            urgency=fused_task.urgency if fused_task else UrgencyLevel.NORMAL,
            confidence=unified_conf,
            task_status=TaskStatus.VALID,
            matched_visual_object=matched_obj,
            reasoning=(
                f"Pronoun '{original_target}' grounded to '{resolved_target}' from conversation history."
                if was_resolved
                else (fused_task.reasoning if fused_task else "Grounded task.")
            ),
        )

        # -------------------------------------------------------------
        # 5. TASK PLANNER DECOMPOSITION
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [TASK PLANNER AGENT]")
        print("-" * 70)

        t0 = time.perf_counter()
        planner_task_dict = {
            "task_status": "VALID",
            "action": eval_task.action,
            "target_object": resolved_target,
            "spatial_sector": eval_task.spatial_sector.value if isinstance(eval_task.spatial_sector, SpatialSector) else eval_task.spatial_sector,
            "proximity": eval_task.proximity.value if isinstance(eval_task.proximity, ProximityLevel) else eval_task.proximity,
            "confidence": eval_task.confidence,
        }
        plan_input = TaskPlannerInput(task_input=planner_task_dict)
        plan_output: TaskPlannerOutput = self.planner_agent.process(plan_input)
        timings["planner_ms"] = (time.perf_counter() - t0) * 1000.0

        if not plan_output.success or not plan_output.plan:
            print(f"  • Plan Generation : FAILED ({plan_output.error_message})")
            print("\n" + "=" * 70)
            print("  [TASK RESULT]: BLOCKED AT TASK PLANNING")
            print("=" * 70 + "\n")
            return {
                "success": False,
                "stage": "PLANNER",
                "reason": plan_output.error_message,
                "timings": timings,
            }

        plan: TaskPlan = plan_output.plan
        print(f"  • Plan ID          : {plan.plan_id}")
        print(f"  • Total Steps      : {len(plan.steps)}")
        print(f"  • Est. Duration    : {plan.total_estimated_duration:.1f} s")
        print("  • Planned Steps:")
        for idx, step in enumerate(plan.steps, start=1):
            act_name = step.action.value if hasattr(step.action, "value") else str(step.action)
            print(f"      {idx}. {act_name:<16} (Target: {step.target or resolved_target or 'None'})")

        # -------------------------------------------------------------
        # 6. STEP-BY-STEP SAFETY EVALUATION & CONTROLLER EXECUTION
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [SAFETY GATED EXECUTION PIPELINE]")
        print("-" * 70)

        effective_obstacle_dist = (
            safety_override_dist
            if safety_override_dist is not None
            else self.safety_demo_distance
        )

        all_steps_executed = True
        step_execution_logs = []

        for idx, step in enumerate(plan.steps, start=1):
            act_name = step.action.value if hasattr(step.action, "value") else str(step.action)
            step_tgt = step.target or resolved_target or "None"
            print(f"\n  >>> STEP [{idx}/{len(plan.steps)}]: {act_name} (Target: {step_tgt})")

            # --- Safety Agent Check ---
            t0 = time.perf_counter()
            safety_input = SafetyAgentInput(
                session_id="realtime_session",
                task=eval_task,
                planner_output={"step": step.to_dict(), "action": act_name, "target": step_tgt},
                obstacle_distance=effective_obstacle_dist,
                human_distance=None,
                target_confidence=eval_task.confidence,
                emergency_stop=False,
            )
            safety_output: SafetyAgentOutput = self.safety_agent.process(safety_input)
            timings[f"safety_step_{idx}_ms"] = (time.perf_counter() - t0) * 1000.0

            decision = safety_output.safety_decision
            is_approved = safety_output.approved_for_execution
            decision_type = (
                decision.decision.value
                if (decision and hasattr(decision.decision, "value"))
                else (decision.decision if decision else "UNSAFE")
            )
            risk_level = (
                decision.risk_level.value
                if (decision and hasattr(decision.risk_level, "value"))
                else (decision.risk_level if decision else "HIGH")
            )
            reason_str = decision.reason if decision else "No decision generated"

            obs_source_str = "DEMO/SIMULATED" if effective_obstacle_dist is not None else "None"
            obs_dist_str = f"{effective_obstacle_dist:.2f} m" if effective_obstacle_dist is not None else "Clear (>1.5m)"

            print(f"      [SAFETY] Obstacle Source   : {obs_source_str}")
            print(f"      [SAFETY] Obstacle Distance : {obs_dist_str}")
            print(f"      [SAFETY] Risk Level        : {risk_level}")
            print(f"      [SAFETY] Decision          : {decision_type} (Approved: {is_approved})")
            print(f"      [SAFETY] Reason            : {reason_str}")

            if not is_approved or decision_type != SafetyDecisionType.SAFE.value:
                print(f"      [SAFETY BLOCK] Step {idx} '{act_name}' was REJECTED by Safety Agent.")
                print(f"      [ROBOT CONTROLLER] Commanding EMERGENCY STOP to preserve safe state.")

                # Publish Emergency Stop to hardware
                self.controller_agent.process(
                    ControllerAgentInput(command=ControllerCommand.STOP.value, is_safety_approved=True)
                )
                all_steps_executed = False
                step_execution_logs.append({"step": idx, "action": act_name, "status": "SAFETY_REJECTED"})
                break

            # --- Robot Controller Execution ---
            # Translate planner action to controller command
            ctrl_cmd = self.ACTION_TO_CONTROLLER_CMD.get(act_name, ControllerCommand.MOVE_FORWARD.value)
            is_manipulation = act_name in self.MANIPULATION_ACTIONS

            t0 = time.perf_counter()
            ctrl_input = ControllerAgentInput(
                command=ctrl_cmd,
                parameters=step.parameters or {},
                is_safety_approved=True,
            )
            ctrl_output: ControllerAgentOutput = self.controller_agent.process(ctrl_input)
            timings[f"controller_step_{idx}_ms"] = (time.perf_counter() - t0) * 1000.0

            backend_name = "ROS2" if self.mode in ("ros2", "full") and isinstance(self.controller_agent.controller.backend, ROS2Interface) else "MOCK"

            if is_manipulation:
                print(f"      [ROBOT] Action   : {act_name} -> {step_tgt}")
                print(f"      [ROBOT] Status   : SIMULATED / NOT IMPLEMENTED (No physical robotic arm attached)")
            else:
                print(f"      [ROBOT] Command  : {ctrl_cmd}")
                print(f"      [ROBOT] Backend  : {backend_name} ({self.cmd_vel_topic})")
                print(f"      [ROBOT] Status   : {ctrl_output.execution_status} (Success: {ctrl_output.success})")

            step_execution_logs.append({"step": idx, "action": act_name, "status": ctrl_output.execution_status})

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        # -------------------------------------------------------------
        # 7. PERFORMANCE & LATENCY TIMINGS
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print("  [PERFORMANCE & LATENCY PROFILING]")
        print("-" * 70)
        print(f"  • Coordinator Fusion Latency : {timings.get('coordinator_ms', 0):.2f} ms")
        print(f"  • Memory Grounding Latency   : {timings.get('memory_ms', 0):.2f} ms")
        print(f"  • Task Planner Latency       : {timings.get('planner_ms', 0):.2f} ms")
        print(f"  • Total End-to-End Pipeline  : {total_elapsed_ms:.2f} ms")

        # -------------------------------------------------------------
        # 8. TASK RESULT BANNER
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        if all_steps_executed:
            print("  [TASK RESULT]: COMPLETED SUCCESSFULLY")
            print(f"  Action '{fused_task.action}' on target '{resolved_target}' executed with all safety gates verified.")
        else:
            print("  [TASK RESULT]: BLOCKED BY SAFETY GUARD")
            print("  Execution was safely terminated to prevent proximity or risk violation.")
        print("=" * 70 + "\n")

        return {
            "success": all_steps_executed,
            "fused_task": fused_task,
            "resolved_target": resolved_target,
            "step_logs": step_execution_logs,
            "timings": timings,
            "total_ms": total_elapsed_ms,
        }

    def run_single_text_command(
        self,
        text_command: str,
        gesture_str: Optional[str] = None,
        vision_objects_str: Optional[str] = None,
        safety_dist: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process a text command directly through VoiceAgent simulation with optional gesture and visual objects."""
        self.logger.info(f"Processing text command: '{text_command}' (Gesture: {gesture_str}, Vision Objects: {vision_objects_str})")
        voice_inp = VoiceAgentInput(
            text_override=text_command,
            audio_format=AudioFormat.TEXT_SIMULATION,
        )
        voice_out = self.voice_agent.process(voice_inp)

        manual_g = None
        if gesture_str:
            g_clean = gesture_str.strip().upper()
            dir_val = "LEFT" if "LEFT" in g_clean else ("RIGHT" if "RIGHT" in g_clean else "FORWARD")
            manual_g = GestureAgentOutput(
                agent_name="GestureAgent",
                agent_type="gesture_agent",
                success=True,
                gesture=g_clean,
                direction=dir_val,
                confidence=0.91,
                is_gesture_detected=True,
            )

        manual_v = None
        if vision_objects_str:
            objs = []
            for idx, item in enumerate(vision_objects_str.split(","), start=1):
                if ":" in item:
                    label, sec_name = item.split(":", 1)
                    sec_clean = sec_name.strip().upper()
                    sec_enum = (
                        SpatialSector.LEFT if sec_clean == "LEFT"
                        else (SpatialSector.RIGHT if sec_clean == "RIGHT" else SpatialSector.CENTER)
                    )
                    objs.append(
                        DetectedObject(
                            object_id=idx,
                            label=label.strip(),
                            confidence=0.88,
                            bbox=BoundingBox(50 * idx, 100, 150 * idx, 300),
                            spatial_sector=sec_enum,
                            proximity=ProximityLevel.NEAR,
                        )
                    )
            manual_v = VisionAgentOutput(
                agent_name="VisionAgent",
                agent_type="vision_agent",
                success=True,
                confidence=0.90,
                detected_objects=objs,
            )

        return self.execute_command_pipeline(
            voice_out,
            manual_vision=manual_v,
            manual_gesture=manual_g,
            safety_override_dist=safety_dist,
        )

    def run_interactive_loop(self):
        """Interactive live demonstration loop listening for voice or keyboard input."""
        print("\n" + "=" * 70)
        print("  HRI LIVE DEMONSTRATION READY")
        print("  Say a command into the microphone (e.g. 'Pick up the bottle', 'Stop').")
        print("  Type 'exit' or press Ctrl+C to quit.")
        print("=" * 70 + "\n")

        has_mic = self.mic_recorder.is_available()
        if not has_mic:
            print("[AUDIO] Physical microphone unavailable.")
            print("[AUDIO] Interactive CLI text input mode active.\n")

        while True:
            try:
                if has_mic:
                    print("[AUDIO] Listening on microphone (speak now)...")
                    audio_buf = self.mic_recorder.record_phrase(
                        on_speech_start=lambda: print("[AUDIO] Speech activity detected! Recording..."),
                        timeout=10.0,
                    )
                    if audio_buf is not None and len(audio_buf) > 0:
                        print("[AUDIO] Processing captured speech audio via VoiceAgent Whisper...")
                        voice_inp = VoiceAgentInput(
                            audio_data=audio_buf,
                            audio_format=AudioFormat.NUMPY_FLOAT32,
                            sample_rate=16000,
                        )
                        voice_out = self.voice_agent.process(voice_inp)
                        if voice_out.transcript:
                            self.execute_command_pipeline(voice_out)
                        else:
                            print("[VOICE] No speech transcribed in audio chunk.\n")
                    else:
                        # If silence timeout, prompt text fallback
                        cmd = input("[TEXT INPUT] Enter command (or press Enter to listen again, 'exit' to quit): ").strip()
                        if cmd.lower() in ("exit", "quit", "q"):
                            break
                        if cmd:
                            self.run_single_text_command(cmd)
                else:
                    cmd = input("HRI Command > ").strip()
                    if cmd.lower() in ("exit", "quit", "q"):
                        break
                    if cmd:
                        self.run_single_text_command(cmd)

            except KeyboardInterrupt:
                print("\nReceived exit signal.")
                break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Multimodal HRI Real-Time End-to-End Multi-Agent Demonstration"
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "full", "ros2"],
        default="simulation",
        help="Demonstration mode: 'simulation' (Mock controller), 'full' (Live sensors + ROS2), 'ros2' (ROS2 backend)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Execute a single text command without requiring live microphone (e.g. --text 'pick it up')",
    )
    parser.add_argument(
        "--gesture",
        default=None,
        help="Inject gesture for demonstration (e.g. POINT_LEFT, POINT_RIGHT, POINT_FORWARD, STOP)",
    )
    parser.add_argument(
        "--vision-objects",
        default=None,
        help="Inject visual objects for demonstration (e.g. 'bottle:LEFT,phone:RIGHT' or 'bottle:CENTER')",
    )
    parser.add_argument(
        "--safety-demo",
        type=float,
        default=None,
        help="Inject a simulated obstacle distance in meters for safety verification (e.g. --safety-demo 0.20)",
    )
    parser.add_argument(
        "--cmd-vel-topic",
        default="/cmd_vel",
        help="ROS2 velocity command topic (default: '/cmd_vel' or '/model/vehicle/cmd_vel')",
    )
    parser.add_argument(
        "--max-vision-age",
        type=float,
        default=3.0,
        help="Maximum vision snapshot age in seconds before marking stale (default: 3.0s)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Enable OpenCV visual camera window overlay",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=False,
        help="Disable OpenCV visual camera window overlay",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to custom config YAML",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    enable_gui = args.gui and not args.no_gui

    orchestrator = RealTimeHRIOrchestrator(
        mode=args.mode,
        cmd_vel_topic=args.cmd_vel_topic,
        max_vision_age=args.max_vision_age,
        safety_demo_distance=args.safety_demo,
        enable_gui=enable_gui,
        config_path=args.config,
    )

    try:
        if args.text:
            orchestrator.run_single_text_command(
                args.text,
                gesture_str=args.gesture,
                vision_objects_str=args.vision_objects,
                safety_dist=args.safety_demo,
            )
        else:
            orchestrator.run_interactive_loop()
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
