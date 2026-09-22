#!/usr/bin/env python3
"""
Run the Gesture Agent with HaGRID pretrained detector, lightweight, or mock backends.
Supports webcam video feed, single synthetic/image frame, and simulation override.
"""

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.gesture.agent import GestureAgent
from src.agents.gesture.hagrid_gesture_recognizer import HagridGestureRecognizer
from src.agents.gesture.recognizer import LightweightGestureRecognizer, MockGestureRecognizer
from src.agents.gesture.schemas import GestureAgentInput, GestureType
from src.common.logger import setup_logger
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run the HRI Gesture Agent.")
    parser.add_argument("--config", default="configs/gesture_config.yaml", help="Path to gesture_config.yaml")
    parser.add_argument("--backend", choices=["hagrid", "lightweight", "mock"], default=None, help="Gesture recognizer backend")
    parser.add_argument("--model-path", default=None, help="Path to HaGRID pretrained model (e.g. models/hagrid/yolov10n_hagrid.pt)")
    parser.add_argument("--device", default="cpu", help="Device for model execution ('cpu' or 'cuda')")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--webcam", action="store_true", help="Run real-time webcam gesture perception stream")
    parser.add_argument("--simulate", default=None, help="Simulate gesture name such as STOP, POINT_LEFT, THUMBS_UP")
    parser.add_argument("--image", default=None, help="Path to input image file")
    parser.add_argument("--frame-id", type=int, default=1)
    return parser.parse_args()


def draw_gesture_overlay(frame: np.ndarray, output) -> np.ndarray:
    """Draw bounding boxes, gesture labels, confidence, and grounded direction on frame."""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Draw vertical sector dividing lines for visual spatial reference
    cv2.line(annotated, (int(w * 0.33), 0), (int(w * 0.33), h), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(annotated, (int(w * 0.67), 0), (int(w * 0.67), h), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(annotated, "LEFT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(annotated, "CENTER", (int(w * 0.45), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(annotated, "RIGHT", (int(w * 0.88), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    for rec in output.recognized_gestures:
        if rec.bbox:
            x1, y1, x2, y2 = int(rec.bbox.x1), int(rec.bbox.y1), int(rec.bbox.x2), int(rec.bbox.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{rec.gesture} ({rec.confidence:.2f})"
            if rec.direction != "NONE" and rec.direction != "UNKNOWN":
                label += f" -> {rec.direction}"

            cv2.putText(annotated, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Header status banner
    status_text = f"Primary: {output.gesture} | Dir: {output.direction} | Conf: {output.confidence:.2f}"
    cv2.putText(annotated, status_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return annotated


def run_webcam_stream(agent: GestureAgent, camera_id: int = 0):
    """Run real-time video stream with OpenCV webcam."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        return

    print("Starting webcam gesture perception. Press 'q' or ESC to exit.")
    frame_id = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            start_t = time.perf_counter()
            agent_input = GestureAgentInput(frame=frame, frame_id=frame_id, source_name="webcam")
            output = agent.process(agent_input)
            latency_ms = (time.perf_counter() - start_t) * 1000.0

            curr_time = time.time()
            fps = 1.0 / max(1e-5, (curr_time - prev_time))
            prev_time = curr_time

            annotated = draw_gesture_overlay(frame, output)
            cv2.putText(
                annotated,
                f"Latency: {latency_ms:.1f}ms | FPS: {fps:.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            cv2.imshow("HRI Gesture Perception", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    logger = setup_logger("RunGesture")
    config = load_config(args.config) if (args.config and Path(args.config).exists()) else {}

    gesture_cfg = config.get("gesture", {})
    backend = args.backend or gesture_cfg.get("backend", "lightweight")
    model_path = args.model_path or gesture_cfg.get("model_path", "models/hagrid/yolov10n_hagrid.pt")
    conf_thresh = args.conf or gesture_cfg.get("confidence_threshold", 0.50)
    device = args.device or gesture_cfg.get("device", "cpu")

    logger.info(f"Initializing Gesture Agent with backend='{backend}'...")

    # Instantiate recognizer backend
    if backend == "mock":
        recognizer = MockGestureRecognizer(confidence_threshold=conf_thresh)
    elif backend == "lightweight":
        recognizer = LightweightGestureRecognizer(confidence_threshold=conf_thresh)
    else:  # hagrid
        recognizer = HagridGestureRecognizer(
            model_path=model_path,
            confidence_threshold=conf_thresh,
            device=device,
        )

    agent = GestureAgent(config=config, recognizer=recognizer)

    try:
        agent.initialize()
    except Exception as e:
        logger.warning(
            f"Failed to initialize '{backend}' recognizer ({e}). Falling back to LightweightGestureRecognizer."
        )
        recognizer = LightweightGestureRecognizer(confidence_threshold=conf_thresh)
        agent = GestureAgent(config=config, recognizer=recognizer)
        agent.initialize()

    if args.webcam:
        cam_id = config.get("camera", {}).get("device_id", 0)
        run_webcam_stream(agent, camera_id=cam_id)
        agent.shutdown()
        return

    # Single-shot execution mode
    if args.simulate:
        agent_input = GestureAgentInput(
            gesture_override=args.simulate,
            frame_id=args.frame_id,
            source_name="simulation",
        )
    elif args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error(f"Image not found: {img_path}")
            return
        frame = cv2.imread(str(img_path))
        agent_input = GestureAgentInput(frame=frame, frame_id=args.frame_id, source_name="image_file")
    else:
        # Synthetic test frame (480x640)
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        # Draw a synthetic hand circle in the right sector to demonstrate spatial grounding
        cv2.circle(frame, (520, 240), 50, (150, 150, 150), -1)
        agent_input = GestureAgentInput(frame=frame, frame_id=args.frame_id, source_name="synthetic")

    start_time = time.perf_counter()
    output = agent.process(agent_input)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print("\n" + "=" * 60)
    print("  GESTURE AGENT PERCEPTION OUTPUT")
    print("=" * 60)
    print(output.to_json(indent=2))
    print("-" * 60)
    logger.info(f"Summary : {output.summary_text}")
    logger.info(f"Latency : {elapsed_ms:.2f} ms")
    print("=" * 60 + "\n")

    agent.shutdown()


if __name__ == "__main__":
    main()