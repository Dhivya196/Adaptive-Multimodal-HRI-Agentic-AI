#!/usr/bin/env python3
"""
CLI entrypoint to run live vision perception pipeline.
Supports live camera feed, video files, synthetic feeds, and static images.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.agents.vision.agent import VisionAgent
from src.agents.vision.detector import MockObjectDetector, YOLOObjectDetector
from src.agents.vision.schemas import BoundingBox, RawDetection, VisionAgentInput, VisionAgentOutput
from src.agents.vision.visualizer import VisionVisualizer
from src.common.logger import setup_logger
from src.pipeline.vision_pipeline import VisionPipeline
from src.utils.camera import CameraManager
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Vision Perception Agent for Adaptive Multimodal HRI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to custom vision_config.yaml")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--video-path", type=str, default=None, help="Path to video file input")
    parser.add_argument("--image-path", type=str, default=None, help="Path to single image input")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic test frame feed")
    parser.add_argument("--mock-detector", action="store_true", help="Use Mock detector (for testing or CI)")
    parser.add_argument("--conf-thresh", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV display window")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process before exiting")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("RunVision")

    config = load_config(args.config)
    if args.conf_thresh is not None:
        config["vision"]["confidence_threshold"] = args.conf_thresh
    if args.headless:
        config["visualizer"]["enable_display"] = False

    logger.info("Initializing Vision Agent perception pipeline...")

    if args.mock_detector:
        logger.info("Using MockObjectDetector with simulated detections.")
        mock_detections = [
            RawDetection(label="bottle", confidence=0.89, bbox=BoundingBox(x1=430, y1=210, x2=490, y2=350)),
            RawDetection(label="person", confidence=0.92, bbox=BoundingBox(x1=160, y1=120, x2=280, y2=410)),
            RawDetection(label="dining table", confidence=0.78, bbox=BoundingBox(x1=50, y1=320, x2=590, y2=460)),
        ]
        detector = MockObjectDetector(
            mock_detections=mock_detections,
            confidence_threshold=config["vision"]["confidence_threshold"],
        )
    else:
        vision_cfg = config["vision"]
        detector = YOLOObjectDetector(
            model_name=vision_cfg.get("model_name", "yolo26n.pt"),
            device=vision_cfg.get("device", "auto"),
            confidence_threshold=vision_cfg.get("confidence_threshold", 0.45),
            iou_threshold=vision_cfg.get("iou_threshold", 0.45),
            target_classes=vision_cfg.get("target_classes"),
        )

    agent = VisionAgent(config=config, detector=detector)

    # Mode 1: Single image inference
    if args.image_path:
        if not os.path.exists(args.image_path):
            logger.error(f"Image path does not exist: {args.image_path}")
            sys.exit(1)

        frame = cv2.imread(args.image_path)
        if frame is None:
            logger.error(f"Failed to read image at: {args.image_path}")
            sys.exit(1)

        agent.initialize()
        output: VisionAgentOutput = agent.process(
            VisionAgentInput(frame=frame, frame_id=1, source_name=args.image_path)
        )

        logger.info("--- Vision Agent Perception Result ---")
        print(output.to_json(indent=2))

        if not args.headless:
            visualizer = VisionVisualizer()
            annotated = visualizer.render(frame, output)
            cv2.imshow("Vision Agent - Image Result", annotated)
            logger.info("Displaying result window. Press any key to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        agent.shutdown()
        return

    # Mode 2: Continuous video stream
    if args.video_path:
        camera = CameraManager(source=args.video_path)
    elif args.synthetic:
        camera = CameraManager(source=0)
        camera.enable_synthetic_mode()
    else:
        camera = CameraManager(source=args.camera_id)

    def on_vision_output(output: VisionAgentOutput):
        if output.detected_objects:
            logger.info(f"[Frame #{output.frame_id}] {output.summary_text}")

    pipeline = VisionPipeline(
        config=config,
        agent=agent,
        camera=camera,
        output_callback=on_vision_output,
    )

    pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
