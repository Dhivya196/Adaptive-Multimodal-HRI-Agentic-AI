#!/usr/bin/env python3
"""
Live Vision and YOLO Diagnostic Script.
Captures live frames from the camera, runs YOLO at multiple confidence thresholds,
saves debug_vision_frame.jpg, and verifies object detection and sector grounding.
"""

from pathlib import Path
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from src.agents.vision.agent import VisionAgent
from src.agents.vision.schemas import SpatialSector, VisionAgentInput
from src.utils.camera import CameraManager
from src.utils.config import load_config


def main():
    print("=" * 70)
    print("         HRI VISION & YOLO LIVE CAMERA DIAGNOSTIC")
    print("=" * 70)

    # 1. Model Inspection
    model_name = "yolo26n.pt"
    print(f"\n[YOLO MODEL INSPECTION]")
    print(f"  Model name        : {model_name}")
    try:
        model = YOLO(model_name)
        print(f"  Model type        : {type(model.model).__name__}")
        print(f"  Total classes     : {len(model.names)}")
        print(f"  Contains 'bottle' : {'bottle' in model.names.values()} (class_id: {[k for k,v in model.names.items() if v=='bottle']})")
        print(f"  Contains 'person' : {'person' in model.names.values()} (class_id: {[k for k,v in model.names.items() if v=='person']})")
        print(f"  Contains 'cup'    : {'cup' in model.names.values()} (class_id: {[k for k,v in model.names.items() if v=='cup']})")
    except Exception as e:
        print(f"  Model loading error: {e}")
        sys.exit(1)

    # 2. Camera Capture
    cam = CameraManager(source=0, width=640, height=480, fps=30)
    if not cam.open():
        print("[CAMERA] Failed to open physical webcam device 0.")
        sys.exit(1)

    print("\n[CAMERA CAPTURE]")
    # Let camera auto-exposure settle for 1 second
    for _ in range(15):
        cam.read_frame()
        time.sleep(0.05)

    ret, frame = cam.read_frame()
    if not ret or frame is None:
        print("[CAMERA] Error: Could not read frame from camera.")
        cam.release()
        sys.exit(1)

    h, w, c = frame.shape
    print(f"  Frame width       : {w}")
    print(f"  Frame height      : {h}")
    print(f"  Channels          : {c}")
    print(f"  dtype             : {frame.dtype}")
    print(f"  Min / Max pixel   : {frame.min()} / {frame.max()}")

    # 3. Save raw debug frame
    debug_path = PROJECT_ROOT / "debug_vision_frame.jpg"
    cv2.imwrite(str(debug_path), frame)
    print(f"  Saved debug frame : {debug_path}")

    # 4. Compare Raw YOLO Detections at Different Confidence Thresholds
    print("\n[RAW YOLO DETECTIONS ACROSS CONFIDENCE THRESHOLDS]")
    thresholds = [0.45, 0.25, 0.15, 0.10, 0.05]
    print(f"{'Threshold':<12} | {'Detected Objects (Class, Conf, BBox Center X, Sector)'}")
    print("-" * 70)

    for conf in thresholds:
        results = model.predict(source=frame, conf=conf, verbose=False)
        det_strs = []
        if results and results[0].boxes:
            for b in results[0].boxes:
                cls_id = int(b.cls[0].item())
                label = model.names.get(cls_id, str(cls_id))
                c_val = float(b.conf[0].item())
                xyxy = b.xyxy[0].tolist()
                cx = (xyxy[0] + xyxy[2]) / 2.0
                norm_x = cx / w
                sector = "LEFT" if norm_x < 0.33 else ("RIGHT" if norm_x > 0.67 else "CENTER")
                det_strs.append(f"{label} (conf={c_val:.2f}, cx={cx:.1f}, {sector})")

        obj_summary = ", ".join(det_strs) if det_strs else "None"
        print(f"conf = {conf:<6.2f} | {obj_summary}")

    # 5. Test Full VisionAgent Process
    print("\n[VISION AGENT OUTPUT WITH CONFIG]")
    cfg = load_config()
    v_agent = VisionAgent(config=cfg)
    v_agent.initialize()
    v_out = v_agent.process(VisionAgentInput(frame=frame, frame_id=1, source_name="camera"))

    print(f"  VisionAgent Success     : {v_out.success}")
    print(f"  Scene Confidence        : {v_out.confidence:.2f}")
    print(f"  Target Candidates       : {v_out.target_candidates}")
    print(f"  Detected Objects Count  : {len(v_out.detected_objects)}")
    for obj in v_out.detected_objects:
        print(f"    - {obj.label:<12} | conf={obj.confidence:.2f} | sector={obj.spatial_sector.value:<7} | proximity={obj.proximity.value:<7} | bbox=[{obj.bbox.x1:.1f}, {obj.bbox.y1:.1f}, {obj.bbox.x2:.1f}, {obj.bbox.y2:.1f}]")

    print("\n" + "=" * 70)
    cam.release()


if __name__ == "__main__":
    main()
