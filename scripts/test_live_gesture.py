#!/usr/bin/env python3
"""
Live Gesture and MediaPipe Diagnostic Script.
Captures live frames from the camera, runs MediaPipe HandLandmarker,
prints detailed 21-landmark data, finger extensions, pointing vectors,
tests gesture classification (POINT_LEFT, POINT_RIGHT, POINT_FORWARD, STOP),
and saves debug_gesture_frame.jpg.
"""

from pathlib import Path
import sys
import time
import math
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.gesture.recognizer import LightweightGestureRecognizer
from src.agents.gesture.schemas import GestureDirection, GestureType
from src.utils.camera import CameraManager


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks and connections on the image."""
    annotated_image = rgb_image.copy()
    if not detection_result or not detection_result.hand_landmarks:
        return annotated_image

    h, w, _ = annotated_image.shape

    # Hand landmark connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (0, 17)                                # Palm base
    ]

    for hand_landmarks in detection_result.hand_landmarks:
        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = hand_landmarks[start_idx]
            pt2 = hand_landmarks[end_idx]
            p1 = (int(pt1.x * w), int(pt1.y * h))
            p2 = (int(pt2.x * w), int(pt2.y * h))
            cv2.line(annotated_image, p1, p2, (0, 255, 0), 2)

        # Draw landmark points
        for idx, lm in enumerate(hand_landmarks):
            px, py = int(lm.x * w), int(lm.y * h)
            color = (255, 0, 0) if idx == 8 else (0, 0, 255)  # Highlight index tip in blue
            cv2.circle(annotated_image, (px, py), 4, color, -1)

    return annotated_image


def main():
    print("=" * 70)
    print("         HRI GESTURE & MEDIAPIPE LIVE CAMERA DIAGNOSTIC")
    print("=" * 70)

    # 1. Initialize Recognizer
    recognizer = LightweightGestureRecognizer(confidence_threshold=0.40)
    recognizer.load_model()

    # 2. Camera Capture
    cam = CameraManager(source=0, width=640, height=480, fps=30)
    if not cam.open():
        print("[CAMERA] Failed to open physical webcam device 0.")
        sys.exit(1)

    print("\n[CAMERA CAPTURE]")
    # Let auto-exposure settle
    for _ in range(15):
        cam.read_frame()
        time.sleep(0.05)

    # 3. Capture frames for testing gesture detection
    print("\n[STREAMING & EVALUATING MEDIAPIPE HAND DETECTION]")
    start_time = time.time()
    frames_checked = 0
    hands_detected_total = 0
    last_debug_print = 0

    saved_debug_frame = False

    # Check for up to 5 seconds of live camera feed
    while time.time() - start_time < 5.0:
        ret, frame = cam.read_frame()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        frames_checked += 1
        h, w, c = frame.shape

        # Verify BGR -> RGB conversion
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run recognition
        gestures = recognizer.recognize(frame=frame)

        # Direct access to landmarker result for detailed inspection
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = recognizer._landmarker.detect(mp_image) if recognizer._landmarker else None

        num_hands = len(detection_result.hand_landmarks) if (detection_result and detection_result.hand_landmarks) else 0

        if num_hands > 0:
            hands_detected_total += 1

            # Save first frame with hand detected
            if not saved_debug_frame:
                raw_debug_path = PROJECT_ROOT / "debug_gesture_frame_raw.jpg"
                cv2.imwrite(str(raw_debug_path), frame)
                print(f"  [SAVED] Raw unmodified gesture frame saved to : {raw_debug_path}")

                annotated = draw_landmarks_on_image(frame, detection_result)
                debug_path = PROJECT_ROOT / "debug_gesture_frame.jpg"
                cv2.imwrite(str(debug_path), annotated)
                print(f"  [SAVED] Annotated gesture frame saved to        : {debug_path}")
                saved_debug_frame = True

            # Print Landmark Debug Data ~once per second
            now = time.time()
            if now - last_debug_print >= 1.0:
                last_debug_print = now
                hand_lms = detection_result.hand_landmarks[0]
                points = [(float(pt.x), float(pt.y), float(pt.z) if hasattr(pt, 'z') else 0.0) for pt in hand_lms]

                wrist = points[0]
                thumb_mcp, thumb_ip, thumb_tip = points[2], points[3], points[4]
                index_mcp, index_pip, index_dip, index_tip = points[5], points[6], points[7], points[8]
                middle_mcp, middle_pip, middle_dip, middle_tip = points[9], points[10], points[11], points[12]
                ring_mcp, ring_pip, ring_dip, ring_tip = points[13], points[14], points[15], points[16]
                pinky_mcp, pinky_pip, pinky_dip, pinky_tip = points[17], points[18], points[19], points[20]

                print("\n" + "-" * 60)
                print("[LANDMARK DEBUG REPORT]")
                print(f"  Hands detected     : {num_hands}")
                print(f"  Landmarks count    : {len(points)}")
                print(f"  Wrist              : ({wrist[0]:.3f}, {wrist[1]:.3f})")
                print(f"  Index:")
                print(f"    MCP = ({index_mcp[0]:.3f}, {index_mcp[1]:.3f})")
                print(f"    PIP = ({index_pip[0]:.3f}, {index_pip[1]:.3f})")
                print(f"    DIP = ({index_dip[0]:.3f}, {index_dip[1]:.3f})")
                print(f"    TIP = ({index_tip[0]:.3f}, {index_tip[1]:.3f})")
                print(f"  Middle:")
                print(f"    MCP = ({middle_mcp[0]:.3f}, {middle_mcp[1]:.3f})")
                print(f"    PIP = ({middle_pip[0]:.3f}, {middle_pip[1]:.3f})")
                print(f"    DIP = ({middle_dip[0]:.3f}, {middle_dip[1]:.3f})")
                print(f"    TIP = ({middle_tip[0]:.3f}, {middle_tip[1]:.3f})")
                print(f"  Ring:")
                print(f"    MCP = ({ring_mcp[0]:.3f}, {ring_mcp[1]:.3f})")
                print(f"    PIP = ({ring_pip[0]:.3f}, {ring_pip[1]:.3f})")
                print(f"    DIP = ({ring_dip[0]:.3f}, {ring_dip[1]:.3f})")
                print(f"    TIP = ({ring_tip[0]:.3f}, {ring_tip[1]:.3f})")
                print(f"  Pinky:")
                print(f"    MCP = ({pinky_mcp[0]:.3f}, {pinky_mcp[1]:.3f})")
                print(f"    PIP = ({pinky_pip[0]:.3f}, {pinky_pip[1]:.3f})")
                print(f"    DIP = ({pinky_dip[0]:.3f}, {pinky_dip[1]:.3f})")
                print(f"    TIP = ({pinky_tip[0]:.3f}, {pinky_tip[1]:.3f})")

                # Finger extension calculation
                def check_ext(tip, pip, mcp):
                    d_tip = (tip[0] - mcp[0])**2 + (tip[1] - mcp[1])**2
                    d_pip = (pip[0] - mcp[0])**2 + (pip[1] - mcp[1])**2
                    return d_tip > d_pip * 1.05

                idx_ext = check_ext(index_tip, index_pip, index_mcp)
                mid_ext = check_ext(middle_tip, middle_pip, middle_mcp)
                rng_ext = check_ext(ring_tip, ring_pip, ring_mcp)
                pnk_ext = check_ext(pinky_tip, pinky_pip, pinky_mcp)

                print("\n  Finger extension:")
                print(f"    Index  = {'YES' if idx_ext else 'NO'}")
                print(f"    Middle = {'YES' if mid_ext else 'NO'}")
                print(f"    Ring   = {'YES' if rng_ext else 'NO'}")
                print(f"    Pinky  = {'YES' if pnk_ext else 'NO'}")

                # Pointing vector (Index TIP - Index MCP)
                dx = index_tip[0] - index_mcp[0]
                dy = index_tip[1] - index_mcp[1]
                angle_deg = math.degrees(math.atan2(dy, dx))

                print("\n  Pointing vector:")
                print(f"    dx              = {dx:+.3f}")
                print(f"    dy              = {dy:+.3f}")
                print(f"    Direction angle = {angle_deg:+.1f} deg")

                # Classification
                if gestures:
                    g = gestures[0]
                    print(f"\n  Classification Result: {g.gesture.value} (direction={g.direction.value}, conf={g.confidence:.2f})")
                else:
                    print("\n  Classification Result: NO_GESTURE / UNKNOWN")
                print("-" * 60)

        time.sleep(0.05)

    print(f"\n[SUMMARY OF LIVE GESTURE TEST]")
    print(f"  Frames processed  : {frames_checked}")
    print(f"  Hands detected    : {hands_detected_total} frame(s)")
    if hands_detected_total == 0:
        print("  Status: No hands detected in field of view during live window.")
    else:
        print("  Status: MediaPipe successfully detected hands and extracted 21 landmarks.")

    cam.release()

    # 4. Fixed Frame GestureAgent Verification
    raw_saved = PROJECT_ROOT / "debug_gesture_frame_raw.jpg"
    if raw_saved.exists():
        print("\n[FIXED SINGLE FRAME VERIFICATION WITH GESTUREAGENT]")
        from src.agents.gesture.agent import GestureAgent
        from src.agents.gesture.schemas import GestureAgentInput
        
        test_frame = cv2.imread(str(raw_saved))
        g_agent = GestureAgent(config={"gesture": {"backend": "lightweight", "confidence_threshold": 0.40}})
        g_agent.initialize()
        g_out = g_agent.process(GestureAgentInput(frame=test_frame, frame_id=999, source_name="fixed_file"))
        
        print(f"  Input frame shape   : {test_frame.shape}")
        print(f"  Gesture detected    : {g_out.is_gesture_detected}")
        print(f"  Primary gesture     : {g_out.gesture}")
        print(f"  Primary direction   : {g_out.direction}")
        print(f"  Confidence          : {g_out.confidence:.2f}")
        print(f"  Summary text        : {g_out.summary_text}")
        if g_out.recognized_gestures:
            top_g = g_out.recognized_gestures[0]
            print(f"  Recognized metadata : {top_g.metadata.get('debug', {})}")

    print("=" * 70)


if __name__ == "__main__":
    main()
