"""Webcam & video capture handler with auto-reconnect and synthetic test feed."""

from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np
import time

from src.common.exceptions import CameraStreamError
from src.common.logger import get_logger


class CameraManager:
    """Manages OpenCV video capture devices, video files, or synthetic frame feeds."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.logger = get_logger("CameraManager")
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self._is_synthetic = False

    def open(self) -> bool:
        """Open the video stream or device."""
        self.logger.info(f"Opening video source: {self.source}...")
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                self.logger.warning(f"Could not open hardware camera at source '{self.source}'.")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Validate that device can actually produce frames (handles busy / dummy devices)
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.logger.warning(f"Camera opened but failed to capture test frame at source '{self.source}'.")
                self.cap.release()
                self.cap = None
                return False

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(f"Camera opened successfully ({actual_w}x{actual_h} @ {self.fps}fps).")
            return True
        except Exception as e:
            self.logger.warning(f"Error opening camera source '{self.source}': {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def enable_synthetic_mode(self) -> None:
        """Enable synthetic test frame generation (for test environments without webcam)."""
        self._is_synthetic = True
        self.logger.info("Enabled synthetic frame generator mode.")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the active stream or synthetic generator."""
        self.frame_count += 1

        if self._is_synthetic:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)

            # Simulated table
            cv2.rectangle(frame, (50, 320), (590, 460), (70, 70, 70), -1)

            # Simulated bottle on right
            cv2.rectangle(frame, (420, 200), (490, 340), (220, 180, 50), -1)
            cv2.putText(frame, "BOTTLE", (425, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Simulated person in center
            cv2.circle(frame, (220, 180), 45, (180, 140, 220), -1)
            cv2.putText(frame, "PERSON", (195, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            cv2.putText(
                frame,
                f"SYNTHETIC STREAM [Frame #{self.frame_count}]",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 120),
                1,
            )
            time.sleep(1.0 / max(1, self.fps))
            return True, frame

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        return True, frame

    def stream(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator that yields (frame_id, frame) continually."""
        while True:
            ret, frame = self.read_frame()
            if not ret or frame is None:
                break
            yield self.frame_count, frame

    def release(self) -> None:
        """Release camera capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.logger.info("Released camera capture device.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
