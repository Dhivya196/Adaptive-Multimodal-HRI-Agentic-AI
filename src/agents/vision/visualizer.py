"""Real-time frame annotation with bounding boxes, spatial sector dividers, and HUD overlays."""

from typing import List, Optional
import cv2
import numpy as np

from src.agents.vision.schemas import DetectedObject, VisionAgentOutput


class VisionVisualizer:
    """Renders structured visual perception overlays, HUD indicators, and bounding boxes."""

    COLOR_PRIMARY = (255, 178, 50)      # Electric Blue
    COLOR_ACCENT = (50, 220, 100)       # Vibrant Green
    COLOR_SECTOR_LINE = (80, 80, 80)    # Subdued Grey
    COLOR_BG_HUD = (25, 25, 25)         # Dark HUD background

    def __init__(
        self,
        window_name: str = "HRI Multimodal Perception - Vision Agent",
        show_spatial_sectors: bool = True,
        show_labels: bool = True,
        show_summary_banner: bool = True,
        sector_split: Optional[List[float]] = None,
    ):
        self.window_name = window_name
        self.show_spatial_sectors = show_spatial_sectors
        self.show_labels = show_labels
        self.show_summary_banner = show_summary_banner
        self.sector_split = sector_split or [0.33, 0.67]

    def _draw_spatial_sectors(self, frame: np.ndarray) -> None:
        """Draw vertical lines dividing the frame into LEFT, CENTER, and RIGHT sectors."""
        h, w = frame.shape[:2]
        x1 = int(w * self.sector_split[0])
        x2 = int(w * self.sector_split[1])

        cv2.line(frame, (x1, 0), (x1, h), self.COLOR_SECTOR_LINE, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, 0), (x2, h), self.COLOR_SECTOR_LINE, 1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1

        cv2.putText(frame, "LEFT SECTOR", (10, h - 15), font, scale, (160, 160, 160), thickness, cv2.LINE_AA)
        cv2.putText(frame, "CENTER SECTOR", (x1 + 10, h - 15), font, scale, (160, 160, 160), thickness, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT SECTOR", (x2 + 10, h - 15), font, scale, (160, 160, 160), thickness, cv2.LINE_AA)

    def _draw_detection(self, frame: np.ndarray, obj: DetectedObject) -> None:
        """Draw bounding box and label badge for an individual object."""
        x1, y1, x2, y2 = obj.bbox.to_xyxy_int()
        label_text = f"#{obj.object_id} {obj.label.upper()} ({obj.confidence:.2f})"
        sector_text = f"[{obj.spatial_sector.value} | {obj.proximity.value}]"

        color = self.COLOR_PRIMARY
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        cx, cy = obj.bbox.center.to_int_tuple()
        cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

        if self.show_labels:
            badge_text = f"{label_text} {sector_text}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            font_thick = 1
            (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, font_thick)

            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            badge_x2 = min(frame.shape[1], x1 + tw + 10)

            cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                frame,
                badge_text,
                (x1 + 5, y1 - 4),
                font,
                font_scale,
                (0, 0, 0),
                font_thick,
                cv2.LINE_AA,
            )

    def _draw_hud_banner(self, frame: np.ndarray, output: VisionAgentOutput) -> None:
        """Draw top HUD bar with system status and scene summary."""
        w = frame.shape[1]
        banner_h = 55

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), self.COLOR_BG_HUD, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.line(frame, (0, banner_h), (w, banner_h), (60, 60, 60), 1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = f"HRI VISION AGENT | Frame: {output.frame_id} | Latency: {output.execution_time_ms:.1f}ms | Objects: {len(output.detected_objects)}"
        cv2.putText(frame, title, (12, 20), font, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        summary = output.summary_text
        if len(summary) > 95:
            summary = summary[:92] + "..."
        cv2.putText(frame, summary, (12, 42), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    def render(self, frame: np.ndarray, output: VisionAgentOutput) -> np.ndarray:
        """Render all annotations onto a copy of the input frame."""
        annotated = frame.copy()

        if self.show_spatial_sectors:
            self._draw_spatial_sectors(annotated)

        for obj in output.detected_objects:
            self._draw_detection(annotated, obj)

        if self.show_summary_banner:
            self._draw_hud_banner(annotated, output)

        return annotated

    def show(self, frame: np.ndarray) -> int:
        """Display the frame in an OpenCV window and return key code."""
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        """Destroy display windows."""
        cv2.destroyAllWindows()
