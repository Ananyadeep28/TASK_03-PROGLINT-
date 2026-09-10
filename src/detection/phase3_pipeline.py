"""
Phase 3: Integration & Visualization Pipeline
Authors: Person 3 (Integration & Visualization)
Collaborators: Person 1 (Detection & Pose), Person 2 (Finite State Machine)

This module integrates:
1. Phase 1 Detector (YOLO11 Object + Pose Detection)
2. Phase 2 Finite State Machine (5-State Bag Interaction Logic)
3. Heads-Up Display (HUD) visualization rendering
4. Output video generation (.mp4)
5. Structured JSON telemetry & event logging
"""

import os
import sys
import json
import time
import cv2
import numpy as np

# Ensure local imports work regardless of execution directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from phase1_detector import Phase1Detector
from phase2_fsm import BagActionFSM


class PipelineVisualizer:
    """Handles rendering of telemetry HUD, object markers, and visual cues."""

    STATE_COLORS = {
        "PLACED (INITIAL)": (255, 191, 0),     # Deep Sky Blue / Cyan (BGR)
        "PICKING": (0, 215, 255),              # Amber / Gold (BGR)
        "PICKED": (30, 30, 230),               # Vibrant Red (BGR)
        "PLACING": (204, 50, 153),             # Purple / Magenta (BGR)
        "PLACED (RETURNED)": (50, 205, 50),    # Lime / Emerald Green (BGR)
    }

    DEFAULT_COLOR = (200, 200, 200)

    @classmethod
    def get_state_color(cls, state: str):
        return cls.STATE_COLORS.get(state, cls.DEFAULT_COLOR)

    @classmethod
    def draw_hud(
        cls,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        fps: float,
        state: str,
        bag_data: dict,
        pose_data: dict,
        hand_dist: float,
        displacement: float,
        initial_pos: tuple,
        latest_event: str
    ) -> np.ndarray:
        """Draws a professional, semi-transparent Heads-Up Display on the frame."""
        h, w = frame.shape[:2]
        state_color = cls.get_state_color(state)
        current_time_sec = frame_idx / fps if fps > 0 else 0.0

        # -------------------------------------------------------------
        # 1. Spatial Graphics: Initial Anchor & Return Zone
        # -------------------------------------------------------------
        if initial_pos is not None:
            ix, iy = int(initial_pos[0]), int(initial_pos[1])
            # Draw anchor crosshair and circle
            cv2.circle(frame, (ix, iy), 12, (255, 200, 0), 2)
            cv2.circle(frame, (ix, iy), 3, (255, 200, 0), -1)
            cv2.line(frame, (ix - 18, iy), (ix + 18, iy), (255, 200, 0), 1)
            cv2.line(frame, (ix, iy - 18), (ix, iy + 18), (255, 200, 0), 1)
            cv2.putText(
                frame,
                "ANCHOR",
                (ix + 16, iy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 2. Spatial Graphics: Bag Bounding Box & Center
        # -------------------------------------------------------------
        if bag_data.get("bag_detected") and bag_data.get("bag_bbox") is not None:
            bx1, by1, bx2, by2 = bag_data["bag_bbox"]
            bcx, bcy = bag_data["bag_center"]
            conf = bag_data.get("bag_confidence", 0.0)

            # Draw colored bounding box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), state_color, 2)

            # Bag center dot
            cv2.circle(frame, (bcx, bcy), 6, state_color, -1)
            cv2.circle(frame, (bcx, bcy), 9, (255, 255, 255), 1)

            # Bag label
            label = f"Bag: {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (bx1, by1 - lh - 8), (bx1 + lw + 8, by1), state_color, -1)
            cv2.putText(
                frame,
                label,
                (bx1 + 4, by1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0) if state != "PICKED" else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Displacement line from anchor to current bag position
            if initial_pos is not None and displacement > 10:
                cv2.line(frame, (int(initial_pos[0]), int(initial_pos[1])), (bcx, bcy), (0, 165, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. Spatial Graphics: Wrists & Hand-to-Bag Interaction Line
        # -------------------------------------------------------------
        active_wrists = []
        for wrist_name in ["left_wrist", "right_wrist"]:
            wrist_pt = pose_data.get(wrist_name)
            conf_val = pose_data.get(f"{wrist_name}_confidence", 0.0)
            if wrist_pt is not None and conf_val >= 0.4:
                wx, wy = int(wrist_pt[0]), int(wrist_pt[1])
                active_wrists.append(((wx, wy), conf_val, wrist_name))
                # Draw wrist marker
                cv2.circle(frame, (wx, wy), 6, (0, 255, 255), -1)
                cv2.circle(frame, (wx, wy), 9, (0, 100, 255), 2)
                tag = "L-Wrist" if "left" in wrist_name else "R-Wrist"
                cv2.putText(
                    frame,
                    tag,
                    (wx + 10, wy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # If bag is detected and hand distance is calculated, draw interaction line to closest wrist
        if bag_data.get("bag_detected") and hand_dist is not None and active_wrists:
            bcx, bcy = bag_data["bag_center"]
            # Find closest wrist point
            closest_wrist = min(
                active_wrists,
                key=lambda w: (w[0][0] - bcx) ** 2 + (w[0][1] - bcy) ** 2
            )
            c_wx, c_wy = closest_wrist[0]

            # Dynamic interaction line
            cv2.line(frame, (bcx, bcy), (c_wx, c_wy), (0, 255, 0), 2, cv2.LINE_AA)

            # Midpoint distance label
            mx = int((bcx + c_wx) / 2)
            my = int((bcy + c_wy) / 2)
            dist_text = f"{hand_dist:.1f}px"
            (dtw, dth), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (mx - 4, my - dth - 4), (mx + dtw + 4, my + 4), (0, 0, 0), -1)
            cv2.putText(
                frame,
                dist_text,
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 4. Telemetry Card: Semi-Transparent Overlay Panel
        # -------------------------------------------------------------
        panel_x = 24
        panel_y = 24
        panel_w = 460
        panel_h = 240

        # Extract ROI for blending
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (20, 22, 28),
            -1,
        )
        # Alpha blend for frosted glass effect
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Accent border & state-colored header strip
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (70, 75, 85),
            1,
        )
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + 8, panel_y + panel_h),
            state_color,
            -1,
        )

        # Header Title
        cv2.putText(
            frame,
            "BAG ACTIVITY MONITOR",
            (panel_x + 22, panel_y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Time & Frame counter
        time_str = f"Time: {current_time_sec:05.2f}s | Frame: {frame_idx:04d}/{total_frames:04d}"
        cv2.putText(
            frame,
            time_str,
            (panel_x + 22, panel_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 185, 195),
            1,
            cv2.LINE_AA,
        )

        # State Badge
        badge_y = panel_y + 80
        badge_h = 32
        badge_w = panel_w - 44
        cv2.rectangle(
            frame,
            (panel_x + 22, badge_y),
            (panel_x + 22 + badge_w, badge_y + badge_h),
            state_color,
            -1,
        )
        text_color = (0, 0, 0) if state not in ["PICKED"] else (255, 255, 255)
        state_label = f"STATE: {state}"
        (slw, slh), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.putText(
            frame,
            state_label,
            (panel_x + 22 + (badge_w - slw) // 2, badge_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            text_color,
            2,
            cv2.LINE_AA,
        )

        # Telemetry stats
        dist_display = f"{hand_dist:.1f} px" if hand_dist is not None else "N/A"
        disp_display = f"{displacement:.1f} px" if displacement is not None else "0.0 px"
        anchor_display = f"({int(initial_pos[0])}, {int(initial_pos[1])})" if initial_pos is not None else "Calibrating..."

        metrics_y = panel_y + 140
        cv2.putText(
            frame,
            f"Hand-Bag Dist : {dist_display}",
            (panel_x + 22, metrics_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Displacement  : {disp_display}",
            (panel_x + 22, metrics_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Initial Anchor: {anchor_display}",
            (panel_x + 22, metrics_y + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (170, 175, 185),
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------
        # 5. Bottom Event Ticker Banner
        # -------------------------------------------------------------
        if latest_event:
            banner_h = 42
            banner_y = h - banner_h - 18
            banner_w = w - 48
            banner_x = 24

            b_overlay = frame.copy()
            cv2.rectangle(
                b_overlay,
                (banner_x, banner_y),
                (banner_x + banner_w, banner_y + banner_h),
                (15, 18, 24),
                -1,
            )
            cv2.addWeighted(b_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(
                frame,
                (banner_x, banner_y),
                (banner_x + banner_w, banner_y + banner_h),
                (60, 65, 75),
                1,
            )
            # Event icon / indicator
            cv2.circle(frame, (banner_x + 22, banner_y + 21), 6, state_color, -1)
            event_text = f"EVENT: {latest_event}"
            cv2.putText(
                frame,
                event_text,
                (banner_x + 38, banner_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (240, 245, 255),
                1,
                cv2.LINE_AA,
            )

        return frame


class Phase3Pipeline:
    """Complete video processing, state management, HUD rendering, and logging pipeline."""

    def __init__(
        self,
        touch_threshold: float = 350.0,
        displacement_threshold: float = 150.0,
        return_tolerance: float = 100.0,
    ):
        print("[Phase 3] Initializing Phase 1 Detector (YOLO11 Object + Pose)...")
        self.detector = Phase1Detector()

        print("[Phase 3] Initializing Phase 2 FSM (5-State Engine)...")
        self.fsm = BagActionFSM(
            touch_threshold=touch_threshold,
            displacement_threshold=displacement_threshold,
            return_tolerance=return_tolerance,
        )

        self.touch_threshold = touch_threshold
        self.displacement_threshold = displacement_threshold
        self.return_tolerance = return_tolerance

        # Internal execution metrics
        self.transition_log = []
        self.frame_telemetry = []

    def process_video(
        self,
        video_input_path: str,
        video_output_path: str = "output/annotated_output.mp4",
        json_log_path: str = "logs/event_log.json",
        show_preview: bool = False,
        progress_callback=None,
    ) -> dict:
        """Executes the full pipeline on an input video file."""
        if not os.path.exists(video_input_path):
            raise FileNotFoundError(f"Input video not found: {video_input_path}")

        # Ensure output directories exist
        out_dir = os.path.dirname(video_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        log_dir = os.path.dirname(json_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[Phase 3] Video Opened: {width}x{height} @ {fps:.2f} FPS ({total_frames} total frames)")

        # Prepare VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            # Fallback to avc1 / XVID if needed
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

        frame_idx = 0
        prev_state = self.fsm.state
        latest_event_str = "Initial resting state"
        start_time = time.time()

        initial_anchor = None
        anchor_samples = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Phase 1: Detection & Pose
                bag_data = self.detector.detect_bag(frame)
                pose_data = self.detector.detect_pose(frame)
                hand_dist = self.detector.calculate_hand_distance(bag_data, pose_data)

                # Stabilize initial anchor across first 20 stable frames while at rest
                if frame_idx < 30 and bag_data.get("bag_detected"):
                    anchor_samples.append(bag_data["bag_center"])
                    median_x = int(np.median([p[0] for p in anchor_samples]))
                    median_y = int(np.median([p[1] for p in anchor_samples]))
                    self.fsm.initial_bag_pos = (median_x, median_y)
                    initial_anchor = self.fsm.initial_bag_pos

                # Calculate displacement from anchor
                displacement = 0.0
                if self.fsm.initial_bag_pos is not None and bag_data.get("bag_center") is not None:
                    ax, ay = self.fsm.initial_bag_pos
                    cx, cy = bag_data["bag_center"]
                    displacement = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5

                # 2. Phase 2: State Machine Update
                current_state = self.fsm.update(bag_data.get("bag_center"), hand_dist)

                # Track transitions
                if current_state != prev_state:
                    timestamp_sec = round(frame_idx / fps, 3)
                    transition_desc = f"{prev_state} -> {current_state}"
                    event_msg = self.fsm.event_log[-1] if self.fsm.event_log else transition_desc
                    latest_event_str = event_msg

                    transition_record = {
                        "frame": frame_idx,
                        "timestamp_seconds": timestamp_sec,
                        "from_state": prev_state,
                        "to_state": current_state,
                        "event_description": event_msg,
                        "hand_distance_px": hand_dist,
                        "bag_displacement_px": round(displacement, 2),
                        "bag_center": bag_data.get("bag_center"),
                    }
                    self.transition_log.append(transition_record)
                    print(
                        f"  [*] Frame {frame_idx:04d} ({timestamp_sec:5.2f}s): "
                        f"{prev_state} ===> {current_state} | Dist: {hand_dist}px"
                    )
                    prev_state = current_state

                # 3. Phase 3: HUD Rendering
                annotated_frame = PipelineVisualizer.draw_hud(
                    frame=frame,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    fps=fps,
                    state=current_state,
                    bag_data=bag_data,
                    pose_data=pose_data,
                    hand_dist=hand_dist,
                    displacement=displacement,
                    initial_pos=self.fsm.initial_bag_pos,
                    latest_event=latest_event_str,
                )

                # 4. Write output frame
                writer.write(annotated_frame)

                # Optional live preview
                if show_preview:
                    display_frame = cv2.resize(annotated_frame, (960, 540))
                    cv2.imshow("Bag Tracking HUD", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[Phase 3] User interrupted playback with 'q'")
                        break

                frame_idx += 1
                if progress_callback and frame_idx % 25 == 0:
                    progress_callback(frame_idx, total_frames)

        finally:
            cap.release()
            writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        elapsed_time = time.time() - start_time
        proc_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0.0
        print(f"[Phase 3] Processing completed: {frame_idx} frames in {elapsed_time:.2f}s ({proc_fps:.1f} FPS)")

        # 5. Build & Save JSON Log
        final_summary = {
            "metadata": {
                "source_video": os.path.abspath(video_input_path),
                "output_video": os.path.abspath(video_output_path),
                "total_frames_processed": frame_idx,
                "video_fps": round(fps, 2),
                "video_resolution": f"{width}x{height}",
                "processing_time_seconds": round(elapsed_time, 2),
                "processing_fps": round(proc_fps, 2),
            },
            "parameters": {
                "touch_threshold_px": self.touch_threshold,
                "displacement_threshold_px": self.displacement_threshold,
                "return_tolerance_px": self.return_tolerance,
            },
            "fsm_results": {
                "initial_bag_position": list(self.fsm.initial_bag_pos) if self.fsm.initial_bag_pos else None,
                "final_bag_position": list(self.fsm.last_known_bag_pos) if self.fsm.last_known_bag_pos else None,
                "final_state": self.fsm.state,
                "total_transitions": len(self.transition_log),
                "all_events": self.fsm.event_log,
            },
            "state_transitions": self.transition_log,
        }

        with open(json_log_path, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=4)

        print(f"[Phase 3] Video saved to: {os.path.abspath(video_output_path)}")
        print(f"[Phase 3] JSON log saved to: {os.path.abspath(json_log_path)}")
        return final_summary