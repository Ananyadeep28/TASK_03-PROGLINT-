"""
Main Project Runner: YOLO11 Bag Interaction Surveillance Pipeline
Integrates Person 1 (Detector), Person 2 (FSM), and Person 3 (HUD & Video/JSON Pipeline)
"""

import argparse
import sys
import os
import time

from src.detection.phase3_pipeline import Phase3Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO11 Bag Interaction & Surveillance Tracking Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=r"C:\Users\MOHAMMAD SAJID\Downloads\VID_20260910_154155778.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/annotated_output.mp4",
        help="Path to save annotated output video (.mp4)",
    )
    parser.add_argument(
        "--log",
        "-l",
        type=str,
        default="logs/event_log.json",
        help="Path to save event log telemetry JSON",
    )
    parser.add_argument(
        "--touch-dist",
        type=float,
        default=350.0,
        help="Hand-to-bag distance threshold (px) to trigger PICKING",
    )
    parser.add_argument(
        "--disp-dist",
        type=float,
        default=150.0,
        help="Bag displacement threshold (px) to trigger PICKED",
    )
    parser.add_argument(
        "--return-tol",
        type=float,
        default=100.0,
        help="Displacement tolerance (px) for returning bag to initial anchor (PLACING)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live preview window while processing",
    )
    return parser.parse_args()


def print_banner():
    banner = r"""
======================================================================
  YOLO11 Bag Interaction Surveillance System
  Person 1 (Detector) | Person 2 (FSM) | Person 3 (HUD & Pipeline)
======================================================================
  FSM States:
    [1] PLACED (INITIAL) -> Hand touches bag (< touch_dist)
    [2] PICKING          -> Bag moves away (> disp_dist)
    [3] PICKED           -> Bag returns to anchor (<= return_tol)
    [4] PLACING          -> Hand releases (> touch_dist)
    [5] PLACED (RETURNED)
======================================================================
"""
    print(banner)


def progress_tracker(current_frame, total_frames):
    percent = int((current_frame / total_frames) * 100) if total_frames > 0 else 0
    bar_length = 30
    filled = int(bar_length * current_frame / total_frames) if total_frames > 0 else 0
    bar = "=" * filled + ">" + " " * (bar_length - filled - 1) if filled < bar_length else "=" * bar_length
    print(f"\rProgress: [{bar}] {percent:3d}% ({current_frame}/{total_frames} frames)", end="", flush=True)


def main():
    args = parse_args()
    print_banner()

    print(f"[Config] Input Video : {args.input}")
    print(f"[Config] Output Video: {args.output}")
    print(f"[Config] Event Log   : {args.log}")
    print(f"[Config] Thresholds  : touch={args.touch_dist}px, disp={args.disp_dist}px, return_tol={args.return_tol}px\n")

    if not os.path.exists(args.input):
        print(f"[Error] Specified input video does not exist: {args.input}")
        sys.exit(1)

    pipeline = Phase3Pipeline(
        touch_threshold=args.touch_dist,
        displacement_threshold=args.disp_dist,
        return_tolerance=args.return_tol,
    )

    print("\n[Execution] Starting video processing...")
    results = pipeline.process_video(
        video_input_path=args.input,
        video_output_path=args.output,
        json_log_path=args.log,
        show_preview=args.show,
        progress_callback=progress_tracker,
    )

    print("\n\n======================================================================")
    print("  Processing Successfully Complete!")
    print("======================================================================")
    print(f"Total Frames Processed : {results['metadata']['total_frames_processed']}")
    print(f"Total Processing Time   : {results['metadata']['processing_time_seconds']} s")
    print(f"Average Processing FPS  : {results['metadata']['processing_fps']} FPS")
    print(f"Final State Reached     : {results['fsm_results']['final_state']}")
    print(f"Total State Transitions : {results['fsm_results']['total_transitions']}")
    print("\nEvent Log History:")
    for evt in results['fsm_results']['all_events']:
        print(f"  * {evt}")
    print(f"\nAnnotated Video Output: {results['metadata']['output_video']}")
    print(f"Telemetry JSON Log    : {os.path.abspath(args.log)}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
