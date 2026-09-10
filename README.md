# YOLO11 Bag Interaction & Surveillance Tracking System

An end-to-end computer vision and behavioral analytics pipeline that tracks bag interactions using **YOLO11 object detection**, **YOLO11-pose estimation**, a **5-State Finite State Machine (FSM)**, and a real-time **Heads-Up Display (HUD)** with structured telemetry logging.

---

## 👥 3-Person Team Architecture

| Role | Module | Responsibility | Core Libraries |
| :--- | :--- | :--- | :--- |
| **Person 1** | [`src/detection/phase1_detector.py`](src/detection/phase1_detector.py) | **Computer Vision**: YOLO11 bag detection, YOLO11-pose wrist keypoints, and Euclidean distance computation. | `ultralytics`, `numpy`, `cv2` |
| **Person 2** | [`src/detection/phase2_fsm.py`](src/detection/phase2_fsm.py) | **Decision Logic**: 5-state finite state machine tracking bag status transitions and displacement. | `math` |
| **Person 3** | [`src/detection/phase3_pipeline.py`](src/detection/phase3_pipeline.py)<br>[`run_project.py`](run_project.py) | **Integration & Visualization**: Video stream capture, HUD rendering, video encoding, JSON telemetry logging, and CLI runner. | `cv2`, `json`, `argparse` |

---

## 🔄 System Pipeline

```text
                     INPUT VIDEO
                          │
                          ▼
               ┌─────────────────────┐
               │      PERSON 1       │
               │ phase1_detector.py  │
               │                     │
               │ YOLO11 Detection    │
               │ YOLO11 Pose         │
               └──────────┬──────────┘
                          │
            bag_center + wrist keypoints
            + minimum hand-bag distance
                          │
                          ▼
               ┌─────────────────────┐
               │      PERSON 2       │
               │ phase2_fsm.py       │
               │                     │
               │ 5-State FSM Engine  │
               └──────────┬──────────┘
                          │
                   current state
                   + displacement
                          │
                          ▼
               ┌─────────────────────┐
               │      PERSON 3       │
               │ phase3_pipeline.py  │
               │ run_project.py      │
               │                     │
               │ HUD Overlay + Loop  │
               └──────────┬──────────┘
                          │
                 ┌────────┴─────────┐
                 ▼                  ▼
        annotated_output.mp4   event_log.json
```

---

## ⚙️ 5-State Bag Action Logic

```text
         ┌─────────────────────────┐
         │     PLACED (INITIAL)    │
         └────────────┬────────────┘
                      │
               hand touches bag
               (dist <= touch_threshold)
                      │
                      ▼
         ┌─────────────────────────┐
         │         PICKING         │
         └────────────┬────────────┘
                      │
               bag moves away
               (displacement > disp_threshold)
                      │
                      ▼
         ┌─────────────────────────┐
         │         PICKED          │
         └────────────┬────────────┘
                      │
            bag returns to anchor
            (displacement <= return_tolerance)
            + hand touches bag
                      │
                      ▼
         ┌─────────────────────────┐
         │         PLACING         │
         └────────────┬────────────┘
                      │
               hand releases bag
               (dist > touch_threshold or None)
                      │
                      ▼
         ┌─────────────────────────┐
         │    PLACED (RETURNED)    │
         └─────────────────────────┘
```

---

## 📂 Project Structure

```text
TASK_03-PROGLINT-/
├── run_project.py               # Master CLI entrypoint (Person 3)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── src/
│   ├── __init__.py
│   └── detection/
│       ├── __init__.py
│       ├── phase1_detector.py   # Person 1: YOLO11 Object & Pose Detector
│       ├── phase2_fsm.py        # Person 2: 5-State Action FSM
│       ├── phase3_pipeline.py   # Person 3: HUD Visualizer & Pipeline
│       ├── test_bag.py          # Person 1 unit test
│       └── test_fsm.py          # Person 2 unit test
│
├── output/                      # Generated annotated videos (.mp4)
│   └── annotated_output.mp4
│
└── logs/                        # Generated telemetry logs (.json)
    └── event_log.json
```

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ananyadeep28/TASK_03-PROGLINT-.git
   cd TASK_03-PROGLINT-
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

*(Weights `yolo11s.pt` and `yolo11n-pose.pt` will be downloaded automatically by Ultralytics on first run.)*

---

## 💻 Running the Pipeline

### Basic Execution
To process a video with default calibrated parameters:
```bash
python run_project.py --input "path/to/video.mp4"
```

### Advanced Execution with Custom Thresholds
```bash
python run_project.py \
    --input "path/to/video.mp4" \
    --output "output/annotated_output.mp4" \
    --log "logs/event_log.json" \
    --touch-dist 350.0 \
    --disp-dist 150.0 \
    --return-tol 100.0 \
    --show
```

### Command-Line Arguments

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--input`, `-i` | `C:\...\VID_*.mp4` | Path to the input video file. |
| `--output`, `-o` | `output/annotated_output.mp4` | Destination path for the annotated video. |
| `--log`, `-l` | `logs/event_log.json` | Destination path for the telemetry JSON log. |
| `--touch-dist` | `350.0` | Euclidean distance threshold (px) between wrist and bag to trigger `PICKING`. |
| `--disp-dist` | `150.0` | Bag displacement threshold (px) from initial anchor to trigger `PICKED`. |
| `--return-tol` | `100.0` | Displacement tolerance (px) for returning bag to initial anchor (`PLACING`). |
| `--show` | `False` | Opens a live OpenCV window showing real-time playback with HUD. |

---

## 📊 Heads-Up Display (HUD) Features

The HUD generated in `output/annotated_output.mp4` includes:
- **Telemetry Card**: Frosted glassmorphism dark overlay showing video time, frame count, real-time hand-bag distance, displacement from anchor, and initial anchor coordinates.
- **State Badges**: Dynamic color-coded status badges:
  - 🔵 **Cyan**: `PLACED (INITIAL)`
  - 🟡 **Amber**: `PICKING`
  - 🔴 **Vibrant Red**: `PICKED`
  - 🟣 **Magenta**: `PLACING`
  - 🟢 **Emerald Green**: `PLACED (RETURNED)`
- **Spatial Markers**:
  - Initial anchor crosshair (`ANCHOR`) indicating the origin position.
  - Bounding box with confidence score.
  - Wrists (`L-Wrist`, `R-Wrist`) with confidence validation.
  - Dynamic line connecting closest wrist to bag center with real-time pixel distance label.
- **Event Banner**: Bottom ticker displaying real-time FSM state transition events.

---

## 📝 Structured JSON Output Example

Each run generates a detailed event log (`logs/event_log.json`):

```json
{
    "metadata": {
        "source_video": "VID_20260910_154155778.mp4",
        "output_video": "output/annotated_output.mp4",
        "total_frames_processed": 1195,
        "video_fps": 59.93,
        "video_resolution": "1920x1080",
        "processing_time_seconds": 95.51,
        "processing_fps": 12.51
    },
    "parameters": {
        "touch_threshold_px": 350.0,
        "displacement_threshold_px": 150.0,
        "return_tolerance_px": 100.0
    },
    "fsm_results": {
        "initial_bag_position": [852, 531],
        "final_bag_position": [817, 513],
        "final_state": "PLACED (RETURNED)",
        "total_transitions": 4,
        "all_events": [
            "Hand reached bag (Picking)",
            "Bag lifted/moved from initial spot (Picked)",
            "Bag returned to original area (Placing)",
            "Hand released; bag placed back at rest (Placed)"
        ]
    },
    "state_transitions": [
        {
            "frame": 172,
            "timestamp_seconds": 2.87,
            "from_state": "PLACED (INITIAL)",
            "to_state": "PICKING",
            "event_description": "Hand reached bag (Picking)",
            "hand_distance_px": 322.76,
            "bag_displacement_px": 25.81,
            "bag_center": [831, 516]
        },
        {
            "frame": 278,
            "timestamp_seconds": 4.639,
            "from_state": "PICKING",
            "to_state": "PICKED",
            "event_description": "Bag lifted/moved from initial spot (Picked)",
            "hand_distance_px": 299.28,
            "bag_displacement_px": 155.54,
            "bag_center": [824, 378]
        },
        {
            "frame": 878,
            "timestamp_seconds": 14.652,
            "from_state": "PICKED",
            "to_state": "PLACING",
            "event_description": "Bag returned to original area (Placing)",
            "hand_distance_px": 303.61,
            "bag_displacement_px": 41.11,
            "bag_center": [821, 504]
        },
        {
            "frame": 896,
            "timestamp_seconds": 14.952,
            "from_state": "PLACING",
            "to_state": "PLACED (RETURNED)",
            "event_description": "Hand released; bag placed back at rest (Placed)",
            "hand_distance_px": null,
            "bag_displacement_px": 36.89,
            "bag_center": [821, 511]
        }
    ]
}
```
