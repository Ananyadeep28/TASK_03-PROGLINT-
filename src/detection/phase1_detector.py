from ultralytics import YOLO


class Phase1Detector:

    def __init__(self):
        self.bag_model = YOLO("yolo11s.pt")
        self.pose_model = YOLO("yolo11n-pose.pt")

    def detect_bag(self, frame):

        results = self.bag_model(
            frame,
            classes=[24],
            conf=0.40,
            verbose=False
        )

        if len(results[0].boxes) == 0:
            return {
                "bag_detected": False,
                "bag_bbox": None,
                "bag_center": None,
                "bag_confidence": None
            }

        box = results[0].boxes.xyxy[0].cpu().numpy()
        confidence = float(results[0].boxes.conf[0].cpu())

        x1, y1, x2, y2 = box

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return {
            "bag_detected": True,
            "bag_bbox": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ],
            "bag_center": [
                center_x,
                center_y
            ],
            "bag_confidence": confidence
        }

    def detect_pose(self, frame):

        results = self.pose_model(frame, verbose=False)

        if len(results[0].keypoints) == 0:
            return {
                "person_detected": False,
                "left_wrist": None,
                "left_wrist_confidence": None,
                "right_wrist": None,
                "right_wrist_confidence": None
            }

        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        confidence = results[0].keypoints.conf[0].cpu().numpy()

        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        return {
            "person_detected": True,
            "left_wrist": [
                int(left_wrist[0]),
                int(left_wrist[1])
            ],
            "left_wrist_confidence": float(confidence[9]),
            "right_wrist": [
                int(right_wrist[0]),
                int(right_wrist[1])
            ],
            "right_wrist_confidence": float(confidence[10])
        }

    def calculate_hand_distance(self, bag, pose):

        if not bag["bag_detected"]:
            return None

        if not pose["person_detected"]:
            return None

        bag_x, bag_y = bag["bag_center"]

        distances = []

        if (
            pose["left_wrist"] is not None
            and pose["left_wrist_confidence"] >= 0.5
        ):
            wrist_x, wrist_y = pose["left_wrist"]

            distance = (
                (wrist_x - bag_x) ** 2
                + (wrist_y - bag_y) ** 2
            ) ** 0.5

            distances.append(distance)

        if (
            pose["right_wrist"] is not None
            and pose["right_wrist_confidence"] >= 0.5
        ):
            wrist_x, wrist_y = pose["right_wrist"]

            distance = (
                (wrist_x - bag_x) ** 2
                + (wrist_y - bag_y) ** 2
            ) ** 0.5

            distances.append(distance)

        if not distances:
            return None

        return round(min(distances), 2)
