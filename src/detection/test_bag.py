import cv2
from detector import BagDetector


INPUT = "data/samples/BAG.mp4"
OUTPUT = "outputs/bag_detection.mp4"


detector = BagDetector()

cap = cv2.VideoCapture(INPUT)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = detector.detect(frame)
    out.write(result)

cap.release()
out.release()

print(f"Saved: {OUTPUT}")