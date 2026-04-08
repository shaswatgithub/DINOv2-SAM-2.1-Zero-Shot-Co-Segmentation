import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO("yolov8n.pt")

# -------------------------------
# VIDEO INPUT (CHANGE THIS)
# -------------------------------
cap = cv2.VideoCapture("/home/shaswat22/python/fight.webm")  # put your video name

# -------------------------------
# OUTPUT VIDEO SAVE
# -------------------------------
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (frame_width, frame_height)
)

# -------------------------------
# TRACKING HISTORY
# -------------------------------
track_history = defaultdict(lambda: deque(maxlen=10))

# -------------------------------
# THRESHOLDS (TUNE HERE)
# -------------------------------
ANOMALY_THRESHOLD = 50

# -------------------------------
# FUNCTIONS
# -------------------------------

def compute_movement(history):
    if len(history) < 2:
        return 0

    x_prev, y_prev = history[-2]
    x_curr, y_curr = history[-1]

    return np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)


def detect_fall(history, box):
    """
    Improved fall detection:
    - Detects falling (fast movement)
    - Detects lying (already fallen)
    """
    if len(history) < 5:
        return False

    # SPEED
    x_prev, y_prev = history[-5]
    x_curr, y_curr = history[-1]
    speed = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)

    # SHAPE
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    aspect_ratio = height / (width + 1e-5)

    # FALL CONDITIONS
    falling = (speed > 30 and aspect_ratio < 0.75)
    lying = (aspect_ratio < 0.6)

    if falling or lying:
        return True

    return False


# -------------------------------
# MAIN LOOP
# -------------------------------
print("🚀 Running Surveillance System... Press 'q' to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO tracking
    results = model.track(frame, persist=True, conf=0.5)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()

        for box, track_id, cls in zip(boxes, track_ids, classes):

            # Only PERSON
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box)

            # CENTER
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # STORE HISTORY
            track_history[track_id].append((cx, cy))

            # MOVEMENT
            movement = compute_movement(track_history[track_id])

            # FALL DETECTION
            is_fall = detect_fall(track_history[track_id], box)

            # ANOMALY
            is_anomaly = movement > ANOMALY_THRESHOLD

            # -----------------------
            # DRAW
            # -----------------------
            color = (0, 255, 0)  # normal

            if is_anomaly:
                color = (0, 165, 255)  # orange

            if is_fall:
                color = (0, 0, 255)  # red

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # LABEL
            label = f"ID {track_id}"

            if is_anomaly:
                label += " | Anomaly"

            if is_fall:
                label += " | FALL!"

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # CENTER DOT
            cv2.circle(frame, (cx, cy), 5, color, -1)

    # TITLE
    cv2.putText(frame, "Smart Surveillance System",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    # SHOW
    cv2.imshow("Surveillance", frame)

    # SAVE
    out.write(frame)

    # EXIT
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Done! Output saved as output.mp4")