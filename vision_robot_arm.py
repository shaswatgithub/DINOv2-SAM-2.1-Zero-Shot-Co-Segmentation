import cv2
import requests
import json
import urllib.parse
from ultralytics import YOLO

# ===============================
# Robot connection
# ===============================

ESP = "http://192.168.4.1"

def robot(cmd):
    url = ESP + "/js?json=" + urllib.parse.quote(json.dumps(cmd))
    try:
        requests.get(url, timeout=0.2)
    except:
        pass


# ===============================
# Load YOLO model
# ===============================

model = YOLO("yolov8n.pt")

# ===============================
# Camera
# ===============================

cap = cv2.VideoCapture("/dev/video2")

target_object = "bottle"

# ===============================
# Robot initial position
# ===============================

robot({"T":100})  # initialize


# ===============================
# Control parameters
# ===============================

BASE_ANGLE = 0
CENTER_MARGIN = 60
AREA_STOP = 150000

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_center = w // 2

    results = model(frame)

    command_sent = False

    for box in results[0].boxes:

        cls = int(box.cls[0])
        label = model.names[cls]

        if label != target_object:
            continue

        x1, y1, x2, y2 = box.xyxy[0]

        object_center = int((x1 + x2) / 2)
        area = (x2 - x1) * (y2 - y1)

        # ===============================
        # LEFT / RIGHT ALIGNMENT
        # ===============================

        if object_center < frame_center - CENTER_MARGIN:

            BASE_ANGLE -= 3

            robot({
                "T":122,
                "b":BASE_ANGLE,
                "s":0,
                "e":90,
                "h":180,
                "spd":10,
                "acc":10
            })

            command_sent = True


        elif object_center > frame_center + CENTER_MARGIN:

            BASE_ANGLE += 3

            robot({
                "T":122,
                "b":BASE_ANGLE,
                "s":0,
                "e":90,
                "h":180,
                "spd":10,
                "acc":10
            })

            command_sent = True


        # ===============================
        # OBJECT CENTERED → MOVE ARM
        # ===============================

        else:

            if area < AREA_STOP:

                robot({
                    "T":104,
                    "x":235,
                    "y":0,
                    "z":220,
                    "t":3.14,
                    "spd":0.2
                })

            else:

                # Object close → grab
                robot({"T":103,"pos":0})

            command_sent = True

    frame = results[0].plot()

    cv2.imshow("Robot Vision", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()