import cv2
import requests
import json
import urllib.parse
import time
from ultralytics import YOLO

# ===============================
# Robot HTTP function
# ===============================

ESP = "http://192.168.4.1"

def robot(cmd):
    url = ESP + "/js?json=" + urllib.parse.quote(json.dumps(cmd))
    try:
        requests.get(url, timeout=0.2)
    except:
        pass


# ===============================
# Load YOLO
# ===============================

model = YOLO("yolov8n.pt")

# ===============================
# Camera
# ===============================

cap = cv2.VideoCapture("/dev/video2")

target = "bottle"

# ===============================
# Robot variables
# ===============================

BASE = 0
XPOS = 220

SEARCH_DIRECTION = 1

STATE = "SEARCH"

DETECT_COUNT = 0
DETECT_THRESHOLD = 3

CENTER_MARGIN = 100
AREA_REACHED = 130000

# initialize robot
robot({"T":100})

time.sleep(2)

# ===============================
# Main loop
# ===============================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_center = w // 2

    results = model(frame)

    bottle_found = False

    obj_center = None
    area = None

    for box in results[0].boxes:

        cls = int(box.cls[0])
        label = model.names[cls]

        if label != target:
            continue

        bottle_found = True

        x1, y1, x2, y2 = box.xyxy[0]

        obj_center = int((x1 + x2) / 2)
        area = (x2 - x1) * (y2 - y1)

        break

    # ===============================
    # Detection stability
    # ===============================

    if bottle_found:
        DETECT_COUNT += 1
    else:
        DETECT_COUNT = 0

    stable_detection = DETECT_COUNT >= DETECT_THRESHOLD

    # ===============================
    # STATE MACHINE
    # ===============================

    if STATE == "SEARCH":

        if stable_detection:
            STATE = "ALIGN"

        else:

            BASE += SEARCH_DIRECTION * 2

            if BASE > 60:
                SEARCH_DIRECTION = -1
            elif BASE < -60:
                SEARCH_DIRECTION = 1

            robot({
                "T":122,
                "b":BASE,
                "s":0,
                "e":90,
                "h":180,
                "spd":8,
                "acc":8
            })

            time.sleep(0.15)

    # ===============================
    # ALIGN WITH OBJECT
    # ===============================

    elif STATE == "ALIGN":

        if not stable_detection:
            STATE = "SEARCH"

        else:

            if obj_center < frame_center - CENTER_MARGIN:

                BASE -= 2

                robot({
                    "T":122,
                    "b":BASE,
                    "s":0,
                    "e":90,
                    "h":180,
                    "spd":8,
                    "acc":8
                })

            elif obj_center > frame_center + CENTER_MARGIN:

                BASE += 2

                robot({
                    "T":122,
                    "b":BASE,
                    "s":0,
                    "e":90,
                    "h":180,
                    "spd":8,
                    "acc":8
                })

            else:

                STATE = "APPROACH"

            time.sleep(0.15)

    # ===============================
    # MOVE TOWARD OBJECT
    # ===============================

    elif STATE == "APPROACH":

        if not stable_detection:
            STATE = "SEARCH"

        else:

            if area < AREA_REACHED:

                XPOS = min(XPOS + 5, 300)

                robot({
                    "T":104,
                    "x":XPOS,
                    "y":0,
                    "z":220,
                    "t":3.14,
                    "spd":0.2
                })

            else:

                print("Bottle reached")
                STATE = "STOP"

            time.sleep(0.2)

    # ===============================
    # STOP STATE
    # ===============================

    elif STATE == "STOP":

        robot({
            "T":122,
            "b":BASE,
            "s":0,
            "e":90,
            "h":180,
            "spd":5,
            "acc":5
        })

        time.sleep(0.5)

    # ===============================
    # Display
    # ===============================

    frame = results[0].plot()

    cv2.putText(frame, f"STATE: {STATE}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Robot Vision", frame)

    if cv2.waitKey(1) == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()