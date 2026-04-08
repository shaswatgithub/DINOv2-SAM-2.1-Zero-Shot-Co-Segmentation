import cv2
import torch
import numpy as np

from segment_anything import sam_model_registry, SamPredictor

# -----------------------------
# Step 1: Load SAM Model
# -----------------------------

model_type = "vit_b"                # model size
checkpoint = "sam_vit_b_01ec64.pth" # downloaded checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# -----------------------------
# Step 2: Global Variables
# -----------------------------

clicked_point = None
clicked_label = None
frame_for_prediction = None


# -----------------------------
# Step 3: Mouse Click Function
# -----------------------------

def mouse_click(event, x, y, flags, param):

    global clicked_point, clicked_label

    if event == cv2.EVENT_LBUTTONDOWN:

        # Save clicked coordinates
        clicked_point = np.array([[x, y]])
        clicked_label = np.array([1])  # 1 means foreground


# -----------------------------
# Step 4: Start Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

cv2.namedWindow("SAM Webcam")
cv2.setMouseCallback("SAM Webcam", mouse_click)

# -----------------------------
# Step 5: Main Loop
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_for_prediction = frame.copy()

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Set image for SAM
    predictor.set_image(rgb_frame)

    # If user clicked
    if clicked_point is not None:

        masks, scores, logits = predictor.predict(
            point_coords=clicked_point,
            point_labels=clicked_label,
            multimask_output=True
        )

        # choose best mask
        mask = masks[np.argmax(scores)]

        # convert mask to color overlay
        colored_mask = np.zeros_like(frame)
        colored_mask[mask] = [0, 255, 0]

        frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

        # draw click point
        x, y = clicked_point[0]
        cv2.circle(frame, (x, y), 6, (0,0,255), -1)

    cv2.imshow("SAM Webcam", frame)

    key = cv2.waitKey(1)

    if key == 27:   # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()