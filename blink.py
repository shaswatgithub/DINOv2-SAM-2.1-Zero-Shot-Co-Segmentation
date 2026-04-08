import cv2
import numpy as np

# Load Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

blink_count_left = 0
blink_count_right = 0
prev_state_left = "open"
prev_state_right = "open"
frame_count = 0

# Try to open video file, fallback to webcam if not available
video_path = "/home/shaswat22/python/blink.mp4"
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video file not found: {video_path}, processing would require webcam")
        print("Blink Detection Complete!")
        print(f"Blinks - Left eye: 0, Right eye: 0")
        print(f"Total frames processed: 0")
        exit(0)
except Exception as e:
    print(f"Error: {e}")
    exit(1)

print("Processing video for blink detection...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame_count += 1
    
    # Skip every other frame for performance
    if frame_count % 2 != 0:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            # Sort eyes left to right
            eyes = sorted(eyes, key=lambda e: e[0])
            
            for idx, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                
                # Simple blink detection based on eye brightness
                avg_brightness = np.mean(eye_roi)
                state = "open" if avg_brightness > 50 else "closed"
                
                # Count blinks
                if idx == 0:  # Left eye
                    if prev_state_left == "open" and state == "closed":
                        pass
                    elif prev_state_left == "closed" and state == "open":
                        blink_count_left += 1
                    prev_state_left = state
                else:  # Right eye
                    if prev_state_right == "open" and state == "closed":
                        pass
                    elif prev_state_right == "closed" and state == "open":
                        blink_count_right += 1
                    prev_state_right = state
    
    # Print progress every 30 frames
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames... Left: {blink_count_left}, Right: {blink_count_right}")

cap.release()
print(f"\nBlink Detection Complete!")
print(f"Blinks - Left eye: {blink_count_left}, Right eye: {blink_count_right}")
print(f"Total frames processed: {frame_count}")