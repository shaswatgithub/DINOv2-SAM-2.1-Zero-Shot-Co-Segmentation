import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

print("Loading models... (First run may take 30-60 seconds)")

# ====================== YOLOE-26n - Open-Vocabulary Detection ======================
detect_model = YOLO("yoloe-26n-seg.pt")
detect_model.export(format="openvino", imgsz=640, half=True)
ov_detect = YOLO("yoloe-26n-seg_openvino_model/")

# ====================== YOLO11n-pose for CPU Pose Estimation ======================
pose_model = YOLO("yolo11n-pose.pt")

track_history = {}  # track_id -> deque of hip y-coordinates

def calculate_anomaly(keypoints, history):
    if keypoints is None or len(keypoints) < 17:
        return 0.0
    
    # COCO keypoints indices
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
    ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
    
    fall_score = max(0, 90 - abs(hip_y - ankle_y)) / 90.0
    movement_score = 0.0
    if len(history) > 8:
        y_std = np.std(list(history))
        movement_score = min(y_std / 35.0, 1.0)
    
    return (fall_score * 0.65) + (movement_score * 0.35)


def process_frame(frame, text_prompts="person, helmet, phone, bag, weapon"):
    start = time.time()
    
    # Update classes for open-vocabulary detection
    classes = [c.strip() for c in text_prompts.split(",") if c.strip()]
    if classes:
        ov_detect.set_classes(classes)
    
    # Detection + Tracking
    detect_results = ov_detect.track(frame, persist=True, tracker="bytetrack.yaml",
                                     conf=0.4, iou=0.6, imgsz=640, verbose=False)
    
    annotated = frame.copy()
    
    if detect_results[0].boxes is not None:
        boxes = detect_results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = detect_results[0].boxes.id.int().cpu().numpy() if detect_results[0].boxes.id is not None else range(len(boxes))
        cls_ids = detect_results[0].boxes.cls.int().cpu().numpy()
        
        for box, tid, cid in zip(boxes, track_ids, cls_ids):
            x1, y1, x2, y2 = box
            label = ov_detect.names[cid]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label} #{tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Pose + Anomaly Detection
    pose_results = pose_model.track(frame, persist=True, conf=0.5, imgsz=640, verbose=False)
    
    if pose_results[0].keypoints is not None and pose_results[0].boxes is not None:
        kpts_data = pose_results[0].keypoints.xy.cpu().numpy()
        boxes = pose_results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = pose_results[0].boxes.id.int().cpu().numpy() if pose_results[0].boxes.id is not None else range(len(kpts_data))
        
        for i, tid in enumerate(track_ids):
            keypoints = kpts_data[i]
            if tid not in track_history:
                track_history[tid] = deque(maxlen=15)
            
            hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
            track_history[tid].append(hip_y)
            
            anomaly_score = calculate_anomaly(keypoints, track_history[tid])
            
            if anomaly_score > 0.62:
                x1, y1 = boxes[i][0], boxes[i][1]
                cv2.putText(annotated, "⚠️ ANOMALY (Fall/Fight)!", 
                            (max(30, x1), max(80, y1-30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    
    # FPS
    fps = 1 / (time.time() - start)
    cv2.putText(annotated, f"FPS: {fps:.1f} (CPU)", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    return annotated


# ====================== Gradio UI (Fixed for Gradio 6.x) ======================
with gr.Blocks(title="Smart Surveillance - CPU Only") as demo:
    gr.Markdown("# 🛡️ Smart Open-Vocabulary Surveillance System\n"
                "YOLOE-26n + YOLO Pose + Anomaly Detection (CPU Optimized)")

    with gr.Row():
        webcam = gr.Image(
            sources=["webcam"],   # Fixed syntax
            type="numpy",
            streaming=True,
            label="Live Webcam"
        )
        output = gr.Image(label="Processed Output")
    
    prompt = gr.Textbox(
        value="person, helmet, phone, bag, weapon",
        label="Objects to Detect (comma separated)"
    )
    
    gr.Markdown("**Instructions**: Allow camera access. Change prompt to detect any objects (zero-shot). "
                "Red alert appears on sudden fall or rapid movement.")

    webcam.stream(
        fn=process_frame,
        inputs=[webcam, prompt],
        outputs=output,
        show_progress=False
    )

if __name__ == "__main__":
    print("🚀 Starting demo... Allow camera access when prompted.")
    demo.launch(share=False)