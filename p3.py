import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

# -----------------------------
# 1. Extract Frames
# -----------------------------
def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture("/home/shaswat22/python/vid2.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError("Error loading video")

    step = max(total_frames // num_frames, 1)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        count += 1
        if len(frames) == num_frames:
            break

    cap.release()

    # If less frames, pad
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.array(frames)


# -----------------------------
# 2. Convert to Tensor
# -----------------------------
def frames_to_tensor(frames):
    frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W)
    frames = frames / 255.0
    tensor = torch.tensor(frames, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)  # (1, T, C, H, W)
    return tensor


# -----------------------------
# 3. Model
# -----------------------------
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained CNN
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.temporal = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(B, T, -1)

        out, _ = self.temporal(features)
        out = out[:, -1, :]  # last timestep

        return self.fc(out)


# -----------------------------
# 4. Prediction Function
# -----------------------------
def predict(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepfakeDetector().to(device)
    model.eval()

    frames = extract_frames(video_path)
    tensor = frames_to_tensor(frames).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)

    real_prob = probs[0][0].item()
    fake_prob = probs[0][1].item()

    print(f"Real Probability: {real_prob:.4f}")
    print(f"Fake Probability: {fake_prob:.4f}")

    if fake_prob > real_prob:
        print("Prediction: FAKE VIDEO")
    else:
        print("Prediction: REAL VIDEO")


# -----------------------------
# 5. Run
# -----------------------------
if __name__ == "__main__":
    video_path = "test_video.mp4"   # <-- PUT YOUR VIDEO HERE
    predict(video_path)