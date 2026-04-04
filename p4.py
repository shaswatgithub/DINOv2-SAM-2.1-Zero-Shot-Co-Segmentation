import torch
import torch.nn as nn
import numpy as np
import librosa

# -----------------------------
# 1. Audio → Mel Spectrogram
# -----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


# -----------------------------
# 2. Preprocess
# -----------------------------
def preprocess(mel):
    # Fix width to 128 (important)
    mel = librosa.util.fix_length(mel, size=128, axis=1)

    # Normalize
    mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)

    # Add channel dimension → (1, 128, 128)
    mel = np.expand_dims(mel, axis=0)

    tensor = torch.tensor(mel, dtype=torch.float32)

    # Add batch → (1, 1, 128, 128)
    tensor = tensor.unsqueeze(0)

    return tensor


# -----------------------------
# 3. Model (Stable CNN)
# -----------------------------
class AudioCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Adaptive pooling → avoids shape crash
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


# -----------------------------
# 4. Classes
# -----------------------------
CLASSES = ["dog_bark", "rain", "siren", "gunshot", "music"]


# -----------------------------
# 5. Prediction
# -----------------------------
def predict(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioCNN(num_classes=len(CLASSES)).to(device)
    model.eval()

    mel = extract_features(file_path)
    tensor = preprocess(mel).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)

    pred = torch.argmax(probs).item()

    print("Prediction:", CLASSES[pred])
    print("Confidence:", probs[0][pred].item())


# -----------------------------
# 6. Run
# -----------------------------
if __name__ == "__main__":
    audio_file = "/home/shaswat22/python/dog.wav"  # <-- your file
    predict(audio_file)