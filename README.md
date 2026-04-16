<img width="1833" height="964" alt="image" src="https://github.com/user-attachments/assets/37174062-f2a7-4953-a865-f21d454b9373" />

---

# 🔬 DINOv2 + SAM 2.1 Zero-Shot Co-Segmentation

🚀 **Live App:**
👉 [https://shaswatgithub-dinov2-sam-2-1-zero-shot-co-segmentat-app1-hchqrt.streamlit.app/](https://shaswatgithub-dinov2-sam-2-1-zero-shot-co-segmentat-app1-hchqrt.streamlit.app/)

📦 **GitHub Repository:**
👉 [https://github.com/shaswatgithub/DINOv2-SAM-2.1-Zero-Shot-Co-Segmentation](https://github.com/shaswatgithub/DINOv2-SAM-2.1-Zero-Shot-Co-Segmentation)

---

## 🧠 Overview

This project presents a **zero-shot co-segmentation system** that automatically extracts the **common object** from two images — without any training, labels, or fine-tuning.

It combines:

* **DINOv2 (ViT-B/14)** → Dense semantic feature extraction
* **SAM 2.1 (Hiera Large)** → Prompt-based segmentation

The system identifies **corresponding semantic regions** across images and uses them as prompts to generate high-quality segmentation masks.

---

## ✨ Features

* 🔍 **Zero-shot learning** (no training required)
* 🧩 Cross-image **semantic correspondence**
* 🎯 High-quality segmentation using SAM 2.1
* 🌡️ Real-time **similarity heatmap visualization**
* 🎛️ Adjustable **Top-K prompt points**
* ⚡ Interactive **Streamlit interface**
* 💻 Supports both **CPU and GPU**

---

## 🖼️ How It Works

### Step-by-step pipeline:

1. Upload **two images** containing a common object
2. Extract **dense patch embeddings** using DINOv2
3. Compute **cross-image cosine similarity**
4. Select **Top-K most similar points**
5. Use these points as **prompts for SAM 2.1**
6. Generate segmentation masks for both images

---

## ⚙️ System Architecture

```text
Image A ──┐
          ├── DINOv2 → Dense Features ──┐
Image B ──┘                             │
                                        ├── Similarity Matching
                                        │       ↓
                                        │   Top-K Points
                                        │
                                        └── SAM 2.1 → Segmentation Masks
```

---

## 🧪 Demo

The app provides:

* 📊 Side-by-side segmentation results
* 🔥 Cross-image similarity heatmap
* 📈 Basic analysis (variance, performance)

---

## 📁 Project Structure

```text
📦 DINOv2-SAM-2.1-Zero-Shot-Co-Segmentation
├── app.py                 # Streamlit web application
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── assets/                # Demo images / visuals (optional)
```

---

## 🛠️ Installation

```bash
git clone https://github.com/shaswatgithub/DINOv2-SAM-2.1-Zero-Shot-Co-Segmentation.git
cd DINOv2-SAM-2.1-Zero-Shot-Co-Segmentation

pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

---

## 🎛️ Usage

1. Upload **two images containing the same object**
2. Adjust **Top-K prompt points**
3. Click **Run Co-Segmentation**
4. View:

   * Segmentation results
   * Similarity heatmap
   * Performance analysis

---

## 📊 Technical Insights

* Uses **patch-level feature matching** instead of pixel-level
* Robust to:

  * Scale variations
  * Lighting differences
  * Background clutter

### Limitation:

* May struggle when:

  * Objects are extremely small
  * Background dominates similarity
  * No clear semantic correspondence exists

---

## 🔮 Future Work

* Multi-object co-segmentation
* Video co-segmentation
* Adaptive prompt selection
* Faster inference optimization

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.

---

## 📜 License

SHASWAT
---

## 👨‍💻 Author

**Shaswat**
Project developed as part of **EE655 (Computer Vision / AI coursework)**

---
