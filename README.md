

```markdown
# DINOv2 + SAM 2.1 Zero-Shot Co-Segmentation

A **Streamlit web application** that performs **zero-shot co-segmentation** on two images using **DINOv2** for dense semantic features and **SAM 2.1** for promptable segmentation.

This project automatically finds corresponding semantic points between two images and segments the common object without any training or fine-tuning.

---

## ✨ Features

- Zero-shot co-segmentation (no labels or training required)
- Uses state-of-the-art DINOv2 (ViT-B/14) for cross-image similarity
- SAM 2.1 (Hiera Large) for high-quality mask generation
- Interactive Streamlit web interface
- Real-time similarity heatmap visualization
- Adjustable number of prompt points (Top-K)
- Works on CPU and GPU

---

## 🖼️ How It Works

1. Upload two images containing the **same object**
2. DINOv2 extracts dense patch features from both images
3. Cross-image semantic similarity is computed to find corresponding regions
4. Top-K most similar points are selected as prompts
5. SAM 2.1 generates accurate segmentation masks for the common object in both images

---

## 🚀 Demo & Deployment

The app is ready to be deployed on **Streamlit Community Cloud**.

**Live Demo**: (Add link after deployment)

---

## 📁 Project Structure

```
DINOv2-SAM-2.1-Zero-Shot-Co-Segmentation/
├── app1.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── results/                # (Generated) Output images and masks
```

---

## 🛠️ Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/shaswatgithub/DINOv2-SAM-2.1-Zero-Shot-Co-Segmentation.git
cd DINOv2-SAM-2.1-Zero-Shot-Co-Segmentation
```

### 2. Create virtual environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate    # On Linux/Mac
# venv\Scripts\activate     # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app locally
```bash
streamlit run app1.py
```

---

## 📦 Requirements

- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- OpenCV
- Streamlit
- DINOv2 & SAM 2.1 models (automatically downloaded on first run)

See `requirements.txt` for full list.

---

## 🎛️ Usage

1. Open the app
2. Upload **Image 1** and **Image 2**
3. Adjust the **Top-K** prompt points (default: 10)
4. Click **"Run Co-Segmentation"**
5. View:
   - Original images
   - Segmented results with overlay
   - Cross-image similarity heatmap
   - Download predicted masks

**Tip**: Best results when both images contain the same prominent object with good lighting and minimal occlusion.

---

## 🧠 Technical Details

- **Feature Extractor**: DINOv2 ViT-B/14 (dense patch tokens)
- **Segmentation Model**: SAM 2.1 Hiera Large
- **Similarity Method**: Normalized cosine similarity + max pooling
- **Prompt Strategy**: Top-K highest similarity points
- **Device Support**: Automatic CPU / CUDA detection

---

## 📊 Future Improvements

- Add SIFT baseline comparison
- Support for multiple objects
- Ground truth evaluation metrics (IoU, Dice, F1)
- Batch processing
- Gradio version
- GPU acceleration optimizations

---

## 👨‍💻 Author

**Shaswat**  
Project for EE655 (Computer Vision / Image Processing)

---
