import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import time

# ====================== REPRODUCIBILITY & DEVICE ======================
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.success(f"Using device: {DEVICE}")

# ====================== LOAD MODELS (cached) ======================
@st.cache_resource
def load_models():
    # DINOv2
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True).to(DEVICE).eval()
    
    # SAM 2.1 via HF
    from transformers import Sam2Model, Sam2Processor
    processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(DEVICE).eval()
    
    return dinov2, processor, model

dinov2, sam_processor, sam_model = load_models()

# ====================== HELPER FUNCTIONS (from your code) ======================
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def get_dinov2_dense_features(image_np, target_size=518):
    img = Image.fromarray(image_np)
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w = (new_w // 14) * 14
    new_h = (new_h // 14) * 14

    img = img.resize((new_w, new_h))
    tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        feats = dinov2.forward_features(tensor)['x_norm_patchtokens'].squeeze(0)

    grid_w = new_w // 14
    grid_h = new_h // 14
    return feats, (grid_w, grid_h), (new_w, new_h)

def compute_cross_image_similarity(feat1, feat2, grid_size):
    feat1 = torch.nn.functional.normalize(feat1, dim=1)
    feat2 = torch.nn.functional.normalize(feat2, dim=1)
    sim = torch.matmul(feat1, feat2.T)
    sim = sim.max(dim=1)[0]
    h, w = grid_size
    return sim.reshape(h, w).cpu().numpy()

def get_topk_prompt_points(heatmap, k=10, original_size=None):
    flat = np.argsort(heatmap.flatten())[-k:]
    y, x = np.unravel_index(flat, heatmap.shape)
    h, w = heatmap.shape
    if original_size is None:
        original_size = (w, h)
    W, H = original_size
    xs = (x / w) * W
    ys = (y / h) * H
    return np.stack([xs, ys], axis=1).astype(int)

def sam_predict(image_np, points):
    if len(points) == 0:
        return np.zeros(image_np.shape[:2], dtype=np.uint8)
    
    labels = np.ones(len(points), dtype=np.int32)
    
    inputs = sam_processor(
        images=Image.fromarray(image_np),
        input_points=[[points.tolist()]],
        input_labels=[[labels.tolist()]],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=True)

    processed_masks = sam_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"]
    )[0]

    # Select best mask (highest iou)
    if isinstance(processed_masks, list) and processed_masks:
        best_mask_info = max(processed_masks, key=lambda x: x.get('predicted_iou', 0))
        mask = best_mask_info['segmentation'].numpy()
    else:
        mask = processed_masks.numpy() if isinstance(processed_masks, torch.Tensor) else np.zeros(image_np.shape[:2])

    mask = np.squeeze(mask)
    if mask.ndim == 3:
        mask = np.mean(mask, axis=0)
    return (mask > 0).astype(np.uint8)

# ====================== STREAMLIT APP ======================
st.set_page_config(page_title="Zero-Shot Co-Segmentation", layout="wide")
st.title("🔬 DINOv2 + SAM 2.1 Zero-Shot Co-Segmentation")
st.markdown("Upload **two images** containing the same object. The app will automatically find corresponding points and segment the common object in both images.")

# Sidebar controls
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of Prompt Points (Top-K)", 5, 30, 10)
run_button = st.sidebar.button("🚀 Run Co-Segmentation", type="primary")

# Image uploaders
col1, col2 = st.columns(2)
with col1:
    st.subheader("Image 1")
    uploaded1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"], key="img1")

with col2:
    st.subheader("Image 2")
    uploaded2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"], key="img2")

if uploaded1 and uploaded2 and run_button:
    with st.spinner("Extracting DINOv2 features + Generating prompts + Running SAM 2..."):
        start_time = time.time()
        
        # Load images
        img1_np = load_image(uploaded1)
        img2_np = load_image(uploaded2)
        
        # Get features
        feat1, grid_size1, _ = get_dinov2_dense_features(img1_np)
        feat2, _, _ = get_dinov2_dense_features(img2_np)
        
        # Compute similarity heatmap
        heatmap = compute_cross_image_similarity(feat1, feat2, grid_size1)
        
        # Get prompt points from Image 1
        points = get_topk_prompt_points(heatmap, k=top_k, original_size=img1_np.shape[:2][::-1])
        
        # Run SAM 2 on both images
        mask1 = sam_predict(img1_np, points)
        mask2 = sam_predict(img2_np, points)
        
        processing_time = time.time() - start_time

    # ====================== DISPLAY RESULTS ======================
    st.success(f"✅ Processing completed in {processing_time:.2f} seconds")

    # Original + Segmented side-by-side
    tab1, tab2, tab3 = st.tabs(["📊 Results", "🔥 Similarity Heatmap", "📈 Analysis"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(uploaded1, caption="Image 1 (Original)", use_container_width=True)
            overlay1 = img1_np.copy()
            overlay1[mask1 > 0] = overlay1[mask1 > 0] * 0.4 + np.array([0, 255, 0]) * 0.6
            st.image(overlay1, caption="Image 1 + Predicted Mask (Ours)", use_container_width=True)
        
        with col_b:
            st.image(uploaded2, caption="Image 2 (Original)", use_container_width=True)
            overlay2 = img2_np.copy()
            overlay2[mask2 > 0] = overlay2[mask2 > 0] * 0.4 + np.array([0, 255, 0]) * 0.6
            st.image(overlay2, caption="Image 2 + Predicted Mask (Ours)", use_container_width=True)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heatmap, cmap='hot')
        ax.set_title("Cross-Image Semantic Similarity Heatmap (DINOv2)")
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

    with tab3:
        st.subheader("Quick Analysis")
        st.write(f"**Top-K points used:** {top_k}")
        st.write(f"**Processing time:** {processing_time:.2f} seconds")
        st.write("**Notes:**")
        std_heatmap = np.std(heatmap.flatten())
        if std_heatmap > 0.15:
            st.warning("High similarity variance – good separation expected")
        else:
            st.info("Low variance – may struggle with uniform regions (e.g., sky, wall)")

        st.caption("This is a **zero-shot** method. No training data or ground truth is needed.")

    # Optional: Download masks
    def mask_to_download(mask, name):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(f"{name}.png")
        with open(f"{name}.png", "rb") as f:
            return f.read()

    st.download_button("Download Mask 1", mask_to_download(mask1, "mask1"), "mask1.png", "image/png")
    st.download_button("Download Mask 2", mask_to_download(mask2, "mask2"), "mask2.png", "image/png")

else:
    st.info("👆 Upload two images and click **Run Co-Segmentation**")

st.caption("Built with DINOv2 (dense features) + SAM 2.1 (promptable segmentation) | Project for EE655")