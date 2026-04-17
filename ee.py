import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.metrics import precision_recall_fscore_support

# ====================== REPRODUCIBILITY ======================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ====================== MODELS ======================
# DINOv2 (official, dense patch features)
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True).to(DEVICE).eval()

# SAM 2 via Hugging Face Transformers (point-prompt ready, 2026 standard)
from transformers import Sam2Model, Sam2Processor
sam_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
sam_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(DEVICE).eval()

# ====================== HELPER FUNCTIONS ======================
def load_image(img_path):
    """Load and convert to RGB numpy."""
    img = Image.open(img_path).convert("RGB")
    return np.array(img), img.size  # (H, W, 3), original size

def get_dinov2_dense_features(image_np, target_size=518):
    img = Image.fromarray(image_np)

    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    # IMPORTANT: keep divisible by 14
    new_w = (new_w // 14) * 14
    new_h = (new_h // 14) * 14

    img = img.resize((new_w, new_h))

    tensor = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).float()/255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        feats = dinov2.forward_features(tensor)['x_norm_patchtokens']

    feats = feats.squeeze(0)

    # correct grid size
    grid_w = new_w // 14
    grid_h = new_h // 14

    return feats, (grid_w, grid_h), (new_w, new_h)
def compute_cross_image_similarity(feat1, feat2, grid_size):
    feat1 = torch.nn.functional.normalize(feat1, dim=1)
    feat2 = torch.nn.functional.normalize(feat2, dim=1)

    sim = torch.matmul(feat1, feat2.T)  # (N1, N2)

    # stable pooling
    sim = sim.max(dim=1)[0]

    h, w = grid_size
    return sim.reshape(h, w).cpu().numpy()

def get_topk_prompt_points(heatmap, k=10, original_size=(1024,1024)):
    flat = np.argsort(heatmap.flatten())[-k:]
    y, x = np.unravel_index(flat, heatmap.shape)

    h, w = heatmap.shape
    H, W = original_size[1], original_size[0]

    xs = (x / w) * W
    ys = (y / h) * H

    return np.stack([xs, ys], axis=1).astype(int)

def sam_predict(image_np, points):
    labels = np.ones(len(points), dtype=np.int32)

    inputs = sam_processor(
        images=Image.fromarray(image_np),
        input_points=[[points.tolist()]], # Corrected nesting here
        input_labels=[[labels.tolist()]], # Corrected nesting here
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=True)

    processed_masks = sam_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"]
    )[0]

    if isinstance(processed_masks, torch.Tensor): # Handle case where post_process_masks returns a tensor (e.g. if no masks were found)
        if processed_masks.numel() == 0: # Empty tensor
            mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        else: # Treat as a single mask if it's a non-empty tensor
            mask = processed_masks.numpy()
            mask = np.squeeze(mask) # Squeeze all singleton dimensions
            if mask.ndim == 3: # If it's still 3D (e.g., C, H, W) after squeeze, take mean across channels
                mask = np.mean(mask, axis=0)
            mask = (mask > 0).astype(np.uint8) # Ensure binary and uint8
        return mask
    elif not processed_masks: # Handle case where no masks are returned (empty list)
        return np.zeros(image_np.shape[:2], dtype=np.uint8)

    # Find the mask with the highest predicted_iou score from the processed_masks
    best_mask_info = max(processed_masks, key=lambda x: x['predicted_iou'])
    mask = best_mask_info['segmentation'].numpy()
    mask = np.squeeze(mask) # Ensure it's 2D (H, W)

    # IMPORTANT FIX: ensure visible mask is binary uint8
    return (mask > 0).astype(np.uint8)

def compute_iou(pred_mask, gt_mask):
    """Binary IoU."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

def compute_metrics(pred_mask, gt_mask):
    """Full segmentation metrics."""
    iou = compute_iou(pred_mask, gt_mask)
    pred_flat = pred_mask.flatten().astype(int)
    gt_flat = gt_mask.flatten().astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(gt_flat, pred_flat, average='binary', zero_division=0)
    return {"mIoU": iou, "Precision": prec, "Recall": rec, "F1": f1}

# ====================== BASELINE: SIFT + SAM2 ======================
def sift_baseline(img1_path, img2_path):
    """Traditional SIFT matching → points → SAM2 (your comparison baseline)."""
    img1 = cv2.imread("/home/shaswat22/python/1.jpeg")
    img2 = cv2.imread("/home/shaswat22/python/2.jpeg")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:20]  # top 20

    points = np.array([kp1[m.queryIdx].pt for m in matches]).astype(int)
    # Use points from img1 on both (cross-image correspondence)
    mask1 = sam_predict(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), points)
    mask2 = sam_predict(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), points)
    return mask1, mask2

# ====================== MAIN PIPELINE ======================
def zero_shot_co_segment(img1_path, img2_path, top_k=10, save_dir="results"):
    Path(save_dir).mkdir(exist_ok=True)
    name = Path(img1_path).stem + "_" + Path(img2_path).stem

    # Load images
    img1_np, orig_size1 = load_image("/home/shaswat22/python/1.jpeg")
    img2_np, orig_size2 = load_image("/home/shaswat22/python/2.jpeg")

    # DINOv2 features
    feat1, grid_size1, resized1 = get_dinov2_dense_features(img1_np)
    feat2, grid_size2, resized2 = get_dinov2_dense_features(img2_np)

    # Semantic similarity
    heatmap = compute_cross_image_similarity(feat1, feat2, grid_size1)

    # Top-K prompts
    points = get_topk_prompt_points(heatmap, k=top_k, original_size=orig_size1)

    # SAM 2 masks
    mask1 = sam_predict(img1_np, points)
    mask2 = sam_predict(img2_np, points)

    # Save visuals for report
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0,0].imshow(img1_np); ax[0,0].set_title("Image 1")
    ax[0,1].imshow(img2_np); ax[0,1].set_title("Image 2")
    ax[0,2].imshow(heatmap, cmap='hot'); ax[0,2].set_title("Semantic Similarity Heatmap")
    ax[1,0].imshow(img1_np); ax[1,0].imshow(mask1, alpha=0.5, cmap='jet'); ax[1,0].set_title("Mask 1 (Ours)")
    ax[1,1].imshow(img2_np); ax[1,1].imshow(mask2, alpha=0.5, cmap='jet'); ax[1,1].set_title("Mask 2 (Ours)")
    ax[1,2].axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}_results.png", dpi=300, bbox_inches='tight')
    plt.close()

    return mask1, mask2, heatmap

# ====================== EVALUATION & ANALYSIS (for 100% marks) ======================
def run_full_evaluation(image_pairs, gt_masks1, gt_masks2, save_dir="results"):
    """
    image_pairs: list of (img1_path, img2_path)
    gt_masks1/2: corresponding ground-truth binary masks (numpy bool)
    """
    results = []
    for i, (p1, p2) in tqdm(enumerate(image_pairs), total=len(image_pairs)):
        # Our method
        mask1_ours, mask2_ours, heatmap_ours = zero_shot_co_segment(p1, p2, top_k=10, save_dir=save_dir)
        metrics1 = compute_metrics(mask1_ours, gt_masks1[i])
        metrics2 = compute_metrics(mask2_ours, gt_masks2[i])

        # SIFT baseline
        mask1_sift, mask2_sift = sift_baseline(p1, p2)
        metrics1_sift = compute_metrics(mask1_sift, gt_masks1[i])
        metrics2_sift = compute_metrics(mask2_sift, gt_masks2[i])

        results.append({
            "Pair": i+1,
            "mIoU_Ours": (metrics1["mIoU"] + metrics2["mIoU"])/2,
            "mIoU_SIFT": (metrics1_sift["mIoU"] + metrics2_sift["mIoU"])/2,
            "F1_Ours": (metrics1["F1"] + metrics2["F1"])/2,
            "F1_SIFT": (metrics1_sift["F1"] + metrics2_sift["F1"])/2,
            "Notes": "High similarity variance" if np.std(heatmap_ours.flatten()) > 0.15 else "Low variance (possible sky failure)"
        })

    df = pd.DataFrame(results)
    avg_row = df.mean(numeric_only=True)
    avg_row["Pair"] = "AVERAGE"
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save for report
    df.to_csv(f"{save_dir}/quantitative_results.csv", index=False)
    print("=== COMPARATIVE TABLE (copy-paste into report) ===")
    print(df.to_markdown(index=False))

    # Ablation: different Top-K
    print("\n=== ABLATION (Top-K) ===")
    for k in [5, 10, 20]:
        # run on first pair only for speed
        _, _, _ = zero_shot_co_segment(image_pairs[0][0], image_pairs[0][1], top_k=k)
        print(f"Top-K={k} completed (check saved figures)")

    return df

# ====================== USAGE EXAMPLE ======================
if __name__ == "__main__":
    # Replace with your own pairs + GT masks (for zero-shot evaluation)
    # Example placeholder – add your 1.jpeg, 2.jpeg and corresponding binary GT masks
    test_pairs = [
        ("data/image1.jpg", "data/image2.jpg"),
        # add more pairs...
    ]
    # gt_masks1 = [np.load("gt1.npy"), ...]  # binary numpy arrays same shape as images
    # gt_masks2 = [np.load("gt2.npy"), ...]

    # Run full evaluation (uncomment when you have GT)
    # df = run_full_evaluation(test_pairs, gt_masks1, gt_masks2)

    # Quick demo run (no GT)
    mask1, mask2, _ = zero_shot_co_segment("1.jpeg", "2.jpeg", top_k=10)
    print("Demo completed – check results/ folder for high-res figures!")
