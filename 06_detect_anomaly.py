# 06_detect_anomaly.py
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent GUI freeze
import matplotlib.pyplot as plt
import os
import cv2  # ← ADD THIS (required for Grad-CAM)

from models.autoencoder import ConvAutoencoder
from models.gradcam import GradCAM  # ← Make sure this exists
from utils.dataset import VideoFrameDataset
from torch.utils.data import DataLoader
# --- ADD LIME ---
import skimage
from utils.lime_explainer import explain_with_lime

lime_img = explain_with_lime(model, orig, device)
plt.imsave(f"results/frame_{i}_lime.png", lime_img)

# Create results folder
os.makedirs("results", exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load("saved_models/autoencoder.pth", map_location=device))
model.eval()

# Load test data
test_dataset = VideoFrameDataset("frames/train")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

anomaly_scores = []
all_labels = []

print("🔍 Detecting anomalies...")

for i, (frame, label) in enumerate(test_loader):
    frame = frame.to(device)
    with torch.no_grad():
        reconstructed = model(frame)
        error = torch.mean((frame - reconstructed) ** 2, dim=1)
        score = error.mean().item()
        anomaly_scores.append(score)
        all_labels.append(label.item())

    # Visualize first 6 frames
    if i < 6:
        orig = frame[0].cpu().numpy().transpose(1, 2, 0)
        recon = reconstructed[0].cpu().numpy().transpose(1, 2, 0)
        err_map = error[0].cpu().numpy()

        # --- Plot Original / Reconstructed / Error ---
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(recon)
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")

        im = axes[2].imshow(err_map, cmap="hot", vmin=0, vmax=0.1)
        axes[2].set_title(f"Error Map\nScore: {score:.4f}")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2])
        plt.tight_layout()
        plt.savefig(f"results/frame_{i}_{'anomaly' if label.item() == 1 else 'normal'}.png")
        plt.close()

        # --- GRAD-CAM ---
        target_layer = model.encoder[-2]  # second-to-last layer
        cam = GradCAM(model, target_layer)
        heatmap = cam(frame)

        # Resize and overlay
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        cam_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        orig_bgr = (orig * 255).astype(np.uint8)
        orig_bgr = cv2.cvtColor(orig_bgr, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(orig_bgr, 0.6, cam_img, 0.4, 0)
        cv2.imwrite(f"results/frame_{i}_gradcam.png", overlay)

# --- Final Graph ---
plt.figure(figsize=(12, 4))
plt.plot(anomaly_scores, label="Anomaly Score", color="red")
plt.axhline(y=np.mean(anomaly_scores[:1200]), color='green', linestyle='--', label="Normal Avg")
plt.xlabel("Frame Index")
plt.ylabel("Reconstruction Error")
plt.title("Anomaly Detection: Reconstruction Error Over Time")
plt.legend()
plt.grid(True)
plt.savefig("results/anomaly_scores.png")
plt.close()

# --- Stats ---
normal_scores = [anomaly_scores[i] for i in range(len(all_labels)) if all_labels[i] == 0]
anomaly_scores_list = [anomaly_scores[i] for i in range(len(all_labels)) if all_labels[i] == 1]

print(f"\n📊 Stats:")
print(f"Normal frames (avg error): {np.mean(normal_scores):.5f}")
print(f"Anomaly frames (avg error): {np.mean(anomaly_scores_list):.5f}")