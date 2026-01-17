# 02_extract_frames.py
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def extract_tif_frames(input_folder, output_folder, label, max_videos=5):
    """Extract .tif frames from UCSDped2 folders and save as .npy"""
    os.makedirs(output_folder, exist_ok=True)
    
    video_folders = [f for f in os.listdir(input_folder) if f.startswith('Train') or f.startswith('Test')]
    video_folders = sorted(video_folders)[:max_videos]  # use first N videos

    frame_count = 0
    for vid_folder in tqdm(video_folders, desc=f"Processing {label} videos"):
        vid_path = os.path.join(input_folder, vid_folder)
        frames = [f for f in os.listdir(vid_path) if f.endswith('.tif')]
        frames = sorted(frames)

        for frame_file in frames:
            # Load .tif image
            img_path = os.path.join(vid_path, frame_file)
            img = Image.open(img_path)
            img = np.array(img)  # shape: (H, W) → grayscale

            # Convert grayscale to RGB (3 channels)
            img_rgb = np.stack([img, img, img], axis=-1)  # (H, W, 3)

            # Resize to 224x224
            img_resized = cv2.resize(img_rgb, (224, 224))

            # Normalize to [0,1]
            img_normalized = img_resized.astype(np.float32) / 255.0

            # Save as .npy
            save_name = f"{label}_{vid_folder}_{frame_file.replace('.tif', '.npy')}"
            np.save(os.path.join(output_folder, save_name), img_normalized)
            frame_count += 1

    print(f"✅ Saved {frame_count} {label} frames to {output_folder}")

# --- MAIN ---
if __name__ == "__main__":
    # Training = all normal (from Train/)
    extract_tif_frames(
        input_folder="data/UCSDped2/Train",
        output_folder="frames/train/normal",
        label="normal",
        max_videos=8  # use 8 normal videos
    )

    # Testing = treat as anomaly (even though some are normal, we'll assume anomalies exist)
    extract_tif_frames(
        input_folder="data/UCSDped2/Test",
        output_folder="frames/train/anomaly",
        label="anomaly",
        max_videos=4  # use 4 test videos as "anomaly"
    )

    print("🎉 Frame extraction complete!")