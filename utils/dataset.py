# utils/dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir):
        self.frames = []
        self.labels = []

        # Add normal frames (label = 0)
        normal_dir = os.path.join(root_dir, "normal")
        if os.path.exists(normal_dir):
            for f in os.listdir(normal_dir):
                if f.endswith(".npy"):
                    self.frames.append(os.path.join(normal_dir, f))
                    self.labels.append(0)

        # Add anomaly frames (label = 1)
        anomaly_dir = os.path.join(root_dir, "anomaly")
        if os.path.exists(anomaly_dir):
            for f in os.listdir(anomaly_dir):
                if f.endswith(".npy"):
                    self.frames.append(os.path.join(anomaly_dir, f))
                    self.labels.append(1)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path = self.frames[idx]
        frame = np.load(frame_path)  # shape: (224, 224, 3)
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  # → (3, 224, 224)
        label = self.labels[idx]
        return frame, label
    
    # utils/dataset.py (add this at the end)

import torchvision.transforms as transforms

class AugmentedNormalDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.frames = []
        normal_dir = os.path.join(root_dir, "normal")
        for f in os.listdir(normal_dir):
            if f.endswith(".npy"):
                self.frames.append(os.path.join(normal_dir, f))
        
        # Define augmentation (only for training!)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        # Load original frame (H, W, 3) in [0,1]
        frame = np.load(self.frames[idx])
        frame_uint8 = (frame * 255).astype(np.uint8)  # convert to uint8 for PIL
        
        # Apply augmentation → returns tensor in [0,1]
        augmented = self.transform(frame_uint8)
        return augmented, augmented  # input = target   