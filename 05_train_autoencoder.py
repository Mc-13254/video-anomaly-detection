# 05_train_autoencoder.py
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from models.autoencoder import ConvAutoencoder
from utils.dataset import AugmentedNormalDataset  # we'll use this

# Load only normal frames WITH augmentation
print("📂 Loading augmented normal-only dataset...")
dataset = AugmentedNormalDataset("frames/train")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"✅ Loaded {len(dataset)} augmented normal frames")

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
print("🚀 Training Autoencoder on NORMAL data only (with augmentation)...")
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# Save
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/autoencoder.pth")
print("✅ Autoencoder saved!")