# 03_train_cnn.py
import torch
from torch.utils.data import DataLoader
from models import SimpleCNN
from utils.dataset import VideoFrameDataset
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset = VideoFrameDataset("frames/train")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"✅ Loaded {len(dataset)} frames for training")

# Create model
model = SimpleCNN(num_classes=2).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
print("🚀 Starting training...")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/cnn_anomaly.pth")
print("✅ Model saved to 'saved_models/cnn_anomaly.pth'")