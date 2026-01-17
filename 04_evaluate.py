# 04_evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from models import SimpleCNN
from utils.dataset import VideoFrameDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = VideoFrameDataset("frames/train")
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Load model
model = SimpleCNN(num_classes=2).to(device)
model.load_state_dict(torch.load("saved_models/cnn_anomaly.pth", map_location=device))
model.eval()

# Predict
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Report
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Anomaly"]))