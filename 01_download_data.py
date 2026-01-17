# 01_download_data.py
import os
import urllib.request
import tarfile

# Create data folder
os.makedirs("data", exist_ok=True)

print("📥 Downloading ShanghaiTech dataset...")
url = "https://github.com/andrewliao11/AnomalyDetectionCVPR2013/releases/download/v1.0/ShanghaiTech.tar.gz"
urllib.request.urlretrieve(url, "data/ShanghaiTech.tar.gz")

print("📦 Extracting...")
with tarfile.open("data/ShanghaiTech.tar.gz", "r:gz") as tar:
    tar.extractall(path="data")

# Clean up .tar.gz file
os.remove("data/ShanghaiTech.tar.gz")

print("✅ Done! Dataset saved in 'data/ShanghaiTech/'")