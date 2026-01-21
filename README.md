#  Video Anomaly Detection
[![Demo Video]](https://github.com/Mc-13254/video-anomaly-detection/blob/main/final_demo_with_subtitles.mp4)

A real-time unsupervised anomaly detection system that explains its decisions using Grad-CAM and LIME.

![Anomaly Detection Example](results/anomaly_scores.png)

A **real-world unsupervised video anomaly detection system** built with PyTorch.  
Trained only on **normal data**, detects anomalies via **reconstruction error**, and explains decisions using **Grad-CAM**.

##  Dataset
- **UCSDped2**: Pedestrian surveillance videos with anomalies (e.g., cars, cyclists in pedestrian zones)
- Train set: 16 normal videos
- Test set: 12 videos with ground-truth anomalies

##  Method
1. **Autoencoder** trained only on normal frames
2. At test time, high reconstruction error → anomaly
3. **Grad-CAM** highlights regions causing high error
4. **Data augmentation** for robustness

##  Results
- Clear separation between normal and anomaly reconstruction error
- Visual explanations show **where** anomalies occur
- No need for anomaly labels during training

##  Sample Outputs
| Original | Reconstructed | Error Map | Grad-CAM |
|----------|---------------|-----------|----------|
| ![Original](results/frame_0_normal.png) | ![Recon](results/frame_0_normal.png) | ![Error](results/frame_0_normal.png) | ![Grad-CAM](results/frame_0_gradcam.png) |

> *(Note: Actual images will appear after running the code)*

##  How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Prepare data
Place UCSDped2 dataset in data/UCSDped2/
Run frame extraction:python 02_extract_frames.py

### 3. Train autoencoder
python 05_train_autoencoder.py

### 4. Detect anomalies + generate visuals
python 06_detect_anomaly.py

### 5. View results
Check the results/ folder for:
anomaly_scores.png → timeline of anomaly scores
frame_X_*.png → visual explanations

### Project Structure

video-anomaly-detection/
├── data/               # Raw dataset (not uploaded)
├── frames/             # Extracted frames (not uploaded)
├── models/             # Autoencoder, Grad-CAM
├── utils/              # Datasets, XAI tools
├── results/            # Output images & graphs
├── 02_extract_frames.py
├── 05_train_autoencoder.py
├── 06_detect_anomaly.py
├── requirements.txt
├── .gitignore
└── README.md

### Features Implemented
 - Full EDA & reporting
 - Data normalization & resizing
 - Baseline model (autoencoder)
 - Data augmentation
 - Evaluation with graphs
 - Explainability (Grad-CAM)
 - Real-world unsupervised anomaly detection