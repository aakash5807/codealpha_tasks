# Handwritten Character Recognition
**CodeAlpha ML Internship — Task 3**

Recognizes handwritten digits (0–9) using a custom CNN trained on the MNIST dataset.

---

## What This Project Does

- Loads the MNIST dataset (60,000 train / 10,000 test images)
- Applies data augmentation (rotation, shift, zoom, shear)
- Trains a 3-block CNN with Batch Normalization and Dropout
- Evaluates with accuracy, confusion matrix, per-class accuracy
- Generates **Grad-CAM saliency maps** showing what the model focuses on
- Saves the trained model and a full visual dashboard

---

## Project Structure

```
handwritten_recognition/
├── handwritten_cnn_complete.py   ← Main script (run this)
├── requirements.txt
└── README.md
```

After running, these are generated:
```
├── handwritten_dashboard.png     ← Full visual dashboard
└── handwritten_cnn.keras         ← Saved model
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the project
python handwritten_cnn_complete.py
```

MNIST downloads automatically (~11MB) on first run.

---

## Model Architecture

```
Input (28×28×1)
    ↓
Conv Block 1 → Conv2D(32) × 2 → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv Block 2 → Conv2D(64) × 2 → BatchNorm → MaxPool → Dropout(0.30)
    ↓
Conv Block 3 → Conv2D(128)    → BatchNorm → Dropout(0.35)
    ↓
Global Average Pooling
    ↓
Dense(256) → Dropout(0.40)
    ↓
Dense(10) + Softmax
```

**Expected accuracy: ~99.2% on test set**

---

## Key Features

| Feature | Detail |
|---|---|
| Dataset | MNIST (70,000 images, 10 classes) |
| Augmentation | Rotation ±8°, Shift 8%, Zoom 8%, Shear 3° |
| Optimizer | Adam (lr=0.001, reduced on plateau) |
| Epochs | Up to 15 (early stopping) |
| Batch Size | 128 |
| Parameters | ~180K |
| Explainability | Grad-CAM heatmaps per digit |

---

## Dashboard Output

The generated `handwritten_dashboard.png` contains:
1. Training & Validation Accuracy curve
2. Training & Validation Loss curve
3. Confusion Matrix (10×10)
4. Per-digit accuracy bar chart
5. Sample correct predictions
6. Sample wrong predictions (with true label)
7. Grad-CAM saliency maps for each digit
8. Model architecture summary
9. Key metrics summary

---

## GitHub Repository Name
`CodeAlpha_HandwrittenCharacterRecognition`
