"""
============================================================
  HANDWRITTEN CHARACTER RECOGNITION
  CodeAlpha ML Internship — Task 

  Dataset: MNIST (auto-downloaded)
  Model  : Custom CNN with Batch Norm + Dropout + Grad-CAM
============================================================
"""

# ── Imports ──────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ── Constants ─────────────────────────────────────────────
IMG_SIZE    = 28
NUM_CLASSES = 10
CLASSES     = [str(i) for i in range(10)]
EPOCHS      = 15
BATCH_SIZE  = 128

PALETTE = {
    "bg":    "#0D1117",
    "panel": "#161B22",
    "acc":   "#58A6FF",
    "green": "#3FB950",
    "red":   "#F85149",
    "gold":  "#E3B341",
    "text":  "#C9D1D9",
    "muted": "#6E7681",
}


# ═══════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════

def load_data():
    print("\n[STEP 1] Loading MNIST dataset ...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize to [0, 1] and add channel dimension → (N, 28, 28, 1)
    x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
    x_test  = x_test.astype("float32")[..., np.newaxis]  / 255.0

    # One-hot encode labels
    y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_oh  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

    print(f"  Train samples : {x_train.shape[0]}")
    print(f"  Test  samples : {x_test.shape[0]}")
    print(f"  Image shape   : {x_train.shape[1:]}")

    return x_train, y_train, x_test, y_test, y_train_oh, y_test_oh


# ═══════════════════════════════════════════════════════════
# STEP 2 — DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════

def get_augmentor():
    """Mild augmentation — rotations, shifts, zoom, shear."""
    return ImageDataGenerator(
        rotation_range    = 8,
        width_shift_range = 0.08,
        height_shift_range= 0.08,
        zoom_range        = 0.08,
        shear_range       = 3.0,
    )


# ═══════════════════════════════════════════════════════════
# STEP 3 — CNN ARCHITECTURE
# ═══════════════════════════════════════════════════════════

def build_model():
    """
    3-block CNN:
      Block 1 → 32 filters  (edge detection)
      Block 2 → 64 filters  (shape composition)
      Block 3 → 128 filters (abstract features)
    Global Average Pooling → Dense head → Softmax
    """
    print("\n[STEP 2] Building CNN architecture ...")

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")

    # ─ Block 1 ─
    x = layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # ─ Block 2 ─
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.30)(x)

    # ─ Block 3 ─
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    # ─ Head ─
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(inp, out, name="HandwrittenCNN")
    model.summary()
    return model


# ═══════════════════════════════════════════════════════════
# STEP 4 — TRAINING
# ═══════════════════════════════════════════════════════════

def train_model(model, x_train, y_train_oh, x_test, y_test_oh):
    print("\n[STEP 3] Training model ...")

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    cb_list = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4,
            patience=3, min_lr=1e-6, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1
        ),
    ]

    augmentor = get_augmentor()
    aug_gen   = augmentor.flow(x_train, y_train_oh, batch_size=BATCH_SIZE)

    history = model.fit(
        aug_gen,
        steps_per_epoch = len(x_train) // BATCH_SIZE,
        epochs          = EPOCHS,
        validation_data = (x_test, y_test_oh),
        callbacks       = cb_list,
        verbose         = 1,
    )
    return history


# ═══════════════════════════════════════════════════════════
# STEP 5 — EVALUATION
# ═══════════════════════════════════════════════════════════

def evaluate_model(model, x_test, y_test, y_test_oh):
    print("\n[STEP 4] Evaluating ...")
    loss, acc = model.evaluate(x_test, y_test_oh, verbose=0)
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    return y_pred


# ═══════════════════════════════════════════════════════════
# STEP 6 — GRAD-CAM (What the CNN sees)
# ═══════════════════════════════════════════════════════════

def get_gradcam(model, img):
    """
    Grad-CAM: computes gradient of the top prediction
    with respect to the last conv layer activations.
    Returns a 28x28 heatmap.
    """
    grad_model = keras.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer("last_conv").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img[np.newaxis], training=False)
        top_class  = tf.argmax(preds[0])
        class_score= preds[:, top_class]

    grads   = tape.gradient(class_score, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam     = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1).numpy()
    cam     = np.maximum(cam, 0)
    cam     = cam / (cam.max() + 1e-8)
    cam_up  = np.array(
        tf.image.resize(cam[..., np.newaxis], (IMG_SIZE, IMG_SIZE))
    ).squeeze()
    return cam_up


# ═══════════════════════════════════════════════════════════
# STEP 7 — DASHBOARD VISUALIZATION
# ═══════════════════════════════════════════════════════════

def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor":   PALETTE["panel"],
        "axes.edgecolor":   PALETTE["muted"],
        "axes.labelcolor":  PALETTE["text"],
        "xtick.color":      PALETTE["muted"],
        "ytick.color":      PALETTE["muted"],
        "text.color":       PALETTE["text"],
        "grid.color":       "#21262D",
        "grid.alpha":       0.5,
        "font.family":      "DejaVu Sans",
        "axes.titlesize":   11,
        "axes.titleweight": "bold",
    })


def plot_dashboard(history, model, x_test, y_test, y_pred, out_dir):
    print("\n[STEP 5] Generating dashboard ...")
    set_dark_style()
    import matplotlib.cm as mpl_cm

    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor(PALETTE["bg"])

    acc_val = accuracy_score(y_test, y_pred)

    # Title
    fig.text(0.5, 0.985,
             "Handwritten Character Recognition — CNN Dashboard",
             ha="center", va="top", fontsize=18,
             fontweight="bold", color=PALETTE["acc"])
    fig.text(0.5, 0.963,
             f"CodeAlpha ML Internship · Task 3  |  MNIST  ·  "
             f"Test Accuracy: {acc_val:.4f} ({acc_val*100:.2f}%)",
             ha="center", va="top", fontsize=10, color=PALETTE["muted"])

    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    # ── 1. Accuracy Curve ──────────────────────────────
    ax1 = fig.add_axes([0.05, 0.77, 0.40, 0.15])
    ax1.set_facecolor(PALETTE["panel"])
    ax1.plot(epochs_range, history.history["accuracy"],
             color=PALETTE["green"], lw=2, label="Train")
    ax1.plot(epochs_range, history.history["val_accuracy"],
             color=PALETTE["acc"], lw=2, ls="--", label="Validation")
    ax1.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax1.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["muted"],
               labelcolor=PALETTE["text"], fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── 2. Loss Curve ──────────────────────────────────
    ax2 = fig.add_axes([0.55, 0.77, 0.40, 0.15])
    ax2.set_facecolor(PALETTE["panel"])
    ax2.plot(epochs_range, history.history["loss"],
             color=PALETTE["red"], lw=2, label="Train")
    ax2.plot(epochs_range, history.history["val_loss"],
             color=PALETTE["gold"], lw=2, ls="--", label="Validation")
    ax2.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["muted"],
               labelcolor=PALETTE["text"], fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── 3. Confusion Matrix ────────────────────────────
    ax3 = fig.add_axes([0.05, 0.50, 0.42, 0.23])
    ax3.set_facecolor(PALETTE["panel"])
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax3,
                cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.4, linecolor=PALETTE["bg"],
                cbar_kws={"shrink": 0.80})
    ax3.set_title("Confusion Matrix")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    # ── 4. Per-class Accuracy ──────────────────────────
    ax4 = fig.add_axes([0.57, 0.50, 0.38, 0.23])
    ax4.set_facecolor(PALETTE["panel"])
    cm_norm     = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    per_class   = cm_norm.diagonal()
    bar_colors  = [
        PALETTE["green"] if v >= 0.99 else
        PALETTE["gold"]  if v >= 0.97 else
        PALETTE["red"]
        for v in per_class
    ]
    bars = ax4.bar(CLASSES, per_class, color=bar_colors, alpha=0.85, zorder=3)
    for bar in bars:
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() - 0.018,
                 f"{bar.get_height():.3f}",
                 ha="center", va="top",
                 fontsize=8, color=PALETTE["bg"], fontweight="bold")
    ax4.set(title="Per-Digit Accuracy", xlabel="Digit",
            ylabel="Accuracy", ylim=[0.93, 1.005])
    ax4.grid(axis="y", alpha=0.3, zorder=0)

    # ── 5. Sample Correct Predictions ─────────────────
    fig.text(0.05, 0.485, "Correct Predictions ↓",
             color=PALETTE["green"], fontsize=9, fontweight="bold")
    correct_idx = np.where(y_pred == y_test)[0]
    rng = np.random.default_rng(42)
    sample_c = rng.choice(correct_idx, 10, replace=False)
    for k, idx in enumerate(sample_c):
        ax = fig.add_axes([0.05 + k * 0.092, 0.385, 0.082, 0.092])
        ax.set_facecolor(PALETTE["panel"])
        ax.imshow(x_test[idx].squeeze(), cmap="gray_r")
        ax.set_title(f"✓ {y_pred[idx]}",
                     color=PALETTE["green"], fontsize=9, pad=2)
        ax.axis("off")

    # ── 6. Sample Wrong Predictions ───────────────────
    fig.text(0.05, 0.375, "Incorrect Predictions ↓",
             color=PALETTE["red"], fontsize=9, fontweight="bold")
    wrong_idx = np.where(y_pred != y_test)[0]
    sample_w  = rng.choice(wrong_idx, 10, replace=False)
    for k, idx in enumerate(sample_w):
        ax = fig.add_axes([0.05 + k * 0.092, 0.275, 0.082, 0.092])
        ax.set_facecolor(PALETTE["panel"])
        ax.imshow(x_test[idx].squeeze(), cmap="gray_r")
        ax.set_title(f"✗ P:{y_pred[idx]} A:{y_test[idx]}",
                     color=PALETTE["red"], fontsize=7, pad=2)
        ax.axis("off")

    # ── 7. Grad-CAM Heatmaps ──────────────────────────
    fig.text(0.5, 0.258,
             "Grad-CAM Saliency — What the CNN focuses on per digit",
             ha="center", fontsize=10, color=PALETTE["gold"], fontweight="bold")
    for digit in range(10):
        sample_idx = np.where(y_test == digit)[0][0]
        img   = x_test[sample_idx]
        cam   = get_gradcam(model, img)
        overlay = np.uint8(255 * mpl_cm.jet(cam)[:, :, :3])

        ax = fig.add_axes([0.03 + digit * 0.096, 0.145, 0.085, 0.105])
        ax.set_facecolor(PALETTE["panel"])
        ax.imshow(img.squeeze(), cmap="gray", alpha=0.5)
        ax.imshow(overlay, alpha=0.5)
        ax.set_title(f"Digit {digit}",
                     color=PALETTE["gold"], fontsize=8, pad=3)
        ax.axis("off")

    # ── 8. Model Architecture Summary (text) ──────────
    arch_text = (
        "CNN Architecture\n"
        "─────────────────\n"
        "Input  28×28×1\n"
        "Conv1  32 filters × 2\n"
        "Pool   MaxPool 2×2\n"
        "Drop   0.25\n"
        "Conv2  64 filters × 2\n"
        "Pool   MaxPool 2×2\n"
        "Drop   0.30\n"
        "Conv3  128 filters\n"
        "Drop   0.35\n"
        "GAP    Global Avg Pool\n"
        "Dense  256 units\n"
        "Drop   0.40\n"
        "Out    10 (Softmax)"
    )
    ax_txt = fig.add_axes([0.05, 0.02, 0.22, 0.12])
    ax_txt.set_facecolor(PALETTE["panel"])
    ax_txt.text(0.05, 0.95, arch_text,
                transform=ax_txt.transAxes,
                va="top", ha="left",
                fontsize=8, color=PALETTE["text"],
                fontfamily="monospace")
    ax_txt.axis("off")

    # ── 9. Key Metrics Box ─────────────────────────────
    metrics_text = (
        f"Key Metrics\n"
        f"─────────────────\n"
        f"Test Accuracy  : {acc_val*100:.2f}%\n"
        f"Train Epochs   : {len(history.history['accuracy'])}\n"
        f"Augmentation   : Yes\n"
        f"Optimizer      : Adam\n"
        f"Batch Size     : {BATCH_SIZE}\n"
        f"Parameters     : ~180K\n"
        f"Dataset        : MNIST\n"
        f"Classes        : 10 digits"
    )
    ax_met = fig.add_axes([0.32, 0.02, 0.22, 0.12])
    ax_met.set_facecolor(PALETTE["panel"])
    ax_met.text(0.05, 0.95, metrics_text,
                transform=ax_met.transAxes,
                va="top", ha="left",
                fontsize=8, color=PALETTE["text"],
                fontfamily="monospace")
    ax_met.axis("off")

    out_path = os.path.join(out_dir, "handwritten_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close()
    print(f"  ✓ Dashboard saved → {out_path}")


# ═══════════════════════════════════════════════════════════
# STEP 8 — SINGLE IMAGE INFERENCE
# ═══════════════════════════════════════════════════════════

def predict_single(model, img_array):
    """
    Predict a single digit image.
    Input : numpy array shape (28, 28) or (28, 28, 1), values in [0,1]
    Output: dict with predicted digit, confidence, top-3 probabilities
    """
    if img_array.ndim == 2:
        img_array = img_array[..., np.newaxis]
    probs  = model.predict(img_array[np.newaxis], verbose=0)[0]
    pred   = int(np.argmax(probs))
    conf   = float(probs[pred])
    top3   = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
    return {
        "predicted"  : pred,
        "confidence" : conf,
        "top_3"      : [(int(d), float(p)) for d, p in top3],
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 55)
    print("  HANDWRITTEN CHARACTER RECOGNITION")
    print("  CodeAlpha ML Internship — Task 3")
    print("=" * 55)

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load data
    x_train, y_train, x_test, y_test, y_train_oh, y_test_oh = load_data()

    # 2. Build model
    model = build_model()

    # 3. Train
    history = train_model(model, x_train, y_train_oh, x_test, y_test_oh)

    # 4. Evaluate
    y_pred = evaluate_model(model, x_test, y_test, y_test_oh)

    # 5. Dashboard
    plot_dashboard(history, model, x_test, y_test, y_pred, out_dir)

    # 6. Save model
    model_path = os.path.join(out_dir, "handwritten_cnn.keras")
    model.save(model_path)
    print(f"\n  ✓ Model saved → {model_path}")

    # 7. Demo prediction
    print("\n── Demo Prediction ──────────────────────────────────")
    idx    = np.random.randint(0, len(x_test))
    result = predict_single(model, x_test[idx])
    print(f"  True Label  : {y_test[idx]}")
    print(f"  Predicted   : {result['predicted']}")
    print(f"  Confidence  : {result['confidence']:.2%}")
    print(f"  Top-3       : {result['top_3']}")
    print("\n" + "=" * 55 + "\n")


if __name__ == "__main__":
    main()
