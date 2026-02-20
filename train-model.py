import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30

DATASET_DIR = "data"
MODEL_SAVE_PATH = "arduino/models/sterlizer-model.keras"

# Fixed: initialize as None — values set dynamically in validate_dataset()
CLASS_NAMES = None
NUM_CLASSES = None

# ──────────────────────────────────────────────
# DATASET VALIDATION
# ──────────────────────────────────────────────

def validate_dataset():
    """Check dataset folder exists and has enough images."""

    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset folder '{DATASET_DIR}' not found!")
        print("📁 Expected structure:")
        print("   data/lattice/")
        print("     UV/      ← flat/smooth implant images")
        print("     Plasma/  ← complex/porous implant images")
        print("     Mist/    ← irregular implant images")
        sys.exit(1)

    # Fixed: load class names dynamically from actual folder names
    global CLASS_NAMES, NUM_CLASSES
    CLASS_NAMES = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if len(CLASS_NAMES) == 0:
        print(f"❌ No subfolders found in '{DATASET_DIR}'!")
        print("   Create UV/, Plasma/, Mist/ folders and add images.")
        sys.exit(1)

    # Fixed: set NUM_CLASSES dynamically, not hardcoded
    NUM_CLASSES = len(CLASS_NAMES)

    print(f"\n📂 Dataset     : {DATASET_DIR}")
    print(f"🏷️  Classes found: {CLASS_NAMES}")

    total_images = 0
    for cls in CLASS_NAMES:
        cls_path = os.path.join(DATASET_DIR, cls)
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        count = len(images)
        total_images += count
        if count < 10:
            print(f"⚠️  Warning: '{cls}' only has {count} images — aim for 50+ per class")
        else:
            print(f"   ✅ {cls}: {count} images")

    print(f"   📊 Total images: {total_images}\n")

    if total_images < 10:
        print("❌ Not enough images to train. Please add more images to your dataset.")
        sys.exit(1)

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_data():
    """Load dataset from data/lattice using modern tf.keras API."""

    train_data = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True               # explicit shuffle for training
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False              # Fixed: explicit no shuffle for validation
    )

    # Normalize pixel values to [0, 1]
    normalization_layer = layers.Rescaling(1.0 / 255)

    train_data = train_data.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_data = val_data.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"✅ Data loaded successfully from {DATASET_DIR}\n")
    return train_data, val_data

# ──────────────────────────────────────────────
# MODEL ARCHITECTURE
# ──────────────────────────────────────────────

def build_model():
    """Build CNN model to classify implant geometry."""

    # Fixed: augmentation layers inlined directly — no nested Sequential
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),

        # Data augmentation — only active during training
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),

        # Block 1 — detect basic edges and textures
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 2 — detect shapes and curves
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 3 — detect complex geometry features
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 4 — high-level feature extraction
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Classify
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model

# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train_model(model, train_data, val_data):
    """Train the model with early stopping and checkpointing."""

    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("🚀 Starting training...\n")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    return history

# ──────────────────────────────────────────────
# EVALUATION & PLOTTING
# ──────────────────────────────────────────────

def plot_results(history):
    """Plot training accuracy and loss curves."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_results.png")

    # Fixed: wrapped in try/except — plt.show() crashes on headless environments
    try:
        plt.show()
    except Exception:
        print("⚠️  Could not display plot — saved to training_results.png instead")

    print("📊 Training graph saved as training_results.png")

# ──────────────────────────────────────────────
# PREDICT A SINGLE IMAGE (for testing)
# ──────────────────────────────────────────────

def predict_image(image_path):
    """Load trained model and predict sterilisation method for a given image."""

    # Fixed: guard against CLASS_NAMES being None if called standalone
    if CLASS_NAMES is None:
        print("❌ CLASS_NAMES not loaded. Run validate_dataset() before predict_image().")
        return None, None

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None, None

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ Trained model not found at '{MODEL_SAVE_PATH}'")
        print("   Please run training first.")
        return None, None

    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    print(f"\n🔬 Implant Image   : {image_path}")
    print(f"📋 Prediction Breakdown:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"   {cls:8s}: {predictions[i]*100:.1f}%")
    print(f"\n✅ Recommended Sterilisation: {predicted_class} ({confidence:.1f}% confidence)")

    return predicted_class, confidence

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # Fixed: os.makedirs moved inside main — not at module level
    os.makedirs("arduino/models", exist_ok=True)

    # Step 1 — Validate dataset and set CLASS_NAMES / NUM_CLASSES
    validate_dataset()

    # Step 2 — Load data
    train_data, val_data = load_data()

    # Step 3 — Build model
    model = build_model()

    # Step 4 — Train model
    history = train_model(model, train_data, val_data)

    # Step 5 — Plot results
    plot_results(history)

    print(f"\n✅ Model saved to : {MODEL_SAVE_PATH}")
    print("🎉 Training complete! You can now run main-control.py")

    # ADD THIS AT THE VERY END OF YOUR FILE
    print("\n--- RUNNING A TEST PREDICTION ---")
    # Replace 'test.jpg' with the path to an image you want to test
    predict_image("test.jpg")