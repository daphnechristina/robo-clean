import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 50 

DATASET_DIR = "data"
MODEL_SAVE_PATH = "arduino/models/sterlizer-model.keras"

CLASS_NAMES = None
NUM_CLASSES = None

# ──────────────────────────────────────────────
# DATASET VALIDATION
# ──────────────────────────────────────────────

def validate_dataset():
    """Check dataset folder exists and has enough images."""
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset folder '{DATASET_DIR}' not found!")
        print("📁 Ensure your structure is: robo-clean/data/[class_folders]")
        sys.exit(1)

    global CLASS_NAMES, NUM_CLASSES
    # Dynamically pick up folder names (UV, Plasma, Mist, etc.)
    CLASS_NAMES = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if len(CLASS_NAMES) == 0:
        print(f"❌ No subfolders found in '{DATASET_DIR}'!")
        sys.exit(1)

    NUM_CLASSES = len(CLASS_NAMES)
    print(f"📂 Dataset Found: {DATASET_DIR}")
    print(f"🏷️  Classes detected: {CLASS_NAMES}")

# ──────────────────────────────────────────────
# DATA LOADING & AUGMENTATION
# ──────────────────────────────────────────────

def load_data():
    """Load and augment dataset using MobileNetV2 preprocessing."""
    
    # Augmentation to help the model generalize from small datasets
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # MobileNetV2 expects specific scaling, not just 1/255
    def preprocess(x, y):
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        return x, y

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# ──────────────────────────────────────────────
# MODEL ARCHITECTURE (TRANSFER LEARNING)
# ──────────────────────────────────────────────

def build_model():
    """Build Transfer Learning model foundation."""
    
    # Pre-trained base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base so we don't destroy pre-learned features
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ──────────────────────────────────────────────
# PREDICTION HELPER
# ──────────────────────────────────────────────

def predict_image(image_path):
    """Load trained model and predict for a specific image."""
    if not os.path.exists(image_path):
        print(f"⚠️  Test image not found at: {image_path}")
        return

    print(f"\n🔬 Running Test Prediction on: {image_path}...")
    
    # Load model and prepare image
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    
    print(f"✅ Result: {CLASS_NAMES[predicted_idx]} ({predictions[predicted_idx]*100:.1f}%)")

# ──────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # 1. Prepare
    validate_dataset()
    train_ds, val_ds = load_data()

    # 2. Build & Train
    model = build_model()
    
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    ]

    print("\n🚀 Starting Training...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # 3. Test (Make sure 'test.png' exists in your folder!)
    predict_image("test2.png")