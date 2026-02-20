import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from arduino_bridge import ArduinoBridge

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

IMG_SIZE = (128, 128)
MODEL_PATH = "models/sterilizer-model.keras"
DATASET_DIR = "data"

# Time in minutes for the sterilization process
STERILIZATION_DURATION = {
    "UV": 30,      
    "PLASMA": 45,  
    "MIST": 35     
}

# ──────────────────────────────────────────────
# MAIN CONTROLLER CLASS
# ──────────────────────────────────────────────

class SterilizationChamber:
    def __init__(self, use_mock_mode=True):
        self.use_mock_mode = use_mock_mode
        self.model = None
        self.class_names = None
        self.arduino = ArduinoBridge()
        self.session_log = []

        print("🚀 Initializing Sterilization Chamber Control System...")
        
        # 1. Load AI model
        if self._load_model():
            print("✅ AI model and class names synchronized.\n")
        else:
            print("❌ Initialization failed. Please check your data/ and models/ folders.\n")
            sys.exit(1)

        # 2. Connect to Hardware
        if use_mock_mode:
            self.arduino.enable_mock_mode()
        else:
            if not self.arduino.connect():
                print("⚠️ Arduino not detected. Switching to Mock Mode for demo.")
                self.arduino.enable_mock_mode()

    def _load_model(self):
        """Load the Transfer Learning model and sync class names."""
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found at {MODEL_PATH}")
            return False

        try:
            # Load the MobileNetV2 based model
            self.model = tf.keras.models.load_model(MODEL_PATH)
            
            # Sync Class Names with the folder structure used in training
            if os.path.exists(DATASET_DIR):
                self.class_names = sorted([
                    d for d in os.listdir(DATASET_DIR)
                    if os.path.isdir(os.path.join(DATASET_DIR, d))
                ])
                return True
            else:
                print("❌ Data directory missing. Cannot verify class labels.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def predict_implant(self, image_path):
        """Predict using MobileNetV2 preprocessing."""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None

        try:
            # Load and Preprocess exactly as done in train-model.py
            img = load_img(image_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # CRITICAL: Use MobileNetV2 preprocessing instead of simple 1/255
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            predictions = self.model.predict(img_array, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(predictions[predicted_idx] * 100)

            # Map folder name to Arduino command
            method = self._map_to_method(predicted_class)

            return {
                "image_path": image_path,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "sterilization_method": method,
                "all_predictions": {
                    cls: float(pred * 100) for cls, pred in zip(self.class_names, predictions)
                }
            }
        except Exception as e:
            print(f"❌ Prediction Error: {e}")
            return None

    def _map_to_method(self, class_name):
            """Maps AI folder names to Arduino command keys."""
            mapping = {
                "metal-standard": "UV",      # Folders must match your data/ directory
                "lattice": "PLASMA",
                "solid-polymer": "MIST"
            }
            return mapping.get(class_name, "UV")

    def execute_sterilization(self, image_path, auto_confirm=False):
        """Full automated cycle synchronized with hardware."""
        print("\n" + "="*50)
        print(f"📸 ANALYZING: {os.path.basename(image_path)}")
        
        result = self.predict_implant(image_path)
        if not result: return

        print(f"🤖 AI Prediction: {result['predicted_class']} ({result['confidence']:.1f}%)")
        print(f"🧪 Recommended Method: {result['sterilization_method']}")

        if not auto_confirm:
            choice = input("\n▶️ Start Sterilization? (y/n): ").lower()
            if choice != 'y': 
                print("⏹️ Cycle aborted.")
                return

        # Hardware Control
        method = result['sterilization_method']
        print(f"\n🚀 Sending command to Arduino... (Triggering {method})")
        
        # This sends '1', '2', or '3'
        if self.arduino.send_command(f"START_{method}"):
            print("⏳ Hardware Sequence Active. Waiting for Arduino...")
            
            # MONITORING LOOP: Listen to Arduino until it finishes
            while True:
                response = self.arduino.read_response()
                if response:
                    print(f"   [Arduino]: {response}")
                    
                    # This matches the EXACT line in your .ino file
                    if "✨ SEQUENCE COMPLETE" in response:
                        break
                    
                    # If you hit an emergency stop
                    if "Emergency Stop" in response:
                        print("🛑 Hardware E-Stop detected!")
                        break
                
                time.sleep(0.1) # Don't melt the CPU while waiting
        
        print(f"\n✅ {method} Cycle Officially Complete.")

        # Logging
        log_entry = {
            "time": time.strftime("%H:%M:%S"),
            "implant": result['predicted_class'],
            "method": method,
            "confidence": f"{result['confidence']:.1f}%"
        }
        self.session_log.append(log_entry)

    def interactive_mode(self):
        print("\n🏥 STERILIZATION CHAMBER READY")
        print("Commands: [image_path], 'log', 'quit'")
        while True:
            cmd = input("\n📸 Enter Image Path: ").strip()
            if cmd.lower() == 'quit': break
            if cmd.lower() == 'log':
                print(json.dumps(self.session_log, indent=2))
                continue
            if cmd: self.execute_sterilization(cmd)

# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Change use_mock_mode=False if your Arduino is plugged in!
    chamber = SterilizationChamber(use_mock_mode=False)
    
    if len(sys.argv) > 1:
        chamber.execute_sterilization(sys.argv[1])
    else:
        chamber.interactive_mode()