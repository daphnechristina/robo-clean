import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from pathlib import Path
from arduino_bridge import ArduinoBridge, SterilizationMethod

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

IMG_SIZE = (128, 128)
MODEL_PATH = "arduino/models/sterlizer-model.keras"
STERILIZATION_DURATION = {
    "UV": 30,               # 30 minutes for metal
    "PLASMA": 45,           # 45 minutes for lattice (needs longer)
    "MIST": 35              # 35 minutes for polymer
}

    # NOTE: Platform rotation is controlled manually via separate 9V battery
    # Arduino does NOT control motor speed

# ──────────────────────────────────────────────
# MAIN CONTROLLER CLASS
# ──────────────────────────────────────────────

class SterilizationChamber:
    """
    Main controller for the sterilization chamber.
    Orchestrates: Image input → AI prediction → Hardware control.
    
    Workflow:
        1. Load trained AI model
        2. Accept user image input (manual scanning)
        3. Predict implant type using AI
        4. Determine sterilization method
        5. Control hardware (rotation, UV/Plasma/Mist)
        6. Log results
    """

    def __init__(self, use_mock_mode=True):
        """
        Initialize sterilization chamber controller.
        
        Args:
            use_mock_mode (bool): If True, simulate Arduino (for testing without hardware)
        """
        self.use_mock_mode = use_mock_mode
        self.model = None
        self.class_names = None
        self.arduino = ArduinoBridge()
        self.current_session = None
        self.session_log = []

        print("🚀 Initializing Sterilization Chamber Control System...")
        
        # Load AI model
        if self._load_model():
            print("✅ AI model loaded successfully\n")
        else:
            print("❌ Failed to load AI model. Check MODEL_PATH.\n")
            sys.exit(1)

        # Connect to Arduino (or enable mock mode)
        if use_mock_mode:
            self.arduino.enable_mock_mode()
        else:
            if not self.arduino.connect():
                print("⚠️  Arduino not connected. Continuing in simulation mode.")
                self.arduino.enable_mock_mode()

    def _load_model(self):
        """
        Load the trained TensorFlow model and class names.
        
        Returns:
            bool: True if model loaded successfully
        """
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found at {MODEL_PATH}")
            print("   Run train-model.py first to train the AI.")
            return False

        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            
            # Load class names from model config or infer from training data
            self._extract_class_names()
            
            print(f"📦 Model loaded from: {MODEL_PATH}")
            print(f"🏷️  Classes: {self.class_names}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def _extract_class_names(self):
        """
        Extract class names from the data directory structure.
        Should match the folders in the /data directory.
        """
        data_dir = "data"
        if os.path.exists(data_dir):
            self.class_names = sorted([
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ])
        else:
            # Fallback if data dir doesn't exist
            self.class_names = ["lattice", "solid-polymer", "metal-standard"]

    # ──────────────────────────────────────────────
    # IMAGE PREDICTION
    # ──────────────────────────────────────────────

    def predict_implant(self, image_path):
        """
        Load image and predict implant type using AI.
        
        Args:
            image_path (str): Path to implant image
        
        Returns:
            dict: {
                "predicted_class": str,
                "confidence": float,
                "all_predictions": dict,
                "sterilization_method": str
            }
        """
        # Validate image exists
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None

        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Run prediction
            predictions = self.model.predict(img_array, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(predictions[predicted_idx] * 100)

            # Map implant type to sterilization method
            sterilization_method = self._map_to_sterilization_method(predicted_class)

            # Prepare result
            result = {
                "image_path": image_path,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_predictions": {
                    cls: float(pred * 100)
                    for cls, pred in zip(self.class_names, predictions)
                },
                "sterilization_method": sterilization_method
            }

            return result
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None

    def _map_to_sterilization_method(self, implant_class):
        """
        Map implant type to sterilization method.
        
        Args:
            implant_class (str): Predicted implant class
        
        Returns:
            str: Sterilization method ("UV", "PLASMA", or "MIST")
        """
        mapping = {
            "metal-standard": "UV",       # Traditional metal → UV light
            "lattice": "PLASMA",          # 3D porous scaffold → Gas plasma
            "solid-polymer": "MIST"       # Polymer → H2O2 mist
        }
        return mapping.get(implant_class, "UV")  # Default to UV

    # ──────────────────────────────────────────────
    # STERILIZATION EXECUTION
    # ──────────────────────────────────────────────

    def execute_sterilization(self, image_path, auto_confirm=False):
        """
        Full pipeline: predict implant → execute sterilization.
        
        Args:
            image_path (str): Path to implant image
            auto_confirm (bool): Skip user confirmation and proceed automatically
        
        Returns:
            dict: Execution result with timing and status
        """
        print("\n" + "="*60)
        print("🔬 STERILIZATION CYCLE INITIATED")
        print("="*60 + "\n")

        # Step 1: Predict implant type
        print(f"📸 Analyzing image: {image_path}")
        prediction = self.predict_implant(image_path)

        if prediction is None:
            print("❌ Prediction failed. Aborting sterilization.")
            return None

        # Step 2: Display prediction results
        self._display_prediction_results(prediction)

        # Step 3: Confirm sterilization method with user
        if not auto_confirm:
            user_input = input("\n✅ Proceed with sterilization? (yes/no): ").strip().lower()
            if user_input not in ["yes", "y"]:
                print("❌ Sterilization cancelled by user.")
                return None

        # Step 4: Execute sterilization
        method = prediction["sterilization_method"]
        duration = STERILIZATION_DURATION[method]

        execution_result = self._run_sterilization_sequence(
            method=method,
            duration_minutes=duration
        )

        # Step 5: Log session
        session_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": prediction,
            "execution": execution_result
        }
        self.session_log.append(session_data)

        # Step 6: Save session log
        self._save_session_log()

        return session_data

    def _display_prediction_results(self, prediction):
        """Pretty-print prediction results."""
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"   Predicted Class: {prediction['predicted_class'].upper()}")
        print(f"   Confidence: {prediction['confidence']:.1f}%")
        print(f"\n   🏷️  Confidence Breakdown:")
        for cls, conf in prediction["all_predictions"].items():
            bar_length = int(conf / 10)
            bar = "█" * bar_length + "░" * (10 - bar_length)
            print(f"      {cls:15s} {bar} {conf:.1f}%")
        print(f"\n   💧 Sterilization Method: {prediction['sterilization_method']}")

    def _run_sterilization_sequence(self, method, duration_minutes):
        """
        Execute sterilization sequence with hardware control.
        NOTE: Platform rotation is manually controlled via separate 9V battery.
        
        Args:
            method (str): Sterilization method
            duration_minutes (int): Duration in minutes
        
        Returns:
            dict: Execution status and timing
        """
        start_time = time.time()

        print(f"\n🚀 EXECUTING STERILIZATION SEQUENCE")
        print(f"   Method: {method}")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   ⚙️  Note: Manually rotate platform using separate 9V battery\n")

        # Start sterilization
        if not self.arduino.send_command(f"START_{method},{duration_minutes}"):
            print("❌ Failed to start sterilization")
            return None

        # Wait and monitor
        print(f"⏳ Sterilization in progress... (estimated {duration_minutes} minutes)")
        print("   In production, this would monitor hardware sensors.")
        print("   For hackathon demo: simulating progress...\n")

        # Simulate progress (in real version, read actual sensor data)
        for i in range(0, 101, 20):
            time.sleep(1)  # Reduced for demo (real: longer intervals)
            bar_length = int(i / 10)
            bar = "█" * bar_length + "░" * (10 - bar_length)
            print(f"   [{bar}] {i}% complete")

        # Stop sterilization
        self.arduino.send_command("STOP_STERILIZATION")

        # Prepare result
        elapsed_time = time.time() - start_time

        result = {
            "method": method,
            "duration_minutes": duration_minutes,
            "status": "COMPLETED",
            "elapsed_seconds": int(elapsed_time),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"\n✅ STERILIZATION COMPLETE")
        print(f"   Total time: {int(elapsed_time)} seconds")
        print(f"   Method: {method}")
        print(f"   💡 Remember to manually stop rotating platform\n")

        return result

    # ──────────────────────────────────────────────
    # SESSION LOGGING
    # ──────────────────────────────────────────────

    def _save_session_log(self):
        """Save session log to JSON file for later analysis."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"sterilization_log_{timestamp}.json")

        try:
            with open(log_file, 'w') as f:
                json.dump(self.session_log, f, indent=2)
            print(f"📝 Session log saved: {log_file}")
        except Exception as e:
            print(f"⚠️  Could not save session log: {e}")

    def get_session_summary(self):
        """Return summary of current session."""
        if not self.session_log:
            return "No sessions logged yet."

        summary = f"\n📊 SESSION SUMMARY ({len(self.session_log)} cycles):\n"
        for i, session in enumerate(self.session_log, 1):
            pred = session["prediction"]
            summary += f"\n   Cycle {i}:"
            summary += f"\n      Class: {pred['predicted_class']}"
            summary += f"\n      Method: {pred['sterilization_method']}"
            summary += f"\n      Time: {session['execution']['elapsed_seconds']}s"

        return summary

    # ──────────────────────────────────────────────
    # INTERACTIVE MODE
    # ──────────────────────────────────────────────

    def interactive_mode(self):
        """
        Interactive CLI for running sterilization cycles.
        User provides image paths one at a time.
        """
        print("\n" + "="*60)
        print("🏥 STERILIZATION CHAMBER - INTERACTIVE MODE")
        print("="*60)
        print("\nEnter image paths to sterilize implants.")
        print("Type 'quit' to exit, 'log' to view session summary.\n")

        while True:
            image_path = input("\n📸 Enter image path (or command): ").strip()

            if image_path.lower() == "quit":
                print(self.get_session_summary())
                print("\n✅ Exiting interactive mode.\n")
                break
            elif image_path.lower() == "log":
                print(self.get_session_summary())
                continue
            elif not image_path:
                print("⚠️  Please enter a path or command.")
                continue

            self.execute_sterilization(image_path, auto_confirm=False)

    # ──────────────────────────────────────────────
    # BATCH MODE (for demo with multiple images)
    # ──────────────────────────────────────────────

    def batch_sterilize(self, image_directory, auto_confirm=True):
        """
        Sterilize all images in a directory.
        
        Args:
            image_directory (str): Path to directory with images
            auto_confirm (bool): Skip confirmation for each image
        """
        if not os.path.isdir(image_directory):
            print(f"❌ Directory not found: {image_directory}")
            return

        image_files = [
            f for f in os.listdir(image_directory)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            print(f"❌ No images found in {image_directory}")
            return

        print(f"\n📦 Found {len(image_files)} images. Processing batch...\n")

        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(image_directory, image_file)
            print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
            self.execute_sterilization(image_path, auto_confirm=auto_confirm)


# ──────────────────────────────────────────────
# DEMO & MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Main entry point for sterilization chamber control.
    
    Usage:
        python main-control.py                  # Interactive mode
        python main-control.py path/to/image    # Single image
        python main-control.py --batch path/    # Batch process directory
    """

    # Initialize chamber (use_mock_mode=True during hackathon)
    chamber = SterilizationChamber(use_mock_mode=True)

    # Parse command-line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--batch" and len(sys.argv) > 2:
            # Batch mode: python main-control.py --batch data/test_images/
            directory = sys.argv[2]
            chamber.batch_sterilize(directory, auto_confirm=True)
        else:
            # Single image mode: python main-control.py path/to/image.jpg
            chamber.execute_sterilization(arg, auto_confirm=False)
    else:
        # Interactive mode
        chamber.interactive_mode()

    # Close connection when done
    if chamber.arduino.is_connected:
        chamber.arduino.disconnect()