import serial
import time
import json
from enum import Enum

# ──────────────────────────────────────────────
# STERILIZATION METHODS
# ──────────────────────────────────────────────

class SterilizationMethod(Enum):
    """Enum for sterilization methods and their Arduino command codes."""
    UV = "UV"                  # For metal-standard implants
    PLASMA = "PLASMA"          # For lattice (3D printed porous)
    MIST = "MIST"              # For solid-polymer implants
    NONE = "NONE"              # No sterilization (test/idle)


class ArduinoBridge:
    """
    Serial communication bridge to Arduino.
    Handles command transmission and response parsing.
    
    Usage:
        bridge = ArduinoBridge(port="/dev/ttyUSB0", baudrate=9600)
        bridge.start_sterilization("UV", duration_minutes=30)
        bridge.rotate_platform(rpm=10)
    """

    def __init__(self, port="/dev/ttyUSB0", baudrate=9600, timeout=2):
        """
        Initialize serial connection to Arduino.
        
        Args:
            port (str): Serial port (e.g., "/dev/ttyUSB0" on Linux/Mac, "COM3" on Windows)
            baudrate (int): Baud rate (must match Arduino code — typically 9600)
            timeout (float): Serial read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.is_connected = False

    def connect(self):
        """Establish serial connection to Arduino."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)  # Arduino resets on serial connect — allow time to boot
            self.is_connected = True
            print(f"✅ Connected to Arduino on {self.port} @ {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"❌ Failed to connect to Arduino: {e}")
            print(f"   Available ports: Check your system's Device Manager or run 'ls /dev/tty*'")
            self.is_connected = False
            return False

    def disconnect(self):
        """Close serial connection."""
        if self.ser and self.is_connected:
            self.ser.close()
            self.is_connected = False
            print("✅ Disconnected from Arduino")

    def send_command(self, command):
        """
        Send a command string to Arduino.
        
        Args:
            command (str): Command to send (e.g., "START_UV,30")
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.is_connected or not self.ser:
            print(f"❌ Not connected to Arduino. Call connect() first.")
            return False

        try:
            self.ser.write((command + "\n").encode())
            print(f"📤 Sent to Arduino: {command}")
            return True
        except Exception as e:
            print(f"❌ Error sending command: {e}")
            return False

    def read_response(self, wait_time=1):
        """
        Read response from Arduino.
        
        Args:
            wait_time (float): Time to wait for response (seconds)
        
        Returns:
            str: Response from Arduino, or None if timeout
        """
        if not self.is_connected or not self.ser:
            return None

        try:
            time.sleep(wait_time)
            if self.ser.in_waiting > 0:
                response = self.ser.readline().decode().strip()
                print(f"📥 Received from Arduino: {response}")
                return response
            return None
        except Exception as e:
            print(f"❌ Error reading response: {e}")
            return None

    # ──────────────────────────────────────────────
    # HIGH-LEVEL STERILIZATION COMMANDS
    # ──────────────────────────────────────────────

    def start_sterilization(self, method, duration_minutes=30):
        """
        Start sterilization cycle.
        
        Args:
            method (str): Sterilization method ("UV", "PLASMA", or "MIST")
            duration_minutes (int): Duration in minutes (default 30)
        
        Returns:
            bool: True if command sent successfully
        """
        if method not in ["UV", "PLASMA", "MIST"]:
            print(f"❌ Invalid sterilization method: {method}")
            return False

        command = f"START_{method},{duration_minutes}"
        return self.send_command(command)

    def stop_sterilization(self):
        """Stop ongoing sterilization cycle."""
        return self.send_command("STOP_STERILIZATION")

    # NOTE: Platform rotation is controlled manually via separate 9V battery
    # No Arduino control of motor speed needed

    def get_chamber_status(self):
        """
        Query Arduino for chamber status (temperature, humidity, etc.).
        
        Returns:
            dict: Parsed JSON status, or None if no response
        """
        self.send_command("GET_STATUS")
        response = self.read_response(wait_time=0.5)
        
        if response:
            try:
                # Expect Arduino to respond with JSON like:
                # {"status": "idle", "temp": 25.5, "humidity": 45}
                status = json.loads(response)
                return status
            except json.JSONDecodeError:
                print(f"⚠️  Could not parse status as JSON: {response}")
                return None
        return None

    def emergency_stop(self):
        """Emergency stop — halts all operations immediately."""
        return self.send_command("EMERGENCY_STOP")

    # ──────────────────────────────────────────────
    # MOCK MODE (for testing without hardware)
    # ──────────────────────────────────────────────

    def enable_mock_mode(self):
        """
        Enable mock mode — simulates Arduino responses for testing.
        Useful during hackathon when hardware isn't fully connected.
        """
        self.mock_mode = True
        print("🔧 Mock mode enabled — Arduino responses will be simulated")

    def disable_mock_mode(self):
        """Disable mock mode and use real serial communication."""
        self.mock_mode = False
        print("🔧 Mock mode disabled — using real Arduino")

    def mock_send_command(self, command):
        """
        Simulate Arduino response for testing purposes.
        
        Args:
            command (str): Command sent (used to determine response)
        
        Returns:
            str: Simulated response
        """
        print(f"📤 [MOCK] Sent to Arduino: {command}")
        time.sleep(0.5)  # Simulate processing delay
        
        if "START_" in command:
            response = "ACK:STERILIZATION_STARTED"
        elif "STOP" in command:
            response = "ACK:STERILIZATION_STOPPED"
        elif "STATUS" in command:
            response = json.dumps({
                "status": "running",
                "method": "UV",
                "temp_c": 22.5,
                "humidity": 45,
                "progress_percent": 50
            })
        else:
            response = "ACK:COMMAND_RECEIVED"
        
        print(f"📥 [MOCK] Received from Arduino: {response}")
        return response


# ──────────────────────────────────────────────
# EXAMPLE USAGE (for testing)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test the Arduino bridge in mock mode.
    Uncomment lines below to test real serial communication.
    """

    # ✅ MOCK MODE TEST (no hardware needed)
    print("=== TESTING ARDUINO BRIDGE IN MOCK MODE ===\n")
    
    bridge = ArduinoBridge()
    bridge.enable_mock_mode()
    
    # Test commands
    bridge.mock_send_command("START_UV,30")
    bridge.mock_send_command("GET_STATUS")
    bridge.mock_send_command("STOP_STERILIZATION")
    
    print("\n✅ Mock mode test complete!")

    # ❌ UNCOMMENT BELOW WHEN HARDWARE IS READY
    # bridge = ArduinoBridge(port="/dev/ttyUSB0")  # Change port for Windows: "COM3"
    # if bridge.connect():
    #     bridge.start_sterilization("UV", duration_minutes=30)
    #     time.sleep(2)
    #     status = bridge.get_chamber_status()
    #     bridge.stop_sterilization()
    #     bridge.disconnect()