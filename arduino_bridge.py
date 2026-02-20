import serial
import serial.tools.list_ports
import time

class ArduinoBridge:
    def __init__(self, baudrate=9600):
        self.serial_connection = None
        self.baudrate = baudrate
        self.is_mock = False
        self.port = None

    def find_arduino_port(self):
        """Automatically detect the Arduino COM port."""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            # Look for common Arduino identifiers (Uno, Mega, Nano CH340)
            if any(ident in p.description for ident in ["Arduino", "CH340", "USB Serial", "Genuino"]):
                return p.device
        return None

    def connect(self):
        """Establish connection to the Arduino."""
        if self.is_mock:
            return True

        self.port = self.find_arduino_port()
        if not self.port:
            print("❌ Arduino NOT found. Check USB connection or drivers.")
            return False

        try:
            # Initialize serial with a 1-second timeout
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            # IMPORTANT: Arduino resets on connection, wait for bootloader
            time.sleep(2) 
            
            # Clear any boot-up text from the buffer
            self.serial_connection.reset_input_buffer()
            
            print(f"🔌 Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return False

    def enable_mock_mode(self):
        """Enable simulation mode (no hardware required)."""
        self.is_mock = True
        print("⚠️  Mock Mode Enabled: Hardware commands will be simulated.")

    def send_command(self, command):
        """
        Translates AI strings into the characters your .ino code expects.
        '1' = UV, '2' = PLASMA, '3' = MIST, 'q' = STOP
        """
        mapping = {
            "START_UV": "1",
            "START_PLASMA": "2",
            "START_MIST": "3",
            "STOP_ALL": "q"
        }

        arduino_char = mapping.get(command)

        if self.is_mock:
            print(f"🤖 [MOCK SEND]: AI requested '{command}', sending '{arduino_char}'")
            return True

        if self.serial_connection and self.serial_connection.is_open:
            if arduino_char:
                try:
                    # Send character as bytes
                    self.serial_connection.write(arduino_char.encode('utf-8'))
                    # Give the Arduino a tiny moment to process
                    time.sleep(0.1) 
                    return True
                except Exception as e:
                    print(f"❌ Serial Write Error: {e}")
            else:
                print(f"⚠️  Unknown command: {command}")
        else:
            print("❌ Not connected to hardware.")
        return False

    def read_response(self):
        """Optional: Reads a line of text sent back from the Arduino."""
        if self.serial_connection and self.serial_connection.is_open:
            if self.serial_connection.in_waiting > 0:
                try:
                    return self.serial_connection.readline().decode('utf-8').strip()
                except:
                    return None
        return None

    def disconnect(self):
        """Close the serial port."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("🔌 Disconnected from Arduino.")

    @property
    def is_connected(self):
        if self.is_mock: return True
        return self.serial_connection is not None and self.serial_connection.is_open