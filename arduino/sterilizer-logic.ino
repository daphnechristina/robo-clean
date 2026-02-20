/*
  ╔═══════════════════════════════════════════════════════════════╗
  ║     STERILIZATION CHAMBER - ARDUINO FIRMWARE                  ║
  ║     Controls: UV/Plasma/Mist Relays and Sensors               ║
  ║                                                               ║
  ║     NOTE: Platform rotation controlled manually via 9V battery║
  ║           Arduino does NOT control the motor                  ║
  ╚═══════════════════════════════════════════════════════════════╝

  PIN CONFIGURATION:
  ─────────────────
  PIN 5  → UV Light relay       (HIGH = on, LOW = off)
  PIN 6  → Plasma relay         (HIGH = on, LOW = off)
  PIN 7  → H2O2 Mist relay      (HIGH = on, LOW = off)
  PIN 13 → Status LED           (optional)
  A0     → Temperature sensor   (optional)
  A1     → Humidity sensor      (optional)

  COMMAND PROTOCOL (from Python):
  ───────────────────────────────
  START_UV,30         → Start UV sterilization for 30 minutes
  START_PLASMA,45     → Start plasma sterilization for 45 minutes
  START_MIST,35       → Start H2O2 mist for 35 minutes
  STOP_STERILIZATION  → Stop sterilization
  GET_STATUS          → Return JSON status
  EMERGENCY_STOP      → Halt all sterilization

  BAUD RATE: 9600
*/

#include <Arduino.h>

// ──────────────────────────────────────────────
// PIN DEFINITIONS
// ──────────────────────────────────────────────

#define UV_RELAY_PIN 5         // UV light control
#define PLASMA_RELAY_PIN 6     // Plasma generator control
#define MIST_RELAY_PIN 7       // H2O2 mist pump control
#define STATUS_LED_PIN 13      // Status indicator LED
#define TEMP_SENSOR_PIN A0     // Temperature analog input
#define HUMIDITY_SENSOR_PIN A1 // Humidity analog input

// ──────────────────────────────────────────────
// STATE MACHINE
// ──────────────────────────────────────────────

enum SterilizationState {
  IDLE,
  STERILIZING_UV,
  STERILIZING_PLASMA,
  STERILIZING_MIST,
  STOPPED
};

// Global variables
SterilizationState currentState = IDLE;
unsigned long sterilizationStartTime = 0;
unsigned long sterilizationDurationMs = 0;
char activeMethod[20] = "";
unsigned long lastLedToggle = 0;

// ──────────────────────────────────────────────
// SETUP
// ──────────────────────────────────────────────

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize relay pins as outputs
  pinMode(UV_RELAY_PIN, OUTPUT);
  pinMode(PLASMA_RELAY_PIN, OUTPUT);
  pinMode(MIST_RELAY_PIN, OUTPUT);
  pinMode(STATUS_LED_PIN, OUTPUT);

  // Initialize all outputs to LOW (safe state)
  digitalWrite(UV_RELAY_PIN, LOW);
  digitalWrite(PLASMA_RELAY_PIN, LOW);
  digitalWrite(MIST_RELAY_PIN, LOW);
  digitalWrite(STATUS_LED_PIN, LOW);

  // Status indicator: blink to indicate startup
  digitalWrite(STATUS_LED_PIN, HIGH);
  delay(300);
  digitalWrite(STATUS_LED_PIN, LOW);
  delay(100);
  digitalWrite(STATUS_LED_PIN, HIGH);
  delay(300);
  digitalWrite(STATUS_LED_PIN, LOW);

  Serial.println("✅ Arduino Ready - Sterilization Chamber Online");
  currentState = IDLE;
}

// ──────────────────────────────────────────────
// MAIN LOOP
// ──────────────────────────────────────────────

void loop() {
  // Check for incoming commands from Python
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    handleCommand(command);
  }

  // Monitor sterilization timer
  if (currentState >= STERILIZING_UV && currentState <= STERILIZING_MIST) {
    checkSterilizationTimer();
  }

  // Update status LED indicator
  updateStatusLED();

  delay(100);  // Main loop cycle time
}

// ──────────────────────────────────────────────
// COMMAND PARSER
// ──────────────────────────────────────────────

void handleCommand(String command) {
  /*
    Parse incoming commands from Python.
    Format: "COMMAND,parameter"
  */

  Serial.print("📥 Received: ");
  Serial.println(command);

  if (command.startsWith("START_UV")) {
    handleStartSterilization(command, "UV", UV_RELAY_PIN);
  }
  else if (command.startsWith("START_PLASMA")) {
    handleStartSterilization(command, "PLASMA", PLASMA_RELAY_PIN);
  }
  else if (command.startsWith("START_MIST")) {
    handleStartSterilization(command, "MIST", MIST_RELAY_PIN);
  }
  else if (command.startsWith("STOP_STERILIZATION")) {
    stopSterilization();
    sendAck("STERILIZATION_STOPPED");
  }
  else if (command.startsWith("GET_STATUS")) {
    sendStatusJSON();
  }
  else if (command.startsWith("EMERGENCY_STOP")) {
    emergencyStop();
    sendAck("EMERGENCY_STOP_EXECUTED");
  }
  else {
    Serial.println("❌ Unknown command");
  }
}

// ──────────────────────────────────────────────
// STERILIZATION CONTROL
// ──────────────────────────────────────────────

void handleStartSterilization(String command, const char* method, int relayPin) {
  /*
    Start sterilization cycle.
    Command format: "START_UV,30" (30 = minutes)
  */

  // Parse duration from command string
  int commaIndex = command.indexOf(',');
  if (commaIndex == -1) {
    Serial.println("❌ Invalid format. Expected: START_METHOD,MINUTES");
    return;
  }

  String durationStr = command.substring(commaIndex + 1);
  int durationMinutes = durationStr.toInt();

  if (durationMinutes <= 0) {
    Serial.println("❌ Invalid duration");
    return;
  }

  // Calculate duration in milliseconds
  sterilizationDurationMs = (unsigned long)durationMinutes * 60 * 1000;
  sterilizationStartTime = millis();

  // Store active method name
  strcpy(activeMethod, method);

  // Set appropriate state
  if (strcmp(method, "UV") == 0) {
    currentState = STERILIZING_UV;
  } 
  else if (strcmp(method, "PLASMA") == 0) {
    currentState = STERILIZING_PLASMA;
  } 
  else if (strcmp(method, "MIST") == 0) {
    currentState = STERILIZING_MIST;
  }

  // Activate relay for selected method
  digitalWrite(relayPin, HIGH);

  Serial.print("🚀 Starting ");
  Serial.print(method);
  Serial.print(" sterilization for ");
  Serial.print(durationMinutes);
  Serial.println(" minutes");

  sendAck("STERILIZATION_STARTED");
}

void checkSterilizationTimer() {
  /*
    Check if sterilization time has elapsed.
    Stop when duration is reached.
  */

  unsigned long elapsedMs = millis() - sterilizationStartTime;

  if (elapsedMs >= sterilizationDurationMs) {
    stopSterilization();
    Serial.println("✅ Sterilization time completed");
  }
}

void stopSterilization() {
  /*
    Stop all sterilization immediately.
    Turn off all relay outputs.
  */

  digitalWrite(UV_RELAY_PIN, LOW);
  digitalWrite(PLASMA_RELAY_PIN, LOW);
  digitalWrite(MIST_RELAY_PIN, LOW);

  currentState = IDLE;
  activeMethod[0] = '\0';

  Serial.println("🛑 Sterilization stopped");
}

// ──────────────────────────────────────────────
// SENSOR READING
// ──────────────────────────────────────────────

float readTemperature() {
  /*
    Read temperature from analog sensor (A0).
    Converts raw 0-1023 value to 0-50°C range.
    Adjust scaling based on your actual sensor!
  */

  int rawValue = analogRead(TEMP_SENSOR_PIN);
  float celsius = (rawValue / 1023.0) * 50.0;
  return celsius;
}

int readHumidity() {
  /*
    Read humidity from analog sensor (A1).
    Converts raw 0-1023 value to 0-100% range.
  */

  int rawValue = analogRead(HUMIDITY_SENSOR_PIN);
  int humidity = (rawValue / 1023.0) * 100;
  return humidity;
}

// ──────────────────────────────────────────────
// STATUS REPORTING
// ──────────────────────────────────────────────

void sendStatusJSON() {
  /*
    Send chamber status as JSON string to Python.
    Format: {"status":"...", "method":"...", "temp_c":X, "humidity":Y, "progress_percent":Z}
  */

  float temp = readTemperature();
  int humidity = readHumidity();
  int progressPercent = 0;

  // Calculate progress percentage if sterilizing
  if (currentState >= STERILIZING_UV && currentState <= STERILIZING_MIST) {
    unsigned long elapsedMs = millis() - sterilizationStartTime;
    progressPercent = constrain((elapsedMs * 100) / sterilizationDurationMs, 0, 100);
  }

  // Build JSON response
  Serial.print("{\"status\":\"");
  
  if (currentState == IDLE) {
    Serial.print("idle");
  } 
  else if (currentState >= STERILIZING_UV && currentState <= STERILIZING_MIST) {
    Serial.print("sterilizing");
  }
  else if (currentState == STOPPED) {
    Serial.print("stopped");
  }

  Serial.print("\",\"method\":\"");
  Serial.print(activeMethod);
  Serial.print("\",\"temp_c\":");
  Serial.print(temp, 1);
  Serial.print(",\"humidity\":");
  Serial.print(humidity);
  Serial.print(",\"progress_percent\":");
  Serial.print(progressPercent);
  Serial.println("}");
}

void sendAck(const char* message) {
  /*
    Send acknowledgment message to Python.
  */

  Serial.print("ACK:");
  Serial.println(message);
}

// ──────────────────────────────────────────────
// EMERGENCY & SAFETY
// ──────────────────────────────────────────────

void emergencyStop() {
  /*
    EMERGENCY STOP - immediately halt all operations.
    Turns off all relays and resets system.
  */

  stopSterilization();
  digitalWrite(STATUS_LED_PIN, LOW);

  currentState = STOPPED;

  Serial.println("🚨 EMERGENCY STOP ACTIVATED - ALL SYSTEMS HALTED");
}

// ──────────────────────────────────────────────
// STATUS LED INDICATOR
// ──────────────────────────────────────────────

void updateStatusLED() {
  /*
    Visual indicator of system state.
    - IDLE: LED off
    - STERILIZING: Fast pulse (300ms toggle)
    - STOPPED: LED off
  */

  unsigned long currentMillis = millis();

  if (currentState == IDLE || currentState == STOPPED) {
    digitalWrite(STATUS_LED_PIN, LOW);
  }
  else if (currentState >= STERILIZING_UV && currentState <= STERILIZING_MIST) {
    // Fast pulse while sterilizing
    if (currentMillis - lastLedToggle > 300) {
      digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
      lastLedToggle = currentMillis;
    }
  }
}

// ──────────────────────────────────────────────
// END OF FIRMWARE
// ──────────────────────────────────────────────
