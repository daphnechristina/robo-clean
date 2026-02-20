void setup() {
#define MIST_LED_PIN 5        // Mist LED on Pin 5
#define PLASMA_LED_PIN 6      // Plasma LED on Pin 6
#define UV_LED_PIN 7          // UV LED on Pin 7
#define FAN_PIN 9             // Relay for DC Fan (Exhaust) on Pin 9
#define SENSOR_PIN A0         // MQ-135 Sensor on A0

#define LED_ON_TIME 5000      // 5 seconds (milliseconds)
#define FAN_ON_TIME 5000      // 5 seconds (milliseconds)
#define SENSOR_MONITOR_TIME 5000 // 5 seconds monitoring window

String inputCommand = "";
bool systemActive = false;
int currentLedPin = 0;
unsigned long startTime = 0;
int systemState = 0;  // 0=idle, 1=led_on, 2=fan_on, 3=sensor_monitor
int fanRunCount = 0;  // Tracks how many times the fan has run per sequence

void setup() {
  // Initialize Serial communication
  Serial.begin(9600);
  
  // Initialize LED pins as outputs
  pinMode(MIST_LED_PIN, OUTPUT);
  pinMode(PLASMA_LED_PIN, OUTPUT);
  pinMode(UV_LED_PIN, OUTPUT);
  
  // Initialize Fan Relay pin as output
  pinMode(FAN_PIN, OUTPUT);
  
  // Turn everything OFF initially
  digitalWrite(MIST_LED_PIN, LOW);
  digitalWrite(PLASMA_LED_PIN, LOW);
  digitalWrite(UV_LED_PIN, LOW);
  
  // ACTIVE-LOW RELAY FIX: HIGH means OFF
  digitalWrite(FAN_PIN, HIGH); 
  
  // Print welcome message
  printWelcome();
}

void loop() {
  // Check for serial input
  if (Serial.available() > 0) {
    char input = Serial.read();
    
    if (input == '1') {
      Serial.println("\n✅ UV LED Selected (Pin 7)");
      startSterilizationSequence(UV_LED_PIN, "UV");
    }
    else if (input == '2') {
      Serial.println("\n✅ Plasma LED Selected (Pin 6)");
      startSterilizationSequence(PLASMA_LED_PIN, "Plasma");
    }
    else if (input == '3') {
      Serial.println("\n✅ Mist LED Selected (Pin 5)");
      startSterilizationSequence(MIST_LED_PIN, "Mist");
    }
    else if (input == 'q' || input == 'Q') {
      Serial.println("\n🛑 Emergency Stop - All OFF");
      stopAll();
      systemActive = false;
    }
  }
  
  // Handle active sequence
  if (systemActive) {
    handleSequence();
  }
  
  delay(100);  // Small delay to prevent overwhelming serial
}

void startSterilizationSequence(int ledPin, String ledName) {
  if (systemActive) {
    Serial.println("⚠️  System already running! Wait for completion.");
    return;
  }
  
  currentLedPin = ledPin;
  systemActive = true;
  systemState = 1;      // Start with LED on
  fanRunCount = 0;      // Reset the fan counter for the new sequence
  startTime = millis();
  
  // Turn on the selected LED
  digitalWrite(currentLedPin, HIGH);
  
  Serial.print("\n🚀 STERILIZATION SEQUENCE STARTED\n");
  Serial.print("   LED: ");
  Serial.println(ledName);
  Serial.println("   ┌─ Turning ON " + ledName + " LED (5s)");
  Serial.println("   ├─ Then: Fan ON via Relay (5s)");
  Serial.println("   └─ Then: Monitor MQ-135 for 5s (Max 2 fan cycles)");
}

void handleSequence() {
  unsigned long elapsed = millis() - startTime;
  
  // STAGE 1: LED ON
  if (systemState == 1) {
    if (elapsed < LED_ON_TIME) {
      digitalWrite(currentLedPin, HIGH);
      
      if (elapsed % 1000 < 100) { 
        Serial.print("   ⏳ LED ON - ");
        Serial.print((LED_ON_TIME - elapsed) / 1000);
        Serial.println("s remaining");
        delay(101); 
      }
    } 
    else {
      digitalWrite(currentLedPin, LOW);
      systemState = 2;
      startTime = millis(); 
      Serial.println("   ✅ LED OFF - Activating Fan Relay");
    }
  }
  
  // STAGE 2: FAN ON 
  else if (systemState == 2) {
    if (elapsed < FAN_ON_TIME) {
      
      // ACTIVE-LOW RELAY FIX: Send LOW to turn the fan ON
      digitalWrite(FAN_PIN, LOW); 
      
      if (elapsed % 1000 < 100) {
        Serial.print("   🔄 Fan ON (Relay Active) - ");
        Serial.print((FAN_ON_TIME - elapsed) / 1000);
        Serial.println("s remaining");
        delay(101);
      }
    } 
    else {
      // ACTIVE-LOW RELAY FIX: Send HIGH to turn the fan OFF
      digitalWrite(FAN_PIN, HIGH); 
      fanRunCount++; 
      
      if (fanRunCount >= 2) {
        Serial.println("   ⚠️ Max fan cycles (2) reached. Ending sequence.");
        stopAll();
        systemActive = false;
        systemState = 0;
        Serial.println("\n✨ SEQUENCE COMPLETE");
        printMenu();
      } 
      else {
        systemState = 3;
        startTime = millis(); 
        Serial.println("   ✅ Fan cycle finished - Monitoring Air Quality for 5s...");
      }
    }
  }
  
  // STAGE 3: SENSOR MONITORING (5-Second Window)
  else if (systemState == 3) {
    if (elapsed < SENSOR_MONITOR_TIME) {
      int sensorValue = analogRead(SENSOR_PIN);
      
      if (elapsed % 1000 < 100) {
        Serial.print("   📊 Monitoring MQ-135 (");
        Serial.print((SENSOR_MONITOR_TIME - elapsed) / 1000);
        Serial.print("s left) - Raw Value: ");
        Serial.println(sensorValue);
        delay(101);
      }
      
      if (sensorValue > 100) {
        Serial.println("   🚨 Air quality > 100 detected! Running exhaust fan a final time...");
        systemState = 2;        
        startTime = millis();   
      }
    } 
    else {
      Serial.println("   ✅ Air is clear (Stayed <= 100).");
      stopAll();
      systemActive = false;
      systemState = 0;
      Serial.println("\n✨ SEQUENCE COMPLETE");
      printMenu();
    }
  }
}

void stopAll() {
  digitalWrite(UV_LED_PIN, LOW);
  digitalWrite(PLASMA_LED_PIN, LOW);
  digitalWrite(MIST_LED_PIN, LOW);
  
  // ACTIVE-LOW RELAY FIX: Send HIGH to make sure fan is OFF
  digitalWrite(FAN_PIN, HIGH); 
}

void printWelcome() {
  Serial.println("\n");
  Serial.println("\n✅ System Initialized");
  Serial.println("   ├─ Pin 5: Mist LED");
  Serial.println("   ├─ Pin 6: Plasma LED");
  Serial.println("   ├─ Pin 7: UV LED");
  Serial.println("   ├─ Pin 9: Relay Module (Active-Low Logic)");
  Serial.println("   └─ A0: MQ-135 Sensor");
  
  printMenu();
}

void printMenu() {
  Serial.println("\n═══════════════════════════════════════════════════");
  Serial.println("📋 SERIAL MENU");
  Serial.println("═══════════════════════════════════════════════════");
  Serial.println("  1 → Start UV LED Sequence (Pin 7)");
  Serial.println("  2 → Start Plasma LED Sequence (Pin 6)");
  Serial.println("  3 → Start Mist LED Sequence (Pin 5)");
  Serial.println("  q → Emergency Stop (All OFF)");
  Serial.println("═══════════════════════════════════════════════════");
  Serial.println("\nWaiting for input...\n");
}
  Serial.println("SYSTEM_READY");
}