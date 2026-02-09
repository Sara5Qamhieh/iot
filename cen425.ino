#include <Servo.h>

// -------------------- PINS --------------------
const int SERVO_PIN = 9;

// Bin 1 ultrasonic
const int TRIG1_PIN = 6;
const int ECHO1_PIN = 7;

// Bin 2 ultrasonic
const int TRIG2_PIN = 4;
const int ECHO2_PIN = 5;

// -------------------- SERVO SETTINGS --------------------
// "Home" is center/neutral where nothing is pushed
const int HOME_ANGLE = 90;

// Move exactly 45 degrees from HOME
const int DELTA_ANGLE = 45;
const int LEFT_ANGLE  = HOME_ANGLE - DELTA_ANGLE;   // 45° to left
const int RIGHT_ANGLE = HOME_ANGLE + DELTA_ANGLE;   // 45° to right

// How long to hold at side so it actually pushes the item
const unsigned long PUSH_HOLD_MS = 600;
const unsigned long RETURN_HOLD_MS = 450;

// -------------------- ULTRASONIC SETTINGS --------------------
const int SAMPLES = 5;

// Full threshold in cm (tune this!)
// Example: if sensor is mounted at top looking down,
// smaller distance means more full. E.g., <= 8cm means full.
const float FULL_THRESHOLD_CM = 8.0;

// How often to print fill levels
const unsigned long REPORT_PERIOD_MS = 400;
unsigned long lastReportMs = 0;

Servo sorter;

// ---------- Ultrasonic helper ----------
long readPulseMicros(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // 30ms timeout ~ 5m max (safe)
  return pulseIn(echoPin, HIGH, 30000);
}

float microsToCm(long us) {
  if (us <= 0) return -1.0f;
  return us / 58.2f;
}

// Take 5 readings and average ONLY valid ones
float averageDistanceCm(int trigPin, int echoPin) {
  float sum = 0.0;
  int valid = 0;

  for (int i = 0; i < SAMPLES; i++) {
    long us = readPulseMicros(trigPin, echoPin);
    float cm = microsToCm(us);

    if (cm > 0 && cm < 400) { // ignore out-of-range junk
      sum += cm;
      valid++;
    }
    delay(25); // small gap between pings
  }

  if (valid == 0) return -1.0f;
  return sum / valid;
}

// ---------- Servo helper ----------
void moveServoTo(int angle) {
  sorter.write(angle);
}

void pushLeft45() {
  moveServoTo(LEFT_ANGLE);
  delay(PUSH_HOLD_MS);
  moveServoTo(HOME_ANGLE);
  delay(RETURN_HOLD_MS);
}

void pushRight45() {
  moveServoTo(RIGHT_ANGLE);
  delay(PUSH_HOLD_MS);
  moveServoTo(HOME_ANGLE);
  delay(RETURN_HOLD_MS);
}

// -------------------- SETUP --------------------
void setup() {
  pinMode(TRIG1_PIN, OUTPUT);
  pinMode(ECHO1_PIN, INPUT);

  pinMode(TRIG2_PIN, OUTPUT);
  pinMode(ECHO2_PIN, INPUT);

  sorter.attach(SERVO_PIN);
  sorter.write(HOME_ANGLE);

  Serial.begin(115200);
  Serial.println("READY");
  Serial.println("Commands: L=push left 45deg, R=push right 45deg, H=home");
}

// -------------------- LOOP --------------------
void loop() {
  // 1) Commands from Raspberry Pi (or Serial Monitor)
  if (Serial.available() > 0) {
    char c = Serial.read();

    if (c == 'L') {
      pushLeft45();
      Serial.println("SORTED:L");
    } else if (c == 'R') {
      pushRight45();
      Serial.println("SORTED:R");
    } else if (c == 'H') {
      moveServoTo(HOME_ANGLE);
      Serial.println("SERVO:HOME");
    }
  }

  // 2) Periodic fill measurement for both bins
  unsigned long now = millis();
  if (now - lastReportMs >= REPORT_PERIOD_MS) {
    lastReportMs = now;

    float bin1_cm = averageDistanceCm(TRIG1_PIN, ECHO1_PIN);
    float bin2_cm = averageDistanceCm(TRIG2_PIN, ECHO2_PIN);

    bool bin1_full = (bin1_cm > 0 && bin1_cm <= FULL_THRESHOLD_CM);
    bool bin2_full = (bin2_cm > 0 && bin2_cm <= FULL_THRESHOLD_CM);

    // Print in a simple parseable format for Raspberry Pi
    Serial.print("BIN1_CM:");
    Serial.print(bin1_cm, 1);
    Serial.print(",FULL:");
    Serial.print(bin1_full ? 1 : 0);

    Serial.print(" | BIN2_CM:");
    Serial.print(bin2_cm, 1);
    Serial.print(",FULL:");
    Serial.println(bin2_full ? 1 : 0);
  }
}
