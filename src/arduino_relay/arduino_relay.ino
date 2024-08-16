// Define the pin for output
const int outputRelay1 = 8;
const int outputRelay2 = 10;

// Define the interval for checking serial input
unsigned long interval = 50;  // In milliseconds

unsigned long long previousMillis = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Set pin relay as output
  pinMode(outputRelay1, OUTPUT);
  pinMode(outputRelay2, OUTPUT);

  // Set pin relay lo low
  digitalWrite(outputRelay1, HIGH);
  digitalWrite(outputRelay2, HIGH);
  delay(100);
  digitalWrite(outputRelay1, LOW);
  digitalWrite(outputRelay2, LOW);
  delay(50);
  digitalWrite(outputRelay1, HIGH);
  digitalWrite(outputRelay2, HIGH);
  delay(100);
  digitalWrite(outputRelay1, LOW);
  digitalWrite(outputRelay2, LOW);
  delay(200);
  digitalWrite(outputRelay1, HIGH);
  digitalWrite(outputRelay2, HIGH);
  delay(500);
  digitalWrite(outputRelay1, LOW);
  digitalWrite(outputRelay2, LOW);
}

void loop() {
  // Current time
  unsigned long long currentMillis = millis();

  // Check if it's time to check serial input
  if (currentMillis - previousMillis >= interval) {
    // Save the last time something was done
    previousMillis = currentMillis;

    // Check if serial data is available to read
    if (Serial.available() > 0) {
      // Read the incoming data
      char incomingData = Serial.read();

      // Process the received data
      switch (incomingData) {
        case '0':
          // Set pin 8 high
          digitalWrite(outputRelay1, HIGH);
          break;
        case '1':
          // Set pin 10 high
          digitalWrite(outputRelay2, HIGH);
          break;
        default:
          // Set pin 8 and 10 low
          digitalWrite(outputRelay1, LOW);
          digitalWrite(outputRelay2, LOW);
          break;
      }
    } else {
      // If no data available
      digitalWrite(outputRelay1, LOW);
      digitalWrite(outputRelay2, LOW);
    }
  }
}