#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <Wire.h> 
#include <LiquidCrystal_I2C.h>

// ================= WIFI =================
const char* ssid = "Hackathon";
const char* password = "Rec#$Hack";
const char* serverURL = "http://172.16.58.129:5000/data";
const char* healthURL = "http://172.16.58.129:5000/health";

// ================= PINS =================
#define DHTPIN 12
#define MQPIN 35
#define FIREPIN 33
#define BUZZER 25
#define ALERT_BTN 32
#define CLK 14
#define DT 27

// ================= OBJECTS =================
DHT dht(DHTPIN, DHT11);
// LCD address 0x27 is standard; if screen stays blank, try 0x3F
LiquidCrystal_I2C lcd(0x27, 16, 2); 

// ================= STATE =================
bool serverOnline = false;
bool wasServerOnline = false;
bool localSilence = false;
String currentStatusMsg = "BOOTING";

unsigned long lastHealthCheck = 0;
unsigned long lastSensorSend = 0;
unsigned long lastEncoderTime = 0;

int localTempThreshold = 32;
int lastCLKState;

// ================= CONSTANTS =================
const int GAS_THRESHOLD = 3000;
const int TEMP_FAILSAFE_MAX = 50;
const unsigned long HEALTH_INTERVAL = 5000;
const unsigned long SENSOR_INTERVAL = 2000;
const unsigned long ENCODER_DEBOUNCE = 5;

// ================= LCD UPDATE FUNCTION =================
void updateLCD(float t, int g, bool online) {
  lcd.clear();
  
  // Row 0: Temp and Gas
  lcd.setCursor(0, 0);
  lcd.print("T:"); 
  if(isnan(t)) lcd.print("--"); else lcd.print((int)t);
  lcd.print("C G:");
  lcd.print(g);

  // Row 1: Server and Alert status
  lcd.setCursor(0, 1);
  if (!online) {
    lcd.print("OFF:"); // Server Offline
  } else {
    lcd.print("ON :"); // Server Online
  }
  lcd.print(currentStatusMsg);
}

// ================= ENCODER ISR =================
void IRAM_ATTR encoderISR() {
  unsigned long now = millis();
  if (now - lastEncoderTime > ENCODER_DEBOUNCE) {
    int clkState = digitalRead(CLK);
    if (clkState != lastCLKState) {
      if (digitalRead(DT) != clkState) localTempThreshold++;
      else localTempThreshold--;
      localTempThreshold = constrain(localTempThreshold, 20, 60);
    }
    lastCLKState = clkState;
    lastEncoderTime = now;
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  
  // FIXED: Changed lcd.init() to lcd.begin() for your library version
  lcd.begin(); 
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("ESP32 STARTING");

  dht.begin();
  pinMode(MQPIN, INPUT);
  pinMode(FIREPIN, INPUT);
  pinMode(BUZZER, OUTPUT);
  pinMode(ALERT_BTN, INPUT_PULLUP);
  pinMode(CLK, INPUT);
  pinMode(DT, INPUT);

  lastCLKState = digitalRead(CLK);
  attachInterrupt(digitalPinToInterrupt(CLK), encoderISR, CHANGE);

  WiFi.begin(ssid, password);
  lcd.setCursor(0, 1);
  lcd.print("WiFi Connect...");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  lcd.clear();
  lcd.print("WiFi Connected!");
  delay(1000);
}

// ================= LOOP =================
void loop() {
  unsigned long now = millis();

  // ---------- Silence Button ----------
  if (digitalRead(ALERT_BTN) == LOW) {
    localSilence = true;
    digitalWrite(BUZZER, LOW);
    currentStatusMsg = "SILENCED";
    delay(300);
  }

  // ---------- Health Check ----------
  if (now - lastHealthCheck > HEALTH_INTERVAL) {
    lastHealthCheck = now;
    HTTPClient http;
    http.begin(healthURL);
    int httpCode = http.GET();
    bool currentStatus = (httpCode == 200);
    http.end();

    if (currentStatus && !wasServerOnline) {
      localSilence = false;
      currentStatusMsg = "OK";
    } 
    else if (!currentStatus) {
      currentStatusMsg = "FAILSAFE";
    }

    serverOnline = currentStatus;
    wasServerOnline = currentStatus;
  }

  // ---------- Sensor Logic & LCD Refresh ----------
  if (now - lastSensorSend > SENSOR_INTERVAL) {
    lastSensorSend = now;
    float temp = dht.readTemperature();
    int gas = analogRead(MQPIN);
    int fire = digitalRead(FIREPIN);

    if (serverOnline && WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(serverURL);
      http.addHeader("Content-Type", "application/json");

      StaticJsonDocument<128> doc;
      doc["temp"] = isnan(temp) ? 0 : temp;
      doc["gas"] = gas;
      doc["fire"] = fire;

      String payload;
      serializeJson(doc, payload);

      int code = http.POST(payload);
      String response = http.getString();
      http.end();

      if (code > 0) {
        if (response.indexOf("FIRE") != -1) {
          digitalWrite(BUZZER, HIGH);
          currentStatusMsg = "!!FIRE!!";
        } else if (response.indexOf("WARNING") != -1) {
          digitalWrite(BUZZER, (millis() % 600 < 300)); // Beeping effect
          currentStatusMsg = "WARNING";
        } else {
          digitalWrite(BUZZER, LOW);
          currentStatusMsg = "NORMAL";
        }
      }
    } 
    else {
      // FAIL-SAFE MODE LOGIC (Server Offline)
      bool fireCondition = (temp > localTempThreshold) || (temp > TEMP_FAILSAFE_MAX) || (gas > GAS_THRESHOLD) || (fire == 0);
      
      if (!localSilence && fireCondition) {
        digitalWrite(BUZZER, HIGH);
        currentStatusMsg = "LOCAL FIRE";
      } else {
        digitalWrite(BUZZER, LOW);
        currentStatusMsg = "NO SRVR";
      }
    }
    
    // Update the physical display
    updateLCD(temp, gas, serverOnline);
  }
}