#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.initialize();
}

void loop() {
  // Read IMU data
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Send data via Serial (JSON format)
  Serial.print("{");
  Serial.print("\"ax\":"); Serial.print(ax); Serial.print(",");
  Serial.print("\"ay\":"); Serial.print(ay); Serial.print(",");
  Serial.print("\"az\":"); Serial.print(az); Serial.print(",");
  Serial.print("\"gx\":"); Serial.print(gx); Serial.print(",");
  Serial.print("\"gy\":"); Serial.print(gy); Serial.print(",");
  Serial.print("\"gz\":"); Serial.print(gz);
  Serial.println("}");

  delay(50); // 20Hz sampling
}