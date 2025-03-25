#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// Complementary filter parameters
float alpha = 0.98;
float pitch = 0, roll = 0;

// Timing control
unsigned long last_time = 0;
const float dt = 0.01; // 100Hz

void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.initialize();
}

void loop() {
  if(micros() - last_time < 10000) return; // 100Hz
  last_time = micros();

  // Read raw data
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Convert to physical units
  float accel[] = {ax/16384.0f, ay/16384.0f, az/16384.0f};
  float gyro[] = {gx/131.0f, gy/131.0f, gz/131.0f};

  // Complementary filter for orientation
  float pitch_acc = atan2(-accel[0], sqrt(accel[1]*accel[1] + accel[2]*accel[2]));
  float roll_acc = atan2(accel[1], accel[2]);
  
  pitch = alpha*(pitch + gyro[1]*dt) + (1-alpha)*pitch_acc;
  roll = alpha*(roll + gyro[0]*dt) + (1-alpha)*roll_acc;

  // Remove gravity components
  float gravity[] = {
    sin(pitch),
    -sin(roll)*cos(pitch),
    cos(roll)*cos(pitch)
  };
  
  // Calculate linear acceleration (world frame)
  float linear_x = accel[0] - gravity[0];
  float linear_y = accel[1] - gravity[1];

  Serial.print("{\"lin_x\":");
  Serial.print(linear_x);
  Serial.print(",\"lin_y\":");
  Serial.print(linear_y);
  Serial.println("}");
}