#!/usr/bin/env python3
import time
from gpiozero import PWMOutputDevice, DigitalOutputDevice

# =========================
# L298N PIN CONFIG (BCM)
# =========================

# Enable pins
ENA = 18   # Front LEFT motor
ENB = 19   # Front RIGHT motor

# Direction pins
IN1 = 5    # Front LEFT
IN2 = 6
IN3 = 16   # Front RIGHT
IN4 = 26

# Encoders (declared but NOT used)
ENC_RIGHT = 27
ENC_LEFT  = 17

# =========================
# CALIBRATION (TIME-BASED)
# =========================
TIME_FOR_360 = 1.46   # seconds for full 360° rotation
TURN_SPEED   = 1.0    # PWM duty cycle (0.0–1.0)

# =========================
# MOTOR CLASS
# =========================
class Motor:
    def __init__(self, en, in1, in2):
        self.en = PWMOutputDevice(en, frequency=2000, initial_value=0)
        self.in1 = DigitalOutputDevice(in1, initial_value=False)
        self.in2 = DigitalOutputDevice(in2, initial_value=False)

    def forward(self, speed):
        self.in1.on()
        self.in2.off()
        self.en.value = speed

    def backward(self, speed):
        self.in1.off()
        self.in2.on()
        self.en.value = speed

    def stop(self):
        self.en.value = 0
        self.in1.off()
        self.in2.off()

# =========================
# ROTATION FUNCTION
# =========================
def rotate(angle_deg, left_motor, right_motor):
    angle = abs(angle_deg)
    turn_time = (angle / 360.0) * TIME_FOR_360

    if angle == 0:
        print("SUCCESS: 0 degree rotation (no movement)")
        return

    if angle_deg > 0:
        # LEFT
        left_motor.backward(TURN_SPEED)
        right_motor.forward(TURN_SPEED)
        direction = "LEFT"
    else:
        # RIGHT
        left_motor.forward(TURN_SPEED)
        right_motor.backward(TURN_SPEED)
        direction = "RIGHT"

    time.sleep(turn_time)

    # AUTO STOP
    left_motor.stop()
    right_motor.stop()

    print(f"SUCCESS: Rotated ~{angle:.1f} degrees {direction}")

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":

    # Create motors
    front_left  = Motor(ENA, IN1, IN2)
    front_right = Motor(ENB, IN3, IN4)

    print("Angle Rotation Program")
    print("Enter angle in degrees:")
    print("(positive = LEFT, negative = RIGHT)")
    print("PRESS q TO QUIT\n")

    try:
        while True:
            user_input = input("Angle: ").strip().lower()

            if user_input == 'q':
                front_left.stop()
                front_right.stop()
                print("Motors stopped. Exiting.")
                break

            try:
                angle = float(user_input)
                rotate(angle, front_left, front_right)
            except ValueError:
                print("Invalid input. Enter a number or q.")

    except KeyboardInterrupt:
        front_left.stop()
        front_right.stop()
        print("\nEmergency stop (Ctrl+C)")
