import lgpio
import time

# L298N Pins (BCM)
IN1, IN2, ENA = 17, 27, 18  # Left motor
IN3, IN4, ENB = 22, 23, 19  # Right motor

PWM_HZ = 200

h = lgpio.gpiochip_open(0)

# Claim outputs
for pin in (IN1, IN2, IN3, IN4, ENA, ENB):
    lgpio.gpio_claim_output(h, pin, 0)

def set_pwm(pin, duty):
    duty = max(0, min(100, duty))
    lgpio.tx_pwm(h, pin, PWM_HZ, duty)

def set_dir(in1, in2, forward: bool):
    lgpio.gpio_write(h, in1, 1 if forward else 0)
    lgpio.gpio_write(h, in2, 0 if forward else 1)

def set_motors(left, right):
    # left/right: -100..100
    lf = left >= 0
    rf = right >= 0

    set_dir(IN1, IN2, lf)
    set_dir(IN3, IN4, rf)

    set_pwm(ENA, abs(left))
    set_pwm(ENB, abs(right))

try:
    print("Forward")
    set_motors(100, 100)
    time.sleep(2)

    print("Turn left (right faster)")
    set_motors(20, 60)
    time.sleep(2)

    print("Turn right (left faster)")
    set_motors(60, 20)
    time.sleep(2)

    print("Stop")
    set_motors(0, 0)
    time.sleep(1)

finally:
    # Stop PWM
    lgpio.tx_pwm(h, ENA, PWM_HZ, 0)
    lgpio.tx_pwm(h, ENB, PWM_HZ, 0)

    # Close gpiochip
    lgpio.gpiochip_close(h)
