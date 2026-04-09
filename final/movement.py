"""
Motor control process.
Reads motor_left / motor_right from shared state and drives L298N via lgpio.
Runs at ~100 Hz, only updates PWM when values change.
"""

import time
import lgpio

# ── L298N pin mapping (BCM) ──
IN1, IN2, ENA = 17, 27, 18
IN3, IN4, ENB = 22, 23, 24
PWM_HZ = 100
MAX_DUTY = 100.0
UPDATE_HZ = 100


def _init_gpio():
    h = lgpio.gpiochip_open(0)
    for pin in (IN1, IN2, IN3, IN4, ENA, ENB):
        lgpio.gpio_claim_output(h, pin, 0)
    return h


def _set_motor(h, in1, in2, en, value):
    """value in [-1, 1]"""
    fwd = value >= 0
    lgpio.gpio_write(h, in1, 1 if fwd else 0)
    lgpio.gpio_write(h, in2, 0 if fwd else 1)
    duty = min(abs(value) * MAX_DUTY, MAX_DUTY)
    lgpio.tx_pwm(h, en, PWM_HZ, duty)


def motor_process(motor_left, motor_right, quit_flag):
    """Target for multiprocessing.Process."""
    h = _init_gpio()
    period = 1.0 / UPDATE_HZ
    prev_l = prev_r = None
    print("[MOTOR] started")

    try:
        while not quit_flag.value:
            l = motor_left.value
            r = motor_right.value

            # Only update hardware when values actually change
            if l != prev_l or r != prev_r:
                _set_motor(h, IN1, IN2, ENA, l)
                _set_motor(h, IN3, IN4, ENB, r)
                prev_l, prev_r = l, r

            time.sleep(period)

    except KeyboardInterrupt:
        pass
    finally:
        # Stop motors
        lgpio.tx_pwm(h, ENA, PWM_HZ, 0)
        lgpio.tx_pwm(h, ENB, PWM_HZ, 0)
        for pin in (IN1, IN2, IN3, IN4):
            lgpio.gpio_write(h, pin, 0)
        lgpio.gpiochip_close(h)
        print("[MOTOR] stopped")
