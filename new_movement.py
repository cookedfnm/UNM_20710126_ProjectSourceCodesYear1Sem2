import lgpio

# ----------------- L298N pins (BCM) -----------------
IN1, IN2, ENA = 17, 27, 18
IN3, IN4, ENB = 22, 23, 19
PWM_HZ = 100
MAX_DUTY = 100.0

hchip = lgpio.gpiochip_open(0)

for pin in (IN1, IN2, IN3, IN4, ENA, ENB):
    lgpio.gpio_claim_output(hchip, pin, 0)

def _set_dir(in1, in2, forward: bool):
    lgpio.gpio_write(hchip, in1, 1 if forward else 0)
    lgpio.gpio_write(hchip, in2, 0 if forward else 1)

def _set_pwm(pin, duty):
    duty = float(max(0.0, min(MAX_DUTY, duty)))
    lgpio.tx_pwm(hchip, pin, PWM_HZ, duty)

def move(left: float, right: float):
    """
    left/right expected in [-1, +1]
    sign = direction, magnitude = speed
    """
    left = float(max(-1.0, min(1.0, left)))
    right = float(max(-1.0, min(1.0, right)))

    _set_dir(IN1, IN2, left >= 0)
    _set_dir(IN3, IN4, right >= 0)

    _set_pwm(ENA, abs(left) * MAX_DUTY)
    _set_pwm(ENB, abs(right) * MAX_DUTY)

def stop():
    move(0.0, 0.0)

def cleanup():
    try:
        stop()
        lgpio.tx_pwm(hchip, ENA, PWM_HZ, 0)
        lgpio.tx_pwm(hchip, ENB, PWM_HZ, 0)
    finally:
        lgpio.gpiochip_close(hchip)

# compatibility with your friend's shutdown line
class _PiWrapper:
    def stop(self):
        cleanup()

pi = _PiWrapper()
