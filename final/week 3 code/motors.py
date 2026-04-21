import RPi.GPIO as GPIO
from . import config as cfg

pwm_l = None
pwm_r = None

def gpio_setup():
    global pwm_l, pwm_r
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (cfg.ENA, cfg.IN1, cfg.IN2, cfg.ENB, cfg.IN3, cfg.IN4):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, False)
    pwm_l = GPIO.PWM(cfg.ENA, cfg.PWM_FREQ)
    pwm_l.start(0)
    pwm_r = GPIO.PWM(cfg.ENB, cfg.PWM_FREQ)
    pwm_r.start(0)

def set_motors(left, right):
    left = max(-100.0, min(100.0, float(left)))
    right = max(-100.0, min(100.0, float(right)))

    GPIO.output(cfg.IN1, left >= 0)
    GPIO.output(cfg.IN2, left < 0)

    GPIO.output(cfg.IN3, right >= 0)
    GPIO.output(cfg.IN4, right < 0)

    pwm_l.ChangeDutyCycle(abs(left))
    pwm_r.ChangeDutyCycle(abs(right))

def stop():
    for pin in (cfg.IN1, cfg.IN2, cfg.IN3, cfg.IN4):
        GPIO.output(pin, False)
    pwm_l.ChangeDutyCycle(0)
    pwm_r.ChangeDutyCycle(0)

def cleanup():
    stop()
    GPIO.cleanup()
