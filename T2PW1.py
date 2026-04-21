#!/usr/bin/env python3
import time
from gpiozero import PWMOutputDevice, DigitalOutputDevice, Button

# PIN CONFIG (BCM)
ENA = 18  # LEFT enable (PWM)
ENB = 19  # RIGHT enable (PWM)
IN1 = 5
IN2 = 6
IN3 = 16
IN4 = 26
ENCL = 17
ENCR = 27

# SETTINGS
PWM_FREQ = 1000
MIN_SPEED = 0.0
MAX_SPEED = 1.0


def clamp(x, lo=MIN_SPEED, hi=MAX_SPEED):
    return max(lo, min(hi, float(x)))


# MOTOR SETUP
ena = PWMOutputDevice(ENA, frequency=PWM_FREQ, initial_value=0)
enb = PWMOutputDevice(ENB, frequency=PWM_FREQ, initial_value=0)

in1 = DigitalOutputDevice(IN1, initial_value=False)
in2 = DigitalOutputDevice(IN2, initial_value=False)
in3 = DigitalOutputDevice(IN3, initial_value=False)
in4 = DigitalOutputDevice(IN4, initial_value=False)

# ENCODERS
encL = Button(ENCL, pull_up=True, bounce_time=0.001)
encR = Button(ENCR, pull_up=True, bounce_time=0.001)

left_count = 0
right_count = 0


def _inc_left():
    global left_count
    left_count += 1


def _inc_right():
    global right_count
    right_count += 1


encL.when_pressed = _inc_left
encR.when_pressed = _inc_right


# MOTOR HELPERS
def set_left_dir(forward: bool):
    if forward:
        in1.on()
        in2.off()
    else:
        in1.off()
        in2.on()


def set_right_dir(forward: bool):
    if forward:
        in3.on()
        in4.off()
    else:
        in3.off()
        in4.on()


def set_speeds(left_speed: float, right_speed: float):
    ena.value = clamp(left_speed)
    enb.value = clamp(right_speed)


def stop(brake=False):
    set_speeds(0, 0)
    if brake:
        in1.on()
        in2.on()
        in3.on()
        in4.on()
    else:
        in1.off()
        in2.off()
        in3.off()
        in4.off()


# MOVES
def forward(speed=0.7):
    set_left_dir(True)
    set_right_dir(True)
    set_speeds(speed, speed)


def backward(speed=0.7):
    set_left_dir(False)
    set_right_dir(False)
    set_speeds(speed, speed)


def turn_left(speed=0.7, angle=60):
    set_left_dir(True)
    set_right_dir(True)
    left = speed * (1 - angle / 100.0)
    right = speed
    set_speeds(left, right)


def turn_right(speed=0.7, angle=60):
    set_left_dir(True)
    set_right_dir(True)
    left = speed
    right = speed * (1 - angle / 100.0)
    set_speeds(left, right)


def move_for(fn, seconds, *args, **kwargs):
    global left_count, right_count

    left_count = 0
    right_count = 0

    start = time.time()
    fn(*args, **kwargs)
    time.sleep(seconds)
    stop()

    dt = time.time() - start
    print(
        f"Ran {fn.__name__} for {dt:.2f}s | "
        f"Encoders: L={left_count} R={right_count} | "
        f"pulses/s: L={left_count/dt:.1f} R={right_count/dt:.1f}"
    )


# DEMO RUN
if __name__ == "__main__":
    try:
        move_for(forward, 2.0, speed=0.70)
        move_for(turn_left, 1.5, speed=0.70, angle=60)
        move_for(turn_right, 1.5, speed=0.70, angle=60)
        move_for(backward, 2.0, speed=0.60)
        stop()

    except KeyboardInterrupt:
        pass

    finally:
        stop()
        print("Done.")
