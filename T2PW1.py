#!/usr/bin/env python3
import time
import math
from gpiozero import PWMOutputDevice, DigitalOutputDevice, Button

# PIN CONFIG (BCM)
ENA = 18
ENB = 19
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

# ======== DISTANCE CALCULATION PARAMETERS ========
WHEEL_DIAMETER = 0.065   # meters
WHEEL_CIRC = math.pi * WHEEL_DIAMETER   # C = πD
PULSES_PER_REV = 20    

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

# ======== DISTANCE FUNCTIONS ========
def get_distance_left():
    return (left_count / PULSES_PER_REV) * WHEEL_CIRC

def get_distance_right():
    return (right_count / PULSES_PER_REV) * WHEEL_CIRC

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
    set_speeds(speed * (1 - angle / 100.0), speed)

def turn_right(speed=0.7, angle=60):
    set_left_dir(True)
    set_right_dir(True)
    set_speeds(speed, speed * (1 - angle / 100.0))

# UPDATED MOVE FUNCTION WITH DISTANCE OUTPUT
def move_for(fn, seconds, *args, **kwargs):
    global left_count, right_count

    left_count = 0
    right_count = 0

    start = time.time()
    fn(*args, **kwargs)
    time.sleep(seconds)
    stop()

    dt = time.time() - start

    # ======== DISTANCE CALCULATION ========
    left_distance = get_distance_left()
    right_distance = get_distance_right()

    print(
        f"Ran {fn.__name__} for {dt:.2f}s | "
        f"Encoders: L={left_count} R={right_count} | "
        f"Distance: L={left_distance:.3f}m R={right_distance:.3f}m"
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
