import time
import new_movement
import feed
import cv2
import numpy as np

# ── Debug overlay via monkey-patch ───────────────────────────────
_orig_get_frame = feed.get_frame_with_overlay

_debug = {
    'state':            'init',
    'entry_direction':  0.0,
    'exit_bias_value':  0.0,
    'pid':              0.0,
    'left':             0.0,
    'right':            0.0,
    'colour_lost':      0,
    'entry_raw_error':  0.0,
    'time_in_colour':   0.0,
}

def _get_frame_debug(running):
    found_line, error, active_mode = _orig_get_frame(running)

    panel = np.zeros((280, 420, 3), dtype=np.uint8)

    STATE_COLOURS = {
        'following_black':  (0,   255, 0),
        'lookahead_colour': (0,   215, 255),
        'following_colour': (0,   100, 255),
        'exit_bias':        (255, 100, 0),
        'init':             (128, 128, 128),
    }
    sc = STATE_COLOURS.get(_debug['state'], (200, 200, 200))

    def row(text, y, colour=(200, 200, 200)):
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1)

    row(f"STATE:        {_debug['state'].upper()}",                                  28,  sc)
    row(f"active_mode:  {active_mode}",                                              56)
    row(f"error:        {error:+.3f}",                                               84)
    row(f"entry_dir:    {_debug['entry_direction']:+.0f}  raw_err: {_debug['entry_raw_error']:+.3f}", 112)
    row(f"exit_bias:    {_debug['exit_bias_value']:+.2f}",                           140)
    row(f"pid:          {_debug['pid']:+.3f}",                                       168)
    row(f"L:{_debug['left']:+.2f}  R:{_debug['right']:+.2f}",                       196)
    row(f"colour_lost:  {_debug['colour_lost']}",                                    224)
    row(f"time_in_col:  {_debug['time_in_colour']:.1f}s",                            252)

    cv2.imshow("DEBUG", panel)
    return found_line, error, active_mode

feed.get_frame_with_overlay = _get_frame_debug

# ── Main loop (mirrors main.py exactly) ──────────────────────────

def clamp(v, mn, mx):
    return mn if v < mn else mx if v > mx else v

def sign(n):
    return (n > 0) - (n < 0)

BASE_SPEED     = 0.3
APPROACH_SPEED = 0.25
COLOUR_SPEED   = 0.25
EXIT_SPEED     = 0.2

KP = 0.4
KI = 0.0001
KD = 0.1

I_CLAMP     = 1.0
SEARCH_TURN = 0.35
DEADBAND    = 0.03
D_FILTER    = 0.7

COLOUR_LOST_FRAMES = 20
MIN_COLOUR_SECS    = 2.0
EXIT_BIAS_STRENGTH = 0.6

STATE_BLACK     = 'following_black'
STATE_LOOKAHEAD = 'lookahead_colour'
STATE_COLOUR    = 'following_colour'
STATE_EXIT_BIAS = 'exit_bias'

COLOUR_MODES    = ('red', 'yellow')
LOOKAHEAD_MODES = ('lookahead_red', 'lookahead_yellow')

print("DEBUG MODE — run debugmain.py")
print("  SPACE = start / pause    Q/ESC = quit")

feed.init_camera()
cv2.namedWindow("DEBUG")
cv2.moveWindow("DEBUG", 0, 500)

running    = False
state      = STATE_BLACK
loop_start = time.perf_counter()

total_error     = 0.0
last_error      = 0.0
last_derivative = 0.0
first_tick      = True

colour_lost_count = 0
entry_errors      = []
entry_direction   = 0.0
active_colour     = None
exit_bias_value   = 0.0
colour_start_time = 0.0

try:
    while True:
        _debug['state']           = state
        _debug['entry_direction'] = entry_direction
        _debug['exit_bias_value'] = exit_bias_value
        _debug['colour_lost']     = colour_lost_count

        found_line, error, active_mode = feed.get_frame_with_overlay(running)

        now        = time.perf_counter()
        dt         = now - loop_start
        loop_start = now
        if dt <= 0:
            dt = 1e-3

        _debug['time_in_colour'] = now - colour_start_time if state == STATE_COLOUR else 0.0

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):
            print("Quit requested.")
            break

        if key == 32:
            running           = not running
            total_error       = 0.0
            last_error        = 0.0
            last_derivative   = 0.0
            first_tick        = True
            state             = STATE_BLACK
            colour_lost_count = 0
            entry_errors      = []
            entry_direction   = 0.0
            active_colour     = None
            exit_bias_value   = 0.0
            colour_start_time = 0.0
            print(f"Running: {running}")

        if not running:
            new_movement.move(0, 0)
            continue

        colour_visible = active_mode in COLOUR_MODES or active_mode in LOOKAHEAD_MODES

        if state == STATE_BLACK:
            if active_mode in COLOUR_MODES:
                state             = STATE_COLOUR
                colour_lost_count = 0
                active_colour     = active_mode
                entry_errors      = []
                colour_start_time = now
                total_error = last_error = last_derivative = 0.0
                first_tick  = True
                print(f"[STATE] BLACK → COLOUR ({active_mode})")
            elif active_mode in LOOKAHEAD_MODES:
                state         = STATE_LOOKAHEAD
                active_colour = active_mode.replace('lookahead_', '')
                entry_errors  = []
                print(f"[STATE] BLACK → LOOKAHEAD ({active_mode})")

        elif state == STATE_LOOKAHEAD:
            if active_mode in COLOUR_MODES:
                state             = STATE_COLOUR
                colour_lost_count = 0
                active_colour     = active_mode
                colour_start_time = now
                total_error = last_error = last_derivative = 0.0
                first_tick  = True
                print(f"[STATE] LOOKAHEAD → COLOUR ({active_mode})")
            elif active_mode == 'black':
                state           = STATE_BLACK
                active_colour   = None
                entry_errors    = []
                entry_direction = 0.0
                print("[STATE] LOOKAHEAD → BLACK (false positive)")

        elif state == STATE_COLOUR:
            if len(entry_errors) == 0:
                entry_direction = sign(error) if abs(error) > DEADBAND else 0.0
                entry_errors.append(error)
                _debug['entry_raw_error'] = error
                print(f"[ENTRY] motor_dir={entry_direction:+.0f} raw_error={error:+.3f} colour={active_colour}")

            time_in_colour = now - colour_start_time
            exit_ready     = time_in_colour >= MIN_COLOUR_SECS

            if exit_ready and active_mode == 'black':
                # Been on colour long enough and now sees black → exit T-junction
                exit_bias_value   = entry_direction * EXIT_BIAS_STRENGTH
                state             = STATE_EXIT_BIAS
                colour_lost_count = 0
                print(f"[STATE] COLOUR → EXIT_BIAS | bias={exit_bias_value:+.2f} time_in={time_in_colour:.1f}s")
                active_colour   = None
                entry_errors    = []
                entry_direction = 0.0
            elif colour_visible:
                # Colour seen — stay on it, reset lost counter
                colour_lost_count = 0
            else:
                # Colour lost — search by turning in entry direction
                # This keeps the car hunting for the colour line rather
                # than giving up and falling back to black
                colour_lost_count += 1

        elif state == STATE_EXIT_BIAS:
            if active_mode == 'black':
                state           = STATE_BLACK
                exit_bias_value = 0.0
                total_error = last_error = last_derivative = 0.0
                first_tick  = True
                print("[STATE] EXIT_BIAS → BLACK (line reacquired)")

        if state == STATE_LOOKAHEAD:
            current_speed = APPROACH_SPEED
        elif state == STATE_COLOUR:
            current_speed = COLOUR_SPEED
        elif state == STATE_EXIT_BIAS:
            current_speed = EXIT_SPEED
        else:
            current_speed = BASE_SPEED

        if state == STATE_EXIT_BIAS:
            pid   = exit_bias_value
            left  = clamp(current_speed - pid, -1.0, 1.0)
            right = clamp(current_speed + pid, -1.0, 1.0)

        elif state == STATE_COLOUR and not colour_visible:
            # Colour lost mid-path — turn in entry direction to search for it
            pid   = entry_direction * SEARCH_TURN
            left  = clamp(current_speed - pid, -1.0, 1.0)
            right = clamp(current_speed + pid, -1.0, 1.0)

        elif found_line:
            smoothed     = 0.7 * last_error + 0.3 * error
            biased_error = clamp(smoothed, -1.0, 1.0)

            if abs(biased_error) < DEADBAND:
                biased_error = 0.0

            total_error += biased_error * dt
            total_error  = clamp(total_error, -I_CLAMP, I_CLAMP)

            if first_tick:
                raw_derivative = 0.0
                first_tick     = False
            else:
                raw_derivative = (biased_error - last_error) / dt

            derivative      = D_FILTER * last_derivative + (1.0 - D_FILTER) * raw_derivative
            last_derivative = derivative

            pid        = KP * biased_error + KI * total_error + KD * derivative
            last_error = biased_error

            left  = clamp(current_speed - pid, -1.0, 1.0)
            right = clamp(current_speed + pid, -1.0, 1.0)

        else:
            pid             = sign(last_error) * SEARCH_TURN
            total_error    *= 0.8
            last_derivative = 0.0
            left  = clamp(current_speed - pid, -1.0, 1.0)
            right = clamp(current_speed + pid, -1.0, 1.0)

        _debug['pid']   = pid
        _debug['left']  = left
        _debug['right'] = right

        new_movement.move(left, right)

except KeyboardInterrupt:
    print("KeyboardInterrupt — stopping.")

finally:
    new_movement.move(0, 0)
    try:
        new_movement.pi.stop()
    except Exception:
        pass
    try:
        feed.close_camera()
    except Exception:
        pass
