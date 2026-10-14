import sys
import time

import pi_servo_hat
import qwiic_oled
import qwiic_proximity

# --- Proximity range that counts as "your snack" (fill these in later) ---
# Raw readings from the VCNL4040: bigger number = closer object.
# Run qwiic_distance.py to see what values your hand/object produces.
SNACK_MIN = 100
SNACK_MAX = 300

# Readings below this mean nothing is in front of the sensor, so the screen
# is cleared instead of showing the warning. Set to 0 to always show it.
IDLE_THRESHOLD = 10

# --- Servo ---
SERVO_CH = 0
REST_ANGLE = 0    # starting position
OPEN_ANGLE = 90   # 90 degrees from rest; swap the two values if it turns the wrong way
HOLD_SECONDS = 5
# Speed of the turn: the servo steps 1 degree, then waits this long.
# 0.02 -> 90 degrees in ~1.8 s. Bigger = slower.
STEP_DELAY = 0.02

POLL_SECONDS = 0.2


def sweep(servo, start, end):
    step = 1 if end > start else -1
    for angle in range(start, end + step, step):
        servo.move_servo_position(SERVO_CH, angle)
        time.sleep(STEP_DELAY)


def show(oled, lines):
    # print() can't handle "\n" in this library, so place each line manually
    oled.clear(oled.PAGE)
    for i, line in enumerate(lines):
        oled.set_cursor(0, i * 10)
        oled.print(line)
    oled.display()


def main():
    prox = qwiic_proximity.QwiicProximity()
    if not prox.connected:
        print("Proximity sensor not connected. Check the Qwiic cable.", file=sys.stderr)
        return
    prox.begin()

    oled = qwiic_oled.QwiicOledDisplay()
    if not oled.connected:
        print("OLED not connected. Check the Qwiic cable.", file=sys.stderr)
        return
    oled.begin()
    oled.clear(oled.ALL)
    oled.set_font_type(0)  # small font: 21 chars x 3 lines on 128x32

    servo = pi_servo_hat.PiServoHat()
    servo.restart()
    servo.move_servo_position(SERVO_CH, REST_ANGLE)

    last_state = None
    while True:
        value = prox.get_proximity()
        print(f"Proximity: {value}")

        if SNACK_MIN <= value <= SNACK_MAX:
            show(oled, ["Hi Enjoy Your Snack!"])
            sweep(servo, REST_ANGLE, OPEN_ANGLE)
            time.sleep(HOLD_SECONDS)
            sweep(servo, OPEN_ANGLE, REST_ANGLE)
            time.sleep(0.5)  # let the servo settle before reading again
            last_state = "snack"
        elif value < IDLE_THRESHOLD:
            if last_state != "idle":
                show(oled, [])
                last_state = "idle"
        else:
            if last_state != "wrong":
                show(oled, ["THIS IS NOT", "YOUR SNACK!"])
                last_state = "wrong"

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
