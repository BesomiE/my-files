import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2

# ==== Motor & LED Setup ====
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

BUTTON_PIN = 17
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

motor_pins = {
    'BR_IN1': 24, 'BR_IN2': 25,
    'BL_IN3': 26, 'BL_IN4': 27,
    'ENA': 18, 'ENB': 19
}
for pin in motor_pins.values():
    GPIO.setup(pin, GPIO.OUT)

pwmA = GPIO.PWM(motor_pins['ENA'], 1000)
pwmB = GPIO.PWM(motor_pins['ENB'], 1000)
pwmA.start(0)
pwmB.start(0)

# LED Pins
RED_LED_PIN = 23
GREEN_LED_PIN = 22
YELLOW_LED_PIN = 16
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(YELLOW_LED_PIN, GPIO.OUT)

# ==== Constants and Configuration ====
FORWARD_SPEED = 100
SLOW_SPEED = 40
TURN_SPEED = 60

COLOR_MIN_PIXELS = 3000
OBJECT_CONFIDENCE_THRESHOLD = 0.7
OBJECT_NEAR_HEIGHT_THRESHOLD = 100

# Avoidance timings (seconds)
AV_BACK = 0.4        # back up a bit
AV_TURN = 1.0        # turn duration
AV_FORWARD = 0.5     # small forward to clear obstacle

# State Machine States
STATE_FORWARD = 0
STATE_STOPPED = 1
STATE_SLOWING = 2
STATE_AVOIDING = 3

current_state = STATE_STOPPED  # Start in a safe, stopped state
avoidance_start_time = 0
avoid_direction = 'right'

# ==== Helpers ====
def set_speed(speed):
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)

def stop():
    for key in ['BR_IN1', 'BR_IN2', 'BL_IN3', 'BL_IN4']:
        GPIO.output(motor_pins[key], GPIO.LOW)
    set_speed(0)

def move_forward(speed=FORWARD_SPEED):
    GPIO.output(motor_pins['BR_IN1'], GPIO.HIGH)
    GPIO.output(motor_pins['BR_IN2'], GPIO.LOW)
    GPIO.output(motor_pins['BL_IN3'], GPIO.HIGH)
    GPIO.output(motor_pins['BL_IN4'], GPIO.LOW)
    set_speed(speed)

def move_backward(speed=SLOW_SPEED):
    GPIO.output(motor_pins['BR_IN1'], GPIO.LOW)
    GPIO.output(motor_pins['BR_IN2'], GPIO.HIGH)
    GPIO.output(motor_pins['BL_IN3'], GPIO.LOW)
    GPIO.output(motor_pins['BL_IN4'], GPIO.HIGH)
    set_speed(speed)

def turn_left(speed=TURN_SPEED):
    GPIO.output(motor_pins['BR_IN1'], GPIO.HIGH)
    GPIO.output(motor_pins['BR_IN2'], GPIO.LOW)
    GPIO.output(motor_pins['BL_IN3'], GPIO.LOW)
    GPIO.output(motor_pins['BL_IN4'], GPIO.HIGH)
    set_speed(speed)

def turn_right(speed=TURN_SPEED):
    GPIO.output(motor_pins['BR_IN1'], GPIO.LOW)
    GPIO.output(motor_pins['BR_IN2'], GPIO.HIGH)
    GPIO.output(motor_pins['BL_IN3'], GPIO.HIGH)
    GPIO.output(motor_pins['BL_IN4'], GPIO.LOW)
    set_speed(speed)

def set_leds(red, green, yellow):
    GPIO.output(RED_LED_PIN, red)
    GPIO.output(GREEN_LED_PIN, green)
    GPIO.output(YELLOW_LED_PIN, yellow)

def draw_color_box(mask, color_name, draw_color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), draw_color, 2)
            cv2.putText(frame, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

# ==== Load TFLite Model and Labels ====
with open("TFLite_model_bbd/labelmap.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# RawID -> label file index remap (BDD100K-style raw order -> your file order)
# Raw: 0 person,1 rider,2 car,3 truck,4 bus,5 train,6 motorcycle,7 bicycle,8 tlight,9 tsign
# File: [0 tsign,1 tlight,2 car,3 rider,4 motor,5 person,6 bus,7 truck,8 bike,9 train]
CLASS_REMAP = [5, 3, 2, 7, 6, 9, 4, 8, 1, 0]

def get_class_name(raw_id: int) -> str:
    idx = int(raw_id)
    if 0 <= idx < len(CLASS_REMAP):
        return class_names[CLASS_REMAP[idx]]
    return f"id_{idx}"

interpreter = Interpreter(model_path="TFLite_model_bbd/detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model input settings
height = input_details[0]['shape'][1]
width  = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

# Toggle if you want to skip RGB preprocessing (not recommended)
USE_RGB_PREPROC = True

# ==== Camera Setup ====
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)}))
picam2.start()

# ==== HSV Color Ranges (BGR -> HSV pipeline) ====
lower_red1 = np.array([0, 150, 120])
upper_red1 = np.array([5, 255, 255])
lower_red2 = np.array([170, 150, 120])
upper_red2 = np.array([180, 255, 255])
lower_green = np.array([40, 70, 70])
upper_green = np.array([80, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

frame_count = 0
last_boxes, last_class_ids, last_scores, last_count = [], [], [], 0

print("Waiting for button press...")
while not GPIO.input(BUTTON_PIN) == GPIO.HIGH:
    time.sleep(0.1)
print("Button pressed. Starting...")

try:
    while True:
        # Capture frame
        request = picam2.capture_request()
        frame = request.make_array("main")     # OpenCV sees this as BGR array
        request.release()

        # --- Vision Processing ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # keep HSV on BGR
        frame_count += 1

        # Color Detection
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        red_detected = cv2.countNonZero(mask_red) > COLOR_MIN_PIXELS
        green_detected = cv2.countNonZero(mask_green) > COLOR_MIN_PIXELS
        yellow_detected = cv2.countNonZero(mask_yellow) > COLOR_MIN_PIXELS

        # Object Detection (reduced frequency for speed)
        if frame_count % 10 == 0:
            if USE_RGB_PREPROC:
                # Use true input size + RGB (improves classification stability)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small_frame = cv2.resize(frame_rgb, (width, height))
            else:
                small_frame = cv2.resize(frame, (width, height))

            input_data = np.expand_dims(small_frame, axis=0)
            if floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                input_data = input_data.astype(np.uint8)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            last_boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            last_class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
            last_scores = interpreter.get_tensor(output_details[2]['index'])[0]
            last_count = int(interpreter.get_tensor(output_details[3]['index'])[0])

        # --- Detection summary / pick nearest instances ---
        person_detected = False
        car_detected = False
        person_near = False
        car_near = False

        nearest_person = None
        nearest_person_h = 0

        nearest_car = None
        nearest_car_h = 0

        for i in range(last_count):
            if last_scores[i] > OBJECT_CONFIDENCE_THRESHOLD:
                # Clamp box to frame
                y1 = int(max(0, last_boxes[i][0] * frame.shape[0]))
                x1 = int(max(0, last_boxes[i][1] * frame.shape[1]))
                y2 = int(min(frame.shape[0], last_boxes[i][2] * frame.shape[0]))
                x2 = int(min(frame.shape[1], last_boxes[i][3] * frame.shape[1]))
                h = max(0, y2 - y1)

                detected_class_name = get_class_name(last_class_ids[i])
                print(f"[DEBUG] raw_id={int(last_class_ids[i])} name={detected_class_name} score={last_scores[i]:.2f} h={h}")

                # Draw detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                label = f"{detected_class_name}: {int(last_scores[i]*100)}%"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if detected_class_name == "person":
                    person_detected = True
                    if h > nearest_person_h:
                        nearest_person_h = h
                        nearest_person = (x1, y1, x2, y2)
                elif detected_class_name == "car":
                    car_detected = True
                    if h > nearest_car_h:
                        nearest_car_h = h
                        nearest_car = (x1, y1, x2, y2)

        # Near checks
        if nearest_person_h > OBJECT_NEAR_HEIGHT_THRESHOLD:
            person_near = True
        if nearest_car_h > OBJECT_NEAR_HEIGHT_THRESHOLD:
            car_near = True

        # --- State Machine Logic (with proper avoidance) ---
        set_leds(red=False, green=False, yellow=False)

        if current_state == STATE_AVOIDING:
            # Timed avoidance sequence; ignore detections/colors until done
            elapsed = time.time() - avoidance_start_time
            cv2.putText(frame, "STATE: AVOIDING", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            set_leds(red=True, green=False, yellow=False)

            if elapsed <= AV_BACK:
                move_backward(speed=SLOW_SPEED)
            elif elapsed <= AV_BACK + AV_TURN:
                if avoid_direction == 'right':
                    turn_right(speed=TURN_SPEED)
                else:
                    turn_left(speed=TURN_SPEED)
            elif elapsed <= AV_BACK + AV_TURN + AV_FORWARD:
                move_forward(speed=SLOW_SPEED)
            else:
                stop()
                current_state = STATE_FORWARD
                set_leds(red=False, green=False, yellow=False)

        else:
            # Not currently avoiding: priorities
            if person_near:
                current_state = STATE_STOPPED
                set_leds(red=True, green=False, yellow=False)
                print("Person very close -> STOP")

            elif car_near:
                # Start avoidance ONCE, using the actual nearest car box
                if nearest_car is not None:
                    x1, _, x2, _ = nearest_car
                    frame_cx = frame.shape[1] / 2
                    car_cx = (x1 + x2) / 2
                    avoid_direction = 'right' if car_cx < frame_cx else 'left'
                else:
                    avoid_direction = 'right'  # safe default

                current_state = STATE_AVOIDING
                avoidance_start_time = time.time()
                set_leds(red=True, green=False, yellow=False)
                print(f"Car near -> START AVOIDANCE to the {avoid_direction}")

            else:
                # No “near” threats; obey traffic lights / general caution
                if red_detected:
                    current_state = STATE_STOPPED
                    set_leds(red=True, green=False, yellow=False)
                    print("Red light -> STOP")
                elif yellow_detected or person_detected or car_detected:
                    current_state = STATE_SLOWING
                    set_leds(red=False, green=False, yellow=True)
                    print("Caution -> SLOW")
                elif green_detected:
                    current_state = STATE_FORWARD
                    set_leds(red=False, green=True, yellow=False)
                    print("Green -> GO")
                else:
                    current_state = STATE_FORWARD
                    print("Clear -> FORWARD")

        # --- Execute actions for non-avoiding states ---
        if current_state == STATE_FORWARD:
            move_forward(speed=FORWARD_SPEED)
            cv2.putText(frame, "STATE: FORWARD", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif current_state == STATE_STOPPED:
            stop()
            cv2.putText(frame, "STATE: STOPPED", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif current_state == STATE_SLOWING:
            move_forward(speed=SLOW_SPEED)
            cv2.putText(frame, "STATE: SLOWING", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # --- Drawing and Displaying (always update UI) ---
        draw_color_box(mask_red, "Red", (0, 0, 255))
        draw_color_box(mask_green, "Green", (0, 255, 0))
        draw_color_box(mask_yellow, "Yellow", (0, 255, 255))

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    stop()
    GPIO.cleanup()
finally:
    stop()
    picam2.close()
    cv2.destroyAllWindows()
    GPIO.cleanup()
