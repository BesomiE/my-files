import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import collections

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

# State Machine States
STATE_FORWARD = 0
STATE_STOPPED = 1
STATE_SLOWING = 2
STATE_AVOIDING = 3

current_state = STATE_STOPPED # Start in a safe, stopped state
avoidance_start_time = 0
avoid_direction = 'right'

# ==== Motor & LED Control Functions ====
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

def smart_avoidance(car_box, frame_width):
    x1, _, x2, _ = car_box
    car_center_x = (x1 + x2) / 2
    frame_center_x = frame_width / 2
    
    # Simple avoidance sequence
    set_leds(red=True, green=False, yellow=False)
    
    if car_center_x < frame_center_x:
        print("Car detected on the left. Avoiding to the right.")
        move_backward(speed=SLOW_SPEED)
        time.sleep(0.4)
        turn_right(speed=TURN_SPEED)
        time.sleep(1.0)
        move_forward(speed=SLOW_SPEED)
        time.sleep(0.5)
    else:
        print("Car detected on the right. Avoiding to the left.")
        move_backward(speed=SLOW_SPEED)
        time.sleep(0.4)
        turn_left(speed=TURN_SPEED)
        time.sleep(1.0)
        move_forward(speed=SLOW_SPEED)
        time.sleep(0.5)
    
    stop()
    print("Avoidance maneuver complete. Resuming normal operations.")
    set_leds(red=False, green=False, yellow=False)

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

interpreter = Interpreter(model_path="TFLite_model_bbd/detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==== Camera Setup ====
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)}))
picam2.start()

# ==== HSV Color Ranges ====
lower_red1 = np.array([0, 150, 120])
upper_red1 = np.array([5, 255, 255])
lower_red2 = np.array([170, 150, 120])
upper_red2 = np.array([180, 255, 255])
lower_green = np.array([40, 70, 70])
upper_green = np.array([80, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Using variables from the old code for drawing, but with improved state management
frame_count = 0
last_boxes, last_class_ids, last_scores, last_count = [], [], [], 0

print("Waiting for button press...")
while not GPIO.input(BUTTON_PIN) == GPIO.HIGH:
    time.sleep(0.1)
print("Button pressed. Starting...")

try:
    while True:
        request = picam2.capture_request()
        frame = request.make_array("main")
        request.release()

        # --- Vision Processing ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_count += 1

        # Color Detection
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        red_detected = cv2.countNonZero(mask_red) > COLOR_MIN_PIXELS
        green_detected = cv2.countNonZero(mask_green) > COLOR_MIN_PIXELS
        yellow_detected = cv2.countNonZero(mask_yellow) > COLOR_MIN_PIXELS
        
        # Object Detection is the same as your old code
        if frame_count % 10 == 0:
            small_frame = cv2.resize(frame, (300, 300))
            input_data = np.expand_dims(small_frame, axis=0)
            if input_details[0]['dtype'] == np.float32:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                input_data = input_data.astype(np.uint8)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            last_boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            last_class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
            last_scores = interpreter.get_tensor(output_details[2]['index'])[0]
            last_count = int(interpreter.get_tensor(output_details[3]['index'])[0])
        
        # Check for detections and set state variables
        person_detected = False
        car_detected = False
        person_near = False
        car_near = False

        for i in range(last_count):
            if last_scores[i] > OBJECT_CONFIDENCE_THRESHOLD:
                detected_class_name = class_names[int(last_class_ids[i])]
                
                # Use bounding box for near-detection logic
                y1 = int(last_boxes[i][0] * frame.shape[0])
                y2 = int(last_boxes[i][2] * frame.shape[0])
                
                if detected_class_name == "person":
                    person_detected = True
                    if (y2 - y1) > OBJECT_NEAR_HEIGHT_THRESHOLD:
                        person_near = True
                elif detected_class_name == "car":
                    car_detected = True
                    if (y2 - y1) > OBJECT_NEAR_HEIGHT_THRESHOLD:
                        car_near = True
        
        # --- State Machine Logic (simplified) ---
        set_leds(red=False, green=False, yellow=False)

        if current_state == STATE_AVOIDING:
            # Continue avoidance maneuver for a set duration
            if time.time() - avoidance_start_time > AVOIDANCE_TIME:
                current_state = STATE_FORWARD
            else:
                # Execute the turn
                if avoid_direction == 'right':
                    turn_right(speed=TURN_SPEED)
                else:
                    turn_left(speed=TURN_SPEED)

        elif person_near:
            # Corrected behavior: Stop, do NOT avoid
            current_state = STATE_STOPPED
            set_leds(red=False, green=False, yellow=True)
            print("Person is very close - stopping")
        elif car_near:
            # Start avoidance maneuver
            current_state = STATE_AVOIDING
            set_leds(red=True, green=False, yellow=False)
            avoidance_start_time = time.time()
            print("Car detected near - starting avoidance")
            
            # Determine avoidance direction based on car's position
            x1 = int(last_boxes[0][1] * frame.shape[1])
            x2 = int(last_boxes[0][3] * frame.shape[1])
            car_center_x = (x1 + x2) / 2
            frame_center_x = frame.shape[1] / 2
            avoid_direction = 'right' if car_center_x < frame_center_x else 'left'
        elif red_detected:
            current_state = STATE_STOPPED
            set_leds(red=True, green=False, yellow=False)
            print("Red light detected - STOP")
        elif yellow_detected:
            current_state = STATE_SLOWING
            set_leds(red=False, green=False, yellow=True)
            print("Yellow light detected - SLOW")
        elif green_detected:
            current_state = STATE_FORWARD
            set_leds(red=False, green=True, yellow=False)
            print("Green light detected - GO")
        else:
            current_state = STATE_FORWARD
            print("No significant detections - moving forward")
        
        # Execute actions based on current state
        if current_state == STATE_FORWARD:
            move_forward(speed=FORWARD_SPEED)
            cv2.putText(frame, "STATE: FORWARD", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif current_state == STATE_STOPPED:
            stop()
            cv2.putText(frame, "STATE: STOPPED", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif current_state == STATE_SLOWING:
            move_forward(speed=SLOW_SPEED)
            cv2.putText(frame, "STATE: SLOWING", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # --- Drawing and Displaying ---
        draw_color_box(mask_red, "Red", (0, 0, 255))
        draw_color_box(mask_green, "Green", (0, 255, 0))
        draw_color_box(mask_yellow, "Yellow", (0, 255, 255))

        for i in range(last_count):
            if last_scores[i] > OBJECT_CONFIDENCE_THRESHOLD:
                x1 = int(last_boxes[i][1] * frame.shape[1])
                y1 = int(last_boxes[i][0] * frame.shape[0])
                x2 = int(last_boxes[i][3] * frame.shape[1])
                y2 = int(last_boxes[i][2] * frame.shape[0])
                detected_class_name = class_names[int(last_class_ids[i])]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                label = f"{detected_class_name}: {int(last_scores[i]*100)}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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
