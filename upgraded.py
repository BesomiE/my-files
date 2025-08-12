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
OBSTACLE_AVOIDANCE_TIME = 2  # seconds

COLOR_MIN_PIXELS = 3000
OBJECT_CONFIDENCE_THRESHOLD = 0.7
OBJECT_NEAR_HEIGHT_THRESHOLD = 100

# State Machine States
STATE_FORWARD = 0
STATE_STOPPED = 1
STATE_SLOWING = 2
STATE_AVOIDING = 3

current_state = STATE_STOPPED # Start in a safe, stopped state

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

# ==== Debouncing for colors and objects ====
color_debounce = {
    "red": collections.deque(maxlen=5),
    "green": collections.deque(maxlen=5),
    "yellow": collections.deque(maxlen=5),
}
object_debounce = collections.deque(maxlen=5)

# Variables for drawing on the frame
last_boxes, last_class_ids, last_scores, last_count = [], [], [], 0

# ==== Main Loop ====
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

        # Color Detection
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        red_detected = cv2.countNonZero(mask_red) > COLOR_MIN_PIXELS
        green_detected = cv2.countNonZero(mask_green) > COLOR_MIN_PIXELS
        yellow_detected = cv2.countNonZero(mask_yellow) > COLOR_MIN_PIXELS

        color_debounce["red"].append(red_detected)
        color_debounce["green"].append(green_detected)
        color_debounce["yellow"].append(yellow_detected)

        stable_red = sum(color_debounce["red"]) > len(color_debounce["red"]) // 2
        stable_green = sum(color_debounce["green"]) > len(color_debounce["green"]) // 2
        stable_yellow = sum(color_debounce["yellow"]) > len(color_debounce["yellow"]) // 2
        
        # Object Detection (every 5 frames for efficiency)
        if len(object_debounce) % 5 == 0:
            small_frame = cv2.resize(frame, (300, 300))
            input_data = np.expand_dims(small_frame, axis=0)
            if input_details[0]['dtype'] == np.float32:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            
            interpreter.set_tensor(input_details[0]['index'], input_data.astype(input_details[0]['dtype']))
            interpreter.invoke()
            
            last_boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            last_class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
            last_scores = interpreter.get_tensor(output_details[2]['index'])[0]
            last_count = int(interpreter.get_tensor(output_details[3]['index'])[0])
            
            object_debounce.clear()
            
            found_person = False
            found_car = False
            
            for i in range(last_count):
                if last_scores[i] > OBJECT_CONFIDENCE_THRESHOLD:
                    detected_class_name = class_names[int(last_class_ids[i])]
                    y1 = int(last_boxes[i][0] * frame.shape[0])
                    y2 = int(last_boxes[i][2] * frame.shape[0])
                    
                    if detected_class_name == "person":
                        found_person = True
                        if (y2 - y1) > OBJECT_NEAR_HEIGHT_THRESHOLD:
                            current_state = STATE_STOPPED
                            print("Person very close - STOP")
                            break
                    elif detected_class_name == "car":
                        found_car = True
                        if (y2 - y1) > OBJECT_NEAR_HEIGHT_THRESHOLD:
                            smart_avoidance(last_boxes[i], frame.shape[1])
                            current_state = STATE_FORWARD
                            break
            
            if found_person:
                object_debounce.append("person")
            elif found_car:
                object_debounce.append("car")
            else:
                object_debounce.append("none")
        
        # --- State Machine Logic ---
        
        set_leds(red=False, green=False, yellow=False)

        if "person" in object_debounce:
            current_state = STATE_STOPPED
            print("Person detected - SLOWING DOWN")
            set_leds(red=False, green=False, yellow=True) # Yellow for slowing down

        elif stable_red:
            current_state = STATE_STOPPED
            print("Red light detected - STOP")
            set_leds(red=True, green=False, yellow=False)

        elif "car" in object_debounce:
            current_state = STATE_SLOWING
            print("Car detected - SLOWING DOWN")
            set_leds(red=False, green=False, yellow=True)

        elif stable_yellow:
            current_state = STATE_SLOWING
            print("Yellow light detected - SLOW")
            set_leds(red=False, green=False, yellow=True)

        elif stable_green:
            current_state = STATE_FORWARD
            print("Green light detected - GO")
            set_leds(red=False, green=True, yellow=False)

        else:
            current_state = STATE_FORWARD
            
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
