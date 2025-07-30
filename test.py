import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2

# ==== Motor Setup ====
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

def set_speed(speed):
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)

def stop():
    for key in ['BR_IN1', 'BR_IN2', 'BL_IN3', 'BL_IN4']:
        GPIO.output(motor_pins[key], GPIO.LOW)
    set_speed(0)

def move_forward(speed=100):
    GPIO.output(motor_pins['BR_IN1'], GPIO.HIGH)
    GPIO.output(motor_pins['BR_IN2'], GPIO.LOW)
    GPIO.output(motor_pins['BL_IN3'], GPIO.HIGH)
    GPIO.output(motor_pins['BL_IN4'], GPIO.LOW)
    set_speed(speed)

def move_backward(speed=60):
    GPIO.output(motor_pins['BR_IN1'], GPIO.LOW)
    GPIO.output(motor_pins['BR_IN2'], GPIO.HIGH)
    GPIO.output(motor_pins['BL_IN3'], GPIO.LOW)
    GPIO.output(motor_pins['BL_IN4'], GPIO.HIGH)
    set_speed(speed)

def turn_left(speed=60):
    GPIO.output(motor_pins['BR_IN1'], GPIO.HIGH)
    GPIO.output(motor_pins['BR_IN2'], GPIO.LOW)
    GPIO.output(motor_pins['BL_IN3'], GPIO.LOW)
    GPIO.output(motor_pins['BL_IN4'], GPIO.HIGH)
    set_speed(speed)

def turn_right(speed=60):
    GPIO.output(motor_pins['BR_IN1'], GPIO.LOW)
    GPIO.output(motor_pins['BR_IN2'], GPIO.HIGH)
    GPIO.output(motor_pins['BL_IN3'], GPIO.HIGH)
    GPIO.output(motor_pins['BL_IN4'], GPIO.LOW)
    set_speed(speed)

# ==== Load Labels ====
with open("TFLite_model_bbd/labelmap.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ==== Load TFLite Model ====
interpreter = Interpreter(model_path="TFLite_model_bbd/detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==== Camera Setup ====
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)}))
picam2.start()

# ==== HSV Color Ranges ====
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
lower_green = np.array([40, 70, 70])
upper_green = np.array([80, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

frame_count = 0
last_class_ids, last_scores, last_count = [], [], 0
yellow_active = False
obstacle_mode = False
last_obstacle_time = 0

def draw_color_box(mask, color_name, draw_color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), draw_color, 2)
            cv2.putText(frame, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

print("Waiting for button press...")
while not GPIO.input(BUTTON_PIN) == GPIO.HIGH:
    time.sleep(0.1)
print("Button pressed. Starting...")

try:
    while True:
        request = picam2.capture_request()
        frame = request.make_array("main")
        request.release()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_count += 1

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        red_detected = cv2.countNonZero(mask_red) > 3000
        green_detected = cv2.countNonZero(mask_green) > 3000
        yellow_detected = cv2.countNonZero(mask_yellow) > 3000

        draw_color_box(mask_red, "Red", (0, 0, 255))
        draw_color_box(mask_green, "Green", (0, 255, 0))
        draw_color_box(mask_yellow, "Yellow", (0, 255, 255))

        if frame_count % 5 == 0:
            small_frame = cv2.resize(frame, (300, 300))
            input_data = np.expand_dims(small_frame, axis=0).astype(np.uint8)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
            count = int(interpreter.get_tensor(output_details[3]['index'])[0])

            last_class_ids, last_scores, last_count = class_ids, scores, count
        else:
            class_ids, scores, count = last_class_ids, last_scores, last_count

        person_detected = False
        person_near = False
        car_detected = False
        car_detected_near = False

        for i in range(count):
            if scores[i] > 0.7:
                x1 = int(boxes[i][1] * frame.shape[1])
                y1 = int(boxes[i][0] * frame.shape[0])
                x2 = int(boxes[i][3] * frame.shape[1])
                y2 = int(boxes[i][2] * frame.shape[0])
                detected_class_name = class_names[int(class_ids[i])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                label = f"{detected_class_name}: {int(scores[i]*100)}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if detected_class_name == "person":
                    person_detected = True
                    if (y2 - y1) > 100:
                        person_near = True
                elif detected_class_name == "car":
                    car_detected = True
                    if (y2 - y1) > 100:
                        car_detected_near = True

        # === Intelligent Decision Logic ===
        if person_near:
            stop()
            print("Person is very close - stopping")
            cv2.putText(frame, "Person Close - STOP", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        if person_detected:
            move_forward(speed=40)
            print("Person detected - slowing down")
            cv2.putText(frame, "Person Detected - SLOW", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            continue

        if red_detected:
            stop()
            print("Red light - stop")
            cv2.putText(frame, "Red Light - STOP", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue

        if car_detected_near:
            move_backward(speed=50)
            time.sleep(0.3)
            turn_right(speed=60)
            time.sleep(0.3)
            obstacle_mode = True
            last_obstacle_time = time.time()
            print("Car very close - avoiding")
            cv2.putText(frame, "Car Close - AVOID", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            

        if car_detected:
            move_forward(speed=50)
            print("Car detected - slowing down")
            cv2.putText(frame, "Car Detected - SLOW", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            continue

        if obstacle_mode:
            turn_right(speed=60)
            if time.time() - last_obstacle_time > 2:
                obstacle_mode = False
            print("Avoiding obstacle...")
            continue

        if yellow_detected:
            move_forward(speed=40)
            yellow_active = True
            print("Yellow light - slowing down")
            cv2.putText(frame, "Yellow Light - SLOW", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            continue

        if yellow_active and not yellow_detected:
            move_forward(speed=100)
            yellow_active = False
            print("Yellow gone - back to full speed")
            cv2.putText(frame, "Speed back to normal", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            continue

        if green_detected:
            move_forward(speed=100)
            print("Green light - moving forward")
            cv2.putText(frame, "Green Light - GO", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            continue

        # Default
        move_forward(speed=100)

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

