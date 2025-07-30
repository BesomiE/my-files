######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras & Ethan Dell
# Date: 10/27/19 & 1/24/21
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.
# ^^^^ Credit to Evan for writing this script. I modified it to interface with a GPIO button and LED and also save out images.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
import pathlib
from threading import Thread
import importlib.util
import datetime
import RPi.GPIO as GPIO

# --- GPIO Setup ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)  # LED
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Button

# --- VideoStream Class ---
class VideoStream:
    def __init__(self, resolution=(640,480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- Parse Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', required=True)
parser.add_argument('--graph', default='detect.tflite')
parser.add_argument('--labels', default='labelmap.txt')
parser.add_argument('--threshold', default=0.5)
parser.add_argument('--resolution', default='1280x720')
parser.add_argument('--edgetpu', action='store_true')
parser.add_argument('--output_path', required=True)
args = parser.parse_args()

# --- Set Up Model ---
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU and GRAPH_NAME == 'detect.tflite':
    GRAPH_NAME = 'edgetpu.tflite'

PATH_TO_CKPT = os.path.join(os.getcwd(), MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(os.getcwd(), MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT,
                          experimental_delegates=[load_delegate('libedgetpu.so.1.0')] if use_TPU else None)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# --- Main Loop ---
try:
    
    while True:
        #if not GPIO.input(17):  # Button pressed
            led_on = True
            outdir = pathlib.Path(args.output_path) / time.strftime('%Y-%m-%d_%H-%M-%S-%Z')
            outdir.mkdir(parents=True)
            GPIO.output(4, True)
            time.sleep(0.1)
            f = []

            frame_rate_calc = 1
            freq = cv2.getTickFrequency()
            videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
            time.sleep(1)

            while True:
                t1 = cv2.getTickCount()
                frame1 = videostream.read()
                frame = frame1.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (width, height))
                input_data = np.expand_dims(frame_resized, axis=0)

                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                boxes = interpreter.get_tensor(output_details[0]['index'])[0]
                classes = interpreter.get_tensor(output_details[1]['index'])[0]
                scores = interpreter.get_tensor(output_details[2]['index'])[0]

                for i in range(len(scores)):
                    if scores[i] > min_conf_threshold and scores[i] <= 1.0:
                        ymin = int(max(1, boxes[i][0] * imH))
                        xmin = int(max(1, boxes[i][1] * imW))
                        ymax = int(min(imH, boxes[i][2] * imH))
                        xmax = int(min(imW, boxes[i][3] * imW))
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                        object_name = labels[int(classes[i])]
                        label = f'{object_name}: {int(scores[i]*100)}%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        label_ymin = max(ymin, labelSize[1] + 10)
                        cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                                      (xmin + labelSize[0], label_ymin + baseLine - 10),
                                      (255, 255, 255), cv2.FILLED)
                        cv2.putText(frame, label, (xmin, label_ymin - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                t2 = cv2.getTickCount()
                time1 = (t2 - t1) / freq
                frame_rate_calc = 1 / time1
                f.append(frame_rate_calc)
                path = outdir / f"{datetime.datetime.now()}.jpg"
                cv2.imwrite(str(path), frame)

                # Uncomment to display frame
                # cv2.imshow("Object Detector", frame)

                if cv2.waitKey(1) == ord('q') or not GPIO.input(17):  # Button released or 'q'
                    print(f"Saved images to: {outdir}")
                    GPIO.output(4, False)
                    videostream.stop()
                    cv2.destroyAllWindows()
                    time.sleep(2)
                    break
finally:
    GPIO.output(4, False)
    GPIO.cleanup()
    try:
        videostream.stop()
    except:
        pass
    cv2.destroyAllWindows()
