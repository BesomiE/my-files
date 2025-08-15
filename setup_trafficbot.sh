#!/bin/bash
set -e

echo "=== Updating system ==="
sudo apt update
sudo apt full-upgrade -y

echo "=== Installing core libraries ==="
sudo apt install -y \
    libcamera-tools \
    rpicam-apps \
    python3-picamera2 \
    python3-opencv \
    python3-rpi.gpio \
    python3-numpy \
    libatlas-base-dev \
    python3-pip

echo "=== Installing TensorFlow Lite Runtime ==="
python3 -m pip install --no-cache-dir --break-system-packages tflite-runtime==2.14.0

echo "=== Enabling camera overlay ==="
sudo sed -i '/^camera_auto_detect=/d;/^dtoverlay=imx219/d' /boot/firmware/config.txt
printf "\ncamera_auto_detect=1\ndtoverlay=imx219\n" | sudo tee -a /boot/firmware/config.txt

echo "=== Adding user to groups for camera/GPU/GPIO access ==="
sudo usermod -aG video,render,input,gpio $USER

echo "=== Setup complete! ==="
echo "Reboot now to apply changes."
