#!/bin/bash
echo "[INFO] Installing project requirements..."

# Update & upgrade
sudo apt update && sudo apt upgrade -y

# Install Raspberry Pi camera dependencies
sudo apt install -y python3-picamera2 libcamera-apps v4l-utils

# Install pip if not installed
sudo apt install -y python3-pip

# Install Python dependencies
if [ -f requirements.txt ]; then
    pip3 install -r requirements.txt
else
    echo "[WARN] requirements.txt not found — skipping."
fi

echo "[INFO] Installation complete."
