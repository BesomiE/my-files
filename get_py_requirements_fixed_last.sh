#!/bin/bash
set -e  # Exit on any error

echo "🔧 Installing system dependencies for TFLite..."
sudo apt update
sudo apt install -y python3-tflite-runtime libatlas-base-dev

echo "🐍 Upgrading pip..."
python3 -m pip install --upgrade pip

echo "📦 Installing Python packages..."
pip3 install numpy RPi.GPIO

echo "✅ Environment setup complete!"
