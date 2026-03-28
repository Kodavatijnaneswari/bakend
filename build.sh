#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
pip install -r requirements.txt --no-cache-dir

# Pre-download YOLO weights for performance
mkdir -p media
if [ ! -f media/yolov8s.pt ]; then
    echo "Pre-downloading YOLOv8s weights..."
    curl -L https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt -o media/yolov8s.pt
fi

python manage.py collectstatic --noinput
python manage.py migrate --noinput
