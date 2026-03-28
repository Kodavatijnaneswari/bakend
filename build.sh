#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
pip install -r requirements.txt --no-cache-dir

python manage.py collectstatic --noinput
python manage.py migrate
