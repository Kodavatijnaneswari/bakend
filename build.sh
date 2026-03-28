#!/usr/bin/env bash
# exit on error
set -o errexit

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python manage.py collectstatic --noinput
python manage.py migrate
