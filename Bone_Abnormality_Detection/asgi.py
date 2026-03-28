import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Bone_Abnormality_Detection.settings')
    os.environ['DJANGO_SETTINGS_MODULE'] = 'Bone_Abnormality_Detection.settings'

    from django.core.asgi import get_asgi_application
    application = get_asgi_application()
    logger.info("ASGI application loaded successfully.")
except Exception as e:
    logger.error(f"CRITICAL ERROR during ASGI loading: {e}")
    print(f"CRITICAL ERROR: {e}", file=sys.stderr)
    raise
