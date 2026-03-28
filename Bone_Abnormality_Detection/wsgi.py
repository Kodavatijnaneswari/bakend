import os
import sys
import logging

# Configure basic logging to ensure we see errors in Render logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Bone_Abnormality_Detection.settings')
    os.environ['DJANGO_SETTINGS_MODULE'] = 'Bone_Abnormality_Detection.settings'

    from django.core.wsgi import get_wsgi_application
    application = get_wsgi_application()
    logger.info("WSGI application loaded successfully.")
except Exception as e:
    logger.error(f"CRITICAL ERROR during WSGI loading: {e}")
    # Also print to stderr just in case
    print(f"CRITICAL ERROR: {e}", file=sys.stderr)
    raise
