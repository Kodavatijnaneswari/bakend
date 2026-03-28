import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Bone_Abnormality_Detection.settings')
os.environ['DJANGO_SETTINGS_MODULE'] = 'Bone_Abnormality_Detection.settings'

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
