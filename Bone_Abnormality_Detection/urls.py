"""
URL configuration for Bone_Abnormality_Detection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include, re_path
from django.views.static import serve
from admins import views
from users import views as u
from users import api_views as api

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('', include('users.urls')),

    # API Endpoints for React Native
    path('api/login/', api.LoginAPIView.as_view(), name='api_login'),
    path('api/detect/', api.DetectionAPIView.as_view(), name='api_detect'),
    path('api/history/<int:userid>/', api.HistoryAPIView.as_view(), name='api_history'),

    path('', u.index, name='index'),
    path('upload_image/', u.upload_image, name='upload_image'),
    path('result/', u.show_result, name='show_result'),
    path('userbase/', u.userbase, name='userbase'),
    path('userlogin/', u.userlogin, name='userlogin'),
    path('training/', u.training, name='training'),
    path('history/', u.history, name='history'),
    path('generate_report/<int:result_id>/', u.generate_report, name='generate_report'),

    path('register/', views.register, name='register'),
    path('view/', views.view, name='view'),
    path('activate/<int:id>', views.activate, name='activate'),
    path('block1/<int:id>', views.block, name='block1'),
    path('adminhome/', views.adminhome, name='adminhome'),
    path('adminlogin/', views.adminlogin, name='adminlogin'),
    path('adminbase/', views.adminbase, name='adminbase'),
    path('delete/<int:id>', views.delete, name='delete'),

    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_ROOT}),
]