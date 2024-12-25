from django.contrib import admin
from django.urls import path, include
from api import views

urlpatterns = [
    path('register', views.register, name='api_register'),
    path('login', views.login, name="api_login"),
    path('login', views.logout, name="api_logout"),
    path('logs', views.logs, name="logs"),
    # path('live', views.live, name="live"),
    # path('live_classification', views.live_classification,name="live_classification"),
    # path('dashboard', views.dashboard, name="dashboard"),
    path('Classify', views.classify, name="Classify"),
    # path('load_video', views.load_video, name="load_video"),
    # path('Logout', views.Logout, name="Logout"),
    path('assign_value/<int:id>', views.assign_value, name="assign_value"),
    path('viewVideo/<int:id>', views.viewVideo, name="viewVideo"),

    # path('write_to_json', views.write_to_json, name="write_to_json"),
    # path('send_json', views.send_json, name="send_json")
]
